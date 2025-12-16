'''
NOT DEVELOPED
Supervised:
- UNet
- Conformer + Decoder
Self-Supervised:
- Conformer (frozen) + Linear Probe: (lin)
- Conformer (frozen) + Decoder: (dec)
- Conformer + Decoder: (full)
'''

import os
import random
import copy
import math
import sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR, CosineAnnealingLR, LambdaLR
import pickle
import wandb

from data_utils import make_wf_dataloaders
from utils import *
from mae_utils import *
from model import ConformerModel, HybridHead
from detection_training import *
from hybrid_training import *

from waveforms.waveforms_module.make_waveform import BatchedLightSimulation
from flash_detection.hybrid_loss import overall_class_acc, overall_class_purity, mined_bce_loss
from flash_detection.evaluation import *

############### SETTINGS ###########################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_PERCENTAGE = 0.1

BACKBONE_CKPT = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mae_con_200k/11.pth"
CURRENT_CKPT_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/con_scratch_0.3/best_photon_model.pth"
SINGLE_FLASH_PATH = '/sdf/home/c/carsmith/sdf_data/self_supervised_flash/data/100ksingle_flash_benchmark.npy'

finetune_type = None
supervised_arc = "conformer"
disable = True

# Define configuration
config = {
    "single_flash": {
        "merged_window_width": True  # Set appropriately
    }
}

batches_per_photon = 2  # Define this based on your data

freeze_backbone = True  # set everybody to frozen 
        
if finetune_type is not None:
    # Self-supervised fine-tuning
    backbone = ConformerModel()
    backbone.to(device)
    checkpoint = torch.load(BACKBONE_CKPT, weights_only=True)
    backbone.load_state_dict(checkpoint['model_state_dict'])

    if finetune_type == "lin":
        freeze_backbone = True
        head = HybridLinearProbe()
        head.to(device)
        head.train()
        optimizable_params = list(head.parameters())  # Convert to list
        
    elif finetune_type == "dec":
        freeze_backbone = True
        head = HybridUNetDecoder()
        head.to(device)
        head.train()
        optimizable_params = list(head.parameters())  # Convert to list
        
    elif finetune_type == "full":
        freeze_backbone = False
        head = HybridUNetDecoder()
        head.to(device)
        head.train()
        backbone.train()
        optimizable_params = list(backbone.parameters()) + list(head.parameters())
        
    else:
        raise ValueError(f"Invalid finetune_type: {finetune_type}")
    
else:  # Supervised training from scratch (finetune_type is None)
    freeze_backbone = False
    if supervised_arc == "conformer":
        backbone = ConformerModel()
        head = HybridUNetDecoder()
        backbone.to(device)
        head.to(device)
        backbone.train()
        head.train()
        optimizable_params = list(backbone.parameters()) + list(head.parameters())
        
    elif supervised_arc == "unet":
        # TODO: Implement UNet
        raise NotImplementedError("UNet not implemented yet")
    else:
        raise ValueError(f"Invalid supervised_arc: {supervised_arc}")

# Freeze backbone if needed
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

####################### UTILS ##############################################
def labelled_collate_fn(norm=False):  # Provide default value
    def collate(batch):
        waveforms, arrival_times, hit_times, photon_bins, photon_list, last_phot_list, label_list = zip(*batch)

        waveforms = torch.stack(waveforms, dim=0)
        token_labels = torch.stack(label_list, dim=0)

        if norm:
            waveforms = (waveforms - waveforms.mean(dim=1, keepdim=True)) / \
                         (waveforms.std(dim=1, keepdim=True) + 1e-8)
        waveforms = waveforms.unsqueeze(1)

        arrival_times = torch.stack(arrival_times, dim=0).unsqueeze(1)
        photon_bins = torch.stack(photon_bins, dim=0).unsqueeze(1)

        hit_times = [torch.tensor(ht, dtype=torch.float32) for ht in hit_times]
        photon_list = [torch.tensor(pl, dtype=torch.float32) for pl in photon_list]
        last_phot_list = [torch.tensor(lp, dtype=torch.float32) for lp in last_phot_list]

        return waveforms, arrival_times, hit_times, photon_bins, photon_list, last_phot_list, token_labels
    return collate
    
def make_sf_dataloader(path=SINGLE_FLASH_PATH, seed=42, batch_size=50, shuffle=False, norm=False):
    load_wfs = np.load(path, allow_pickle=True)
    full_data = load_wfs.item()
    dataset = WaveformDataset(full_data)
    collate_func = labelled_collate_fn(norm)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        collate_fn=collate_func,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    return loader

# SINGLE FLASH BENCHMARK
merge_loader = make_sf_dataloader()
if finetune_type == None:
    finetune_type == "scratch"
model_name = f"{supervised_arc}_{finetune_type}_{DATA_PERCENTAGE}" if finetune_type else supervised_arc

single_phot_results = {
    "merge_acc": [],
    "merge_pure": [],
    "interval": [],
    "reco_frac": []
}

print("Single Flash Loader Length: ", len(merge_loader))       

epochs = 1
for epoch in range(epochs):
    acc_progress = tqdm(merge_loader, desc=f"Scanning {epoch+1}/{epochs}", leave=False, position=0)

    # temporary accumulators
    interval_bins = []
    reco_frac = 0.0
    merged_acc = 0.0
    merged_pure = 0.0
    avg_interval = None
    batch_count = 0

    with torch.no_grad():
        for i, (data, target, hit_times, photon_target, photon_list, _, _) in enumerate(acc_progress):
            data, target = data.to(device), target.to(device)
            data = torch.arcsinh(data / 5.0)
            
            embeddings = backbone(data, mode="just_embeddings")
            reg_output, class_output = head(embeddings)
            merged_mask = merge_bins(class_output, skip_tol=5)

            # interval widths
            for b in range(merged_mask.shape[0]):
                mask_row = merged_mask[b, 0]
                intervals = mask_to_intervals(mask_row)
                widths = [(e - s + 1) for (s, e) in intervals]
                interval_bins.extend(widths)

            merged_acc += merged_class_acc(merged_mask, hit_times, device)
            merged_pure += merged_class_purity(merged_mask, hit_times, device)
            batch_count += 1

            if (i + 1) % batches_per_photon == 0:
                reco_frac /= batches_per_photon
                merged_acc /= batches_per_photon
                merged_pure /= batches_per_photon
                
                if config["single_flash"]["merged_window_width"]: 
                    avg_interval = np.mean(interval_bins) if interval_bins else 0.0

                single_phot_results["interval"].append(avg_interval)
                single_phot_results["merge_acc"].append(merged_acc)
                single_phot_results["merge_pure"].append(merged_pure)

                # reset accumulators
                interval_bins = []
                reco_frac = 0.0
                merged_acc = 0.0
                merged_pure = 0.0
                    
np.save(f"{model_name}_single_flash.npy", single_phot_results, allow_pickle=True)