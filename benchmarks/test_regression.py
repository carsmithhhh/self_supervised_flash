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
from hybrid_training import *
from waveforms.waveforms_module.make_waveform import BatchedLightSimulation
from flash_detection.hybrid_loss import overall_class_acc, overall_class_purity, mined_bce_loss
from flash_detection.evaluation import *

############### SETTINGS ###########################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_PERCENTAGE = 0.3

SINGLE_FLASH_PATH = '/sdf/home/c/carsmith/sdf_data/self_supervised_flash/data/100ksingle_flash_benchmark.npy'
BACKBONE_CKPT = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mae_con_200k/11.pth"
CURRENT_CKPT_PATH = "/sdf/home/c/carsmith/sdf_data/SSF_finetune/con_scratch_0.3/best_photon_model.pth"

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

def mask_to_intervals(mask_b):
    """
    Extract intervals from binary mask.
    Returns list of (start, end) tuples where both are INCLUSIVE indices.
    """
    intervals = []
    in_interval = False
    start = 0
    T = len(mask_b)
    
    for i in range(T):
        if mask_b[i] == 1 and not in_interval:
            in_interval = True
            start = i
        elif mask_b[i] == 0 and in_interval:
            in_interval = False
            intervals.append((start, i-1))  # INCLUSIVE end
    
    if in_interval:
        intervals.append((start, T-1))  # INCLUSIVE end
    
    return intervals


def create_interval_ids(merged_mask):
    """
    Convert binary mask to interval IDs for vectorized operations.
    Returns: (B, T) tensor where each position has its interval ID (or -1 if not in interval)
    """
    B, _, T = merged_mask.shape
    interval_ids = torch.full((B, T), -1, dtype=torch.long, device=merged_mask.device)
    
    for b in range(B):
        mask_b = merged_mask[b, 0]
        intervals = mask_to_intervals(mask_b)
        
        for interval_id, (start, end) in enumerate(intervals):
            interval_ids[b, start:end+1] = interval_id
    
    return interval_ids


def confidence_weighted_photons_in_intervals_vectorized(reg_output, class_output, merged_mask,
                                                        keep_grads=False, method='softmax'):
    """
    Vectorized version: Aggregate photon predictions using classification confidence.
    
    reg_output:  (B, 1, T)
    class_output: (B, 1, T)
    merged_mask: (B, 1, T)
    """
    B, _, T = reg_output.shape
    device = reg_output.device
    
    # Compute confidence scores (B, 1, T)
    if method == 'softmax':
        confidence = class_output  # Will apply softmax per interval later
    elif method == 'sigmoid':
        confidence = torch.sigmoid(class_output)
    else:
        confidence = torch.sigmoid(class_output)
    
    # Flatten to (B, T)
    reg_flat = reg_output.squeeze(1)  # (B, T)
    conf_flat = confidence.squeeze(1)  # (B, T)
    
    # Create interval IDs: (B, T) with -1 for non-interval positions
    interval_ids = create_interval_ids(merged_mask)
    
    results = []
    
    for b in range(B):
        reg_b = reg_flat[b]  # (T,)
        conf_b = conf_flat[b]  # (T,)
        ids_b = interval_ids[b]  # (T,)
        
        # Get valid positions (in an interval)
        valid_mask = ids_b >= 0
        if not valid_mask.any():
            results.append(torch.tensor([], device=device))
            continue
        
        valid_ids = ids_b[valid_mask]  # (N_valid,)
        valid_reg = reg_b[valid_mask]  # (N_valid,)
        valid_conf = conf_b[valid_mask] if method != 'softmax' else class_output[b, 0][valid_mask]  # (N_valid,)
        
        num_intervals = valid_ids.max().item() + 1
        
        if method == 'softmax':
            # Compute softmax per interval using scatter_max for normalization
            # First, get max logit per interval for numerical stability
            max_logits = torch.full((num_intervals,), float('-inf'), device=device)
            max_logits.scatter_reduce_(0, valid_ids, valid_conf, reduce='amax', include_self=False)
            
            # Subtract max for stability
            stable_conf = valid_conf - max_logits[valid_ids]
            exp_conf = torch.exp(stable_conf)
            
            # Sum exp per interval
            sum_exp = torch.zeros(num_intervals, device=device)
            sum_exp.scatter_add_(0, valid_ids, exp_conf)
            
            # Normalize to get softmax weights
            weights = exp_conf / (sum_exp[valid_ids] + 1e-8)
            
        elif method == 'sigmoid' or method is None:
            # Normalize confidence within each interval
            sum_conf = torch.zeros(num_intervals, device=device)
            sum_conf.scatter_add_(0, valid_ids, valid_conf)
            weights = valid_conf / (sum_conf[valid_ids] + 1e-8)
            
        elif method == 'weighted_max':
            # Find argmax confidence per interval
            max_conf_per_interval = torch.full((num_intervals,), float('-inf'), device=device)
            max_idx_per_interval = torch.zeros(num_intervals, dtype=torch.long, device=device)
            
            for i in range(len(valid_ids)):
                interval_id = valid_ids[i].item()
                if valid_conf[i] > max_conf_per_interval[interval_id]:
                    max_conf_per_interval[interval_id] = valid_conf[i]
                    max_idx_per_interval[interval_id] = i
            
            # Create one-hot weights
            weights = torch.zeros_like(valid_conf)
            for interval_id in range(num_intervals):
                idx = max_idx_per_interval[interval_id]
                weights[idx] = 1.0
                
        elif method == 'top_k':
            # For each interval, find top-k and weight by confidence
            interval_preds = []
            for interval_id in range(num_intervals):
                interval_mask = valid_ids == interval_id
                interval_conf = valid_conf[interval_mask]
                interval_reg = valid_reg[interval_mask]
                
                k = min(3, len(interval_conf))
                top_k_vals, top_k_idx = torch.topk(interval_conf, k)
                top_k_preds = interval_reg[top_k_idx]
                top_k_weights = top_k_vals / (top_k_vals.sum() + 1e-8)
                
                interval_preds.append(torch.sum(top_k_preds * top_k_weights))
            
            if not keep_grads:
                results.append(torch.stack([p.detach() for p in interval_preds]))
            else:
                results.append(torch.stack(interval_preds))
            continue
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Weighted sum: multiply and scatter_add
        weighted_reg = valid_reg * weights
        interval_sums = torch.zeros(num_intervals, device=device)
        interval_sums.scatter_add_(0, valid_ids, weighted_reg)
        
        if not keep_grads:
            results.append(interval_sums.detach())
        else:
            results.append(interval_sums)
    
    return results


def max_photons_in_intervals_vectorized(photon_target, merged_mask, keep_grads=False):
    """
    Vectorized version: Find max photon count within each interval.
    
    photon_target: (B, 1, T)
    merged_mask: (B, 1, T)
    """
    B, _, T = photon_target.shape
    device = photon_target.device
    
    # Flatten
    target_flat = photon_target.squeeze(1).float()  # (B, T)
    
    # Create interval IDs
    interval_ids = create_interval_ids(merged_mask)
    
    results = []
    
    for b in range(B):
        target_b = target_flat[b]  # (T,)
        ids_b = interval_ids[b]  # (T,)
        
        # Get valid positions
        valid_mask = ids_b >= 0
        if not valid_mask.any():
            results.append(torch.tensor([], device=device))
            continue
        
        valid_ids = ids_b[valid_mask]
        valid_target = target_b[valid_mask]
        
        num_intervals = valid_ids.max().item() + 1
        
        # Scatter max
        interval_maxs = torch.full((num_intervals,), float('-inf'), device=device)
        interval_maxs.scatter_reduce_(0, valid_ids, valid_target, reduce='amax', include_self=False)
        
        if not keep_grads:
            results.append(interval_maxs.detach())
        else:
            results.append(interval_maxs)
    
    return results


# SINGLE FLASH BENCHMARK
merge_loader = make_sf_dataloader()
model_name = f"{supervised_arc}_{finetune_type}_reg_{DATA_PERCENTAGE}" if finetune_type else supervised_arc

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
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)
            data = torch.arcsinh(data / 5.0)
            
            embeddings = backbone(data, mode="just_embeddings")
            reg_output, class_output = head(embeddings)
                
            merged_mask = merge_bins(class_output, skip_tol=5)

            # Compute interval widths using the SAME function
            for b in range(merged_mask.shape[0]):
                mask_row = merged_mask[b, 0]
                intervals = mask_to_intervals(mask_row)
                widths = [(end - start + 1) for (start, end) in intervals]
                interval_bins.extend(widths)

            # Both functions now use the same interval extraction
            interval_pred_sums = confidence_weighted_photons_in_intervals_vectorized(
                reg_output, class_output, merged_mask,
                keep_grads=False, method='sigmoid'
            )
            
            interval_true_sums = max_photons_in_intervals_vectorized(
                photon_target, merged_mask, keep_grads=False
            )
            
            # Safe concatenation - they're already tensors from torch.stack
            pred = torch.cat([x for x in interval_pred_sums if len(x) > 0]).float()
            true = torch.cat([x for x in interval_true_sums if len(x) > 0]).float()
            
            # They should match now!
            assert len(pred) == len(true), f"Mismatch at batch {i}: pred={len(pred)}, true={len(true)}"
            
            mask = (true > 0) & torch.isfinite(pred) & torch.isfinite(true)
        
            if mask.any():
                batch_reco_frac = torch.mean(pred[mask] / true[mask]).item()
                reco_frac += batch_reco_frac
            
            # Debug output
            print(f"pred: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
            print(f"true: min={true.min():.6f}, max={true.max():.6f}, mean={true.mean():.6f}")
            print(f"mask.sum(): {mask.sum()} out of {len(pred)}")
            if mask.any():
                ratio = pred[mask] / true[mask]
                print(f"ratio: min={ratio.min():.6f}, max={ratio.max():.6f}, mean={ratio.mean():.6f}")

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
                single_phot_results["reco_frac"].append(reco_frac)
                single_phot_results["merge_acc"].append(merged_acc)
                single_phot_results["merge_pure"].append(merged_pure)

                # reset accumulators
                interval_bins = []
                reco_frac = 0.0
                merged_acc = 0.0
                merged_pure = 0.0
                    
np.save(f"{model_name}_single_flash.npy", single_phot_results, allow_pickle=True)
