import copy
import math
import pickle
import sys
sys.path.append('..')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F

from tqdm import tqdm

from data_utils import *
from linear_probe import *
from flash_detection.hybrid_loss import *
import flash_detection.evaluation
from flash_detection.evaluation import *

def _compute_grad_norm(parameters, norm_type=2.0):
    """Compute gradient norm for given parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    
    if norm_type == float('inf'):
        total_norm = max(g.abs().max() for g in grads)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g, norm_type) for g in grads]), 
            norm_type
        )
    
    return total_norm

def mae_pretrain(model, train_loader, val_loader, optimizer, scheduler, epochs, device, logger=None):
    results = {
        'train_loss': [],
        'eval_acc': []
    }

    for epoch in range(epochs):
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)
        for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(train_progress):
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)

            #### TRANSFORMING WITH ARCSINH
            data = torch.arcsinh(data / 5)
            
            x_recon, target, mask = model(data, mode='masked_reco')
            mask_per_token = mask.mean(dim=-1) 
            mask_wave = mask_per_token.repeat_interleave(10, dim=1).unsqueeze(1) # token size

            ########### New 11.29 - Weighted loss for token classes ##############################
            class_weights = torch.tensor([1.0, 100.0, 2.0]).to(device)  # [background, rising_edge, tail]
            token_weights = class_weights[token_labels]
            token_weights_wave = token_weights.unsqueeze(1).repeat_interleave(10, dim=2)
            
            # reco_error = ((x_recon - target) ** 2) * mask_wave
            # masked_loss = reco_error.sum() / mask_wave.sum().clamp(min=1)
            reco_error = ((x_recon - target) ** 2) * mask_wave * token_weights_wave
            masked_loss = reco_error.sum() / (mask_wave * token_weights_wave).sum().clamp(min=1)
            ######################################################################################
            
            # update weights
            masked_loss.backward() 
            optimizer.step()

            # Logging
            train_loss += masked_loss.item()
            train_progress.set_postfix({"train_loss": train_loss/(i+1)})

            if (i % 100) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch*(len(train_loader)),
                    "train_loss": train_loss / (i+1),
                    "grad_norm": _compute_grad_norm(model.parameters()),
                    "lr": optimizer.param_groups[0]['lr'],
                })
            optimizer.zero_grad()

        train_loss /= len(train_loader) 
        results['train_loss'].append(train_loss)

        # Evaluation - Train Probe
        if (epoch + 1) % 1 == 0:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            val_classifier = NonlinearValClassifier(d_model=256).to(device)
            val_optimizer = torch.optim.Adam(val_classifier.parameters(), lr=1e-3)
        
            ### ADDED 11.29 - WEIGHT tail & rising edge POSITIVE CLASSES in CrossEntropy, not BCE
            class_counts = torch.tensor([0.53, 0.02, 0.45]) # bg, rise, tail
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # Train for a single epoch
            val_classifier.train()
            for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(val_loader):
                data, token_labels = data.to(device), token_labels.to(device)
                data = torch.arcsinh(data / 5) # use same transformation as in training
                
                val_optimizer.zero_grad()
                
                # Extract embeddings from frozen student network
                with torch.no_grad():
                    embeddings = model(data, mode='just_embeddings')  # [B, L, d_model]
                
                logits = val_classifier(embeddings)                     # [B, L, 1]
                logits_flat = logits.view(-1, logits.size(-1))  # [B*L, num_classes]
                labels_flat = token_labels.view(-1).long()  # [B*L]
                
                loss = criterion(logits_flat, labels_flat)
                
                loss.backward()
                val_optimizer.step()

            # Evaluation - Evaluate Nonlinear Probe
            val_classifier.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for i, (data, _, _, _, _, _, token_labels) in enumerate(val_loader):
                    data, token_labels = data.to(device), token_labels.to(device)
                    data = torch.arcsinh(data / 5) # same transformation as in training
                    embeddings = model(data, mode='just_embeddings')
                    logits = val_classifier(embeddings)
                    preds = torch.argmax(logits, dim=-1) 
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(token_labels.cpu())
            
            all_preds = torch.cat(all_preds, dim=0).flatten()
            all_labels = torch.cat(all_labels, dim=0).flatten()
            
            accuracy = (all_preds == all_labels).float().mean()
            logger.log({"val_accuracy": accuracy})
            
            ## log class-wise accuracy
            for class_idx in range(3):
                class_mask = all_labels == class_idx
                if class_mask.sum() > 0:
                    class_acc = (all_preds[class_mask] == all_labels[class_mask]).float().mean()
                    logger.log({f"val_accuracy_class_{class_idx}": class_acc})

            results['eval_acc'].append(accuracy)

        if (epoch + 1) % 2 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mae_60mask_full/{epoch}.pth")
        
        scheduler.step()
            
    return results   


                
                

            
            

        
                
            
        

    

    
