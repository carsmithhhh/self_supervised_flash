#!/usr/bin/env python3
"""
Fine-tune a frozen Conformer backbone with a HybridHead on waveform data.

Usage:
    python finetune.py
"""

import os
import random
import copy
import math
import pickle
import sys
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt  # kept if you later want plotting
import wandb

from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR, CosineAnnealingLR, LambdaLR

# local project imports — ensure your PYTHONPATH is set so these resolve
from data_utils import make_wf_dataloaders
from utils import *
from mae_utils import *
from model import ConformerModel, HybridHead
from waveforms.waveforms_module.make_waveform import BatchedLightSimulation
from flash_detection.hybrid_loss import overall_class_acc, overall_class_purity, mined_bce_loss
import flash_detection.evaluation
from torch.cuda.amp import autocast


# --------------------------
# Configuration
# --------------------------
SEED = 42
BATCH_SIZE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/data/200k_3labels.npy"
CHECKPOINT_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/hybrid_contrast/5.pth"

EPOCHS = 50
LR = 1e-4

# W&B config (set to None to skip logging)
WANDB_PROJECT = "dino_waveforms"
WANDB_RUN_NAME = "finetune_cont3_frozen_40k"


# --------------------------
# Reproducibility
# --------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

g = torch.Generator()
g.manual_seed(SEED)


# --------------------------
# Utilities
# --------------------------
def _compute_grad_norm(parameters, norm_type: float = 2.0):
    """Compute gradient norm for given parameters (returns python float)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    if norm_type == float("inf"):
        total_norm = max(g.abs().max().item() for g in grads)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type).item()
    return float(total_norm)

# --------------------------
# Training fn
# --------------------------
def fine_tune(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int = 50,
    unfreeze_after_epoch: int = 10,
    backbone_lr: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 10,
    logger = None,
):
    
    # Freeze backbone initially
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head.train()
    head.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    results = {
        "train_loss": [], "train_acc": [], "train_pure": [], "train_reg_rmse": [],
        "eval_loss": [], "eval_acc": [], "eval_pure": [], "eval_reg_rmse": [],
    }

    for epoch in range(epochs):
        # Gradual unfreezing
        # if epoch == unfreeze_after_epoch:
        #     print(f"Unfreezing backbone at epoch {epoch}")
        #     backbone.train()
        #     for p in backbone.parameters():
        #         p.requires_grad = True
        #     optimizer.add_param_group({'params': backbone.parameters(), 'lr': backbone_lr})
        
        # Set modes
        # if epoch < unfreeze_after_epoch:
        #     backbone.eval()
        # else:
        #     backbone.train()
        backbone.eval()
        head.train()
        
        # Regression loss warmup
        reg_weight = min(1.0, epoch / 5.0)
        
        running_loss = 0.0
        running_acc = 0.0
        running_pure = 0.0
        running_reg_rmse = 0.0
        batches = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (train)", leave=False, disable=True)

        for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(train_iter):
            data = data.to(device)
            photon_target = photon_target.to(device)
            data = torch.arcsinh(data / 5.0)

            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                if epoch < unfreeze_after_epoch:
                    with torch.no_grad():
                        embeddings = backbone(data, mode="just_embeddings")
                else:
                    embeddings = backbone(data, mode="just_embeddings")
                
                class_output, reg_output = head(embeddings)

                # Size adjustment (should be unnecessary with proper head design)
                if class_output.shape[-1] != data.shape[-1]:
                    diff = data.shape[-1] - class_output.shape[-1]
                    if diff > 0:
                        class_output = F.pad(class_output, (0, diff))
                        reg_output = F.pad(reg_output, (0, diff))
                    else:
                        class_output = class_output[..., : data.shape[-1]]
                        reg_output = reg_output[..., : data.shape[-1]]

                # Compute loss with regression weight
                loss_tuple = mined_loss(data, hit_times, photon_list, class_output, 
                                           reg_output, epoch, device, include_reg=True,
                                           reg_weight=reg_weight)
                loss = loss_tuple[0] if isinstance(loss_tuple, (tuple, list)) else loss_tuple

            # Backward with mixed precision
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     [p for p in list(head.parameters()) + (list(backbone.parameters()) if epoch >= unfreeze_after_epoch else []) if p.requires_grad],
            #     max_norm=grad_clip
            # )
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # Metrics (outside autocast for numerical stability)
            with torch.no_grad():
                acc = overall_class_acc(hit_times, class_output, device)
                purity = overall_class_purity(hit_times, class_output, device)
                try:
                    reg_rmse = regression_rmse(hit_times, photon_target, reg_output, class_output, device)
                except Exception:
                    reg_rmse = 0.0

            running_loss += loss.item()
            running_acc += acc
            running_pure += purity
            running_reg_rmse += float(reg_rmse)
            batches += 1

            train_iter.set_postfix({
                "loss": running_loss / batches,
                "acc": running_acc / batches,
                "purity": running_pure / batches,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            })

        # Epoch averages
        train_loss = running_loss / batches if batches > 0 else 0.0
        train_acc = running_acc / batches if batches > 0 else 0.0
        train_pure = running_pure / batches if batches > 0 else 0.0
        train_reg_rmse = running_reg_rmse / batches if batches > 0 else 0.0

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_pure"].append(train_pure)
        results["train_reg_rmse"].append(train_reg_rmse)

        # Validation
        backbone.eval()
        head.eval()
        eval_batches = 0
        eval_loss = 0.0
        eval_acc = 0.0
        eval_pure = 0.0
        eval_reg_rmse = 0.0
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (val)", leave=False, disable=True)
            for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(val_iter):
                data = data.to(device)
                photon_target = photon_target.to(device)
                data = torch.arcsinh(data / 5.0)
                
                with autocast():
                    embeddings = backbone(data, mode="just_embeddings")
                    class_output, reg_output = head(embeddings)

                    if class_output.shape[-1] != data.shape[-1]:
                        diff = data.shape[-1] - class_output.shape[-1]
                        if diff > 0:
                            class_output = F.pad(class_output, (0, diff))
                            reg_output = F.pad(reg_output, (0, diff))
                        else:
                            class_output = class_output[..., : data.shape[-1]]
                            reg_output = reg_output[..., : data.shape[-1]]

                    loss_tuple = mined_loss(data, hit_times, photon_list, class_output, 
                                               reg_output, epoch, device, include_reg=True)
                    loss_val = loss_tuple[0] if isinstance(loss_tuple, (tuple, list)) else loss_tuple

                acc = overall_class_acc(hit_times, class_output, device)
                purity = overall_class_purity(hit_times, class_output, device)
                try:
                    reg_rmse = regression_rmse(hit_times, photon_target, reg_output, class_output, device)
                except Exception:
                    reg_rmse = 0.0

                eval_loss += loss_val.item()
                eval_acc += acc
                eval_pure += purity
                eval_reg_rmse += float(reg_rmse)
                eval_batches += 1

        eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0.0
        eval_acc = eval_acc / eval_batches if eval_batches > 0 else 0.0
        eval_pure = eval_pure / eval_batches if eval_batches > 0 else 0.0
        eval_reg_rmse = eval_reg_rmse / eval_batches if eval_batches > 0 else 0.0

        results["eval_loss"].append(eval_loss)
        results["eval_acc"].append(eval_acc)
        results["eval_pure"].append(eval_pure)
        results["eval_reg_rmse"].append(eval_reg_rmse)

        # Logging
        if logger is not None:
            logger.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_pure": train_pure,
                "train_reg_rmse": train_reg_rmse,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "eval_pure": eval_pure,
                "eval_reg_rmse": eval_reg_rmse,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "reg_weight": reg_weight,
            })

        # LR scheduler (if ReduceLROnPlateau)
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(eval_loss)

        # Early stopping
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'head_state_dict': head.state_dict(),
                'backbone_state_dict': backbone.state_dict() if epoch >= unfreeze_after_epoch else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': eval_loss,
            }, 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return results


# --------------------------
# Main
# --------------------------
def main():
    # Data loaders
    train_loader, val_loader, test_loader = make_wf_dataloaders(
        DATA_PATH, batch_size=BATCH_SIZE, val_ratio=0.1, test_ratio=0.7, generator=g, norm=False
    )

    # Load backbone and checkpoint
    backbone = ConformerModel()
    device = torch.device(DEVICE)
    backbone.to(device)
    ckpt = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            backbone.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")
    else:
        print("Warning: checkpoint path does not exist, starting from random backbone weights")

    # Create head and optimizer (train only head)
    head = HybridHead()
    head.to(device)

    # optimizer = optim.Adam(head.parameters(), lr=LR)
    # scheduler = None  # optionally create a scheduler
    optimizer = optim.Adam(head.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # W&B init (optional)
    logger = None
    if WANDB_PROJECT is not None:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
            reinit=True,
        )
        wandb.watch(head, log="all", log_freq=100)
        logger = wandb_run

    # Print head parameter count
    total_params = sum(p.numel() for p in head.parameters())
    print(f"Head parameters: {total_params:,}")

    # Run fine-tuning
    results = fine_tune(backbone, head, train_loader, val_loader, optimizer, scheduler, device, epochs=EPOCHS, logger=logger)

    # Save results locally
    out_path = "finetune_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {out_path}")

    if logger is not None:
        logger.finish()


def mined_loss(
    data, 
    hit_times, 
    photon_list, 
    class_output, 
    reg_output, 
    epoch, 
    device, 
    include_reg=True,
    reg_weight=1.0,  # Add this parameter for warmup scheduling
    logger=None
):
    """
    Computes classification and regression loss with hard negative mining.
    
    Args:
        data: Input waveform [N, L] or [N, 1, L]
        hit_times: List of hit times for each sample
        photon_list: List of photon counts for each hit
        class_output: Classification logits [N, 1, L]
        reg_output: Regression output [N, 1, L]
        epoch: Current epoch (unused but kept for compatibility)
        device: Device to put tensors on
        include_reg: Whether to include regression loss
        reg_weight: Weight for regression loss (for warmup)
        logger: Optional logger
    
    Returns:
        tuple: (total_loss, sampled_indices, masked_target, masked_class_output, 
                class_output, target, masked_reg_output, masked_photon_target)
    """
    
    # Ensure data is 2D
    if data.dim() == 3:
        data = data.squeeze(1)
    
    N, L = data.shape
    
    # Initialize targets
    target = torch.zeros((N, L), dtype=torch.float32, device=device)
    photon_target = torch.zeros((N, L), dtype=torch.float32, device=device)
    sampled_indices = torch.zeros((N, L), dtype=torch.bool, device=device)
    
    # Constants
    wf_width = 900  # 900 ns window for hard negatives
    num_hard_neg = 500
    num_random_neg = 100
    offset = 0
    
    rng = np.random.default_rng()
    
    # Build targets and sample indices
    for i, times in enumerate(hit_times):
        # Handle various empty cases
        if times is None:
            continue
        
        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        elif isinstance(times, (list, np.ndarray)):
            times = np.asarray(times).flatten()
            if len(times) == 0 or np.all(times < 0):
                continue
        else:
            continue
        
        # Process valid hit times
        hit_indices = []
        for j, t in enumerate(times):
            if t < 0:
                continue
            
            t_idx = int(np.clip(t + offset, 0, L - 1))
            
            # Set targets
            target[i, t_idx] = 1.0
            
            # Handle photon counts safely
            if j < len(photon_list[i]):
                photon_num = photon_list[i][j]
                photon_target[i, t_idx] = float(photon_num)
            
            sampled_indices[i, t_idx] = True
            hit_indices.append(t_idx)
        
        if len(hit_indices) == 0:
            continue
        
        # Hard negative mining: negatives within wf_width of any hit
        wf_neg_candidates = set()
        for t_idx in hit_indices:
            start = max(0, t_idx)
            end = min(L, t_idx + wf_width + 1)
            wf_neg_candidates.update(range(start, end))
        
        # Remove actual hit indices from candidates
        wf_neg_candidates.difference_update(hit_indices)
        wf_neg_candidates = list(wf_neg_candidates)
        
        if len(wf_neg_candidates) > 0:
            n_sample = min(num_hard_neg, len(wf_neg_candidates))
            chosen_wf_neg = rng.choice(wf_neg_candidates, size=n_sample, replace=False)
            sampled_indices[i, chosen_wf_neg] = True
        
        # Random negative mining: negatives far from any hit
        all_indices = set(range(L))
        forbidden = set(hit_indices).union(wf_neg_candidates)
        random_neg_candidates = list(all_indices - forbidden)
        
        if len(random_neg_candidates) > 0:
            n_sample = min(num_random_neg, len(random_neg_candidates))
            chosen_rand_neg = rng.choice(random_neg_candidates, size=n_sample, replace=False)
            sampled_indices[i, chosen_rand_neg] = True
    
    # Extract masked predictions and targets for classification
    masked_class_output = class_output.squeeze(1)[sampled_indices]
    masked_target = target[sampled_indices]
    
    # ============================================
    # Classification Loss
    # ============================================
    if masked_target.numel() == 0:
        class_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        # Dynamic positive weighting to handle class imbalance
        n_pos = masked_target.sum().item()
        n_total = masked_target.numel()
        n_neg = n_total - n_pos
        
        if n_pos > 0:
            pos_weight_val = n_neg / n_pos
        else:
            pos_weight_val = 1.0
        
        # Clamp to reasonable range to avoid extreme weights
        pos_weight_val = np.clip(pos_weight_val, 1.0, 100.0)
        
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        class_loss = criterion(masked_class_output, masked_target)
    
    # ============================================
    # Regression Loss
    # ============================================
    reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    masked_reg_output = torch.tensor([], device=device)
    masked_photon_target = torch.tensor([], device=device)
    
    if include_reg:
        # Only compute regression loss where classification predicts positive
        with torch.no_grad():
            # Use sigmoid to get probabilities
            class_probs = torch.sigmoid(class_output.squeeze(1))
            reg_mask = class_probs > 0.5
        
        if reg_mask.any():
            masked_reg_output = reg_output.squeeze(1)[reg_mask]
            masked_photon_target = photon_target[reg_mask]
            
            if masked_reg_output.numel() > 0:
                # Apply softplus to ensure positive outputs
                # (if your head doesn't already do this)
                positive_reg_output = F.softplus(masked_reg_output)
                
                # Poisson NLL loss for count data
                # log_input=False because we're passing raw counts (after softplus)
                reg_criterion = torch.nn.PoissonNLLLoss(
                    log_input=False, 
                    full=False,  # Use simplified version
                    reduction='mean'
                )
                reg_loss = reg_criterion(positive_reg_output, masked_photon_target)
                
                # Alternative: Huber loss if Poisson isn't working well
                # reg_criterion = torch.nn.HuberLoss(delta=1.0)
                # reg_loss = reg_criterion(positive_reg_output, masked_photon_target)
    
    # ============================================
    # Combined Loss
    # ============================================
    scale_factor = 0.1 * reg_weight  # Apply warmup weight
    total_loss = class_loss + scale_factor * reg_loss
    
    # Optional: Log loss components
    if logger is not None:
        logger.log({
            'class_loss': class_loss.item(),
            'reg_loss': reg_loss.item(),
            'pos_weight': pos_weight_val if masked_target.numel() > 0 else 0,
            'n_sampled': masked_target.numel(),
            'n_positive': masked_target.sum().item() if masked_target.numel() > 0 else 0,
            'n_reg_samples': masked_reg_output.numel(),
        })
    
    return (
        total_loss, 
        sampled_indices, 
        masked_target, 
        masked_class_output, 
        class_output, 
        target, 
        masked_reg_output, 
        masked_photon_target
    )

def regression_rmse(hit_times, photon_bins, reg_output, class_output, device, 
                   use_mse=False, conf_threshold=0.5):
    """
    Compute regression RMSE over detected photon bins with high classification confidence.
    
    Parameters:
        hit_times: list/array of true hit indices per sample (length B)
        photon_bins: [B, L] or [B, 1, L] ground truth photon counts
        reg_output: [B, 1, L] per-bin regression predictions
        class_output: [B, 1, L] per-bin classification logits
        device: torch device
        use_mse: if True, use MSE instead of RMSE
        conf_threshold: only evaluate bins where sigmoid(class_output) > threshold
    
    Returns:
        float: average RMSE/MSE across all valid samples
    """
    
    with torch.no_grad():
        # Handle dimensions
        if reg_output.dim() == 3:
            reg_output = reg_output.squeeze(1)  # [B, L]
        if photon_bins.dim() == 3:
            photon_bins = photon_bins.squeeze(1)  # [B, L]
        if class_output.dim() == 3:
            class_output = class_output.squeeze(1)  # [B, L]
        
        reg_output = reg_output.to(device)
        photon_bins = photon_bins.to(device)
        class_output = class_output.to(device)
        
        B, L = reg_output.shape
        
        # Get classification probabilities
        class_probs = torch.sigmoid(class_output)
        
        batch_errors = []
        
        for i in range(B):
            # Get true hit indices
            if hit_times[i] is None:
                continue
            
            if torch.is_tensor(hit_times[i]):
                true_hit_idx = hit_times[i].detach().cpu().numpy().flatten()
            elif np.isscalar(hit_times[i]):
                true_hit_idx = [hit_times[i]]
            else:
                true_hit_idx = np.asarray(hit_times[i]).flatten()
            
            true_hit_idx = [int(t) for t in true_hit_idx if t >= 0 and t < L]
            
            if len(true_hit_idx) == 0:
                continue
            
            # Only evaluate at hits where model is confident
            confident_mask = class_probs[i, true_hit_idx] > conf_threshold
            
            if not confident_mask.any():
                continue
            
            # Filter to confident predictions
            confident_hits = torch.tensor(true_hit_idx, device=device)[confident_mask]
            
            preds = F.softplus(reg_output[i, confident_hits])
            targets = photon_bins[i, confident_hits].float()
            
            squared_error = (preds - targets) ** 2
            mse_val = torch.mean(squared_error)
            
            if use_mse:
                batch_errors.append(mse_val.item())
            else:
                rmse_val = torch.sqrt(mse_val + 1e-8)
                batch_errors.append(rmse_val.item())
        
        return sum(batch_errors) / len(batch_errors) if len(batch_errors) > 0 else 0.0

if __name__ == "__main__":
    main()
