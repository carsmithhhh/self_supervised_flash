'''
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
from waveforms.waveforms_module.make_waveform import BatchedLightSimulation
from flash_detection.hybrid_loss import overall_class_acc, overall_class_purity, mined_bce_loss
import flash_detection.evaluation

############### SETTINGS ###########################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/data/200k_3labels.npy"
BACKBONE_CKPT = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mae_con_200k/11.pth"
CURRENT_CKPT_PATH = "/sdf/home/c/carsmith/sdf_data/SSF_finetune/con_full_0.9/"
os.makedirs(CURRENT_CKPT_PATH, exist_ok=True)

finetune_type = "full"
supervised_arc = "conformer"
disable = True

DATA_PERCENTAGE = 0.9
SEED = 42
EPOCHS = 100
VAL_EPOCHS = 2
LR = 1e-4
BATCH_SIZE = 50

WANDB_PROJECT = "hybrid_FT"
WANDB_RUN_NAME = "con_full_0.9"

####################################################################################################

def train_both(backbone, head, train_loader, val_loader, optimizer, scheduler, 
                    device, epochs, freeze_backbone=True, finetune_type="lin", 
                    logger=None, val_epochs=2, log_interval=100, checkpoint_path=None, 
                    max_grad_norm=1.0, use_poisson_loss=True, disable_tqdm=disable):
    """
    Train photon regression model with proper logging and checkpointing.
    
    Args:
        backbone: Encoder backbone model
        head: Regression head (linear probe or decoder)
        freeze_backbone: Whether to freeze backbone during training
        finetune_type: "lin" for linear probe, "dec" or "full" for dual-head decoder
        logger: W&B logger or None
        log_interval: Log metrics every N iterations
        checkpoint_path: Directory to save checkpoints (None = no checkpoints)
        max_grad_norm: Maximum gradient norm for clipping
        use_poisson_loss: Use Poisson NLL (True) for linear probe
        disable_tqdm: Disable progress bars
    """
    results = {
        "train_loss": [], 
        "train_det_loss": [], 
        "train_reg_loss": [],
        "train_mae": [],
        "train_rel_err": [],
        "train_det_true_acc": [],
        "eval_loss": [], 
        "eval_det_loss": [],
        "eval_reg_loss": [],
        "eval_mae": [],
        "eval_rel_err": [],
        "eval_det_true_acc": []
    }
    
    global_step = 0
    best_val_mae = float('inf')
    
    # Choose loss function for linear probe
    if finetune_type == "lin":
        if use_poisson_loss:
            linear_loss_fn = photon_regression_poisson_loss
            print("Using Poisson NLL loss for linear probe")
        else:
            # linear_loss_fn = photon_regression_log_loss
            print("Using log-scale MSE loss for linear probe")

    for epoch in range(epochs):
        # Set training modes
        if not freeze_backbone:
            backbone.train()
        else:
            backbone.eval()
        head.train()
        
        # Training loop accumulators
        epoch_loss = 0.0
        epoch_detection_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_mae = 0.0
        epoch_rel_err = 0.0
        epoch_det_true_acc = 0.0
        
        running_loss = 0.0
        running_detection_loss = 0.0
        running_reg_loss = 0.0
        running_mae = 0.0
        running_rel_err = 0.0
        running_det_true_acc = 0.0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (train)", 
                         leave=False, disable=disable_tqdm)
        
        all_batch_ids = set()
        
        for batch_idx, (data, _, hit_times, photon_target, photon_list, _, _) in enumerate(train_iter):
            all_batch_ids.add(batch_idx)
            data = data.to(device)
            data = torch.arcsinh(data / 5.0)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = backbone(data, mode="just_embeddings")
            
            # Dual-head decoder: regression + detection
            reg_output, class_output = head(embeddings)
            loss, detection_loss, regression_loss, mae, relative_error, det_true_acc = photon_regression_dual_head_loss(
                reg_output, class_output, hit_times, photon_list, data, device
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                trainable_params = [p for p in list(head.parameters()) + 
                                   (list(backbone.parameters()) if not freeze_backbone else []) 
                                   if p.requires_grad and p.grad is not None]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            
            # Compute gradient norm (after clipping)
            grad_norm = _compute_grad_norm(
                [p for p in list(head.parameters()) + 
                 (list(backbone.parameters()) if not freeze_backbone else []) 
                 if p.requires_grad and p.grad is not None]
            )
            
            optimizer.step()
            
            # Accumulate metrics
            batch_loss = loss.item()
            batch_det_loss = detection_loss.item()
            batch_reg_loss = regression_loss.item()
            batch_mae = mae.item()
            batch_rel_err = relative_error.item()
            batch_det_true_acc = det_true_acc.item()
            
            epoch_loss += batch_loss
            epoch_detection_loss += batch_det_loss
            epoch_reg_loss += batch_reg_loss
            epoch_mae += batch_mae
            epoch_rel_err += batch_rel_err
            epoch_det_true_acc += batch_det_true_acc
            
            running_loss += batch_loss
            running_detection_loss += batch_det_loss
            running_reg_loss += batch_reg_loss
            running_mae += batch_mae
            running_rel_err += batch_rel_err
            running_det_true_acc += batch_det_true_acc
            
            global_step += 1
            
            # Log every log_interval iterations
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_det_loss = running_detection_loss / log_interval
                avg_reg_loss = running_reg_loss / log_interval
                avg_mae = running_mae / log_interval
                avg_rel_err = running_rel_err / log_interval
                avg_det_true_acc = running_det_true_acc / log_interval
                
                if logger is not None:
                    log_dict = {
                        "train/total_loss": avg_loss,
                        "train/reg_loss": avg_reg_loss,
                        "train/mae": avg_mae,
                        "train/rel_err": avg_rel_err,
                        "train/det_true_acc": avg_det_true_acc,
                        "train/grad_norm": grad_norm,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "global_step": global_step,
                        "epoch": epoch
                    }
                    if finetune_type != "lin":
                        log_dict["train/det_loss"] = avg_det_loss
                    logger.log(log_dict)
                
                train_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'mae': f'{avg_mae:.2f}',
                    'rel_err': f'{avg_rel_err:.2%}',
                    'det_true_acc': f'{avg_det_true_acc:.2%}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                # Reset running metrics
                running_loss = 0.0
                running_detection_loss = 0.0
                running_reg_loss = 0.0
                running_mae = 0.0
                running_rel_err = 0.0
                running_det_true_acc = 0.0
        
        # End of epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_det_loss = epoch_detection_loss / len(train_loader)
        avg_epoch_reg_loss = epoch_reg_loss / len(train_loader)
        avg_epoch_mae = epoch_mae / len(train_loader)
        avg_epoch_rel_err = epoch_rel_err / len(train_loader)
        avg_epoch_det_true_acc = epoch_det_true_acc / len(train_loader)
        
        results["train_loss"].append(avg_epoch_loss)
        results["train_det_loss"].append(avg_epoch_det_loss)
        results["train_reg_loss"].append(avg_epoch_reg_loss)
        results["train_mae"].append(avg_epoch_mae)
        results["train_rel_err"].append(avg_epoch_rel_err)
        results["train_det_true_acc"].append(avg_epoch_det_true_acc)
        
        # Validation
        if epoch % val_epochs == 0:
            backbone.eval()
            head.eval()
            eval_loss = 0.0
            eval_detection_loss = 0.0
            eval_det_true_acc = 0.0
            eval_reg_loss = 0.0
            eval_mae = 0.0
            eval_rel_err = 0.0
            
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (val)", 
                               leave=False, disable=disable_tqdm)
                
                for data, _, hit_times, photon_target, photon_list, _, _ in val_iter:
                    data = data.to(device)
                    data = torch.arcsinh(data / 5.0)
                    
                    # Forward pass
                    embeddings = backbone(data, mode="just_embeddings")
                    reg_output, class_output = head(embeddings)
                    loss, detection_loss, regression_loss, mae, relative_error, det_true_acc = photon_regression_dual_head_loss(
                        reg_output, class_output, hit_times, photon_list, data, device
                    )
                    
                    eval_loss += loss.item()
                    eval_detection_loss += detection_loss.item()
                    eval_reg_loss += regression_loss.item()
                    eval_det_true_acc += det_true_acc.item()
                    eval_mae += mae.item()
                    eval_rel_err += relative_error.item()
            
            # Average validation metrics
            avg_eval_loss = eval_loss / len(val_loader)
            avg_eval_det_loss = eval_detection_loss / len(val_loader)
            avg_eval_reg_loss = eval_reg_loss / len(val_loader)
            avg_eval_mae = eval_mae / len(val_loader)
            avg_eval_rel_err = eval_rel_err / len(val_loader)
            avg_eval_det_true_acc = eval_det_true_acc / len(val_loader)
            
            results["eval_loss"].append(avg_eval_loss)
            results["eval_det_loss"].append(avg_eval_det_loss)
            results["eval_reg_loss"].append(avg_eval_reg_loss)
            results["eval_mae"].append(avg_eval_mae)
            results["eval_rel_err"].append(avg_eval_rel_err)
            results["eval_det_true_acc"].append(avg_eval_det_true_acc)

            if logger is not None:
                log_dict = {
                    "val/total_loss": avg_eval_loss,
                    "val/reg_loss": avg_eval_reg_loss,
                    "val/mae": avg_eval_mae,
                    "val/rel_err": avg_eval_rel_err,
                    "val/det_true_acc": avg_eval_det_true_acc,
                    "global_step": global_step,
                    "epoch": epoch
                }
                if finetune_type != "lin":
                    log_dict["val/det_loss"] = avg_eval_det_loss
                logger.log(log_dict)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, Train MAE={avg_epoch_mae:.2f}, "
                  f"Val Loss={avg_eval_loss:.4f}, Val MAE={avg_eval_mae:.2f}, Val Rel Err={avg_eval_rel_err:.2%}")
            
            # Save best model based on MAE (most interpretable for regression)
            if checkpoint_path and avg_eval_mae < best_val_mae:
                best_val_mae = avg_eval_mae
                torch.save({
                    'epoch': epoch,
                    'backbone_state_dict': backbone.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': avg_eval_loss,
                    'val_mae': avg_eval_mae,
                    'val_rel_err': avg_eval_rel_err,
                    'val_det_true_acc': avg_eval_det_true_acc,
                    'freeze_backbone': freeze_backbone,
                    'finetune_type': finetune_type,
                }, os.path.join(checkpoint_path, "best_photon_model.pth"))
                print(f"✓ Saved best model (val_mae: {best_val_mae:.2f})")
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs validation metric
                if epoch % val_epochs == 0:
                    scheduler.step(avg_eval_mae)  # Use MAE for scheduling
            else:
                scheduler.step()
        
        # Save periodic checkpoint
        if checkpoint_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth"))

    return results
    

############################# Loss ###################################################
def photon_regression_poisson_loss(photon_pred, hit_times, photon_targets, device):
    """
    Poisson NLL loss for photon regression - statistically appropriate for photon counting.
    """
    N, _, L = photon_pred.shape
    
    photon_target = torch.zeros((N, L), dtype=torch.float32, device=device)
    regression_mask = torch.zeros((N, L), dtype=torch.bool, device=device)
    
    for i, times in enumerate(hit_times):
        if times is None or len(times) == 0:
            continue
        
        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        else:
            times = np.asarray(times).flatten()
        
        for j, t in enumerate(times):
            if t < 0:
                continue
            
            t_idx = int(np.clip(t, 0, L - 1))
            photon_target[i, t_idx] = photon_targets[i][j]
            regression_mask[i, t_idx] = True
    
    if regression_mask.sum() > 0:
        pred_photons = photon_pred.squeeze(1)[regression_mask]
        true_photons = photon_target[regression_mask]
        
        # Poisson NLL loss
        # PyTorch expects log predictions, so take log of predicted rates
        # Add small epsilon to prevent log(0)
        log_pred = torch.log(pred_photons + 1e-8)
        loss = F.poisson_nll_loss(log_pred, true_photons, log_input=True, full=False, reduction='mean')
        
        # Metrics in linear scale
        mae = F.l1_loss(pred_photons, true_photons)
        relative_error = ((pred_photons - true_photons).abs() / (true_photons + 1e-6)).mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        mae = torch.tensor(0.0, device=device)
        relative_error = torch.tensor(0.0, device=device)
    
    return loss, mae, relative_error


def photon_regression_dual_head_loss(photon_pred, detection_logits, hit_times, photon_targets, data, device):
    """
    Combined loss for photon regression and detection using Poisson NLL for regression.
    
    Args:
        photon_pred: [B, 1, L] predicted photon counts (must be positive)
        detection_logits: [B, 1, L] detection logits (if dual-head)
        hit_times: list of hit times per batch
        photon_targets: [B, num_hits] actual photon counts at each hit
        data: [B, 1, L] input waveform
    """
    data = data.squeeze(1)
    N, L = data.shape
    
    # Create target tensors
    detection_target = torch.zeros((N, L), dtype=torch.float32, device=device)
    photon_target = torch.zeros((N, L), dtype=torch.float32, device=device)
    regression_mask = torch.zeros((N, L), dtype=torch.bool, device=device)
    
    for i, times in enumerate(hit_times):
        if times is None or len(times) == 0:
            continue
        
        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        else:
            times = np.asarray(times).flatten()
        
        for j, t in enumerate(times):
            if t < 0:
                continue
            
            t_idx = int(np.clip(t, 0, L - 1))
            detection_target[i, t_idx] = 1.0
            photon_target[i, t_idx] = photon_targets[i][j]  # Ground truth photon count
            regression_mask[i, t_idx] = True
    
    # Detection loss (BCE) - only where we have signals
    # detection_loss = F.binary_cross_entropy_with_logits(
    #     detection_logits.squeeze(1),
    #     detection_target,
    #     reduction='mean'
    # )
    detection_loss, det_true_acc = mined_bce_loss(data, hit_times, detection_logits)
    
    # Regression loss (Poisson NLL) - only at signal locations
    if regression_mask.sum() > 0:
        pred_photons = photon_pred.squeeze(1)[regression_mask]
        true_photons = photon_target[regression_mask]
        
        # Poisson NLL loss - statistically appropriate for photon counting
        # PyTorch's poisson_nll_loss expects log of the predicted rate when log_input=True
        log_pred = torch.log(pred_photons + 1e-8)  # Add epsilon to prevent log(0)
        regression_loss = F.poisson_nll_loss(
            log_pred, 
            true_photons, 
            log_input=True,  # Indicates we're passing log predictions
            full=False,  # Don't include Stirling approximation (faster)
            reduction='mean'
        )
        
        # Compute metrics
        mae = F.l1_loss(pred_photons, true_photons)
        relative_error = ((pred_photons - true_photons).abs() / (true_photons + 1e-6)).mean()
    else:
        regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mae = torch.tensor(0.0, device=device)
        relative_error = torch.tensor(0.0, device=device)
    
    # Combined loss with weighting
    # Note: Poisson NLL is already on a reasonable scale, but you may want to tune this
    total_loss = detection_loss + 1.0 * regression_loss  # Adjust weight as needed
    
    return total_loss, detection_loss, regression_loss, mae, relative_error, det_true_acc

def mined_bce_loss(data, hit_times, class_output):
    data = data.squeeze(1)
    N, L = data.shape
    target = torch.zeros((N, L), dtype=torch.float32).to(device)
    wf_width = 700 # rough
    rng = np.random.default_rng()

    sampled_indices = torch.zeros((N, L), dtype=torch.bool, device=device)

    for i, times in enumerate(hit_times):
        if (
            times is None 
            or (isinstance(times, (list, np.ndarray)) and len(times) == 0)
            or (isinstance(times, (list, np.ndarray)) and np.all(np.array(times) < 0))
        ):
            continue  # no flashes

        if torch.is_tensor(times):
            times = times.detach().cpu().numpy().flatten()
        elif np.isscalar(times):
            times = [times]
        else:
            times = np.asarray(times).flatten()

        hit_indices = []
        for j, t in enumerate(times):
            if t < 0:
                pass
            else:
                t_idx = int(np.clip(t, 0, L - 1))
                target[i, t_idx] = 1.0
                sampled_indices[i, t_idx] = True
                hit_indices.append(t_idx)

        # Hard negative mining: 500 negative bins within wf_width of any hit time (but not the hit time itself)
        wf_neg_candidates = set()
        for t_idx in hit_indices:
            start = max(0, t_idx)
            end = min(L, t_idx + wf_width + 1)
            wf_neg_candidates.update(range(start, end))
            
        wf_neg_candidates.difference_update(hit_indices)
        wf_neg_candidates = list(wf_neg_candidates)
        if len(wf_neg_candidates) > 0:
            chosen_wf_neg = rng.choice(wf_neg_candidates, size=min(500, len(wf_neg_candidates)), replace=False)
            sampled_indices[i, chosen_wf_neg] = True

        # Random negative mining: 100 random bins outside wf_width of any hit and not a hit
        all_indices = set(range(L))
        forbidden = set(hit_indices).union(wf_neg_candidates)
        random_neg_candidates = list(all_indices - forbidden)
        if len(random_neg_candidates) > 0:
            chosen_rand_neg = rng.choice(random_neg_candidates, size=min(100, len(random_neg_candidates)), replace=False)
            sampled_indices[i, chosen_rand_neg] = True
            
    masked_class_output = class_output.squeeze(1)[sampled_indices]  # shape: [num_selected]
    masked_target = target[sampled_indices]
    
    if masked_target.numel() == 0:
        class_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        n_pos = masked_target.sum().item()
        n_total = masked_target.numel()
        n_neg = n_total - n_pos
        pos_weight_val = n_neg / max(1, n_pos)
        
        pos_weight = torch.tensor([pos_weight_val], device=masked_target.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        class_loss = criterion(masked_class_output, masked_target)

    predictions = (torch.sigmoid(masked_class_output) > 0.5).float()
    true_accuracy = (predictions == masked_target).float().mean()

    return class_loss, true_accuracy

############################# HEAD ARCHITECTURES ###################################################

class SupervisedUNet(nn.Module):
    def __init__(self, unet, head, d_model=256):
        super().__init__()
        # TODO
        pass
    def forward(self, x):
        # TODO
        pass

class HybridLinearProbe(nn.Module):
    """
    Simple linear probe for photon count regression.
    Predicts photon counts at each timestep independently.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.detection_head = nn.Linear(d_model, 1)
        self.regression_head = nn.Linear(d_model, 1)
        
    def forward(self, x):  # x: [B, L, C] where L is seq_len, C is d_model
        batch_size, seq_len, d_model = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)  # [B, C, 8000]
        x = x.permute(0, 2, 1)  # [B, 8000, C]
        photon_counts = self.regression_head(x)  # [B, 8000, 1]
        photon_counts = photon_counts.permute(0, 2, 1)  # [B, 1, 8000]
        photon_counts = F.relu(photon_counts)  # or F.softplus for smoother gradients - either way non negative
        
        det_logits = self.detection_head(x)
        det_logits = det_logits.permute(0, 2, 1)

        return photon_counts, det_logits

class HybridUNetDecoder(nn.Module):
    """
    Hybrid U-Net decoder combining detection and photon regression.
    Uses skip connections and dual prediction heads.
    ~750K parameters with d_model=256
    """
    def __init__(self, d_model=256, token_size=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        
        # Encoder path (compress token features)
        self.enc1 = self._make_encoder_block(d_model, 128, kernel_size=3)
        self.enc2 = self._make_encoder_block(128, 64, kernel_size=3)
        
        # Bottleneck with depthwise separable convolution
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # Decoder path with progressive upsampling
        # Stage 1: 2x upsample
        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(64 + 128, 64, kernel_size=3)
        
        # Stage 2: 5x upsample (to reach 10x total)
        self.up2 = nn.ConvTranspose1d(64, 64, kernel_size=5, stride=5)
        self.dec2 = self._make_decoder_block(64 + 256, 96, kernel_size=5)
        
        # Shared feature refinement
        self.shared_refine = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        # Photon regression head
        self.photon_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(2, 16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        
    def _make_encoder_block(self, in_ch, out_ch, kernel_size=3):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, 
                     padding=kernel_size//2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
    
    def _make_decoder_block(self, in_ch, out_ch, kernel_size=3):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, 
                     padding=kernel_size//2),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C] - tokenized sequence
            
        Returns:
            photon_counts: [B, 1, L*token_size] - photon count predictions
            detection_logits: [B, 1, L*token_size] - detection logits
        """
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        # Encoder with skip connections
        skip1 = x  # [B, 256, L]
        x = self.enc1(x)  # [B, 128, L]
        skip2 = x  # [B, 128, L]
        x = self.enc2(x)  # [B, 64, L]
        
        # Bottleneck
        x = self.bottleneck(x)  # [B, 64, L]
        
        # Decoder Stage 1: 2x upsample
        x = self.up1(x)  # [B, 64, L*2]
        skip2_up = F.interpolate(skip2, size=x.shape[2], 
                                mode='linear', align_corners=False)
        x = torch.cat([x, skip2_up], dim=1)  # [B, 192, L*2]
        x = self.dec1(x)  # [B, 64, L*2]
        
        # Decoder Stage 2: 5x upsample (total 10x)
        x = self.up2(x)  # [B, 64, L*10]
        skip1_up = F.interpolate(skip1, size=x.shape[2], 
                                mode='linear', align_corners=False)
        x = torch.cat([x, skip1_up], dim=1)  # [B, 320, L*10]
        x = self.dec2(x)  # [B, 96, L*10]
        
        # Shared feature refinement
        shared_features = self.shared_refine(x)  # [B, 64, L*10]
        
        # Dual prediction heads
        detection_logits = self.detection_head(shared_features)  # [B, 1, L*10]
        photon_raw = self.photon_head(shared_features)  # [B, 1, L*10]
        
        # Apply non-negativity constraint using softplus
        # No max_photons cap - unbounded regression
        photon_counts = F.softplus(photon_raw)
        
        # Optional: gate photon predictions by detection confidence
        # This helps the model learn to predict zero photons where there's no signal
        detection_prob = torch.sigmoid(detection_logits)
        photon_counts = photon_counts * detection_prob
        
        return photon_counts, detection_logits
    
class TransformerHybridDecoder(nn.Module):
    """
    Transformer-based decoder with cross-attention upsampling.
    Combines self-attention for global context with learned queries for upsampling.
    ~1M parameters with d_model=256
    """
    def __init__(self, d_model=256, token_size=10, dropout=0.1, num_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        self.num_heads = num_heads
        
        # Efficient token refinement with self-attention
        self.token_attention = nn.ModuleList([
            EfficientTransformerBlock(d_model, num_heads, dropout, ffn_ratio=2)
            for _ in range(num_layers)
        ])
        
        # Learnable upsampling queries
        # These act as "anchor points" at the target resolution
        self.upsample_queries = nn.Parameter(
            torch.randn(1, token_size, d_model) / math.sqrt(d_model)
        )
        
        # Cross-attention for upsampling
        self.cross_attention = CrossAttentionUpsampler(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ffn_ratio=2
        )
        
        # Position encoding for upsampled features
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=10000)
        
        # Single refinement block after upsampling
        self.refine_attention = EfficientTransformerBlock(d_model, num_heads, dropout, ffn_ratio=2)
        
        # Shared feature projection (smaller)
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Detection head (lightweight)
        self.detection_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Photon regression head (lightweight)
        self.photon_head = nn.Sequential(
            nn.Linear(64, 24),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, L, C] - tokenized sequence
            
        Returns:
            photon_counts: [B, 1, L*token_size] - photon count predictions
            detection_logits: [B, 1, L*token_size] - detection logits
        """
        B, L, C = x.shape
        
        # Token-level self-attention for global context
        for attn_block in self.token_attention:
            x = attn_block(x)  # [B, L, C]
        
        # Prepare upsampling queries: one set per token position
        queries = self.upsample_queries.expand(B, L, -1, -1)  # [B, L, token_size, C]
        queries = queries.reshape(B, L * self.token_size, C)  # [B, L*10, C]
        
        # Cross-attention upsampling: queries attend to token features
        upsampled = self.cross_attention(queries, x)  # [B, L*10, C]
        
        # Add positional encoding
        upsampled = self.pos_encoding(upsampled)
        
        # Refine with self-attention at high resolution
        upsampled = self.refine_attention(upsampled)  # [B, L*10, C]
        
        # Shared feature projection
        shared_features = self.shared_proj(upsampled)  # [B, L*10, 64]
        
        # Dual prediction heads
        detection_logits = self.detection_head(shared_features)  # [B, L*10, 1]
        photon_raw = self.photon_head(shared_features)  # [B, L*10, 1]
        
        # Transpose to [B, 1, L*10] for consistency with conv-based decoders
        detection_logits = detection_logits.transpose(1, 2)  # [B, 1, L*10]
        photon_raw = photon_raw.transpose(1, 2)  # [B, 1, L*10]
        
        # Unbounded non-negative regression
        photon_counts = F.softplus(photon_raw)
        
        # Gate by detection confidence
        detection_prob = torch.sigmoid(detection_logits)
        photon_counts = photon_counts * detection_prob
        
        return photon_counts, detection_logits


class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block with reduced FFN size."""
    def __init__(self, d_model, num_heads, dropout=0.1, ffn_ratio=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Smaller FFN (2x instead of 4x)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CrossAttentionUpsampler(nn.Module):
    """Cross-attention for upsampling: queries attend to token features."""
    def __init__(self, d_model, num_heads, dropout=0.1, ffn_ratio=2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
        # Smaller FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, queries, keys_values):
        """
        Args:
            queries: [B, L*token_size, C] - upsampling queries
            keys_values: [B, L, C] - token features
        """
        # Cross-attention: queries attend to token features
        attn_out, _ = self.cross_attn(queries, keys_values, keys_values)
        queries = self.norm(queries + attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for 1D sequences."""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, C]
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerPhotonDecoder(nn.Module):
    """
    Transformer decoder for photon regression with long-range context.
    ~1.1M parameters
    """
    def __init__(self, d_model=256, num_layers=3, num_heads=8, 
                 token_size=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        
        # Transformer layers for temporal context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection before upsampling
        self.pre_upsample = nn.Linear(d_model, d_model // 2)
        
        # Learnable upsampling
        self.upsample = nn.ConvTranspose1d(
            d_model // 2, d_model // 2,
            kernel_size=token_size, stride=token_size
        )
        
        # Post-upsample refinement
        self.refine = nn.Sequential(
            nn.Conv1d(d_model // 2, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        
        # Dual heads
        self.detection_head = nn.Conv1d(32, 1, kernel_size=1)
        self.photon_head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        
    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :L, :]
        
        # Transformer for global context
        x = self.transformer(x)  # [B, L, C]
        
        # Project and upsample
        x = self.pre_upsample(x)  # [B, L, C//2]
        x = x.permute(0, 2, 1)  # [B, C//2, L]
        x = self.upsample(x)  # [B, C//2, L*10]
        
        # Refine
        x = self.refine(x)  # [B, 32, L*10]
        
        # Predictions
        detection_logits = self.detection_head(x)  # [B, 1, L*10]
        photon_raw = self.photon_head(x)  # [B, 1, L*10]
        
        # Apply constraints
        photon_counts = F.softplus(photon_raw)
        detection_prob = torch.sigmoid(detection_logits)
        photon_counts = photon_counts * detection_prob
        
        return photon_counts, detection_logits

############################# Utilities ###################################################

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

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def print_run_description(finetune_type, supervised_arc, backbone_ckpt, data_path, epochs, batch_size, lr, freeze_backbone, total_params):
    """Print a detailed description of the current training run."""
    
    # def fprint(*args, **kwargs):
    #     print(*args, **kwargs, flush=True)
    
    fprint("\n" + "="*80)
    fprint("TRAINING RUN CONFIGURATION")
    fprint("="*80)
    
    # Training Mode
    if finetune_type is not None:
        fprint(f"\n📋 MODE: Fine-tuning (Self-Supervised)")
        fprint(f"   └─ Type: {finetune_type}")
        if finetune_type == "lin":
            fprint(f"      └─ Linear Probe (frozen backbone + linear head)")
        elif finetune_type == "dec":
            fprint(f"      └─ Decoder (frozen backbone + decoder head)")
        elif finetune_type == "full":
            fprint(f"      └─ Full Fine-tuning (trainable backbone + decoder head)")
        fprint(f"   └─ Backbone checkpoint: {backbone_ckpt}")
        fprint(f"   └─ Backbone frozen: {freeze_backbone}")
    else:
        fprint(f"\n📋 MODE: Supervised Training (from scratch)")
        fprint(f"   └─ Architecture: {supervised_arc}")
        if supervised_arc == "conformer":
            fprint(f"      └─ Conformer + Decoder")
        elif supervised_arc == "unet":
            fprint(f"      └─ UNet")
    
    # Model Info
    fprint(f"\n🏗️  MODEL:")
    fprint(f"   └─ Total trainable parameters: {total_params:,}")
    fprint(f"   └─ Device: {device}")
    
    # Data Info
    fprint(f"\n📊 DATA:")
    fprint(f"   └─ Dataset: {data_path}")
    fprint(f"   └─ Batch size: {batch_size}")
    fprint(f"   └─ Train/Val/Test split: {DATA_PERCENTAGE*100}% / 10% / {(1-DATA_PERCENTAGE-0.1)*100}%")
    
    # Training Hyperparameters
    fprint(f"\n⚙️  HYPERPARAMETERS:")
    fprint(f"   └─ Epochs: {epochs}")
    fprint(f"   └─ Learning rate: {lr}")
    fprint(f"   └─ Optimizer: Adam (weight_decay=1e-6)")
    fprint(f"   └─ Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)")
    fprint(f"   └─ Validation frequency: every {VAL_EPOCHS} epoch(s)")
    fprint(f"   └─ Loss: Mined BCE (hard negative mining)")
    
    # Logging
    if WANDB_PROJECT is not None:
        fprint(f"\n📈 LOGGING:")
        fprint(f"   └─ W&B Project: {WANDB_PROJECT}")
        fprint(f"   └─ W&B Run: {WANDB_RUN_NAME}")
        fprint(f"   └─ Log interval: every 100 iterations")
    else:
        print(f"\n📈 LOGGING: Disabled (no W&B)")
    
    # Reproducibility
    fprint(f"\n🎲 REPRODUCIBILITY:")
    fprint(f"   └─ Random seed: {SEED}")
    
    fprint("\n" + "="*80 + "\n")


############################# MAIN ###################################################
def main():
    # Define SEED first!
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    # Load data
    train_loader, val_loader, test_loader = make_wf_dataloaders(
        DATA_PATH, batch_size=BATCH_SIZE, val_ratio=0.1, test_ratio=(1 - DATA_PERCENTAGE - 0.1), generator=g, norm=False
    )

    # Initialize models based on configuration
    freeze_backbone = False  # Will be set based on finetune_type
    
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
    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

    # Setup optimizer and scheduler
    optimizer = optim.Adam(optimizable_params, lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Setup wandb logging
    logger = None
    if WANDB_PROJECT is not None:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "finetune_type": finetune_type,
                "supervised_arc": supervised_arc,
                "freeze_backbone": freeze_backbone,
                "log_interval": 100,
                "task": "photon_regression",  # Add task type
            },
            reinit=True,
        )
        wandb.watch(head, log="all", log_freq=100)
        logger = wandb_run

    # Count parameters (must iterate over list, not generator)
    total_params = sum(p.numel() for p in optimizable_params if p.requires_grad)

    # Print run description
    print_run_description(
        finetune_type=finetune_type,
        supervised_arc=supervised_arc,
        backbone_ckpt=BACKBONE_CKPT if finetune_type is not None else None,
        data_path=DATA_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        freeze_backbone=freeze_backbone,
        total_params=total_params
    )

    # ============== CALL THE TRAINING METHOD ==============
    results = train_both(  # Changed from train_detection to train_regression
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        freeze_backbone=freeze_backbone,
        finetune_type=finetune_type if finetune_type is not None else "full",  # Pass finetune type
        logger=logger,
        val_epochs=VAL_EPOCHS,
        log_interval=100,
        checkpoint_path=CURRENT_CKPT_PATH,
        max_grad_norm=1.0,
        use_poisson_loss=True,  # Set to False for log-scale MSE
        disable_tqdm=True  # Set to True for cluster jobs
    )
    # =====================================================

    # Save final results
    os.makedirs(CURRENT_CKPT_PATH, exist_ok=True)  # Ensure directory exists
    out_path = os.path.join(CURRENT_CKPT_PATH, "regression_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results to {out_path}")

    # Save final model
    final_checkpoint_path = os.path.join(CURRENT_CKPT_PATH, "final_photon_model.pth")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'results': results,
        'config': {
            'finetune_type': finetune_type,
            'supervised_arc': supervised_arc,
            'freeze_backbone': freeze_backbone,
            'epochs': EPOCHS,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'task': 'photon_regression',
        }
    }, final_checkpoint_path)
    print(f"✓ Saved final model to {final_checkpoint_path}")

    if logger is not None:
        logger.finish()
        
    return results

if __name__ == "__main__":
    # Make sure these are defined at the top of your script
    SEED = 42  # ADD THIS!
    
    # Run training
    results = main()