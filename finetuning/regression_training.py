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
CURRENT_CKPT_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/conformer_dec_reg_0.3/"
os.makedirs(CURRENT_CKPT_PATH, exist_ok=True)

finetune_type = "dec"
supervised_arc = "conformer"
disable = True

DATA_PERCENTAGE = 0.3
SEED = 42
EPOCHS = 50
VAL_EPOCHS = 2
LR = 1e-4
BATCH_SIZE = 50

WANDB_PROJECT = "dino_waveforms"
WANDB_RUN_NAME = "conformer_dec_reg_0.3"

####################################################################################################


def train_regression(backbone, head, train_loader, val_loader, optimizer, scheduler, 
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
        use_poisson_loss: Use Poisson NLL (True) or log-scale MSE (False) for linear probe
        disable_tqdm: Disable progress bars
    """
    results = {
        "train_loss": [], 
        "train_det_loss": [], 
        "train_reg_loss": [],
        "train_mae": [],
        "train_mse": [],
        "train_rel_err": [],
        "eval_loss": [], 
        "eval_det_loss": [],
        "eval_reg_loss": [],
        "eval_mae": [],
        "eval_mse": [],
        "eval_rel_err": []
    }
    
    global_step = 0
    best_val_mae = float('inf')
    
    # Choose loss function for linear probe
    if finetune_type == "lin":
        if use_poisson_loss:
            linear_loss_fn = photon_regression_poisson_loss
            print("Using Poisson NLL loss for linear probe")
        else:
            linear_loss_fn = photon_regression_log_loss
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
        epoch_mse = 0.0
        epoch_rel_err = 0.0
        
        running_loss = 0.0
        running_detection_loss = 0.0
        running_reg_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_rel_err = 0.0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (train)", 
                         leave=False, disable=disable_tqdm)
        
        for batch_idx, (data, _, hit_times, photon_target, photon_list, _, _) in enumerate(train_iter):
            data = data.to(device)
            data = torch.arcsinh(data / 5.0)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = backbone(data, mode="just_embeddings")
            
            # Compute loss based on head type
            if finetune_type == "lin":
                # Linear probe: only regression output
                reg_output = head(embeddings)
                regression_loss, mae, mse, relative_error = linear_loss_fn(
                    reg_output, hit_times, photon_list, device
                )
                loss = regression_loss
                detection_loss = torch.tensor(0.0, device=device)
            else:
                # Dual-head decoder: regression + detection
                reg_output, class_output = head(embeddings)
                loss, detection_loss, regression_loss, mae, relative_error = photon_regression_dual_head_loss(
                    reg_output, class_output, hit_times, photon_list, data, device
                )
                mse = torch.tensor(0.0, device=device)  # MSE not computed in dual-head loss
            
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
            batch_mse = mse.item()
            batch_rel_err = relative_error.item()
            
            epoch_loss += batch_loss
            epoch_detection_loss += batch_det_loss
            epoch_reg_loss += batch_reg_loss
            epoch_mae += batch_mae
            epoch_mse += batch_mse
            epoch_rel_err += batch_rel_err
            
            running_loss += batch_loss
            running_detection_loss += batch_det_loss
            running_reg_loss += batch_reg_loss
            running_mae += batch_mae
            running_mse += batch_mse
            running_rel_err += batch_rel_err
            
            global_step += 1
            
            # Log every log_interval iterations
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_det_loss = running_detection_loss / log_interval
                avg_reg_loss = running_reg_loss / log_interval
                avg_mae = running_mae / log_interval
                avg_mse = running_mse / log_interval
                avg_rel_err = running_rel_err / log_interval
                
                if logger is not None:
                    log_dict = {
                        "train/total_loss": avg_loss,
                        "train/reg_loss": avg_reg_loss,
                        "train/mae": avg_mae,
                        "train/rel_err": avg_rel_err,
                        "train/grad_norm": grad_norm,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "global_step": global_step,
                        "epoch": epoch
                    }
                    if finetune_type != "lin":
                        log_dict["train/det_loss"] = avg_det_loss
                    if finetune_type == "lin":
                        log_dict["train/mse"] = avg_mse
                    logger.log(log_dict)
                
                train_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'mae': f'{avg_mae:.2f}',
                    'rel_err': f'{avg_rel_err:.2%}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                # Reset running metrics
                running_loss = 0.0
                running_detection_loss = 0.0
                running_reg_loss = 0.0
                running_mae = 0.0
                running_mse = 0.0
                running_rel_err = 0.0
        
        # End of epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_det_loss = epoch_detection_loss / len(train_loader)
        avg_epoch_reg_loss = epoch_reg_loss / len(train_loader)
        avg_epoch_mae = epoch_mae / len(train_loader)
        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_rel_err = epoch_rel_err / len(train_loader)
        
        results["train_loss"].append(avg_epoch_loss)
        results["train_det_loss"].append(avg_epoch_det_loss)
        results["train_reg_loss"].append(avg_epoch_reg_loss)
        results["train_mae"].append(avg_epoch_mae)
        results["train_mse"].append(avg_epoch_mse)
        results["train_rel_err"].append(avg_epoch_rel_err)
        
        # Validation
        if epoch % val_epochs == 0:
            backbone.eval()
            head.eval()
            eval_loss = 0.0
            eval_detection_loss = 0.0
            eval_reg_loss = 0.0
            eval_mae = 0.0
            eval_mse = 0.0
            eval_rel_err = 0.0
            
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (val)", 
                               leave=False, disable=disable_tqdm)
                
                for data, _, hit_times, photon_target, photon_list, _, _ in val_iter:
                    data = data.to(device)
                    data = torch.arcsinh(data / 5.0)
                    
                    # Forward pass
                    embeddings = backbone(data, mode="just_embeddings")
                    
                    # Compute loss based on head type
                    if finetune_type == "lin":
                        reg_output = head(embeddings)
                        regression_loss, mae, mse, relative_error = linear_loss_fn(
                            reg_output, hit_times, photon_list, device
                        )
                        loss = regression_loss
                        detection_loss = torch.tensor(0.0, device=device)
                    else:
                        reg_output, class_output = head(embeddings)
                        loss, detection_loss, regression_loss, mae, relative_error = photon_regression_dual_head_loss(
                            reg_output, class_output, hit_times, photon_list, data, device
                        )
                        mse = torch.tensor(0.0, device=device)
                    
                    eval_loss += loss.item()
                    eval_detection_loss += detection_loss.item()
                    eval_reg_loss += regression_loss.item()
                    eval_mae += mae.item()
                    eval_mse += mse.item()
                    eval_rel_err += relative_error.item()
            
            # Average validation metrics
            avg_eval_loss = eval_loss / len(val_loader)
            avg_eval_det_loss = eval_detection_loss / len(val_loader)
            avg_eval_reg_loss = eval_reg_loss / len(val_loader)
            avg_eval_mae = eval_mae / len(val_loader)
            avg_eval_mse = eval_mse / len(val_loader)
            avg_eval_rel_err = eval_rel_err / len(val_loader)
            
            results["eval_loss"].append(avg_eval_loss)
            results["eval_det_loss"].append(avg_eval_det_loss)
            results["eval_reg_loss"].append(avg_eval_reg_loss)
            results["eval_mae"].append(avg_eval_mae)
            results["eval_mse"].append(avg_eval_mse)
            results["eval_rel_err"].append(avg_eval_rel_err)

            if logger is not None:
                log_dict = {
                    "val/total_loss": avg_eval_loss,
                    "val/reg_loss": avg_eval_reg_loss,
                    "val/mae": avg_eval_mae,
                    "val/rel_err": avg_eval_rel_err,
                    "global_step": global_step,
                    "epoch": epoch
                }
                if finetune_type != "lin":
                    log_dict["val/det_loss"] = avg_eval_det_loss
                if finetune_type == "lin":
                    log_dict["val/mse"] = avg_eval_mse
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
        mse = F.mse_loss(pred_photons, true_photons)
        relative_error = ((pred_photons - true_photons).abs() / (true_photons + 1e-6)).mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        mae = torch.tensor(0.0, device=device)
        mse = torch.tensor(0.0, device=device)
        relative_error = torch.tensor(0.0, device=device)
    
    return loss, mae, mse, relative_error


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
    detection_loss = F.binary_cross_entropy_with_logits(
        detection_logits.squeeze(1),
        detection_target,
        reduction='mean'
    )
    
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
        mse = F.mse_loss(pred_photons, true_photons)
        relative_error = ((pred_photons - true_photons).abs() / (true_photons + 1e-6)).mean()
    else:
        regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mae = torch.tensor(0.0, device=device)
        mse = torch.tensor(0.0, device=device)
        relative_error = torch.tensor(0.0, device=device)
    
    # Combined loss with weighting
    # Note: Poisson NLL is already on a reasonable scale, but you may want to tune this
    total_loss = detection_loss + 1.0 * regression_loss  # Adjust weight as needed
    
    return total_loss, detection_loss, regression_loss, mae, relative_error

############################# HEAD ARCHITECTURES ###################################################

class SupervisedConformer(nn.Module):
    def __init__(self, conformer, head, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.conformer = conformer
        self.head = head

    def forward(self, x):
        embeddings = conformer(x, mode='just_embeddings')
        det_logits = head(embeddings)
        return det_logits

class SupervisedUNet(nn.Module):
    def __init__(self, unet, head, d_model=256):
        super().__init__()
        # TODO
        pass
    def forward(self, x):
        # TODO
        pass

class PhotonRegressionLinearProbe(nn.Module):
    """
    Simple linear probe for photon count regression.
    Predicts photon counts at each timestep independently.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.regression_head = nn.Linear(d_model, 1)
        
    def forward(self, x):  # x: [B, L, C] where L is seq_len, C is d_model
        batch_size, seq_len, d_model = x.shape
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)  # [B, C, 8000]
        x = x.permute(0, 2, 1)  # [B, 8000, C]
        photon_counts = self.regression_head(x)  # [B, 8000, 1]
        photon_counts = photon_counts.permute(0, 2, 1)  # [B, 1, 8000]
        photon_counts = F.relu(photon_counts)  # or F.softplus for smoother gradients - either way non negative
        
        return photon_counts

class PhotonRegressionDecoder(nn.Module):
    """
    Multi-scale decoder for photon count regression.
    Uses separate pathways for detection confidence and photon estimation.
    ~900K parameters with d_model=256
    """
    def __init__(self, d_model=256, token_size=10, dropout=0.1, max_photons=1000):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        self.max_photons = max_photons
        
        # Token-level context refinement
        self.token_refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
            nn.GELU(),
        )
        
        # Multi-scale feature extraction
        self.scale1 = self._make_scale_branch(d_model, 128, kernel_size=3)  # Fine details
        self.scale2 = self._make_scale_branch(d_model, 128, kernel_size=5)  # Medium context
        self.scale3 = self._make_scale_branch(d_model, 128, kernel_size=7)  # Broad context
        
        # Learned upsampling for each scale
        self.upsample1 = nn.ConvTranspose1d(128, 64, kernel_size=token_size, stride=token_size)
        self.upsample2 = nn.ConvTranspose1d(128, 64, kernel_size=token_size, stride=token_size)
        self.upsample3 = nn.ConvTranspose1d(128, 64, kernel_size=token_size, stride=token_size)
        
        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),  # 64*3 = 192
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # Dual-head architecture: detection + regression
        # Detection head (helps localize where photons are)
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
        
    def _make_scale_branch(self, in_ch, out_ch, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),  # Project to out_ch first
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=out_ch),  # Depthwise
            nn.Conv1d(out_ch, out_ch, kernel_size=1),  # Pointwise
            nn.GroupNorm(16, out_ch),
            nn.GELU(),
        )
    
    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        
        # Token-level refinement
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.token_refine(x) + x  # Residual
        
        # Multi-scale processing
        feat1 = self.scale1(x)  # [B, 128, L]
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # Upsample to full resolution
        up1 = self.upsample1(feat1)  # [B, 64, L*10]
        up2 = self.upsample2(feat2)
        up3 = self.upsample3(feat3)
        
        # Fuse multi-scale features
        fused = torch.cat([up1, up2, up3], dim=1)  # [B, 192, L*10]
        refined = self.fusion(fused)  # [B, 64, L*10]
        
        # Dual prediction
        detection_logits = self.detection_head(refined)  # [B, 1, L*10]
        photon_raw = self.photon_head(refined)  # [B, 1, L*10]
        
        # Apply non-negativity and reasonable upper bound
        # Using softplus for smooth gradients near zero
        photon_counts = F.softplus(photon_raw)
        
        # Optional: gate photon predictions by detection confidence
        # This helps avoid predicting photons where there's no signal
        detection_prob = torch.sigmoid(detection_logits)
        photon_counts = photon_counts * detection_prob
        
        return photon_counts, detection_logits

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

def print_run_description(finetune_type, supervised_arc, backbone_ckpt, data_path, epochs, batch_size, lr, freeze_backbone, total_params):
    """Print a detailed description of the current training run."""
    
    def fprint(*args, **kwargs):
        print(*args, **kwargs, flush=True)
    
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
    fprint(f"   └─ Train/Val/Test split: 20% / 10% / 70%")
    
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
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    # Load data
    train_loader, val_loader, test_loader = make_wf_dataloaders(
        DATA_PATH, batch_size=BATCH_SIZE, val_ratio=0.1, test_ratio=0.7, generator=g, norm=False
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
            head = PhotonRegressionLinearProbe()
            head.to(device)
            head.train()
            optimizable_params = list(head.parameters())  # Convert to list
            
        elif finetune_type == "dec":
            freeze_backbone = True
            head = PhotonRegressionDecoder()
            head.to(device)
            head.train()
            optimizable_params = list(head.parameters())  # Convert to list
            
        elif finetune_type == "full":
            freeze_backbone = False
            head = PhotonRegressionDecoder()
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
            head = PhotonRegressionDecoder()
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
    results = train_regression(  # Changed from train_detection to train_regression
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

    