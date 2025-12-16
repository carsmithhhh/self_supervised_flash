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
CURRENT_CKPT_PATH = "/sdf/home/c/carsmith/sdf_data/self_supervised_flash/full_detection_0.1/"
os.makedirs(CURRENT_CKPT_PATH, exist_ok=True)

finetune_type = None
supervised_arc = "conformer"
disable = True

DATA_PERCENTAGE = 0.1
SEED = 42
EPOCHS = 50
VAL_EPOCHS = 2
LR = 1e-4
BATCH_SIZE = 50

WANDB_PROJECT = "dino_waveforms"
WANDB_RUN_NAME = "conformer_full_det_0.1"

####################################################################################################


# assume we have a "model", and optimizer defined with appropriate parameters
def train_detection(backbone, head, train_loader, val_loader, optimizer, scheduler, 
                   device, epochs, freeze_backbone=True, logger=None, val_epochs=2, 
                   log_interval=100, checkpoint_path=None, max_grad_norm=1.0):
    """
    Train detection model with proper logging and checkpointing.
    
    Args:
        freeze_backbone: Whether to freeze backbone during training
        log_interval: Log metrics every N iterations
        checkpoint_path: Directory to save checkpoints (None = no checkpoints)
        max_grad_norm: Maximum gradient norm for clipping
    """
    results = {
        "train_loss": [], 
        "train_true_acc": [], 
        "eval_loss": [], 
        "eval_true_acc": []
    }
    
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Set training modes
        if not freeze_backbone:
            backbone.train()
        else:
            backbone.eval()
        head.train()
        
        # Training loop
        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_acc = 0.0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (train)", leave=False, disable=disable)
        for batch_idx, (data, _, hit_times, _, _, _, _) in enumerate(train_iter):
            data = data.to(device)
            data = torch.arcsinh(data / 5.0)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = backbone(data, mode="just_embeddings")
            class_output = head(embeddings)
            
            # Compute loss
            loss, true_acc = mined_bce_loss(data, hit_times, class_output)
            
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
            batch_acc = true_acc.item()
            epoch_loss += batch_loss
            epoch_acc += batch_acc
            running_loss += batch_loss
            running_acc += batch_acc
            
            global_step += 1
            
            # Log every log_interval iterations
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_acc = running_acc / log_interval
                
                if logger is not None:
                    logger.log({
                        "train/loss": avg_loss,
                        "train/acc": avg_acc,
                        "train/grad_norm": grad_norm,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "global_step": global_step,
                        "epoch": epoch
                    })
                
                train_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.4f}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                running_loss = 0.0
                running_acc = 0.0
        
        # End of epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_acc / len(train_loader)
        results["train_loss"].append(avg_epoch_loss)
        results["train_true_acc"].append(avg_epoch_acc)
        
        # Validation
        if epoch % val_epochs == 0:
            backbone.eval()
            head.eval()
            eval_loss = 0.0
            eval_acc = 0.0
            
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (val)", leave=False, disable=disable)
                for data, _, hit_times, _, _, _, _ in val_iter:
                    data = data.to(device)
                    data = torch.arcsinh(data / 5.0)
                    
                    embeddings = backbone(data, mode="just_embeddings")
                    class_output = head(embeddings)
                    
                    loss, true_acc = mined_bce_loss(data, hit_times, class_output)
                    eval_loss += loss.item()
                    eval_acc += true_acc.item()
            
            avg_eval_loss = eval_loss / len(val_loader)
            avg_eval_acc = eval_acc / len(val_loader)
            
            if logger is not None:
                logger.log({
                    "val/loss": avg_eval_loss,
                    "val/acc": avg_eval_acc,
                    "global_step": global_step,
                    "epoch": epoch
                })
            
            results["eval_loss"].append(avg_eval_loss)
            results["eval_true_acc"].append(avg_eval_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                  f"Train Acc={avg_epoch_acc:.4f}, Val Loss={avg_eval_loss:.4f}, "
                  f"Val Acc={avg_eval_acc:.4f}")
            
            # Save best model
            if checkpoint_path and avg_eval_loss < best_val_loss:
                best_val_loss = avg_eval_loss
                torch.save({
                    'epoch': epoch,
                    'backbone_state_dict': backbone.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': avg_eval_loss,
                    'val_acc': avg_eval_acc,
                    'freeze_backbone': freeze_backbone
                }, os.path.join(checkpoint_path, "best_model.pth"))
                print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs validation loss
                if epoch % val_epochs == 0:
                    scheduler.step(avg_eval_loss)
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

class DetectionLinearProbe(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 1)
    
    def forward(self, x):  # X comes in [B, L, C] where L is seq_len, C is d_model
        batch_size, seq_len, d_model = x.shape
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)  # [batch, 8000, d_model]
        det_logits = self.linear1(x)  # [batch, 8000, 1]
        det_logits = det_logits.permute(0, 2, 1)  # [batch, 1, 8000]
        return det_logits

class UNetDetectionDecoder(nn.Module):
    """
    U-Net inspired decoder with progressive upsampling.
    ~700K parameters with d_model=256
    """
    def __init__(self, d_model=256, token_size=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_size = token_size
        
        # Encoder path (compress token features)
        self.enc1 = self._make_encoder_block(d_model, 128, kernel_size=3)
        self.enc2 = self._make_encoder_block(128, 64, kernel_size=3)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # Decoder path with progressive upsampling
        # Stage 1: 2x upsample
        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(64 + 128, 64, kernel_size=3)  # 64 from up1 + 128 from skip2
        
        # Stage 2: 5x upsample (to reach 10x total)
        self.up2 = nn.ConvTranspose1d(64, 64, kernel_size=5, stride=5)
        self.dec2 = self._make_decoder_block(64 + 256, 128, kernel_size=5)  # 64 from up2 + 256 from skip1
        
        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        
        # Detection head
        self.detect_head = nn.Conv1d(32, 1, kernel_size=1)
        
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
    
    def forward(self, x):  # x: [B, L, C]
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
        # Interpolate skip2 to match spatial dimension
        skip2_up = F.interpolate(skip2, size=x.shape[2], mode='linear', align_corners=False)  # [B, 128, L*2]
        x = torch.cat([x, skip2_up], dim=1)  # [B, 64+128=192, L*2]
        x = self.dec1(x)  # [B, 64, L*2]
        
        # Decoder Stage 2: 5x upsample (total 10x)
        x = self.up2(x)  # [B, 64, L*10]
        # Interpolate skip1 to match spatial dimension
        skip1_up = F.interpolate(skip1, size=x.shape[2], mode='linear', align_corners=False)  # [B, 256, L*10]
        x = torch.cat([x, skip1_up], dim=1)  # [B, 64+256=320, L*10]
        x = self.dec2(x)  # [B, 128, L*10]
        
        # Final refinement and detection
        x = self.final_refine(x)  # [B, 32, L*10]
        logits = self.detect_head(x)  # [B, 1, L*10]
        
        return logits


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
            head = DetectionLinearProbe()
            head.to(device)
            head.train()
            optimizable_params = head.parameters()
            
        elif finetune_type == "dec":
            freeze_backbone = True
            head = UNetDetectionDecoder()
            head.to(device)
            head.train()
            optimizable_params = head.parameters()
            
        elif finetune_type == "full":
            freeze_backbone = False
            head = UNetDetectionDecoder()
            head.to(device)
            head.train()
            backbone.train()
            optimizable_params = list(backbone.parameters()) + list(head.parameters())
            
        else:
            raise ValueError("Invalid finetune_type.")
        
    else:  # Supervised training from scratch
        freeze_backbone = False
        if supervised_arc == "conformer":
            backbone = ConformerModel()
            head = UNetDetectionDecoder()
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
            },
            reinit=True,
        )
        wandb.watch(head, log="all", log_freq=100)
        logger = wandb_run

    # Count parameters
    total_params = sum(p.numel() for p in optimizable_params)

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
    results = train_detection(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        freeze_backbone=freeze_backbone,  # Pass as parameter
        logger=logger,
        val_epochs=VAL_EPOCHS,
        log_interval=100,  # Log every 100 iterations
        checkpoint_path=CURRENT_CKPT_PATH,  # Save checkpoints here
        max_grad_norm=1.0  # Gradient clipping threshold
    )
    # =====================================================

    # Save final results
    out_path = os.path.join(CURRENT_CKPT_PATH, "finetune_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results to {out_path}")

    # Save final model
    final_checkpoint_path = os.path.join(CURRENT_CKPT_PATH, "final_model.pth")
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results': results,
        'config': {
            'finetune_type': finetune_type,
            'supervised_arc': supervised_arc,
            'freeze_backbone': freeze_backbone,
            'epochs': EPOCHS,
            'lr': LR,
            'batch_size': BATCH_SIZE,
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

    