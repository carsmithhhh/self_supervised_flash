import copy
import math
import pickle

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

def cosine_lambda(step, total_steps, lambda_start=0.996, lambda_end=1.0):
    """Compute EMA momentum λ at current step using a cosine schedule."""
    cos_inner = math.pi * step / total_steps
    return lambda_end - 0.5 * (lambda_end - lambda_start) * (1 + math.cos(cos_inner))

def teacher_temp_warmup(step, end_steps, t_start=0.04, t_end=0.07):
    step = min(step, end_steps)  # clamp to end_steps
    return t_start + (step / end_steps) * (t_end - t_start)

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

def pretrain(student, teacher, train_loader, val_loader, optimizer, scheduler, device, epochs=10, logger=None):

    results = {
        'dino_loss': [],
        'mask_loss': [],
        'total_train_loss': [],
        'eval_acc': []
    }

    step = 0
    total_steps = len(train_loader) * epochs
    eps = 1e-6

    # DINO Settings
    center = torch.zeros(student.num_prototypes, device=device)
    momentum = 0.9
    temp_student = 0.1

    for epoch in range(epochs):
        student.train()
        optimizer.zero_grad()
        train_loss = 0.0
        epoch_dino_loss = 0.0
        epoch_mask_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)
        
        for i, (data, target, hit_times, photon_target, photon_list, token_labels) in enumerate(train_progress):
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)

            # get student & teacher logits
            # s1, s2 = student(data, mode='dino_pretraining', masking='no_mask'), student(data, mode='dino_pretraining', masking='mask')
            
            # with torch.no_grad():
            #     t1, t2 = teacher(data, mode='dino_pretraining', masking='no_mask'), teacher(data, mode='dino_pretraining', masking='mask')
                
            ############# Vanilla DINO (no online clustering, as in DinoSR) #############
            # temp_teacher = teacher_temp_warmup(step, len(train_loader)*3)
            # s1_probs, s2_probs = F.softmax(s1 / temp_student, dim=-1), F.softmax(s2 / temp_student, dim=-1) # [25, 80, 20]
            # t1_probs, t2_probs = F.softmax((t1 - center) / temp_teacher, dim=-1), F.softmax((t2 - center) / temp_teacher, dim=-1)
            
            # # sum over all tokens in batch
            # d_loss = 0.5 * (
            #     ((s2_probs+eps).log() * t1_probs).sum() + 
            #     ((s1_probs+eps).log() * t2_probs).sum()
            # )
            # d_loss = -d_loss / s1_probs.numel() # this whole thing is effectively kl-divergence
            # epoch_dino_loss += d_loss.item()
            
            # center = momentum * center + (1 - momentum) * torch.cat([t1, t2], dim=0).mean(dim=(0,1))

            ############## OUTDATED Masked Reconstruction Task - on Embeddings #######################
            # embeddings, target_embeddings, mask = student(data, mode='masked_reco')
            # mask_per_token = mask.mean(dim=-1) 
            # cosine_sim = F.cosine_similarity(embeddings, target_embeddings, dim=-1)
            # masked_loss = ((1 - cosine_sim) * mask_per_token).sum() / (mask_per_token.sum() + 1e-6)
            # epoch_mask_loss += masked_loss.item()

            ############## Masked Reconstruction Task On Waveforms #######################
            x_recon, target, mask = student(data, mode='masked_reco')
            mask_per_token = mask.mean(dim=-1) 
            mask_wave = mask_per_token.repeat_interleave(10, dim=1).unsqueeze(1) # token size
            reco_error = ((x_recon - target) ** 2) * mask_wave
            masked_loss = reco_error.sum() / mask_wave.sum().clamp(min=1)
            epoch_mask_loss += masked_loss.item()
            
            ############### Backprop & Updating Teacher Network ############################
            # update student
            batch_loss = masked_loss # + d_loss
            batch_loss.backward() 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # EMA teacher
            step += 1
            lambda_coef = cosine_lambda(step, total_steps)
            # with torch.no_grad():
            #     for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            #         t_param.data.mul_(lambda_coef).add_(s_param.data, alpha=1-lambda_coef)
            
            # Logging
            train_loss += batch_loss
            train_progress.set_postfix({"train_loss": train_loss/(i+1)})

            if (i % 10) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch*(len(train_loader)),
                    "train_loss": train_loss / (i+1),
                    "dino_loss": epoch_dino_loss / (i+1),
                    "mask_loss": epoch_mask_loss / (i+1),
                    "grad_norm": _compute_grad_norm(student.parameters()),
                    "lr": optimizer.param_groups[0]['lr'],
                    "lambda_coef": lambda_coef
                })

        train_loss /= len(train_loader) 
        epoch_dino_loss /= len(train_loader)
        epoch_mask_loss /= len(train_loader)
        results['total_train_loss'].append(train_loss)
        results['dino_loss'].append(epoch_dino_loss)
        results['mask_loss'].append(epoch_mask_loss)


        # Evaluation - Linear Probe
        if (epoch + 1) % 1 == 0:
            student.eval()
            val_classifier = ValClassifier(d_model=256).to(device)
            val_optimizer = torch.optim.Adam(val_classifier.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            # Train for a single epoch
            val_classifier.train()
            for i, (data, target, hit_times, photon_target, photon_list, token_labels) in enumerate(val_loader):
                data, token_labels = data.to(device), token_labels.to(device)
                
                val_optimizer.zero_grad()
                
                # Extract embeddings from frozen student network
                with torch.no_grad():
                    embeddings = student(data, mode='just_embeddings')  # [B, L, d_model]
                
                logits = val_classifier(embeddings)                     # [B, L, 1]
                loss = criterion(logits.squeeze(-1), token_labels.float())  # [B, L]
                
                loss.backward()
                val_optimizer.step()

            # Evaluate
            val_classifier.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for i, (data, _, _, _, _, token_labels) in enumerate(val_loader):
                    data, token_labels = data.to(device), token_labels.to(device)
                    embeddings = student(data, mode='just_embeddings')
                    logits = val_classifier(embeddings)
                    preds = torch.sigmoid(logits).squeeze(-1)
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(token_labels.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            accuracy = ((all_preds > 0.5) == all_labels).float().mean()
            logger.log({
                "val_accuracy": accuracy
            })
            results['eval_acc'].append(accuracy)

        torch.save({
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mask_unbal_overfit/{epoch}.pth")

    return results