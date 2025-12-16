'''
OUTDATED - use training method in 'hybrid_pretrain2.py'
'''

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

def cosine_lambda(step, total_steps, lambda_start=0.996, lambda_end=1.0):
    """Compute EMA momentum λ at current step using a cosine schedule."""
    cos_inner = math.pi * step / total_steps
    return lambda_end - 0.5 * (lambda_end - lambda_start) * (1 + math.cos(cos_inner))

def teacher_temp_warmup(step, end_steps, t_start=0.04, t_end=0.07):
    step = min(step, end_steps)  # clamp to end_steps
    return t_start + (step / end_steps) * (t_end - t_start)


def hybrid_pretrain(student, teacher, train_loader, val_loader, optimizer, scheduler, device, epochs=10, logger=None):
    results = {
        'dino_loss': [],
        'mae_loss': [],
        'hybrid_loss': [],
        'eval_acc': [],
        'temp_loss': []
    }

    step = 0
    total_steps = len(train_loader) * epochs
    eps = 1e-6

    # DINO Settings
    center = torch.zeros(student.num_prototypes).to(device)
    momentum = 0.9
    temp_student = 0.1

    for epoch in range(epochs):
        for param in student.parameters():
            param.requires_grad = True
        student.train()
        hybrid_loss = 0.0
        dino_loss = 0.0
        mae_loss = 0.0
        temp_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)
        for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(train_progress):
            data, token_labels = data.to(device), token_labels.to(device)

            #### TRANSFORMING WITH ARCSINH
            data = torch.arcsinh(data / 5)

            # get student & teacher logits
            s1, s2 = student(data, mode='dino_pretraining', masking='no_mask'), student(data, mode='dino_pretraining', masking='mask')
            
            with torch.no_grad():
                t1, t2 = teacher(data, mode='dino_pretraining', masking='no_mask'), teacher(data, mode='dino_pretraining', masking='mask')

             ############# Vanilla DINO (no online clustering, as in DinoSR) #############
            temp_teacher = teacher_temp_warmup(step, len(train_loader)*3)
            s1_probs, s2_probs = F.softmax(s1 / temp_student, dim=-1), F.softmax(s2 / temp_student, dim=-1) # [25, 80, 20]
            t1_probs, t2_probs = F.softmax((t1 - center) / temp_teacher, dim=-1), F.softmax((t2 - center) / temp_teacher, dim=-1)
            
            # sum over all tokens in batch
            class_weights = torch.tensor([1.0, 100.0, 2.0]).to(device)
            token_weights = class_weights[token_labels].unsqueeze(-1)

            kl_12 = (t1_probs * (s2_probs+eps).log())
            kl_21 = (t2_probs * (s1_probs+eps).log())
            kl_12 = kl_12 * token_weights
            kl_21 = kl_21 * token_weights
            d_loss = -0.5 * (kl_12.sum() + kl_21.sum())
            d_loss = -d_loss / s1_probs.numel() # this whole thing is effectively kl-divergence
            
            center = momentum * center + (1 - momentum) * torch.cat([t1, t2], dim=0).mean(dim=(0,1))

            ############# MAE Reconstruction ##########################
            x_recon, target, mask = student(data, mode='masked_reco')
            mask_per_token = mask.mean(dim=-1) 
            mask_wave = mask_per_token.repeat_interleave(10, dim=1).unsqueeze(1) # token size
            class_weights = torch.tensor([1.0, 100.0, 2.0]).to(device)  # [background, rising_edge, tail]
            token_weights = class_weights[token_labels]
            token_weights_wave = token_weights.unsqueeze(1).repeat_interleave(10, dim=2)

            reco_error = ((x_recon - target) ** 2) * mask_wave * token_weights_wave
            masked_loss = reco_error.sum() / (mask_wave * token_weights_wave).sum().clamp(min=1)

            #################### Temporal Contrastive Loss #################################
            embeddings = student(data, mode='just_embeddings')
            temporal_loss = temporal_contrastive_loss_efficient(embeddings, token_labels, temperature=0.1)

            ################################################################################

            loss = d_loss + masked_loss + (5 * temporal_loss)

            loss.backward()
            optimizer.step()

            # EMA teacher
            step += 1
            lambda_coef = cosine_lambda(step, total_steps)
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                    t_param.data.mul_(lambda_coef).add_(s_param.data, alpha=1-lambda_coef)

            # Logging
            hybrid_loss += loss.item()
            dino_loss += d_loss.item()
            mae_loss += masked_loss.item()
            temp_loss += temporal_loss.item()
            
            train_progress.set_postfix({"train_loss": hybrid_loss/(i+1)})

            if (i % 10) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch*(len(train_loader)),
                    "train_loss": hybrid_loss / (i+1),
                    "temp_loss": temp_loss / (i+1),
                    "grad_norm": _compute_grad_norm(student.parameters()),
                    "lr": optimizer.param_groups[0]['lr'],
                    "lambda_coef": lambda_coef
                })
            optimizer.zero_grad()

        results['dino_loss'].append(dino_loss / len(train_loader))
        results['mae_loss'].append(mae_loss / len(train_loader))
        results['hybrid_loss'].append(hybrid_loss / len(train_loader))
        results['temp_loss'].append(temp_loss / len(train_loader))
        
        # Evaluation - Train Probe
        if (epoch + 1) % 1 == 0:
            for param in student.parameters():
                param.requires_grad = False
            student.eval()
            accuracy, class_accuracies = knn_validation_efficient(
                student, val_loader, device, k=3, 
                query_chunk_size=50,      # Smaller query chunks
                ref_chunk_size=500,       # Smaller reference chunks  
                max_samples=10000         # Only use 10k tokens for validation
            )
            
            # val_classifier = NonlinearValClassifier(d_model=256).to(device)
            # val_optimizer = torch.optim.Adam(val_classifier.parameters(), lr=1e-3)
        
            # ### ADDED 11.29 - WEIGHT tail & rising edge POSITIVE CLASSES in CrossEntropy, not BCE
            # class_counts = torch.tensor([0.53, 0.02, 0.45]) # bg, rise, tail
            # class_weights = 1.0 / class_counts
            # class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
            # class_weights = class_weights.to(device)
            # criterion = nn.CrossEntropyLoss(weight=class_weights)

            # # Train for a single epoch
            # val_classifier.train()
            # for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(val_loader):
            #     data, token_labels = data.to(device), token_labels.to(device)
            #     data = torch.arcsinh(data / 5) # use same transformation as in training
                
            #     val_optimizer.zero_grad()
                
            #     # Extract embeddings from frozen student network
            #     with torch.no_grad():
            #         embeddings = student(data, mode='just_embeddings')  # [B, L, d_model]
                
            #     logits = val_classifier(embeddings)                     # [B, L, 1]
            #     logits_flat = logits.view(-1, logits.size(-1))  # [B*L, num_classes]
            #     labels_flat = token_labels.view(-1).long()  # [B*L]
                
            #     loss = criterion(logits_flat, labels_flat)
                
            #     loss.backward()
            #     val_optimizer.step()

            # # Evaluation - Evaluate Nonlinear Probe
            # val_classifier.eval()
            # all_preds, all_labels = [], []
            # with torch.no_grad():
            #     for i, (data, _, _, _, _, _, token_labels) in enumerate(val_loader):
            #         data, token_labels = data.to(device), token_labels.to(device)
            #         data = torch.arcsinh(data / 5) # same transformation as in training
            #         embeddings = student(data, mode='just_embeddings')
            #         logits = val_classifier(embeddings)
            #         preds = torch.argmax(logits, dim=-1) 
                    
            #         all_preds.append(preds.cpu())
            #         all_labels.append(token_labels.cpu())
            
            # all_preds = torch.cat(all_preds, dim=0).flatten()
            # all_labels = torch.cat(all_labels, dim=0).flatten()
            
            # accuracy = (all_preds == all_labels).float().mean()
            
            logger.log({"knn_val_accuracy": accuracy})
            for i in range(3):
                logger.log({f"knn_val_class{i}": class_accuracies[f'class_{i}']})
            
            ## log class-wise accuracy
            # for class_idx in range(3):
            #     class_mask = all_labels == class_idx
            #     if class_mask.sum() > 0:
            #         class_acc = (all_preds[class_mask] == all_labels[class_mask]).float().mean()
            #         logger.log({f"val_accuracy_class_{class_idx}": class_acc})

            results['eval_acc'].append(accuracy)

        if (epoch + 1) % 2 == 0:
            torch.save({
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"/sdf/home/c/carsmith/sdf_data/self_supervised_flash/hybrid_contrast/{epoch}.pth")
        
        scheduler.step()
            
    return results   


def knn_validation_efficient(student, val_loader, device, k=20, chunk_size=1000):
    """
    Memory-efficient k-NN using chunked similarity computation.
    """
    student.eval()
    
    # Extract all embeddings and labels
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, _, _, _, _, _, token_labels in val_loader:
            data = data.to(device)
            data = torch.arcsinh(data / 5)
            embeddings = student(data, mode='just_embeddings')
            
            # Move to CPU immediately to free GPU memory
            all_embeddings.append(embeddings.reshape(-1, embeddings.size(-1)).cpu())
            all_labels.append(token_labels.reshape(-1).cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D] on CPU
    labels = torch.cat(all_labels, dim=0)  # [N] on CPU
    N = embeddings.size(0)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute k-NN in chunks to avoid OOM
    all_predictions = []
    
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk_emb = embeddings[start_idx:end_idx].to(device)  # [chunk_size, D]
        
        # Compute similarities for this chunk
        similarities = chunk_emb @ embeddings.T.to(device)  # [chunk_size, N]
        
        # Find k nearest neighbors
        topk_vals, topk_indices = similarities.topk(k+1, dim=-1)
        topk_indices = topk_indices[:, 1:].cpu()  # exclude self, move to CPU
        
        # Vote among neighbors
        neighbor_labels = labels[topk_indices]  # [chunk_size, k]
        predictions = torch.mode(neighbor_labels, dim=-1).values
        
        all_predictions.append(predictions)
        
        # Clean up
        del chunk_emb, similarities, topk_vals, topk_indices
        torch.cuda.empty_cache()
    
    all_predictions = torch.cat(all_predictions, dim=0)
    accuracy = (all_predictions == labels).float().mean()
    
    # Class-wise accuracy
    class_accs = {}
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_accs[f'class_{class_idx}'] = (all_predictions[mask] == labels[mask]).float().mean()
    
    return accuracy, class_accs

def temporal_contrastive_loss_efficient(embeddings, token_labels, temperature=0.1, num_samples=512, boundary_weight=2.0):
    """
    Memory-efficient version using sampling instead of full N×N matrix.
    """
    B, L, D = embeddings.shape
    embeddings_flat = embeddings.reshape(-1, D)  # [B*L, D]
    labels_flat = token_labels.reshape(-1)  # [B*L]
    N = embeddings_flat.size(0)
    
    # If dataset is small enough, use full matrix
    if N <= 512:
        return temporal_contrastive_loss(embeddings, token_labels, temperature)
    
    # Sample anchors instead of using all tokens
    anchor_indices = torch.randperm(N, device=embeddings.device)[:num_samples]
    anchor_embeddings = embeddings_flat[anchor_indices]  # [num_samples, D]
    anchor_labels = labels_flat[anchor_indices]  # [num_samples]
    
    # Normalize
    anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
    all_embeddings = F.normalize(embeddings_flat, dim=-1)
    
    # Compute similarity only for anchors vs all
    sim_matrix = anchor_embeddings @ all_embeddings.T / temperature  # [num_samples, N]
    
    # Positive mask: same label
    label_matrix = anchor_labels.unsqueeze(1) == labels_flat.unsqueeze(0)  # [num_samples, N]
    
    # Find boundaries for anchors
    anchor_batch = anchor_indices // L
    anchor_token = anchor_indices % L
    all_batch = torch.arange(N, device=embeddings.device) // L
    all_token = torch.arange(N, device=embeddings.device) % L
    
    is_same_batch = anchor_batch.unsqueeze(1) == all_batch.unsqueeze(0)
    is_adjacent = (anchor_token.unsqueeze(1) - all_token.unsqueeze(0)).abs() <= 2
    is_self = anchor_indices.unsqueeze(1) == torch.arange(N, device=embeddings.device).unsqueeze(0)
    is_boundary = is_same_batch & is_adjacent & ~label_matrix & ~is_self
    
    # Remove self from positives
    label_matrix = label_matrix & ~is_self
    
    # InfoNCE
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * label_matrix.float()).sum(dim=1)
    
    # Weighted negatives
    neg_weights = torch.ones_like(sim_matrix)
    neg_weights[is_boundary] = boundary_weight
    neg_sim = (exp_sim * (~label_matrix).float() * neg_weights).sum(dim=1)    
    
    # Only compute loss for anchors with positives
    has_positives = label_matrix.sum(dim=1) > 0
    if has_positives.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    loss = -torch.log(pos_sim[has_positives] / (pos_sim[has_positives] + neg_sim[has_positives] + 1e-8))
    
    return loss.mean()
