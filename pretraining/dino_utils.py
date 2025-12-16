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
    
def dino_pretrain(student, teacher, train_loader, val_loader, optimizer, scheduler, device, epochs=10, logger=None):
    results = {
        'dino_loss': [],
        'eval_acc': []
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
        train_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)
        for i, (data, target, hit_times, photon_target, photon_list, last_phot_list, token_labels) in enumerate(train_progress):
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)

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
            # d_loss = 0.5 * (
            #     ((s2_probs+eps).log() * t1_probs).sum() + 
            #     ((s1_probs+eps).log() * t2_probs).sum()
            # )
            d_loss = -d_loss / s1_probs.numel() # this whole thing is effectively kl-divergence
            
            center = momentum * center + (1 - momentum) * torch.cat([t1, t2], dim=0).mean(dim=(0,1))

            d_loss.backward()
            optimizer.step()
            scheduler.step()

            # EMA teacher
            step += 1
            lambda_coef = cosine_lambda(step, total_steps)
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                    t_param.data.mul_(lambda_coef).add_(s_param.data, alpha=1-lambda_coef)

            # Logging
            train_loss += d_loss.item()
            train_progress.set_postfix({"train_loss": train_loss/(i+1)})

            if (i % 10) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch*(len(train_loader)),
                    "train_loss": train_loss / (i+1),
                    "grad_norm": _compute_grad_norm(student.parameters()),
                    "lr": optimizer.param_groups[0]['lr'],
                    "lambda_coef": lambda_coef
                })
            optimizer.zero_grad()

        train_loss /= len(train_loader) 
        results['dino_loss'].append(train_loss)

        # Evaluation - Train Probe
        if (epoch + 1) % 1 == 0:
            for param in student.parameters():
                param.requires_grad = False
            student.eval()
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
                    embeddings = student(data, mode='just_embeddings')  # [B, L, d_model]
                
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
                    embeddings = student(data, mode='just_embeddings')
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
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"/sdf/home/c/carsmith/sdf_data/self_supervised_flash/dino_weighted/{epoch}.pth")
        
        scheduler.step()
            
    return results   

            

            