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

def pretrain(student, teacher, train_loader, val_loader, optimizer, device, epochs=10, logger=None):
    student.train()
    optimizer.zero_grad()

    results = {
        'train_loss': [],
        'eval_loss': []
    }

    step = 0
    total_steps = len(train_loader) * epochs
    eps = 1e-6

    # DINO Settings
    center = 0.0
    momentum = 0.9
    temp_student = 0.1

    for epoch in range(epochs):
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=0, disable=True)
        
        for i, (data, target, hit_times, photon_target, photon_list) in enumerate(train_progress):
            data, target, photon_target = data.to(device), target.to(device), photon_target.to(device)

            # get student & teacher logits
            s1, s2 = student(data, mode='pretraining', masking='no_mask'), student(data, mode='pretraining', masking='mask')
            
            with torch.no_grad():
                t1, t2 = teacher(data, mode='pretraining', masking='no_mask'), teacher(data, mode='pretraining', masking='mask')
                
            ############## Vanilla DINO (no online clustering, as in DinoSR) ##############
            temp_teacher = teacher_temp_warmup(step, len(train_loader)*3)
            s1_probs, s2_probs = F.softmax(s1 / temp_student, dim=-1), F.softmax(s2 / temp_student, dim=-1) # [25, 80, 20]
            t1_probs, t2_probs = F.softmax((t1 - center) / temp_teacher, dim=-1), F.softmax((t2 - center) / temp_teacher, dim=-1)
            
            # sum over all tokens in batch
            loss = 0.5 * (
                ((s2_probs+eps).log() * t1_probs).sum() + 
                ((s1_probs+eps).log() * t2_probs).sum()
            )
            loss = -loss / s1_probs.numel() # this whole thing is effectively kl-divergence
            
            center = momentum * center + (1 - momentum) * torch.cat([t1, t2]).mean(dim=0)

            # update student
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

            # EMA teacher
            step += 1
            lambda_coef = cosine_lambda(step, total_steps)
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                    t_param.data.mul_(lambda_coef).add_(s_param.data, alpha=1-lambda_coef)

            train_loss += loss.item()
            train_progress.set_postfix({"train_loss": train_loss/(i+1)})

            if (i % 5) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch*(len(train_loader)),
                    "train_loss": train_loss / (i+1),
                    "grad_norm": _compute_grad_norm(student.parameters()),
                    "lr": optimizer.param_groups[0]['lr'],
                    "lambda_coef": lambda_coef
                })

        train_loss /= len(train_loader) 
        results['train_loss'].append(train_loss)

    return results