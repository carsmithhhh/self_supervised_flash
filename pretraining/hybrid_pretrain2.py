import copy
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def temporal_contrastive_loss_efficient(embeddings, token_labels, temperature=0.1, 
                                       num_samples=512, boundary_weight=2.0):
    """Memory-efficient temporal contrastive loss using sampling."""
    B, L, D = embeddings.shape
    embeddings_flat = embeddings.reshape(-1, D)
    labels_flat = token_labels.reshape(-1)
    N = embeddings_flat.size(0)
    
    if N <= 512:
        # For small batches, compute full matrix
        embeddings_norm = F.normalize(embeddings_flat, dim=-1)
        sim_matrix = embeddings_norm @ embeddings_norm.T / temperature
        label_matrix = labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)
        mask_diagonal = torch.eye(N, device=embeddings.device).bool()
        label_matrix = label_matrix & ~mask_diagonal
        
        position_matrix = torch.arange(N, device=embeddings.device)
        batch_indices = position_matrix // L
        token_indices = position_matrix % L
        is_same_batch = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)
        is_adjacent = (token_indices.unsqueeze(1) - token_indices.unsqueeze(0)).abs() <= 2
        is_boundary = is_same_batch & is_adjacent & ~label_matrix & ~mask_diagonal
        
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * label_matrix.float()).sum(dim=1)
        
        neg_weights = torch.ones_like(sim_matrix)
        neg_weights[is_boundary] = boundary_weight
        neg_sim = (exp_sim * (~label_matrix).float() * neg_weights).sum(dim=1)
        
        has_positives = label_matrix.sum(dim=1) > 0
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        loss = -torch.log(pos_sim[has_positives] / (pos_sim[has_positives] + neg_sim[has_positives] + 1e-8))
        return loss.mean()
    
    # Sample anchors for large batches
    anchor_indices = torch.randperm(N, device=embeddings.device)[:num_samples]
    anchor_embeddings = F.normalize(embeddings_flat[anchor_indices], dim=-1)
    anchor_labels = labels_flat[anchor_indices]
    all_embeddings = F.normalize(embeddings_flat, dim=-1)
    
    sim_matrix = anchor_embeddings @ all_embeddings.T / temperature
    label_matrix = anchor_labels.unsqueeze(1) == labels_flat.unsqueeze(0)
    
    anchor_batch = anchor_indices // L
    anchor_token = anchor_indices % L
    all_batch = torch.arange(N, device=embeddings.device) // L
    all_token = torch.arange(N, device=embeddings.device) % L
    
    is_same_batch = anchor_batch.unsqueeze(1) == all_batch.unsqueeze(0)
    is_adjacent = (anchor_token.unsqueeze(1) - all_token.unsqueeze(0)).abs() <= 2
    is_self = anchor_indices.unsqueeze(1) == torch.arange(N, device=embeddings.device).unsqueeze(0)
    is_boundary = is_same_batch & is_adjacent & ~label_matrix & ~is_self
    
    label_matrix = label_matrix & ~is_self
    
    exp_sim = torch.exp(sim_matrix)
    pos_sim = (exp_sim * label_matrix.float()).sum(dim=1)
    
    neg_weights = torch.ones_like(sim_matrix)
    neg_weights[is_boundary] = boundary_weight
    neg_sim = (exp_sim * (~label_matrix).float() * neg_weights).sum(dim=1)
    
    has_positives = label_matrix.sum(dim=1) > 0
    if has_positives.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    loss = -torch.log(pos_sim[has_positives] / (pos_sim[has_positives] + neg_sim[has_positives] + 1e-8))
    return loss.mean()


def knn_validation_efficient(student, val_loader, device, k=20, 
                            query_chunk_size=100, ref_chunk_size=1000, 
                            max_samples=50000):
    """
    Ultra memory-efficient k-NN validation with double chunking.
    Computes similarities in small blocks to avoid OOM.
    """
    student.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, _, _, _, _, _, token_labels in val_loader:
            data = data.to(device)
            data = torch.arcsinh(data / 5)
            embeddings = student(data, mode='just_embeddings')
            
            # Move to CPU immediately
            all_embeddings.append(embeddings.reshape(-1, embeddings.size(-1)).cpu())
            all_labels.append(token_labels.reshape(-1).cpu())
    
    embeddings = F.normalize(torch.cat(all_embeddings, dim=0), dim=-1)
    labels = torch.cat(all_labels, dim=0)
    N = embeddings.size(0)
    
    # Subsample if dataset is too large
    if N > max_samples:
        print(f"Subsampling validation set from {N} to {max_samples} tokens")
        indices = torch.randperm(N)[:max_samples]
        embeddings = embeddings[indices]
        labels = labels[indices]
        N = max_samples
    
    all_predictions = []
    
    # Iterate over query chunks
    for query_start in range(0, N, query_chunk_size):
        query_end = min(query_start + query_chunk_size, N)
        query_chunk = embeddings[query_start:query_end]  # [query_chunk_size, D] on CPU
        
        # For each query chunk, compute similarities with reference chunks
        all_similarities = []
        
        for ref_start in range(0, N, ref_chunk_size):
            ref_end = min(ref_start + ref_chunk_size, N)
            
            # Move both chunks to GPU, compute similarity, move result back to CPU
            query_gpu = query_chunk.to(device)
            ref_gpu = embeddings[ref_start:ref_end].to(device)
            
            sim_block = query_gpu @ ref_gpu.T  # [query_chunk_size, ref_chunk_size]
            all_similarities.append(sim_block.cpu())
            
            del query_gpu, ref_gpu, sim_block
            torch.cuda.empty_cache()
        
        # Concatenate all similarity blocks for this query chunk
        similarities = torch.cat(all_similarities, dim=1)  # [query_chunk_size, N]
        
        # Find k nearest neighbors
        topk_indices = similarities.topk(k+1, dim=-1)[1][:, 1:]  # exclude self
        
        # Vote among neighbors
        neighbor_labels = labels[topk_indices]
        predictions = torch.mode(neighbor_labels, dim=-1).values
        all_predictions.append(predictions)
        
        del similarities, topk_indices, neighbor_labels
    
    all_predictions = torch.cat(all_predictions, dim=0)
    accuracy = (all_predictions == labels).float().mean()
    
    # Class-wise accuracy
    class_accs = {}
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            class_accs[f'class_{class_idx}'] = (all_predictions[mask] == labels[mask]).float().mean()
    
    return accuracy, class_accs


def hybrid_pretrain_optimized(student, teacher, train_loader, val_loader, optimizer, 
                              scheduler, device, epochs=10, logger=None, 
                              use_amp=False, gradient_accumulation_steps=1):
    """Memory-optimized training loop with mixed precision and gradient accumulation."""
    
    results = {
        'dino_loss': [],
        'mae_loss': [],
        'hybrid_loss': [],
        'eval_acc': [],
        'temp_loss': []
    }

    step = 0
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    eps = 1e-6
    scaler = GradScaler() if use_amp else None

    # DINO Settings
    center = torch.zeros(student.num_prototypes).to(device)
    momentum = 0.9
    temp_student = 0.1

    for epoch in range(epochs):
        student.train()
        hybrid_loss = 0.0
        dino_loss = 0.0
        mae_loss = 0.0
        temp_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                            leave=False, position=0, disable=True)
        
        for i, (data, target, hit_times, photon_target, photon_list, 
                last_phot_list, token_labels) in enumerate(train_progress):
            
            data, token_labels = data.to(device), token_labels.to(device)
            data = torch.arcsinh(data / 5)
            
            # Use automatic mixed precision
            with autocast(enabled=use_amp):
                # DINO forward passes
                s1 = student(data, mode='dino_pretraining', masking='no_mask')
                s2 = student(data, mode='dino_pretraining', masking='mask')
                
                with torch.no_grad():
                    t1 = teacher(data, mode='dino_pretraining', masking='no_mask')
                    t2 = teacher(data, mode='dino_pretraining', masking='mask')
                
                # DINO loss
                temp_teacher = 0.04 + (min(step * gradient_accumulation_steps, 
                                          len(train_loader)*3) / (len(train_loader)*3)) * 0.03
                s1_probs = F.softmax(s1 / temp_student, dim=-1)
                s2_probs = F.softmax(s2 / temp_student, dim=-1)
                t1_probs = F.softmax((t1 - center) / temp_teacher, dim=-1)
                t2_probs = F.softmax((t2 - center) / temp_teacher, dim=-1)
                
                class_weights = torch.tensor([1.0, 100.0, 2.0], device=device)
                token_weights = class_weights[token_labels].unsqueeze(-1)
                
                kl_12 = (t1_probs * (s2_probs + eps).log()) * token_weights
                kl_21 = (t2_probs * (s1_probs + eps).log()) * token_weights
                d_loss = -0.5 * (kl_12.sum() + kl_21.sum()) / s1_probs.numel()
                d_loss = -d_loss
                
                # Update center (outside autocast for stability)
                with torch.no_grad():
                    center = momentum * center + (1 - momentum) * torch.cat([t1, t2], dim=0).mean(dim=(0,1))
                
                # MAE Reconstruction
                x_recon, target_recon, mask = student(data, mode='masked_reco')
                mask_per_token = mask.mean(dim=-1)
                mask_wave = mask_per_token.repeat_interleave(10, dim=1).unsqueeze(1)
                token_weights_wave = class_weights[token_labels].unsqueeze(1).repeat_interleave(10, dim=2)
                
                reco_error = ((x_recon - target_recon) ** 2) * mask_wave * token_weights_wave
                masked_loss = reco_error.sum() / (mask_wave * token_weights_wave).sum().clamp(min=1)
                
                # Temporal Contrastive Loss (extract embeddings once)
                embeddings = student(data, mode='just_embeddings')
                temporal_loss_val = temporal_contrastive_loss_efficient(
                    embeddings, token_labels, temperature=0.1, num_samples=512
                )
                
                # Total loss with gradient accumulation scaling
                # loss = (d_loss + masked_loss + (5 * temporal_loss_val)) / gradient_accumulation_steps
                loss = (d_loss + masked_loss) / gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # EMA teacher update
                step += 1
                lambda_coef = 1.0 - 0.5 * (1.0 - 0.996) * (1 + math.cos(math.pi * step / total_steps))
                with torch.no_grad():
                    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                        t_param.data.mul_(lambda_coef).add_(s_param.data, alpha=1-lambda_coef)
            
            # Logging (scale back for actual loss values)
            hybrid_loss += loss.item() * gradient_accumulation_steps
            dino_loss += d_loss.item()
            mae_loss += masked_loss.item()
            # temp_loss += temporal_loss_val.item()
            
            train_progress.set_postfix({"train_loss": hybrid_loss/(i+1)})
            
            # Periodic cache clearing
            if (i % 50) == 0:
                torch.cuda.empty_cache()
            
            if (i % 10) == 0 and logger is not None:
                logger.log({
                    "iter": i + epoch * len(train_loader),
                    "train_loss": hybrid_loss / (i+1),
                    "lr": optimizer.param_groups[0]['lr'],
                })
            
            # Clean up large tensors
            # del s1, s2, t1, t2, x_recon, target_recon, mask, embeddings
        
        results['dino_loss'].append(dino_loss / len(train_loader))
        logger.log({"dino_loss": dino_loss / len(train_loader)})
        results['mae_loss'].append(mae_loss / len(train_loader))
        logger.log({"mae_loss": mae_loss / len(train_loader)})
        results['hybrid_loss'].append(hybrid_loss / len(train_loader))
        # results['temp_loss'].append(temp_loss / len(train_loader))
        
        # Validation
        if (epoch + 1) % 1 == 0:
            torch.cuda.empty_cache()  # Clear before validation
            
            accuracy, class_accuracies = knn_validation_efficient(
                student, val_loader, device, k=3, 
                query_chunk_size=100, ref_chunk_size=1000, max_samples=50000
            )
            
            logger.log({"knn_val_accuracy": accuracy})
            for i in range(3):
                logger.log({f"knn_val_class{i}": class_accuracies[f'class_{i}']})
            
            results['eval_acc'].append(accuracy)
        
        if (epoch + 1) % 2 == 0:
            torch.save({
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"/sdf/home/c/carsmith/sdf_data/self_supervised_flash/mae_dino_200k/{epoch}.pth")
        
        scheduler.step()
    
    return results