import sys
sys.path.append('..')
sys.path.append('../..')

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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, random_split

from data_utils import *
from utils import *
from mae_utils import *
from hybrid_utils import *
from hybrid_pretrain2 import *
from model import *
from waveforms.waveforms_module.make_waveform import BatchedLightSimulation

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using CUDA

g = torch.Generator()
g.manual_seed(seed)

path = '/sdf/home/c/carsmith/sdf_data/self_supervised_flash/data/200k_3labels.npy'
train_loader, val_loader, test_loader = make_wf_dataloaders(path, batch_size=25, val_ratio=0.1, test_ratio=0, generator=g, norm=False)

device = 'cuda'
student = ConformerModel()
teacher = copy.deepcopy(student) # identical starting weights
student, teacher = student.to(device), teacher.to(device)
total_params = sum(p.numel() for p in student.parameters())
print(f"Total number of parameters: {total_params:,}")

epochs = 100

# DINO learning hyperparameters
#batch_size=50
total_steps = len(train_loader) * epochs
batch_size = train_loader.batch_size
# lr_base_rate = 0.0005 * batch_size / 256 # from DINO paper
# warmup_steps = total_steps * 0.03 # 3% of training from DINO paper

# optimizer = torch.optim.Adam(student.parameters(), lr=lr_base_rate)
# warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
# decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
# scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

# MAE learning hyperparameters
lr_base_rate = 1e-3
min_lr = 2e-4
warmup_epochs = 5

batch_size = train_loader.batch_size
optimizer = torch.optim.AdamW(student.parameters(), lr=lr_base_rate, betas=(0.9, 0.999), eps=1e-8)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader)) # simple cosine decay w/ out warmup

###### Version that doesn't have lr=0 for epoch 0
def lr_lambda(epoch):
    base_lr = optimizer.defaults["lr"]
    final_scale = min_lr / base_lr

    if epoch < warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        return final_scale + warmup_factor * (1 - final_scale)

    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return cosine_decay * (1 - final_scale) + final_scale

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

logger = wandb.init(
    project="dino_waveforms",
    name="mae_dino_200k",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr_base_rate,
    }
)
wandb.watch(student, log="all", log_freq=100)

results = hybrid_pretrain_optimized(student, teacher, train_loader, val_loader, optimizer, scheduler, device, epochs, logger=logger)
wandb.finish()