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
from dino_utils import *
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
train_loader, val_loader, test_loader = make_wf_dataloaders(path, batch_size=50, val_ratio=0.05, test_ratio=0.85, generator=g, norm=False)

device = 'cuda'
student = ConformerModel()
teacher = copy.deepcopy(student) # identical starting weights
student, teacher = student.to(device), teacher.to(device)
total_params = sum(p.numel() for p in student.parameters())
print(f"Total number of parameters: {total_params:,}")

epochs = 50

# DINO learning hyperparameters
total_steps = len(train_loader) * epochs
batch_size = train_loader.batch_size
lr_base_rate = 0.0005 * batch_size / 256 # from DINO paper
warmup_steps = total_steps * 0.03 # 3% of training from DINO paper

optimizer = torch.optim.Adam(student.parameters(), lr=lr_base_rate)
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


logger = wandb.init(
    project="dino_waveforms",
    name="dino1",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr_base_rate,
    }
)
wandb.watch(student, log="all", log_freq=100)

results = dino_pretrain(student, teacher, train_loader, val_loader, optimizer, scheduler, device, epochs, logger=logger)
wandb.finish()