import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

from utils import *
from torchaudio.models import Conformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # won’t be trained

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.device)  # [seq_len, 1, d_model]

class MultiLevelTokenizer(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_sizes=[20, 50, 100, 400], pool_stride=10, window_len=8000, token_size=10, downsample='conv_mlp_lite'):
        """
        Args:
            in_channels: input feature channels (e.g., 1 for waveform)
            hidden_dim: embedding size per conv branch
            kernel_sizes: list of temporal kernel sizes (in samples)
            stride: how many samples = 100 ns (controls downsampling rate)
        """
        super().__init__()
        num_tokens = window_len // token_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=hidden_dim, 
                      kernel_size=k, 
                      stride=1, 
                      padding=k//2)
            for k in kernel_sizes
        ])

        if downsample == 'pool':
            self.downsample = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        elif downsample == 'linear_mlp':
            self.downsample = nn.Sequential(
                nn.Linear(window_len, num_tokens*5),
                nn.GELU(),
                nn.Linear(num_tokens*5, num_tokens),
            )
        elif downsample == 'single_conv':
            self.downsample = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=pool_stride,
                stride=pool_stride,
                padding=0 # want exactly window_len // token_size tokens
            )
        # 2-Layer Conv MLP
        elif downsample == 'conv_mlp_lite':
            self.downsample = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=token_size, stride=token_size, padding=0, groups=hidden_dim, bias=False),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            )
        
        self.proj = nn.Conv1d(
            in_channels=len(kernel_sizes) * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, in_channels, L)
        """
        conv_outs = []
        for conv in self.convs:
            feat = F.relu(conv(x))         # (B, d_model, 8000)
            feat = self.downsample(feat)   # [B, 80, d_model]
            conv_outs.append(feat)

        out = torch.cat(conv_outs, dim=1)  # (B, d_model * n_kernels, 80)
        out = self.proj(out)

        return out

# different defaults than flash_detection version
class ConformerModel(nn.Module):
    def __init__(self, in_channels=1, d_model=256, num_heads=8, num_layers=4, token_size=10, window_len = 8000, tokens='multi-level', kernel_sizes=[20, 50, 100, 400], ffn_factor=8, downsample='conv_mlp_lite', dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.window_len = window_len
        self.token_size = token_size
        self.tokens = tokens
        self.kernel_sizes = kernel_sizes

        # For DINO
        self.num_prototypes = 20
        self.prototypes = nn.Parameter(torch.rand(self.d_model, self.num_prototypes))

        if self.tokens == 'multi-level':
            self.tokenizer = MultiLevelTokenizer(
                in_channels=1,
                hidden_dim=d_model,
                kernel_sizes=kernel_sizes,
                window_len=window_len,
                token_size=token_size,
                downsample=downsample
            )
        else:
            self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=10, stride=10)
    
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        self.conformer = Conformer(input_dim=d_model, num_heads=num_heads, ffn_dim=(ffn_factor*d_model), num_layers=num_layers, depthwise_conv_kernel_size=21, dropout=dropout)
      
        self.upsample = nn.Linear(d_model, d_model)

        # Classification Head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression Head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x, mode='pretraining', masking='no_mask'):
        B, in_channels, window_len = x.shape

        x = self.tokenizer(x)          # [B, d_model, 80]
        L_tokens = x.size(-1)

        if mode == 'pretraining' and masking == 'mask':
            waveformMask = RandomMasking(masking_p=0.5)
            _, x = waveformMask(x)
        
        x = x.permute(2, 0, 1)         # [80, B, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)         #[B, 80, d_model]

        lengths = torch.full((B,), window_len // self.token_size, dtype=torch.long, device=x.device)
        
        # lengths is not used for padding sequences to same length, since all waveforms are same length
        x, _ = self.conformer(x, lengths)          # [B, 80, d_model]

        if mode == 'pretraining':
            prototypes = F.normalize(self.prototypes, dim=0)
            return x @ prototypes # logits over prototype classes
            
        elif mode == 'finetuning':
            x = self.upsample(x)        # [B, d_model, 8000]
            x = x.permute(0, 2, 1)         # [B, d_model, 80]
            x = F.interpolate(x, size=8000, mode="linear", align_corners=False)
    
            # Class & Reg Heads
            class_logits = self.class_l2(F.relu(self.class_l1(x)))
            reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))
    
            return class_logits, reg_logits

        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of: ['pretraining', 'finetuning']."
            ) 