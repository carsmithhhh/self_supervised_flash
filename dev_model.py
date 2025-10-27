import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

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
    def __init__(self, in_channels, hidden_dim, kernel_sizes=[20, 50, 100, 400], pool_stride=16, window_len=8000, token_size=16, downsample='conv_mlp_lite'):
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

# 2-layer MLP w/ dropout
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-Attention with Residual Connection
        x = x + self.dropout(self.self_attn(x, x, x))
        x = self.norm1(x)
        
        # Feedforward with Residual Connection
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=4, d_model=256, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
