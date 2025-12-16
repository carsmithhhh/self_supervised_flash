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
        # 2-Layer Conv MLP
        # elif downsample == 'conv_mlp_lite':
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
        out = F.layer_norm(out, out.shape[-2:]) # TESTING LAYERNORM HERE - NEW 11.7
 
        return out

# different defaults than flash_detection version
class ConformerModel(nn.Module):
    def __init__(self, in_channels=1, d_model=256, num_heads=8, num_layers=4, token_size=10, window_len = 8000, tokens='multi-level', kernel_sizes=[20, 50, 100, 400], ffn_factor=8, downsample='conv_mlp_lite', dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.d_decoder = d_model // 4
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
            self.tokenizer = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=token_size, stride=token_size)
    
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_len)

        self.conformer = Conformer(input_dim=d_model, num_heads=num_heads, ffn_dim=(ffn_factor*d_model), num_layers=num_layers, depthwise_conv_kernel_size=21, dropout=dropout)
      
        self.upsample = nn.Linear(d_model, d_model)

        # For Masked Reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc_to_dec = nn.Linear(d_model, self.d_decoder) # project from encoder channel dim to decoder channel dim
        
        self.positional_encoding_dec = PositionalEncoding(d_model=self.d_decoder, max_len=window_len // token_size)

        ### OLD - TransformerDecoder Based, No Bidirectional Attention
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=self.d_decoder,
        #     nhead=4,
        #     dim_feedforward=self.d_decoder * 4,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # self.decoder_norm = nn.LayerNorm(self.d_decoder)
        # self.reco_out = nn.Linear(self.d_decoder, token_size)
        ##### NEW - Transformer Encoder Reco Head
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_decoder,
            nhead=4,
            dim_feedforward=self.d_decoder * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.decoder_norm = nn.LayerNorm(self.d_decoder)
        self.reco_out = nn.Linear(self.d_decoder, token_size)
        
        # Classification Head (2-layer MLP)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

        # Regression Head (2-layer MLP)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, data, mode='dino_pretraining', masking='no_mask'):
        B, in_channels, window_len = data.shape
        lengths = torch.full((B,), window_len // self.token_size, dtype=torch.long, device=data.device)

        x_tokens = self.tokenizer(data)          # [B, d_model, 800]
        L_tokens = x_tokens.size(-1)

        if mode == 'dino_pretraining' and masking == 'no_mask':
            x = x_tokens.permute(2, 0, 1)  
            x = self.positional_encoding(x) # [800, d_model, B]
            x = x.permute(1, 0, 2) 

            x, _ = self.conformer(x, lengths) 

            prototypes = F.normalize(self.prototypes, dim=0)
            return x @ prototypes # logits over prototype classes for DINO

        if mode == 'dino_pretraining' and masking == 'mask':
            waveformMask = RandomMasking(masking_p=0.6, mode='dino')
            _, x = waveformMask(x_tokens)

            x = x.permute(2, 0, 1) 
            x = self.positional_encoding(x)
            x = x.permute(1, 0, 2) 

            x, _ = self.conformer(x, lengths) 

            prototypes = F.normalize(self.prototypes, dim=0)
            return x @ prototypes # logits over prototype classes for DINO

        if mode == 'masked_reco':
            x = x_tokens.permute(2, 0, 1)
            x = self.positional_encoding(x)
            x = x.permute(1, 0, 2)

            # Encode ALL tokens
            x_encoded, _ = self.conformer(x, lengths)       # [B, L, d_model]

            # Mask some tokens
            waveformMask = RandomMasking(masking_p=0.75, mode='masked_reco')
            x_masked, mask = waveformMask(x_encoded)
            mask = mask.type_as(x_masked)
            x_masked = x_masked * (1 - mask) + self.mask_token * mask

            x_dec_proj = self.enc_to_dec(x_masked)                 # [B, 800, d_decoder]
            x_full = self.enc_to_dec(x_encoded)

            # Use embeddings of all tokens as memory for decoder
            x_dec = x_dec_proj.permute(1, 0, 2)                    # [L, B, d_decoder)
            x_dec = self.positional_encoding_dec(x_dec)            

            x_full = x_full.permute(1, 0, 2)
            x_dec = self.decoder(x_dec)             # [L, B, d_decoder]
            x_dec = self.decoder_norm(x_dec)
            x_recon_tokens = self.reco_out(x_dec)                  # [L, B, token_size]

            x_recon_tokens = x_recon_tokens.permute(1, 0, 2)  # [B, L, token_size]
            B, L, token_size = x_recon_tokens.shape
            L_target = L * token_size
            x_recon = x_recon_tokens.reshape(B, 1, L_target)  # [B, 1, L_target]
            
            target = data # [B, 1, L]
            
            return x_recon, target, mask

        if mode == 'just_embeddings':
            x = x_tokens.permute(2, 0, 1) 
            x = self.positional_encoding(x)
            x = x.permute(1, 0, 2) 

            x, _ = self.conformer(x, lengths) 
            return x
                
        elif mode == 'finetuning':
            x = self.upsample(x_tokens)        # [B, d_model, 8000]
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

class DetectionHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        self.upsample = nn.Linear(d_model, d_model)
        self.class_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.class_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)
        class_logits = self.class_l2(F.relu(self.class_l1(x)))
        return class_logits

class RegressionHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        self.upsample = nn.Linear(d_model, d_model)
        self.reg_l1 = nn.Conv1d(d_model, d_model // 2, 1)
        self.reg_l2 = nn.Conv1d(d_model // 2, 1, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=8000, mode="linear", align_corners=False)
        reg_logits = self.reg_l2(F.relu(self.reg_l1(x)))
        return reg_logits

class HybridHead(nn.Module):
    def __init__(self, d_model=256, num_context_layers=2, dropout=0.1, token_size=10):
        super().__init__()
        self.d_model = d_model
        
        # Temporal context aggregation before upsampling
        self.context_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)  # Depthwise
            for _ in range(num_context_layers)
        ])
        self.context_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_context_layers)
        ])
        
        # Pre-upsample projection (optional but helps)
        self.pre_upsample_proj = nn.Linear(d_model, d_model)
        
        # Upsampling with transposed conv for learnable interpolation
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, 
            kernel_size=token_size, stride=token_size,  # upsampling to 1ns
            padding=0
        )
        
        # Post-upsample refinement
        self.post_upsample_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.post_upsample_norm = nn.LayerNorm(d_model)
        
        # Classification head with more capacity
        self.class_head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model // 2),  # Better than BatchNorm for variable length
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 4, 1, kernel_size=1)
        )
        
        # Regression head (shared initial features with classification)
        self.reg_head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 4, 1, kernel_size=1),
            nn.Softplus()  # Ensure positive photon counts
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Careful initialization for the upsampling layer
        nn.init.xavier_uniform_(self.upsample.weight)
        if self.upsample.bias is not None:
            nn.init.zeros_(self.upsample.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) - token embeddings from backbone
        Returns:
            class_logits: (batch, 1, seq_len * 10) - per-nanosecond classification
            reg_logits: (batch, 1, seq_len * 10) - per-nanosecond photon counts
        """
        batch_size, seq_len, d_model = x.shape
        
        # Temporal context aggregation (operates on token level)
        x_ctx = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        for conv, norm in zip(self.context_layers, self.context_norms):
            residual = x_ctx
            x_ctx = conv(x_ctx)
            x_ctx = x_ctx.permute(0, 2, 1)  # (batch, seq_len, d_model)
            x_ctx = norm(x_ctx)
            x_ctx = F.gelu(x_ctx)
            x_ctx = x_ctx.permute(0, 2, 1)  # (batch, d_model, seq_len)
            x_ctx = x_ctx + residual  # Residual connection
        
        # Pre-upsample projection
        x_ctx = x_ctx.permute(0, 2, 1)  # (batch, seq_len, d_model)
        x_ctx = self.pre_upsample_proj(x_ctx)
        x_ctx = x_ctx.permute(0, 2, 1)  # (batch, d_model, seq_len)
        
        # Learnable upsampling (10x)
        x_up = self.upsample(x_ctx)  # (batch, d_model, seq_len * 10)
        
        # Post-upsample refinement
        x_up = self.post_upsample_conv(x_up)
        x_up = x_up.permute(0, 2, 1)  # (batch, seq_len * 10, d_model)
        x_up = self.post_upsample_norm(x_up)
        x_up = F.gelu(x_up)
        x_up = x_up.permute(0, 2, 1)  # (batch, d_model, seq_len * 10)
        
        # Task-specific heads
        class_logits = self.class_head(x_up)  # (batch, 1, seq_len * 10)
        reg_logits = self.reg_head(x_up)      # (batch, 1, seq_len * 10)
        
        return class_logits, reg_logits
