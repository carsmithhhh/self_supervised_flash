import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math

from utils import *

# A single linear layer for binary classification of input tokens 
# Train on embeddings every x training epochs as a way to evaluate quality of embeddings
class ValClassifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear_layer = nn.Linear(d_model, 3)

    def forward(self, x):
        logits = self.linear_layer(x)
        return logits


class NonlinearValClassifier(nn.Module): # Nonlinear probe
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, 3)

    def forward(self, x):
        logits = self.linear_2(F.gelu(self.linear_1(x)))
        return logits