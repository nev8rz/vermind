
from typing import Tuple,Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ 
    enhanced LayerNorm: bias or not
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        if bias:
            self.beta = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return F.layer_norm(
            x,
            normalized_shape=(self.hidden_size,),
            weight=self.gamma,
            bias=self.beta,
            eps=self.eps,
        ).type_as(x)

class RMSNorm(nn.Module):
    """ RMSNorm implements the Root Mean Square Layer Normalization.
    Reference: https://arxiv.org/abs/1910.07467
    公式:  x_norm = x / (||x||_2 / sqrt(d)) * gamma
    default eps = 1e-6
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x,
            normalized_shape=(self.hidden_size,),
            weight=self.gamma,
            eps=self.eps,
        ).type_as(x)  
        
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w2(x)) * self.w1(x))
