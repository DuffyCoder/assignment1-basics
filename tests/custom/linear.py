import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 weights: Float[Tensor, " d_out d_in"] | None = None, 
                 device: str = None, 
                 dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            self.weights = weights
        else:
            self._initialize_weights(device, dtype)
    
    def _initialize_weights(self, device: str, dtype: torch.dtype):
        d_out, d_in = self.out_features, self.in_features
        self.weights = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, std=2 / (d_out * d_in))
    
    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
        
    