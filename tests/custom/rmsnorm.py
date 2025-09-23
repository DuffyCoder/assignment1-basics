import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 weights: Float[Tensor, " d_model"] | None = None, 
                 eps: float = 1e-5, 
                 device: str = None, 
                 dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if weights is not None:
            self.weights = weights
        else:
            self._initialize_weights(device, dtype)
    
    def _initialize_weights(self, device: str, dtype: torch.dtype):
        self.weights = nn.Parameter(torch.empty(self.d_model, device=device, dtype=dtype))
        nn.init.normal_(self.weights)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Calculate RMS: sqrt(mean(x^2) + eps)
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        rms_x = torch.sqrt(mean_square + self.eps)
        # Normalize and scale
        normalized = x / rms_x
        return normalized * self.weights