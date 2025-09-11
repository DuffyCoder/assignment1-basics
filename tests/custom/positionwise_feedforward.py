import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from einops import einsum
from .linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self._initialize_weights(device, dtype)
    
    def _initialize_weights(self, device: str, dtype: torch.dtype):
        self.w1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device, dtype)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1_output = self.w1(x)  # [... d_ff]
        silu = w1_output * torch.sigmoid(w1_output)  # SiLU activation: x * sigmoid(x)
        w3_output = self.w3(x)  # [... d_ff]
        gated = silu * w3_output  # Element-wise multiplication
        return self.w2(gated)  # Project back to d_model