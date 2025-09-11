import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.matrix = [self._matrix(i) for i in range(self.max_seq_len)]
        self.register_buffer("matrix", torch.stack(self.matrix), persistent=False)
        
    def _thetas(self, i: int, k: int) -> Tensor:
        return torch.tensor(i / (self.theta ** ((2 * k - 1) / self.d_k)), device=self.device, dtype=self.dtype)
    
    def _sub_matrix(self, i: int, k: int) -> Tensor:
        return torch.tensor([[torch.cos(self._thetas(i, k)), - torch.sin(self._thetas(i, k))],
                             [torch.sin(self._thetas(i, k)), torch.cos(self._thetas(i, k))]], 
                            device=self.device, dtype=self.dtype)
        
    def _matrix(self, i: int) -> Tensor:
        return torch.ones([self._sub_matrix(i, k) for k in range(self.d_k // 2)], device=self.device, dtype=self.dtype)
    
    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], 
                token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        pass
    