import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum
from typing import Optional, Union

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self._initialize_weights()

    def _initialize_weights(self):
        rope_matrix = torch.stack([self._rope_matrix(i) for i in range(self.max_seq_len)])
        self.register_buffer("rope_matrix", rope_matrix, persistent=False)

    def _thetas(self, i: int, k: int) -> Tensor:
        theta_value = i * (self.theta ** (-2 * k / self.d_k))
        return torch.tensor(theta_value, device=self.device, dtype=self.dtype)

    def _sub_matrix(self, i: int, k: int) -> Tensor:
        theta = self._thetas(i, k)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return torch.tensor(
            [[cos_theta, -sin_theta],
             [sin_theta, cos_theta]], device=self.device, dtype=self.dtype)

    def _rope_matrix(self, i: int) -> Tensor:
        sub_matrices = [self._sub_matrix(i, k) for k in range(self.d_k // 2)]
        return torch.block_diag(*sub_matrices)

    def forward(self, x: Float[Tensor, " ... seq_len d_k"],
                token_positions: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len d_k"]:

        return einsum(self.rope_matrix[token_positions], x,
                       "seq_len d_k_out d_k_in, ... seq_len d_k_in -> ... seq_len d_k_out")

