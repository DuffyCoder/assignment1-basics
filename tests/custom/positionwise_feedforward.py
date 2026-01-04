import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from einops import einsum
from .linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1_output = self.w1(x)
        silu = torch.nn.functional.silu(w1_output)
        w3_output = self.w3(x)
        gated = silu * w3_output
        return self.w2(gated)
