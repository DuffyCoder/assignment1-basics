import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Float[Tensor, " d_out d_in"] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._init_weights(weights, device, dtype)

    def _init_weights(
        self,
        weights: Float[Tensor, " d_out d_in"] | None,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        target_dtype = dtype or torch.float32
        if weights is not None:
            param = torch.as_tensor(weights, dtype=target_dtype, device=device)
            self.weights = nn.Parameter(param.clone().contiguous())
        else:
            self.weights = nn.Parameter(
                torch.empty(self.out_features, self.in_features, device=device, dtype=target_dtype)
            )
            nn.init.normal_(self.weights, mean=0.0, std=0.02)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
