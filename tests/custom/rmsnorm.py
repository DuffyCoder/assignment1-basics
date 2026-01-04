import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        weights: Float[Tensor, " d_model"] | None = None,
        eps: float = 1e-5,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self._init_weights(weights, device, dtype)

    def _init_weights(
        self,
        weights: Float[Tensor, " d_model"] | None,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        target_dtype = dtype or torch.float32
        if weights is not None:
            param = torch.as_tensor(weights, dtype=target_dtype, device=device)
            self.weights = nn.Parameter(param.clone().contiguous())
        else:
            self.weights = nn.Parameter(torch.ones(self.d_model, device=device, dtype=target_dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        rms_x = torch.sqrt(mean_square + self.eps)
        normalized = x / rms_x
        return normalized * self.weights
