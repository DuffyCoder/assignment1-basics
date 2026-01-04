import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum
from jaxtyping import Float, Int

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weights: Float[Tensor, " num_embeddings embedding_dim"] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._init_weights(weights, device, dtype)

    def _init_weights(
        self,
        weights: Float[Tensor, " num_embeddings embedding_dim"] | None,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        target_dtype = dtype or torch.float32
        if weights is not None:
            param = torch.as_tensor(weights, dtype=target_dtype, device=device)
            self.weight = nn.Parameter(param.clone().contiguous())
        else:
            self.weight = nn.Parameter(
                torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=target_dtype)
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        if x.dtype not in (torch.long, torch.int, torch.int64):
            x = x.long()
        return self.weight[x]
