import torch
from torch import Tensor
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._initialize_weights(device, dtype)
        
    def _initialize_weights(self, device: str, dtype: torch.dtype):
        d_in, d_out = self.num_embeddings, self.embedding_dim
        self.weight = nn.Parameter(torch.empty(d_in, d_out, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=2 / (d_in * d_out))
        
    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return self.weight[x]