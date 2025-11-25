import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum, rearrange
from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear

class TransformerLM(nn.Module):
    def __init__(self,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
    temperature: float = 1.0,
    top_p: float = 0.0,
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.in_indices = in_indices
        self.temperature = temperature
        self.top_p = top_p
        self.embedding = Embedding(
            self.vocab_size,
            self.d_model,
            self.weights['token_embeddings.weight'],
        )
        self.rmsnorm = RMSNorm(self.d_model, self.weights['ln_final.weight'])
        self.output = Linear(self.d_model, self.vocab_size, self.weights['lm_head.weight'])
        
        
    def forward(self):
        x = self.embedding(self.in_indices)
        for layer in range(self.num_layers):
            prefix = f'layers.{layer}'
            layer_weights = {
                k.replace(prefix + '.', ''): v
                for k, v in self.weights.items()
                if k.startswith(prefix)
            }
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.context_length,
                self.rope_theta,
                layer_weights,
                x,
                self.temperature,
                self.top_p,
            )()

        output = self.output(self.rmsnorm(x))
        return output
        
        
