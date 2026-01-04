import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .multihead_self_attention import MultiheadSelfAttention
from .positionwise_feedforward import PositionwiseFeedForward
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch seq_len d_model"], token_positions: Int[Tensor, " batch seq_len"]) -> Float[Tensor, " batch seq_len d_model"]:
        attn_out = self.attn(self.norm1(x), token_positions=token_positions)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
