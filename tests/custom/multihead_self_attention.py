import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from jaxtyping import Float, Int

from .linear import Linear
from .rope import RoPE
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        rope_theta: float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.use_rope = rope_theta is not None
        self.rope = (
            RoPE(theta=rope_theta, d_k=self.head_dim, max_seq_len=context_length, device=device, dtype=dtype)
            if self.use_rope
            else None
        )
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " batch seq_len d_model"],
        token_positions: Int[Tensor, " batch seq_len"] | None = None,
    ) -> Float[Tensor, " batch seq_len d_model"]:
        batch_size, seq_len, _ = x.shape
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.num_heads)

        if self.use_rope and self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
                token_positions = token_positions.expand(batch_size, seq_len)
            token_positions = token_positions.unsqueeze(1).expand(batch_size, self.num_heads, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = self.causal_mask[:seq_len, :seq_len]
        mask = mask.to(device=x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn_module = ScaledDotProductAttention(
            q=q,
            k=k,
            v=v,
            mask=mask,
        )
        attn_output = attn_module()
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        return self.o_proj(attn_output)
