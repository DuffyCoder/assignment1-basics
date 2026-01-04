import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float | None = 10000.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model
        self.rope_theta = rope_theta

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=self.d_ff,
                    context_length=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: Int[torch.Tensor, " batch seq_len"]) -> Float[torch.Tensor, " batch seq_len vocab_size"]:
        if input_ids.dim() != 2:
            raise ValueError("输入需要是二维 token ids 张量，形状 [batch, seq_len]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"序列长度 {seq_len} 超过了模型 context_length={self.context_length}")

        token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        token_positions = token_positions.expand(batch_size, seq_len)

        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, token_positions=token_positions)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return logits
