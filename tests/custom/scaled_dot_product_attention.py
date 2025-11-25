import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum
from .softmax import Softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self, 
                 q: Float[Tensor, "batch_size ... seq_len d_k"], 
                 k: Float[Tensor, "batch_size ... seq_len d_k"], 
                 v: Float[Tensor, "batch_size ... seq_len d_v"], 
                 mask: Float[Tensor, "batch_size ... seq_len seq_len"] | None = None,
                 temperature: float = 1.0,
                 top_p: float = 0.0):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.mask = mask
        self.softmax = Softmax(temperature=temperature, top_p=top_p)
        self.d_k = q.shape[-1]
    
    def mask_score(self, 
                   score: Float[Tensor, "batch_size ... seq_len seq_len"]):
        if self.mask is not None:
            score = score.masked_fill(self.mask == False, -1e9)
        return score
    
    def forward(self):
        # 计算 Q @ K^T 并应用缩放因子
        attn = einsum(self.q, self.k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
        attn = attn / (self.d_k ** 0.5)  # 添加缩放因子 1/sqrt(d_k)

        # 在 softmax 之前应用 mask
        attn = self.mask_score(attn)

        # 应用 softmax
        attn_score = self.softmax(attn, dim=-1)

        # 计算最终输出 attention_weights @ V
        return einsum(attn_score, self.v,
                      "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    
