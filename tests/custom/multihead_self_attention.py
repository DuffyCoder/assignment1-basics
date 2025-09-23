import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import rearrange, einsum

from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 q_proj_weight: Float[Tensor, "d_model d_in"],
                 k_proj_weight: Float[Tensor, "d_model d_in"],
                 v_proj_weight: Float[Tensor, "d_model d_in"],
                 o_proj_weight: Float[Tensor, "d_model d_v"],
                 in_features: Float[Tensor, " ... seq_len d_in"],
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight
        self.in_features = in_features
        self.seq_len = in_features.shape[-2]

    def forward(self):
        q = k = v = self.in_features
        q = einsum(q, self.q_proj_weight, 
                   " ... seq_len d_in, d_model d_in -> ... seq_len d_model")
        q = rearrange(q, " ... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                      num_heads=self.num_heads)
        
        k = einsum(k, self.k_proj_weight, 
                   " ... seq_len d_in, d_model d_in -> ... seq_len d_model")
        k = rearrange(k, " ... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
                      num_heads=self.num_heads)
        
        v = einsum(v, self.v_proj_weight, 
                   " ... seq_len d_in, d_model d_in -> ... seq_len d_model")
        v = rearrange(v, " ... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", 
                      num_heads=self.num_heads)
        
        mask = torch.tril(torch.ones(
            self.seq_len, self.seq_len))
        
        attn = ScaledDotProductAttention(q, k, v, mask)
        attn_output = attn()
        attn_output = rearrange(attn_output, 
                                " ... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        
        output = einsum(attn_output, self.o_proj_weight,
                        " ... seq_len d_v, d_model d_v -> ... seq_len d_model")
        return output
        