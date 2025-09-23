import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum, rearrange
from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 weights: dict[str, Tensor],
                 in_features: Float[Tensor, " batch seq_len d_model"]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights
        self.in_features = in_features
        self.token_positions = torch.arange(self.in_features.shape[-2])
        self.rmsnorm_1 = RMSNorm(self.d_model, 
                                self.weights['ln1.weight'])
        self.rmsnorm_2 = RMSNorm(self.d_model, 
                                self.weights['ln2.weight'])
        self.mha = MultiheadSelfAttention(
            self.d_model, 
            self.num_heads, 
            self.weights['attn.q_proj.weight'], 
            self.weights['attn.k_proj.weight'], 
            self.weights['attn.v_proj.weight'], 
            self.weights['attn.output_proj.weight'], 
            self.rmsnorm_1(self.in_features), 
            self.max_seq_len, 
            self.theta, 
            self.token_positions)
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_ff, self.weights)
        
    def forward(self):
        x_attn = self.mha() + self.in_features
        x_ffn = self.ffn(self.rmsnorm_2(x_attn)) + x_attn
        return x_ffn
    
    