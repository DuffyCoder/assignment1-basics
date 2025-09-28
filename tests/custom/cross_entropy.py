import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,
                inputs: Float[Tensor, " batch_size vocab_size"], 
                targets: Int[Tensor, " batch_size"]):
        self.inputs = inputs
        self.targets = targets
        
        x_max = torch.max(self.inputs, dim=-1, keepdim=True)[0]
        x_shifted = self.inputs - x_max
        log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=-1, keepdim=True))
        logits = - (x_shifted - log_sum_exp)
        
        row_indices = torch.arange(self.targets.size(0))
        return logits[row_indices, self.targets].mean()