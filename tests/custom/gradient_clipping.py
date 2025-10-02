import torch
import torch.nn as nn
from collections.abc import Iterable

class GradientClipping(nn.Module):
    def __init__(
        self,
        max_l2_norm: float
    ):
        super().__init__()
        self.max_l2_norm = max_l2_norm
    
    def forward(self, parameters: Iterable[nn.Parameter]):
        if len(parameters) == 0:
            return
        grads = [param.grad for param in parameters if param.grad is not None]
        total_norm = torch.norm(torch.stack(grads), p=2)
        if total_norm > self.max_l2_norm:
            scale = self.max_l2_norm / (total_norm + 1e-6)
            for grad in grads:
                grad.mul_(scale)
