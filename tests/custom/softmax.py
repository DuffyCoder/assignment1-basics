import torch
from jaxtyping import Float, Int
import torch.nn as nn
from torch import Tensor

class Softmax(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 top_p: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.top_p = top_p
        
    def forward(self, x: Float[Tensor, " ... d_out"], dim: Int):
        # 为了数值稳定性，沿指定维度计算最大值
        x_max = torch.max(x, dim=dim, keepdim=True)[0]

        # 减去最大值以避免数值溢出
        x_shifted = x - x_max

        # 计算指数
        exp_x = torch.exp(x_shifted / self.temperature)

        # 沿指定维度求和进行归一化
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

        # 返回softmax结果
        softmax_probs = exp_x / sum_exp_x
        
        if self.top_p <= 0 or self.top_p >= 1:
            return softmax_probs
        
        # Sort probabilities descending along the target dimension
        sorted_probs, sorted_indices = torch.sort(
            softmax_probs, dim=dim, descending=True
        )

        # Keep tokens whose cumulative probability before them is still below top_p
        cumulative_probs = torch.cumsum(sorted_probs, dim=dim)
        keep_mask = (cumulative_probs - sorted_probs) < self.top_p
        filtered_sorted = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))

        # Scatter filtered probabilities back to original order and re-normalize
        filtered_probs = torch.zeros_like(softmax_probs)
        filtered_probs.scatter_(dim, sorted_indices, filtered_sorted)
        normalizer = torch.clamp(filtered_probs.sum(dim=dim, keepdim=True), min=1e-12)
        return filtered_probs / normalizer
        