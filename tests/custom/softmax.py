import torch
from jaxtyping import Float, Int
import torch.nn as nn
from torch import Tensor

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Float[Tensor, " ... d_out"], dim: Int):
        # 为了数值稳定性，沿指定维度计算最大值
        x_max = torch.max(x, dim=dim, keepdim=True)[0]

        # 减去最大值以避免数值溢出
        x_shifted = x - x_max

        # 计算指数
        exp_x = torch.exp(x_shifted)

        # 沿指定维度求和进行归一化
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

        # 返回softmax结果
        return exp_x / sum_exp_x
        