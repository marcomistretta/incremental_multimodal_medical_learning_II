# Start with some standard imports.
import torch
import torch.nn as nn
from torch.nn.functional import relu


class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.layer = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        # self.layer = nn.Sequential(nn.Linear(128, 128))

    def forward(self, x):
        x = self.layer(x)
        return x

    # def update(self, optimizer, threshold, lr):
    #     beta1 = 0.9
    #     beta2 = 0.999
    #     eps = 1e-8
    #     for param in self.layer.parameters():
    #         if param.grad is not None:
    #             state = optimizer.state[param]
    #             if "exp_avg" not in state:
    #                 state["exp_avg"] = torch.zeros_like(param.data)
    #             if "exp_avg_sq" not in state:
    #                 state["exp_avg_sq"] = torch.zeros_like(param.data)
    #             if "step" not in state:
    #                 state["step"] = 0
    #             update = state["step"] + 1
    #             exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    #             grad = param.grad
    #             exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    #             exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    #             threshold_check = torch.abs(exp_avg) / (torch.sqrt(exp_avg_sq) + eps)
    #             mask = threshold_check > threshold
    #             param.data.add_(mask.float() * exp_avg * lr * update)
    #             state["step"] = update