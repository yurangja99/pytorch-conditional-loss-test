import torch
from torch import nn

class CustomReLULoss:
  def __init__(self):
    pass

  def get_loss(self, y: torch.Tensor, gt: torch.Tensor):
    return nn.MSELoss()(y, gt)

  def get_conditional_loss(self, x: torch.Tensor, y: torch.Tensor, gt: torch.Tensor):
    mask = (-1 < x) * (x < 1)
    loss = torch.pow(y - gt, 2)
    loss[mask] = 0.0
    return loss.mean()