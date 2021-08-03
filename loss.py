import torch
from torch import nn

class CustomReLULoss:
  def __init__(self):
    pass

  def get_loss(
    self, 
    y_cls: torch.Tensor, 
    y_reg: torch.Tensor, 
    gt_cls: torch.Tensor, 
    gt_reg: torch.Tensor
  ):
    loss_cls = nn.BCELoss()(y_cls, gt_cls)
    loss_reg = nn.MSELoss()(y_reg, gt_reg)
    return loss_cls + loss_reg, loss_cls, loss_reg

  def get_conditional_loss(
    self, 
    y_cls: torch.Tensor, 
    y_reg: torch.Tensor, 
    gt_cls: torch.Tensor, 
    gt_reg: torch.Tensor
  ):
    loss_cls = nn.BCELoss()(y_cls, gt_cls)

    err_reg = torch.pow(y_reg - gt_reg, 2) * (gt_cls > 0.5)
    loss_reg = err_reg.mean()

    return loss_cls + loss_reg, loss_cls, loss_reg

