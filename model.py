import torch
from torch import nn

class CustomReLUNetwork(nn.Module):
  def __init__(self):
    '''
    Fields
    - self.common_layers: common layers to get features
    - self.classification_layers: features to classification
    - self.regression_layers: features to regression
    '''

    super(CustomReLUNetwork, self).__init__()

    self.common_layers = nn.Sequential(
      nn.Linear(2, 8),
      nn.ReLU(),
      nn.Linear(8, 16),
      nn.ReLU(),
      nn.Linear(16, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU()
    )
    self.classification_layers = nn.Sequential(
      nn.Linear(8, 4),
      nn.ReLU(),
      nn.Linear(4, 1),
      nn.Sigmoid()
    )
    self.regression_layers = nn.Sequential(
      nn.Linear(8, 4),
      nn.ReLU(),
      nn.Linear(4, 1)
    )
  
  def forward(self, x: torch.Tensor):
    features = self.common_layers(x)
    y_cls = self.classification_layers(features)
    y_reg = self.regression_layers(features)
    return y_cls, y_reg

