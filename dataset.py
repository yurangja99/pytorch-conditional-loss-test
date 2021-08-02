import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CustomReLUDataset(Dataset):
  def __init__(self, item_cnt: int = 100, height: int = 4, width: int = 4):
    '''
    Parameters
    - item_cnt
    - height
    - width

    Fields
    - self.x: random data of size (batch_size, height, width)
    - self.y: labels for self.x
    '''

    super(CustomReLUDataset, self).__init__()

    self.x = torch.randn(item_cnt, height, width)
    self.y = ((self.x <= -1) * self.x) + \
      (-1 < self.x) * (self.x < 1) * torch.randn(self.x.shape) + \
      (1 <= self.x) * torch.pow(self.x, 2)

  def __len__(self):
    assert self.x.shape == self.y.shape
    return self.x.shape[0]

  def __getitem__(self, index: int):
    return {
      'x': self.x[index],
      'y': self.y[index]
    }

