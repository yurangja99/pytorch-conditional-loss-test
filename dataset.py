import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CustomReLUDataset(Dataset):
  def __init__(self, item_cnt: int = 100):
    '''
    Parameters
    - item_cnt

    Fields
    - self.x: random data of size (batch_size, 2)
    - self.y_cls: classification labels for self.x of size (batch_size, 1)
    - self.y_reg: regression labels for self.x of size (batch_size, 1)
    '''

    super(CustomReLUDataset, self).__init__()

    x = torch.unsqueeze(torch.linspace(-2.0, 2.0, item_cnt), dim=-1)
    
    y_cls = ((-1.0 <= x) * (x <= 0.0) + (1.0 <= x)) * 1.0
    y_reg = (x < -1.0) * torch.randn(x.shape) + \
      (-1.0 <= x) * (x <= 0.0) * (-4.0 * x) + \
      (0.0 < x) * (x < 1.0) * torch.randn(x.shape) + \
      (1.0 <= x) * 16.0 * torch.pow(x - 1.5, 2)
    
    self.x = torch.cat([x, x ** 2], dim=-1)
    self.y_cls = y_cls
    self.y_reg = y_reg

  def __len__(self):
    assert self.x.shape[0] == self.y_cls.shape[0]
    assert self.x.shape[0] == self.y_reg.shape[0]
    return self.x.shape[0]

  def __getitem__(self, index: int):
    return {
      'x': self.x[index],
      'y_cls': self.y_cls[index], 
      'y_reg': self.y_reg[index]
    }

