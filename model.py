import torch
from torch import nn

class CustomReLUNetwork(nn.Module):
  def __init__(self, input_height: int = 4, input_width: int = 4):
    '''
    Parameters
    - input_height
    - input_width

    Fields
    - self.output_height: same with input_height
    - self.output_width: same with input_width
    - self.input_pixels: input width x input height
    - self.custom_relu: custom sequential model for ReLU operation. 
    '''

    super(CustomReLUNetwork, self).__init__()
    
    self.input_pixels = input_height * input_width
    self.output_height = input_height
    self.output_width = input_width

    self.custom_relu = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.input_pixels, self.input_pixels),
      nn.ReLU(),
      nn.Linear(self.input_pixels, self.input_pixels)
    )
  
  def forward(self, x: torch.Tensor):
    y = self.custom_relu(x)
    return y.view((-1, self.output_height, self.output_width))

