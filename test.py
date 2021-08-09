import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt

from model import CustomReLUNetwork
from dataset import CustomReLUDataset
from loss import CustomReLULoss

#######################
# Assert cuda is available
#######################
assert torch.cuda.is_available()

#######################
# Config
#######################
# dataset
TEST_ITEM_CNT = 100

# testing
USE_CONDITIONAL_LOSS = True

# testing resources
IS_PT_FILE = False
MODEL_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  'train-21-08-09-20-13-56',
  'final_weights'
)
SAVE_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  'test-{}'.format(datetime.now(timezone(timedelta(hours=9))).strftime('%y-%m-%d-%H-%M-%S'))
)

#######################
# Make output directory
#######################
os.mkdir(SAVE_PATH)

#######################
# Log config
#######################
with open(os.path.join(SAVE_PATH, 'readme.txt'), 'w') as f:
  f.write('=== TEST ===\n')
  f.write('Test data count {}\n'.format(TEST_ITEM_CNT))
  f.write('Is pt file (otherwise, weight) {}\n'.format(IS_PT_FILE))
  f.write('Model path {}\n'.format(MODEL_PATH))
  f.write('Use conditional(custom) loss: {}\n'.format(USE_CONDITIONAL_LOSS))
  f.write('Save at {}\n'.format(SAVE_PATH))

#######################
# Setting models and datasets
#######################
test_dataset = CustomReLUDataset(item_cnt=TEST_ITEM_CNT)

test_dataloader = DataLoader(
  test_dataset, 
  batch_size=1, 
  shuffle=False, 
  num_workers=2, 
  drop_last=False
)

if IS_PT_FILE:
  network = torch.load(MODEL_PATH)
else:
  network = CustomReLUNetwork()
  checkpoint = torch.load(MODEL_PATH)
  if type(checkpoint) == dict:
    checkpoint = checkpoint['state_dict']
  network.load_state_dict(checkpoint)

network.cuda()

loss_calculator = CustomReLULoss()

#######################
# Initialize result
#######################
csv = pd.DataFrame(columns=['index', 'loss'])
x_all = []
gt_cls_all = []
gt_reg_all = []
y_cls_all = []
y_reg_all = []

#######################
# Testing
#######################
with torch.no_grad():
  # initialize test loss
  test_loss = 0
  test_loss_cls = 0
  test_loss_reg = 0
  test_cnt = 0

  for idx, data in enumerate(tqdm(test_dataloader)):
    # get test data
    x = data['x'].cuda()
    gt_cls = data['y_cls'].cuda()
    gt_reg = data['y_reg'].cuda()

    # predict y_pred and calculate loss
    y_cls, y_reg = network(x)
    if USE_CONDITIONAL_LOSS:
      loss, loss_cls, loss_reg = loss_calculator.get_conditional_loss(
        y_cls=y_cls, 
        y_reg=y_reg, 
        gt_cls=gt_cls, 
        gt_reg=gt_reg
      )
    else:
      loss, loss_cls, loss_reg = loss_calculator.get_loss(
        y_cls=y_cls, 
        y_reg=y_reg, 
        gt_cls=gt_cls, 
        gt_reg=gt_reg
      )
    test_loss += loss.item()
    test_loss_cls += loss_cls.item()
    test_loss_reg += loss_reg.item()

    # append y and y_pred to result set
    x_all += [batch[0] for batch in x.tolist()]
    gt_cls_all += [batch[0] for batch in gt_cls.tolist()]
    gt_reg_all += [batch[0] for batch in gt_reg.tolist()]
    y_cls_all += [batch[0] for batch in y_cls.tolist()]
    y_reg_all += [batch[0] for batch in y_reg.tolist()]

    # append loss to csv
    csv = csv.append(
      {
        'index': idx, 
        'loss': loss.item(), 
        'loss_cls': loss_cls.item(), 
        'loss_reg': loss_reg.item()
      }, 
      ignore_index=True
    )

    test_cnt += 1
  
  # append val loss
  csv = csv.append(
    {
      'index': 'AVG', 
      'loss': test_loss / test_cnt, 
      'loss_cls': test_loss_cls / test_cnt, 
      'loss_reg': test_loss_reg / test_cnt
    },
    ignore_index=True
  )

#######################
# Save testing result
#######################
# save loss
csv.to_csv(
  os.path.join(SAVE_PATH, 'loss-test.csv'), 
  header=True, 
  index=False
)

# visualize results
plt.title('Classification Result')
plt.plot(
  [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0], 
  [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], 
  'k--', label='optimal'
)
plt.scatter(x_all, gt_cls_all, label='label')
plt.scatter(x_all, y_cls_all, label='pred')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'result-cls.png'))
plt.close()

plot_x1 = np.linspace(-1.0, 0.0, 100)
plot_y1 = -4.0 * plot_x1
plot_x2 = np.linspace(1.0, 2.0, 100)
plot_y2 = 16.0 * np.power(plot_x2 - 1.5, 2)

plt.title('Regression Result')
plt.plot(plot_x1, plot_y1, 'k--', label='optimal')
plt.plot(plot_x2, plot_y2, 'k--')
plt.scatter(x_all, gt_reg_all, label='label')
plt.scatter(x_all, y_reg_all, label='pred')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'result-reg.png'))
plt.close()