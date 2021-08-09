import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
TRAIN_ITEM_CNT = 1000
VAL_ITEM_CNT = 100
TRAIN_SHUFFLE = True
TRAIN_NUM_WORKERS = 8
TRAIN_DROP_LAST = False

# training
BATCH_SIZE = 32
MAX_EPOCHS = 400
INIT_LR = 0.001
WEIGHT_DECAY = 0.00005
LR_DROP_MILESTONES = [250, 350]
VAL_STEP = 10
SAVE_STEP = 100
USE_CONDITIONAL_LOSS = True

# training result
SAVE_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  'train-{}'.format(datetime.now(timezone(timedelta(hours=9))).strftime('%y-%m-%d-%H-%M-%S'))
)

#######################
# Make output directory
#######################
os.mkdir(SAVE_PATH)

#######################
# Log config
#######################
with open(os.path.join(SAVE_PATH, 'readme.txt'), 'w') as f:
  f.write('=== DATA ===\n')
  f.write('Train data count {}\n'.format(TRAIN_ITEM_CNT))
  f.write('Val data count {}\n\n'.format(VAL_ITEM_CNT))
  f.write('=== TRAINING ===\n')
  f.write('Shuffle: {}\n'.format(TRAIN_SHUFFLE))
  f.write('# of workers: {}\n'.format(TRAIN_NUM_WORKERS))
  f.write('Drop last: {}\n'.format(TRAIN_DROP_LAST))
  f.write('Batch size: {}\n'.format(BATCH_SIZE))
  f.write('Max epochs: {}\n'.format(MAX_EPOCHS))
  f.write('Initial lr: {}\n'.format(INIT_LR))
  f.write('Weight decay: {}\n'.format(WEIGHT_DECAY))
  f.write('Lr drop milestones: {}\n'.format(LR_DROP_MILESTONES))
  f.write('Validation step: {}\n'.format(VAL_STEP))
  f.write('Save weights step: {}\n'.format( SAVE_STEP))
  f.write('Use conditional(custom) loss: {}\n'.format(USE_CONDITIONAL_LOSS))
  f.write('Save path: {}\n'.format(SAVE_PATH))

#######################
# Setting models and datasets
#######################
train_dataset = CustomReLUDataset(item_cnt=TRAIN_ITEM_CNT)
val_dataset = CustomReLUDataset(item_cnt=VAL_ITEM_CNT)

train_dataloader = DataLoader(
  train_dataset, 
  batch_size=BATCH_SIZE, 
  shuffle=TRAIN_SHUFFLE, 
  num_workers=TRAIN_NUM_WORKERS, 
  drop_last=TRAIN_DROP_LAST
)
val_dataloader = DataLoader(
  val_dataset, 
  batch_size=1, 
  shuffle=False, 
  num_workers=2, 
  drop_last=False
)

network = CustomReLUNetwork().cuda()

loss_calculator = CustomReLULoss()
optimizer = torch.optim.Adam(
  network.parameters(), 
  lr=INIT_LR, 
  weight_decay=WEIGHT_DECAY
)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
  optimizer, 
  milestones=LR_DROP_MILESTONES
)

writer = SummaryWriter(log_dir=os.path.join('runs', os.path.basename(SAVE_PATH)))

train_loss_history = []
train_loss_cls_history = []
train_loss_reg_history = []
val_loss_history = []
val_loss_cls_history = []
val_loss_reg_history = []

#######################
# Plot Model Structure
#######################
writer.add_graph(network, torch.rand(1, 2).cuda())

#######################
# Training
#######################
for epoch in tqdm(range(1, MAX_EPOCHS + 1)):
  # switch network to train mode
  network.train()

  # initialize train_loss
  train_loss = 0
  train_loss_cls = 0
  train_loss_reg = 0
  train_cnt = 0

  for data in train_dataloader:
    # initizlie optimizer
    optimizer.zero_grad()

    # get train data
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
    train_loss += loss.item()
    train_loss_cls += loss_cls.item()
    train_loss_reg += loss_reg.item()

    # train
    loss.backward()
    optimizer.step()

    train_cnt += 1
  
  # update learning rate
  lr_scheduler.step()

  # append train loss
  train_loss_history.append(train_loss / train_cnt)
  train_loss_cls_history.append(train_loss_cls / train_cnt)
  train_loss_reg_history.append(train_loss_reg / train_cnt)
  writer.add_scalar('Loss/train', train_loss / train_cnt, epoch)
  writer.add_scalar('Loss_cls/train', train_loss_cls / train_cnt, epoch)
  writer.add_scalar('Loss_reg/train', train_loss_reg / train_cnt, epoch)

  # save weights in specific steps
  if epoch % SAVE_STEP == 0:
    torch.save(
      {
        'epoch': epoch, 
        'state_dict': network.state_dict(), 
        'optimizer': optimizer.state_dict()
      }, 
      f=os.path.join(SAVE_PATH, 'epoch_{}'.format(epoch))
    )
  
  # validation
  if epoch % VAL_STEP == 0:
    # switch network to evaluation mode
    network.eval()

    with torch.no_grad():
      # initialize val_loss
      val_loss = 0
      val_loss_cls = 0
      val_loss_reg = 0
      val_cnt = 0
      
      for data in val_dataloader:
        # get val data
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
        val_loss += loss.item()
        val_loss_cls += loss_cls.item()
        val_loss_reg += loss_reg.item()

        val_cnt += 1
      
      # append val loss
      val_loss_history.append(val_loss / val_cnt)
      val_loss_cls_history.append(val_loss_cls / val_cnt)
      val_loss_reg_history.append(val_loss_reg / val_cnt)
      writer.add_scalar('Loss/val', val_loss / val_cnt, epoch)
      writer.add_scalar('Loss_cls/val', val_loss_cls / val_cnt, epoch)
      writer.add_scalar('Loss_reg/val', val_loss_reg / val_cnt, epoch)

# flush and close tensorboard writer
writer.flush()
writer.close()

#######################
# Save training results
#######################
# save final weights
torch.save(
  network.state_dict(), 
  f=os.path.join(SAVE_PATH, 'final_weights')
)

# save pt file
scripted_model=torch.jit.script(network)
scripted_model.save(os.path.join(SAVE_PATH, 'model.pt'))
optimal_scripted_model = optimize_for_mobile(scripted_model)
optimal_scripted_model.save(os.path.join(SAVE_PATH, 'optimal_model.pt'))

# save losses
train_epochs = list(range(1, MAX_EPOCHS + 1))
val_epochs = list(range(VAL_STEP, MAX_EPOCHS + 1, VAL_STEP))
pd.DataFrame({
  'epoch': train_epochs, 
  'loss': train_loss_history, 
  'loss_cls': train_loss_cls_history, 
  'loss_reg': train_loss_reg_history
}).to_csv(
  os.path.join(SAVE_PATH, 'loss-train.csv'), 
  header=True, 
  index=False
)
pd.DataFrame({
  'epoch': val_epochs, 
  'loss': val_loss_history, 
  'loss_cls': val_loss_cls_history, 
  'loss_reg': val_loss_reg_history
}).to_csv(
  os.path.join(SAVE_PATH, 'loss-val.csv'), 
  header=True, 
  index=False
)

# plot losses
plt.title('Train / Validation Loss')
plt.plot(train_epochs, train_loss_history, label='train')
plt.plot(val_epochs, val_loss_history, label='val')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'loss.png'))
plt.close()

plt.title('Train Loss')
plt.plot(train_epochs, train_loss_history, label='total')
plt.plot(train_epochs, train_loss_cls_history, label='cls')
plt.plot(train_epochs, train_loss_reg_history, label='reg')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'loss-train.png'))
plt.close()

plt.title('Val Loss')
plt.plot(val_epochs, val_loss_history, label='total')
plt.plot(val_epochs, val_loss_cls_history, label='cls')
plt.plot(val_epochs, val_loss_reg_history, label='reg')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'loss-val.png'))
plt.close()

