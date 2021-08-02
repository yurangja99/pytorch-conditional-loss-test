# Conditional Loss Test

## Purpose
- Purpose of this repository is to check whether conditional loss according to input values is possible in PyTorch model. 
- In this project, my model will be trained to act like below.
  - ```x <= -1``` : model is trained to ```y = x```
  - ```-1 < x < 1```: I don't train for this case (random)
  - ```1 <= x``` : model is trained to ```y = x * x```

## Usage
### Install packages
- Type ```pip3 install -r requirements.txt``` at root directory. 

### Train
- First, open ```train.py``` and set your configurations. 
- Type ```python3 train.py``` at root directory. 
- (If you want to run it as background process, type ```nohup python3 train.py > [sometrainname].log 2> [sometrainname].err &```)
- After the process, check output folder. Followings are examples of training results. 
[to be added]

### Test
- First, open ```test.py``` and set your configurations. 
- Type ```python3 test.py``` at root directory. 
- (If you want to run it as background process, type ```nohup python3 test.py > [sometestname].log 2> [sometestname].err &```)
- After the process, check output folder. Followings are examples of training results. 
[to be added]

## Codes
### [dataset.py](dataset.py)
- Define dataset for this repository. 
- Dataset
  - ```x```: torch.Tensor of (item_cnt, height, width)
  - ```y```: torch.Tensor of (item_cnt, height, width)
    - ```x <= -1```: ```y = x```
    - ```-1 < x < 1```: ```y = randn()```
    - ```1 <= x```: ```y = pow(x, 2)```

### [model.py](model.py)
- Define model for this repository. 
- Layers
  - ```Input```: ```(batch_size, height, width)```
  - ```Flatten```: ```(batch_size, height * width)```
  - ```Linear```: ```(batch_size, height * width)```
  - ```ReLU``` (normalization)
  - ```Linear```: ```(batch_size, height * width)```
  - ```View```: ```(batch_size, height, width)```

### [loss.py](loss.py)
- Define losses for this repository. 
- Losses
  - ```get_loss()```
    - normal ```MSELoss()```
    - For ```x``` in (-1, 1), model don't have to be trained so loss should be 0, but in this case, loss is not null according to y labels. 
  - ```get_conditional_loss()```
    - my custom loss
    - is auto_grad available for this loss? (purpose of this repository)
    - For ```x``` in (-1, 1), loss is set to zero. 

### [train.py](train.py)
- Define train process for this repository. 
- You can change configs as you want. 

### [test.py](train.py)
- Define test process for this repository. 
- You can change configs as you want. 
