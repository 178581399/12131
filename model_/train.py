import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import argparse
import time
import copy
import numpy as np

import model_vgg
from data_set import data_set_train, data_set_val
from train_code import train_model

# parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
# parser.add_argument('--model_p', type = str, default = 'vgg11', help = 'CNN architecture')
# parser.add_argument('--dataset_train', type = str, default = './CK/train', help = 'dataset_train')
# parser.add_argument('--dataset_val', type = str, default = './CK/val', help = 'dataset_val')
# parser.add_argument('--fold', default = 1, type = int, help = 'k fold number')
# parser.add_argument('--bs', default = 128, type = int, help = 'batch_size')
# parser.add_argument('--lr', default = 0.01, type = float, help = 'learning rate')
# parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
# opt = parser.parse_args()

bs = 128
lr = 0.0001
num_epochs = 100
model_p = 'vgg11'
dataset_train = './CK/train/'
dataset_val = './CK/val/'

model_ft = model_vgg.VGG()
model_res = models.vgg11(True)

model_ft_dict = model_ft.state_dict()
model_res_dict = model_res.state_dict()

# 选取可以更新的参数
model_dict = {k: v for k, v in model_res_dict.items() 
                if k in model_ft_dict 
                and v.shape == model_ft_dict[k].shape}
# 更新参数
model_ft_dict.update(model_dict)
model_ft.load_state_dict(model_ft_dict)

# 获取数据集
dataset_loader_train = data_set_train(bs, dataset_train)
dataset_loader_val = data_set_val(bs, dataset_val)
datasetloaders = {'train': dataset_loader_train, 'val': dataset_loader_val}

# 将模型加到GPU中
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ft.to(device)

dataset_sizes = {'train': 800, 'val': 181}

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=0.01)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)

model_k = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, datasetloaders, dataset_sizes, device, num_epochs)

# 保存模型
torch.save(model_k.state_dict(), './params.pkl')














