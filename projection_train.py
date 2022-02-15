'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import torch.utils.data
from PIL import Image
import numpy as np

import os
import sys
import time
import argparse

from resnet_projector import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr = 1e-2
image_size = 256
train_path = './data/train/'
test_path = './data/test/'
batch_size = 64
# Data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.png')]
        print('==> Preparing data..')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


train_dataset = Dataset(train_path)
test_dataset = Dataset(test_path)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         num_workers=8)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         num_workers=8)
eig_val = np.load(train_path+'eig_val.npy')
smallest_eig_val = eig_val[-1]
print(smallest_eig_val)
eig_vec = np.load(train_path+'eig_vec.npy')
data_sample = np.load(train_path+'data_sample.npy')
true_keys = np.load(train_path+'key.npy')
mean = np.mean(data_sample,axis=0)
key_length = 64
starting_axis = 512-64
# Model
print('==> Building model..')
net = resnet50()
net = net.to(device)

def loss_function(output, label):
    a = output.unsqueeze(1) - label.unsqueeze(0)
    return torch.sqrt(torch.sum(a*a))/batch_size

def percentage_loss(output, label):
    a = abs((output.unsqueeze(1) - label.unsqueeze(0))/smallest_eig_val)
    return torch.sum(a)/(512*batch_size)

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
save_dir = './classifier_checkpoint'
isExist = os.path.exists(save_dir)
if not isExist:
    os.makedirs(save_dir)

# Training
def train(epoch, batch_size):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    net.train()
    train_label = np.load('./data/train/key.npy')
    train_label = torch.tensor(train_label)
    train_label = torch.split(train_label, batch_size)
    for batch, target in zip(trainloader, train_label):
        optimizer.zero_grad()
        inputs = batch.to(device)
        target = target.to(device)
        outputs = net(inputs)
        loss = loss_function(outputs, target)
        per_loss = percentage_loss(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('l2 loss: {}'.format(loss))
    if (epoch + 1) % 1 == 0:
        gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch + 1) + ".pth")
        torch.save(net.state_dict(), gen_save_file)


num_epoch = 200
for epoch in range(start_epoch, start_epoch+num_epoch):
    train(epoch, batch_size)
