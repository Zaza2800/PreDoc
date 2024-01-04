import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms 
from tqdm import tqdm
import click

from datamaestro import prepare_dataset
from torch.utils.data import TensorDataset, DataLoader


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var

def calculate_entropy(model, dataloader):
    model.eval()
    total_entropy = 0
    total_samples = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.float()
            output = model(X)
            probabilities = F.softmax(output, dim=1)
            log_probabilities = F.log_softmax(output, dim=1)
            entropy = -(probabilities * log_probabilities).sum(dim=1).mean().item()
            total_entropy += entropy * X.size(0)
            total_samples += X.size(0)

################
##### Data #####
################

ds = prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

## Data Augmentation ##
trans = transforms.Compose([transforms.ToPILImage(), 
                                     transforms.ColorJitter(brightness=0.5), 
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor()])

transform_img = torch.stack([trans(img) for img in train_img])
train_labels_t= train_labels


train_size = len(train_img)
test_size  = len(test_img) 

percentage = 0.05
num_train = int(train_size*percentage)
num_test  = int(test_size*percentage)

train_indices = torch.randperm(train_size)[:num_train]
test_indices = torch.randperm(test_size)[:num_test]

new_train_img = train_img[train_indices]
new_train_labels = train_labels[train_indices]

new_test_img = test_img[test_indices]
new_test_labels = test_labels[test_indices]

## Concatenate for data augmentation
#new_train_img = torch.tensor(new_train_img)
#print(type(new_train_img))
#new_train_img = torch.cat((new_train_img,transform_img.squeeze(dim=1)), dim = 0)
#new_train_labels = torch.cat((new_train_labels,train_labels), dim = 0)
## Normalize ###
new_train_img = torch.from_numpy(new_train_img)
new_train_labels = torch.from_numpy(new_train_labels)

new_test_img = torch.from_numpy(new_test_img)
new_test_labels = torch.from_numpy(new_test_labels)
new_train_img = new_train_img /255
new_test_img = new_test_img /255

new_train_img = torch.prod(new_train_img, dim=2)
new_test_img = torch.prod(new_test_img, dim=2) 

################
#### Model #####
################

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.softmax= nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.softmax(self.layer4(x))
        return x

    
 

input_size = new_train_img.shape[1]
hidden_size = 100
output_size = 10
batch_size = 300
nb_epoch = 1000

model_dl = DNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_dl.parameters(), lr=0.00001)
#not working well
#l1_lambda = 0.001 
l1_lambda = 0

dataset_train = TensorDataset(new_train_img, new_train_labels)
dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size = batch_size, shuffle=True)

dataset_test = TensorDataset(new_test_img, new_test_labels)
dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size = batch_size, shuffle=True)


LOSS_train = []
LOSS_test  = []

log_dir = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
writer = SummaryWriter(log_dir)

for epoch in range (nb_epoch):
  total_loss_train = 0
  model_dl.train()
  for X_batch, Y_batch in dataloader_train :
      optimizer.zero_grad()
      X_batch = X_batch.float()
      prediction = model_dl(X_batch)
      loss = criterion(prediction,Y_batch)
      l1_loss = 0
      for param in model_dl.parameters(): 
        l1_loss += torch.norm(param,p=1)
        
      loss += l1_lambda*l1_loss
      loss.backward()
      total_loss_train += loss.item()
      for name, param in model_dl.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'{name}_grad', param.grad.clone().cpu().data.numpy(), epoch)

      optimizer.step()

  average_loss_train = total_loss_train / len(dataloader_train)
  LOSS_train.append(average_loss_train)

  total_loss_test = 0
  model_dl.eval()
  for X_test, Y_test in dataloader_test : 
    X_test = X_test.float()
    prediction = model_dl(X_test)
    loss = criterion(prediction, Y_test)
    total_loss_test += loss.item()
    
  average_loss_test = total_loss_test / len(dataloader_test)
  LOSS_test.append(average_loss_test)
  writer.add_scalar('Loss/train', average_loss_train, epoch)
  writer.add_scalar('Loss/test', average_loss_test, epoch)
    
  #make entropie


  print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss trainning : {average_loss_train:.4f}, Loss test : {average_loss_test:.4f}')
    
writer.close()
import matplotlib.pyplot as plt

plt.plot(range(nb_epoch), LOSS_train, range(nb_epoch), LOSS_test)
plt.xlabel('nb_epoch')
plt.ylabel('average_loss')
plt.savefig('notuning.png')
plt.show()

