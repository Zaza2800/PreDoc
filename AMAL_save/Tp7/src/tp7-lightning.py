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
from pytorch_lightning.loggers import TensorBoardLogger

from datamaestro import prepare_dataset
from torch.utils.data import TensorDataset, DataLoader

####Lightning 
import pytorch_lightning as pl 
from pytorch_lightning import Trainer

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
learning_rate = 0.00001
batch_size    = 300
class DNN(pl.LightningModule):
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
    
    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = F.cross_entropy(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs
    
    def train_loader(self): 
        ds = prepare_dataset("com.lecun.mnist")

        train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
        test_img, test_labels = ds.test.images.data(), ds.test.labels.data()
        
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
        
        new_train_img = torch.from_numpy(new_train_img)
        new_train_labels = torch.from_numpy(new_train_labels)

        new_test_img = torch.from_numpy(new_test_img)
        new_test_labels = torch.from_numpy(new_test_labels)
        new_train_img = new_train_img /255
        new_test_img = new_test_img /255

        new_train_img = torch.prod(new_train_img, dim=2)
        new_test_img = torch.prod(new_test_img, dim=2) 
        
        dataset_train = TensorDataset(new_train_img, new_train_labels)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size = self.batch_size, shuffle=True)
        dataset_test = TensorDataset(new_test_img, new_test_labels)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size = self.batch_size, shuffle=True)

        return dataloader_train, dataloader_test   
        

        
        
        
    
  
 
    
if __name__ == '__main__':
    LOG_PATH = "/tmp/runs/lightning_logs"
    

    trainer = Trainer(fast_dev_run=True,logger=LOG_PATH,max_epochs=100) # test if model works
    input_size = new_train_img.shape[1]
    hidden_size = 100
    output_size = 10
    model   = DNN(input_size, hidden_size, output_size)
    logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)
    trainer.fit(model)



