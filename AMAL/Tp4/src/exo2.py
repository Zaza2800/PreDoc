from utils import RNN, device, SampleMetroDataset
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np 

torch.autograd.set_detect_anomaly(True)

# Nombre de stations utilisé
CLASSES = 2
#Longueur des séquences
LENGTH = 10
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 64

HIDDEN_SIZE = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matrix_train, matrix_test = torch.load(open("hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)



#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
rnn = RNN(input_size=DIM_INPUT, hidden_size=HIDDEN_SIZE, output_size=CLASSES).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)


NB_ITERS = 25
loss_train_per_epoch = np.zeros(NB_ITERS)
loss_test_per_epoch = np.zeros(NB_ITERS)

acc_train_per_epoch = np.zeros(NB_ITERS)
acc_test_per_epoch = np.zeros(NB_ITERS)


def train_loop(data, device, HIDDEN_SIZE, model, criterion, epoch_loss, epoch_acc): 
    for i, (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)  # x.shape  batch_size*length*input_dim
        x = torch.permute(x, (1, 0, 2))
        optimizer.zero_grad()

        h = torch.zeros((x.shape[1], HIDDEN_SIZE), device=device)  # taille batch_size*hidden_size
        output_sequence = model.forward(x, h)
        last_hidden_state = output_sequence[-1, :, :].to(device)
        y_hat = model.decode(last_hidden_state)

        loss = criterion(y_hat, y)    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        acc = torch.sum(y_hat.argmax(1) == y)/len(y)
        epoch_acc += acc

    return epoch_loss, epoch_acc

def test_loop(data, device, HIDDEN_SIZE, model, criterion, epoch_loss, epoch_acc):
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            x = torch.permute(x, (1, 0, 2))

            h = torch.zeros((x.shape[1], HIDDEN_SIZE), device=device)  # taille batch_size*hidden_size
            output_sequence = model.forward(x, h)
            last_hidden_state = output_sequence[-1, :, :].to(device)
            y_hat = model.decode(last_hidden_state)
            loss = criterion(y_hat, y)
            epoch_loss += loss.item()

            acc = torch.sum(y_hat.argmax(1) == y)/len(y)
            epoch_acc += acc
    
    return epoch_loss, epoch_acc 

for epoch in range(NB_ITERS):
    print("\n---> Epoch :", epoch+1, "/",NB_ITERS)
    epoch_loss_train = 0
    epoch_loss_test  = 0

    epoch_acc_train = 0
    epoch_acc_test = 0

    
    epoch_loss_train, epoch_acc_train = train_loop(data_train, device, HIDDEN_SIZE, rnn, criterion, epoch_loss_train, epoch_acc_train)
    epoch_loss_test, epoch_acc_test = test_loop(data_test, device, HIDDEN_SIZE, rnn, criterion, epoch_loss_test, epoch_acc_test)
    
        

    loss_train_per_epoch[epoch] = epoch_loss_train / len(data_train)
    loss_test_per_epoch[epoch] = epoch_loss_test / len(data_test)

    acc_train_per_epoch[epoch] = epoch_acc_train / len(data_train)
    acc_test_per_epoch[epoch] = epoch_acc_test / len(data_test)



fig, axes = plt.subplots((2), figsize = (8,12))

axes[0].plot(range(1,len(loss_train_per_epoch)+1), loss_train_per_epoch, label="train loss /epoch")
axes[0].plot(range(1,len(loss_test_per_epoch)+1), loss_test_per_epoch, label="test loss /epoch")

axes[1].plot(range(1,len(acc_train_per_epoch)+1), acc_train_per_epoch, label="acc_train_per_epoch")
axes[1].plot(range(1,len(acc_test_per_epoch)+1), acc_test_per_epoch, label="acc_test_per_epoch")

axes[0].legend()
axes[1].legend()

axes[0].set(title = 'losses on train & test', xlabel = 'epoch', ylabel = 'loss')
axes[1].set(title = 'accuracies on train & test', xlabel = 'epoch', ylabel = 'accuracy')

plt.savefig('result_plot_exo2.png')
plt.show()