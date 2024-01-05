from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

import matplotlib.pyplot as plt
import numpy as np 

# Nombre de stations utilisé
CLASSES = 3
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 64

HIDDEN_SIZE = 20



matrix_train, matrix_test = torch.load(open("hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)




#  TODO:  Question 3 : Prédiction de séries temporelles



rnn = RNN(input_size=DIM_INPUT, hidden_size=HIDDEN_SIZE, output_size=DIM_INPUT)
rnn.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

NB_ITERS = 25
loss_train_per_epoch = np.zeros(NB_ITERS)
loss_test_per_epoch = np.zeros(NB_ITERS)


def train_loop(data, device, HIDDEN_SIZE, CLASSES, model, criterion, epoch_loss): 
    for i, (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)  # x.shape  batch_size*length*input_dim
        x = torch.permute(x, (1,0,2,3))
        y = torch.permute(y, (1,0,2,3))

        for c in range(CLASSES): 
            x_c = x[:,:,c,:].to(device) # torch.Size([32, 19, 2])
            y_c = y[:,:,c,:].to(device)

            optimizer.zero_grad()
            
            h = torch.zeros((x_c.shape[1], HIDDEN_SIZE), device=device)  # taille batch_size*hidden_size
            output_sequence = model.forward(x_c, h)
            last_hidden_state = output_sequence[-1, :, :].to(device)
            y_hat = model.decode(last_hidden_state)

            loss = criterion(y_hat, y_c)    
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss

def test_loop(data, device, HIDDEN_SIZE, model, criterion, epoch_loss):
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            x = torch.permute(x, (1,0,2,3))
            y = torch.permute(y, (1,0,2,3))

            for c in range(CLASSES): 
                x_c = x[:,:,c,:].to(device) # torch.Size([32, 19, 2])
                y_c = y[:,:,c,:].to(device)

                optimizer.zero_grad() 

                h = torch.zeros((x_c.shape[1], HIDDEN_SIZE), device=device)  # taille batch_size*hidden_size
                output_sequence = model.forward(x_c, h)
                last_hidden_state = output_sequence[-1, :, :].to(device)
                y_hat = model.decode(last_hidden_state)
            
                loss = criterion(y_hat, y_c)
                epoch_loss += loss.item()

        return epoch_loss

for epoch in range(NB_ITERS):
    print("\n---> Epoch :", epoch, "/",NB_ITERS)
    epoch_loss_train = 0
    epoch_loss_test = 0

    epoch_loss_train = train_loop(data_train, device, HIDDEN_SIZE, CLASSES, rnn, criterion, epoch_loss_train)
    epoch_loss_train = train_loop(data_test, device, HIDDEN_SIZE, CLASSES, rnn, criterion, epoch_loss_test)

    loss_train_per_epoch[epoch] = epoch_loss_train / len(data_train)
    loss_test_per_epoch[epoch] = epoch_loss_test / len(data_test)



fig, ax = plt.subplots(figsize = (8,8))

ax.plot(range(1,len(loss_train_per_epoch)+1), loss_train_per_epoch, label="loss_train_per_epoch")
ax.plot(range(1,len(loss_test_per_epoch)+1), loss_test_per_epoch, label="loss_test_per_epoch")
ax.legend()
ax.set(title = 'losses on train & test', xlabel = 'epoch', ylabel = 'loss')

plt.savefig('result_plot_exo3.png')
plt.show()