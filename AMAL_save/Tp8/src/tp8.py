import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import torch.optim as optim
import torch.nn.functional as F
from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO:
        
class EmotionCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_size, dropout_prob):
        super(EmotionCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        max_indices = []
        x = self.embedding(x.long()).unsqueeze(1)  # Add channel dimension
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        for i in range(len(x)):
            pooled, indices = nn.functional.max_pool1d(x[i], kernel_size=x[i].size(2), stride=500, return_indices = True)
            x[i] = pooled.squeeze(2)
            max_indices.append(indices.tolist())
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        output = self.softmax(x)
        return output, max_indices
        




embedding_dim = 128
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 3  # Three categories for emotions
dropout_prob = 0.5

model =  EmotionCNN(vocab_size, embedding_dim, num_filters, filter_sizes, output_size, dropout_prob)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 10e-4)

## Training 

def training_dl(nb_epoch, model_dl, dataloader, optimizer, criterion):
  LOSS = []
  ACCURACY = []
  Y_esti_list = []


  for epoch in range(nb_epoch):
      total_loss = 0
      correct_predictions = 0
      total_samples = 0
      max_indices_list_neutral  = []
      max_indices_list_negative = []
      max_indices_list_positive = []
      for X_batch, Y_batch in dataloader:
          X_batch = X_batch.type(torch.LongTensor)  # Convert input to FloatTensor if needed
          Y_batch = Y_batch.type(torch.LongTensor)
          optimizer.zero_grad()
          prediction, max_indices = model_dl(X_batch)
          _, Y_esti = torch.max(prediction, 1)
          if epoch == (nb_epoch-1):
              for i in range (len(Y_esti)):
                  print(Y_batch[i])
                  if Y_esti[i].item() == 0 : 
                      max_indices_list_negative.append(max_indices)
                  elif Y_esti[i].item() == 1:
                      max_indices_list_neutral.append(max_indices)
                  else : 
                      max_indices_list_positive.append(max_indices)
                  
            

          # Calcul de la perte
          loss = criterion(prediction, Y_batch)
          loss.backward()
          total_loss += loss.item()
          optimizer.step()

          # Calcul de l'exactitude
          _, Y_esti = torch.max(prediction, 1)
          correct_predictions += torch.sum(torch.eq(Y_esti, (Y_batch))).item()
          total_samples += len(Y_batch)

      # Calcul de la perte moyenne et de l'exactitude moyenne pour l'époque
      average_loss = total_loss / len(dataloader)
      accuracy = correct_predictions / total_samples

      # Enregistrement des valeurs
      LOSS.append(average_loss)
      ACCURACY.append(accuracy)

      print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
      if epoch == (nb_epoch-1):
          print('Vocabulary neutral: ', torch.unique(torch.tensor(max_indices_list_neutral)))
          print('Vocabulary negative:', torch.unique(torch.tensor(max_indices_list_negative)))
          print('Vocabulary positive:', torch.unique(torch.tensor(max_indices_list_positive)))

  return LOSS, ACCURACY, model_dl
LOSS, ACCURACY, model = training_dl(nb_epoch = 25, model_dl = model, dataloader= val_iter, optimizer=optimizer, criterion=criterion)