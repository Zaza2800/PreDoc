import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 

class EmbeddingLayer(nn.Module):
    def __init__(self, n_symboles, output_size):
        super().__init__()
        self.n_symboles = n_symboles
        self.output_size = output_size

        self.projection = torch.nn.Linear(self.n_symboles, self.output_size)

    def forward(self, x, label=False):
        x_one_hot =  torch.nn.functional.one_hot(x, self.n_symboles).float()
        if not label:
            x_proj = self.projection(x_one_hot)
            return x_proj
        return x_one_hot


# ENTRAINEMENT


BATCH_SIZE = 128
HIDDEN_SIZE = 100
DIM_OUTPUT = len(id2lettre)  # = 96
NB_ITERS = 10
data_trump = DataLoader(TrumpDataset(open("trump_full_speech.txt","rb").read().decode(),maxlen=100), batch_size= BATCH_SIZE, shuffle=True)


EMBED_DIM = 75 # < 96
embedding = EmbeddingLayer(len(id2lettre), EMBED_DIM)
embedding.to(device)


rnn = RNN(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=DIM_OUTPUT)
rnn.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)
embed_opt = torch.optim.Adam(embedding.parameters(), lr=0.1)


loss_train_per_epoch = []

def train_loop(data, device, HIDDEN_SIZE, embedding, model, criterion): 
    epoch_loss = 0
    for i, (x, y) in enumerate(data):
        x, y = x.to(device), y.to(device)


        x = embedding(x).to(device)
        y = embedding(y, label=True).to(device)

        x = torch.permute(x, (1,0,2))
        y = torch.permute(y, (1,0,2))
        
        optimizer.zero_grad() 
        embed_opt.zero_grad()
        
        h = torch.zeros((x.shape[1], HIDDEN_SIZE), device=device)
        output = model.forward(x, h)
        output = output.to(device)

        y_hat = model.decode(output)

        loss = criterion(y_hat, y)
        loss.backward()

        optimizer.step()
        embed_opt.step()

        epoch_loss += loss.item()


    return epoch_loss

for epoch in range(NB_ITERS):
    print("\n---> Epoch :", epoch, "/",NB_ITERS)
    epoch_loss_train = train_loop(data_trump, device, HIDDEN_SIZE, embedding, rnn, criterion)
    loss_train_per_epoch.append(epoch_loss_train / len(data_trump))

print('Entrainement finis')
fig, ax = plt.subplots(figsize = (8,8))

ax.plot(range(1,len(loss_train_per_epoch)+1), loss_train_per_epoch, label="loss_train_per_epoch")
ax.legend()
ax.set(title = 'losses on train', xlabel = 'epoch', ylabel = 'loss')

plt.savefig('result_plot_exo4_train.png')
plt.show()

# GENERATION
print("Generation de texte :")

caractere = torch.tensor(torch.randint(len(lettre2id), (1,))).long().to(device)
texte = [caractere]

for i in range(100):
    h = torch.zeros((1, HIDDEN_SIZE), device=device)
    x = embedding(texte[-1])
    output = rnn.one_step(x, h)
    output = rnn.decode(output)
    caractere = output.argmax(1)
    texte.append(caractere)
texte = torch.cat(texte, dim=0)
texte = texte.tolist()
print(code2string(texte))