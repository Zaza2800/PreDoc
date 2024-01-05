
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    loss = CrossEntropyLoss()
    mask = (output != padcar)
    error = loss(output[mask], target[mask])
    return error.mean()

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_x = nn.Parameter(torch.randn(hidden_size,input_size))
        self.W_h = nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.W_d = nn.Parameter(torch.randn(output_size,hidden_size))

        self.b_h = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(output_size))

        self.h0 = torch.zeros(self.hidden_size)

    def one_step(self, x, h):
        ''' x : un batch des sequences de taille batch*dim
        h: batch des états cachés de taille batch*latent
        sortie de taille batch*latent
        '''
        f = torch.nn.functional.linear(x, self.W_x, bias=None) + torch.nn.functional.linear(h, self.W_h, bias=self.b_h)
        h_new = torch.tanh(f)
        return h_new
    

    def forward(self, x, h):
        ''' x : taille length*batch*input_dim
        h: taille batch*hidden_size
        sortie= : taille length*batch*hidden_size
        '''  # dans exo2 : x de taille batch_size*length*input_dim
        
        b,l,i = x.size()
        _, hidden_size = h.size()  

        output = torch.zeros(l,b,hidden_size)

        for i in range(l):
            output[i] = self.one_step(x[:,i, :], h)
            h = output[i]

        return output
    
    def decode(self, h):
        '''
        h: taille batch*latent
        sortie de taille batch*output
        '''  # dans exo2 : h de taille length*batchsize*hidden_size
        return torch.nn.functional.linear(h, self.W_d, bias=self.b_d)


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

        self.f_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        self.i_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        self.C_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        self.o_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=True)
        self.decode_linear = nn.Linear(hidden_size, output_size)

    def one_step(self, x, h, C):
        concat = torch.cat([h, x], dim=1) 
        f_t = torch.sigmoid(self.f_linear(concat))
        i_t = torch.sigmoid(self.i_linear(concat))
        C_t = f_t*C + i_t*torch.tanh(self.C_linear(concat))

        o_t = torch.sigmoid(self.o_linear(concat))
        h_t = o_t*torch.tanh(C_t)
        return torch.tensor(h_t, requires_grad=True)
    
    def forward(self, x, h_0, C_0):
        l, b = x.shape[0], x.shape[1]

        if h_0 == None:
            h_t = torch.zeros(b,self.size_latent)
        else:
            h_t = h_0

        if C_0==None:
            C_tm1 = torch.zeros(b,self.size_latent)
        else:
            C_tm1 = C_0

        output = torch.zeros((l, b, self.size_latent))
        for t in range(l):
            h_tm1, C_tm1 = h_t, C_t
            xt = x[t]
            h_t, C_t = self.one_step(xt, h_tm1, C_tm1)
            output[t,:,:]= h_t
        return output
    
    def decode(self, h):
        return self.decode_linear(h)


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.z_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.r_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.w_linear = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.decode_linear = nn.Linear(hidden_size, output_size)
    
    def one_step(self, x, h):
        concat = torch.cat((h, x), dim=1) 
        z_t = torch.sigmoid(self.z_linear(concat))
        r_t = torch.sigmoid(self.r_linear(concat))
        concat2 = torch.cat((r_t*h, x), dim=1)
        h_hat = torch.tanh(self.w_linear(concat2))
        h_t = (1-z_t)*h + z_t*h_hat
        return torch.tensor(h_t, requires_grad=True)

    def forward(self, x, h):
        b,l,i = x.size()
        _, hidden_size = h.size()  

        output = torch.zeros(l,b,hidden_size)

        for i in range(l):
            output[i] = self.one_step(x[:,i, :], h)
            h = output[i]

        return output
    
    def decode(self, h):
        return self.decode_linear(h)
    
    
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


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
PATH = "student_tp5/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################
# ENTRAINEMENT
##########################################################################################

BATCH_SIZE = 128
HIDDEN_SIZE = 100
DIM_OUTPUT = len(id2lettre)  # sortie du RNN = nombre de symboles considérés   # = 96

# prendre un RNN qui permet de engendrer un discours à la Trump :
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), batch_size= BATCH_SIZE, shuffle=True)


EMBED_DIM = 75 # < len(id2lettre = 96)
embedding = nn.Embedding(len(id2lettre), EMBED_DIM)

# rnn = RNN(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=EMBED_DIM)
# rnn.to(device)

gru = GRU(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE, output_size=EMBED_DIM)

criterion = maskedCrossEntropy
#optimizer = torch.optim.Adam(rnn.parameters(), lr=0.05)
optimizer = torch.optim.Adam(gru.parameters(), lr=0.05)
softmax = nn.Softmax(dim=1)


loss_train_per_epoch = []

for epoch in range(25):
    print("\n__________Epoch :", epoch, "___________________________________")
    epoch_loss = 0
    epoch_loss_test = 0
    for i, (x, y) in enumerate(data_trump):
        x = x.long()

        x_projete = embedding(x)
        y_projete = embedding(y)

        optimizer.zero_grad()

        h = torch.zeros((x_projete.size(0), HIDDEN_SIZE), device=device)

        #output_sequence = rnn.forward(x_projete, h)
        output_sequence = gru.forward(x_projete, h)
        #last_hidden_state = output_sequence.to(device)
        
        #y_hat = rnn.decode(last_hidden_state)
        y_hat = gru.decode(output_sequence)
        y_hat = y_hat.permute(1, 0, 2)
        y_hat = torch.flatten(y_hat, end_dim=-2)
        y_projete = torch.flatten(y_projete, end_dim=-2)

        loss = criterion(y_hat, y_projete, PAD_IX)

        #opti
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.sum()
        loss_train_per_epoch.append(epoch_loss / len(data_trump))

        torch.cuda.empty_cache()
        
        if i % 10 ==0:
            print("Epoch %d, Batch %d, Loss %f" % (epoch, i,  loss.item()))
    
    print(code2string(y[0]))
    print(code2string(y_hat.argmax(2)[0,:]))


##########################################################################################
# GENERATION
##########################################################################################
sentence = generate(gru, embedding, gru.decode, EOS_IX, "", 200)
print(code2string(sentence))
# for l in range(25):
#     h = torch.zeros((1, HIDDEN_SIZE), device=device)
#     caractere = torch.tensor(torch.randint(len(lettre2id), (1,))).long().to(device)
#     texte = [caractere]
#     for i in range(100):
#         x = embedding(texte[-1])
#         #output = rnn.one_step(x, h)
#         output = gru.one_step(x, h)
#         #output = rnn.decode(output)
#         output = gru.decode(output)
#         caractere = output.argmax(1)
#         texte.append(caractere)
#     texte = texte[1:]
#     texte = torch.cat(texte, dim=0)
#     texte = texte.tolist()
#     print(code2string(texte))