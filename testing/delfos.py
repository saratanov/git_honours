#copy of Delfos model for pka prediction
#21st September

import torch
from torch import nn
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
import numpy as np

#################
#DATA PROCESSING#
#################

#modified sentence2vec function to return lists of word vectors
def sentences2vecs(sentences, model, unseen=None):
    """Generate vectors for each word in a sentence sentence (list) in a list of sentences.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032
    Returns
    -------
    list of arrays, each sentence -> array of word vectors
    """
    keys = set(model.wv.key_to_index)
    bigveclist = []
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        veclist = []
        if unseen:
            veclist.append([model.wv.get_vector(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence])
        else:
            veclist.append([model.wv.get_vector(y) for y in sentence 
                            if y in set(sentence) & keys])
        vecarray = np.concatenate(veclist, axis=1)
        vectensor = torch.Tensor(vecarray)
        bigveclist.append(vectensor)
    return bigveclist

#load mol2vec model
mol2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')

#define dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, solute, solvent, labels):
        self.labels = labels
        self.solute = solute
        self.solvent = solvent
        self.list_IDs = list_IDs
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        
        X = (self.solute[ID],self.solvent[ID])
        y = self.labels[ID]
        
        return X, y

def sentence_generator(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sentence = mol2alt_sentence(mol,1)
    return sentence

#return dataset of vector sentences for each solute solvent pair
def delfos_data(sol_smiles, solv_smiles):
    data = []
    size = len(sol_smiles)
    for mols in [sol_smiles,solv_smiles]:
        sentences = [sentence_generator(x) for x in mols]
        vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
        data.append(vecs)
    pairs = [(data[0][i],data[1][i]) for i in range(size)]
    return pairs

def collate_double(batch):
    '''
    Padds batch of variable length
    '''
    sol_batch = [torch.Tensor(t[0][0]) for t in batch]
    sol_batch = torch.nn.utils.rnn.pad_sequence(sol_batch)
    solv_batch = [torch.Tensor(t[0][1]) for t in batch]
    solv_batch = torch.nn.utils.rnn.pad_sequence(solv_batch)
    targets = torch.Tensor([t[1].item() for t in batch])
    
    return [sol_batch, solv_batch, targets]

#loader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_double)
"""    
epochs = 10
n_features = 300
n_hidden = 100
losslist = []
for x in range(epochs):
    for (sol,solv,t) in loader:
        output = dmodel(sol,solv) 
        loss = criterion(output, t)  
        losslist.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('step : ' , x , 'loss : ' , loss.item())
    
import matplotlib.pyplot as plt
plt.plot(losslist)
"""

##############
### MODELS ###
##############

#attention alignment (alpha) of H wrt G
def alpha(G,H):
    alpha = torch.exp(H@torch.t(G))
    norm = torch.sum(alpha, dim=1)
    norm = torch.pow(norm, -1)
    alpha = alpha * norm[:, None]
    return alpha

#context of H
#inH = H;P, where P is the emphasised hidden state H / solvent context
def att(G,H):
    P = alpha(G,H)@G
    inH = torch.stack((H,P),2)
    return inH

#module for variable maxpooling with interaction vectors
class maxpool_int(nn.Module):
    def __init__(self, L):
        super(maxpool_int, self).__init__()
        self.maxpool = nn.MaxPool3d((L,1,2))
    def forward(self, X):
        return self.maxpool(X)

#module for variable maxpooling without interaction vectors
class maxpool(nn.Module):
    def __init__(self, L):
        super(maxpool, self).__init__()
        self.maxpool = nn.MaxPool2d((L,1))
    def forward(self, X):
        return self.maxpool(X)

#delfos model
class dnet(nn.Module):
    def __init__(self, n_features=300, D=150, FF=2000, interaction=True):
        super(dnet, self).__init__()
        self.features = n_features
        self.dim = D
        self.hidden = FF
        self.interaction = interaction
    
        self.biLSTM_X = nn.LSTM(self.features, self.dim, bidirectional=True)
        self.biLSTM_Y = nn.LSTM(self.features, self.dim, bidirectional=True)
        
        self.FF = nn.Linear(4*self.dim, self.hidden)
        self.out = nn.Linear(self.hidden, 1)
    
    def forward(self,Y,X):
        #max sequence lengths
        N = X.shape[0] #solvent
        M = Y.shape[0] #solute
        #batch size
        B = X.shape[1] 
        
        #biLSTM to get hidden states
        H, hcX = self.biLSTM_X(X, None) #NxBx2D tensor - solvent hidden state
        G, hcY = self.biLSTM_Y(Y, None) #MxBx2D tensor - solute hidden state
        
        if self.interaction:
            #calculate attention, then concatenate with hidden states H and G
            inH = torch.stack([att(G[:,b,:],H[:,b,:]) for b in range(B)],0) #BxNx2Dx2
            inG = torch.stack([att(H[:,b,:],G[:,b,:]) for b in range(B)],0) #BxMx2Dx2
        
            #maxpool concatenated tensors
            maxpool_X = maxpool_int(N)
            maxpool_Y = maxpool_int(M)
            u = maxpool_X(inH).view(B,2*self.dim)  #Bx2D - solvent vector
            v = maxpool_Y(inG).view(B,2*self.dim)  #Bx2D - solute vector
        
        else:
            #maxpool tensors
            maxpool_X = maxpool(N)
            maxpool_Y = maxpool(M)
            u = maxpool_X(torch.transpose(H,0,1)).view(B,2*self.dim)  #Bx2D - solvent vector
            v = maxpool_Y(torch.transpose(G,0,1)).view(B,2*self.dim)  #Bx2D - solute vector
        
        #feed forward neural network
        NN = torch.cat((u,v),1) #Bx4D - concatenated solvent/solute vector
        NN = self.FF(NN) #Bxhidden
        NN = nn.functional.relu(NN)
        output = self.out(NN)
        return output

#criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(dmodel.parameters(), lr=0.0002, momentum=0.9, nesterov=True)