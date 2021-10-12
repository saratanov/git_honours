#copy of Delfos model for pka prediction
#21st September

import torch
from torch import nn
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
import numpy as np
import chemprop_ish as c

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

def delfos_data_1(smiles):
    sentences = [sentence_generator(x) for x in smiles]
    vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
    return vecs    

##############
### MODELS ###
##############

#delfos model
class dnet(nn.Module):
    def __init__(self, features=300, RNN_hidden=256, NN_hidden=1024, NN_depth=1, interaction=None, readout='max', dropout=0.1, activation='ReLU'):
        super(dnet, self).__init__()
        self.features = features
        self.dim = RNN_hidden
        self.NN_hidden = NN_hidden
        self.NN_depth = NN_depth
        self.interaction = interaction
        self.dropout = dropout
        self.activation = activation
        self.readout = readout
    
        self.biLSTM_X = nn.LSTM(self.features, self.dim, bidirectional=True)
        self.biLSTM_Y = nn.LSTM(self.features, self.dim, bidirectional=True)
        
        activation = c.get_activation_function(self.activation)
        
        # create NN layers
        if self.NN_depth == 1:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(4*self.dim, 1)
            ]
        else:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(4*self.dim, self.NN_hidden)
            ]
            for _ in range(self.NN_depth - 2):
                ffn.extend([
                    activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(self.NN_hidden, self.NN_hidden),
                ])
            ffn.extend([
                activation,
                nn.Dropout(self.dropout),
                nn.Linear(self.NN_hidden, 1),
            ])
        self.ffn = nn.Sequential(*ffn)
    
    def forward(self,Y,X):
        #max sequence lengths
        N = X.shape[0] #solvent
        M = Y.shape[0] #solute
        #batch size
        B = X.shape[1] 
        
        #biLSTM to get hidden states
        H, hcX = self.biLSTM_X(X, None) #NxBx2D tensor - solvent hidden state
        G, hcY = self.biLSTM_Y(Y, None) #MxBx2D tensor - solute hidden state
        
        if self.interaction in ['exp','tanh']:
            #calculate attention, then concatenate with hidden states H and G
            cats = [int_func(H[:,b,:],G[:,b,:],self.interaction) for b in range(B)]
            inH = torch.stack([cats[b][0] for b in range(B)],0) #Bx2Nx2D
            inG = torch.stack([cats[b][1] for b in range(B)],0) #Bx2Nx2D
         #   inH = torch.stack([att(G[:,b,:],H[:,b,:]) for b in range(B)],0) #Bx2Nx2D
         #   inG = torch.stack([att(H[:,b,:],G[:,b,:]) for b in range(B)],0) #Bx2Mx2D
        
        else:
            inH = torch.transpose(H,0,1) #BxNx2D
            inG = torch.transpose(G,0,1) #BxMx2D
        
        if self.readout == 'max':
            #maxpool concatenated tensors
            u = torch.max(inH,1)[0]
            v = torch.max(inG,1)[0]
        if self.readout == 'mean':
            u = torch.mean(inH,1)
            v = torch.mean(inG,1)
        if self.readout == 'sum':
            u = torch.sum(inH,1)
            v = torch.sum(inG,1)
        
        #feed forward neural network
        encodings = torch.cat((u,v),1) #Bx4D - concatenated solvent/solute vector
        output = self.ffn(encodings) #Bx1
        return output

#criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(dmodel.parameters(), lr=0.0002, momentum=0.9, nesterov=True)

def int_func(X,Y,func):
    if func == 'exp':
        #interaction map
        I = torch.exp(torch.mm(X,Y.t()))
        #inverse of sum of columns for I and transpose I
        x_norm = torch.pow(torch.sum(I, dim=1),-1)
        y_norm = torch.pow(torch.sum(I.t(), dim=1),-1)
        #interaction maps with normalised columns
        I_x = I * x_norm[:,None] 
        I_y = I.t() * y_norm[:,None]
        #contexts
        X_context = torch.mm(I_x,Y)
        Y_context = torch.mm(I_y,X)
    if func == 'tanh':
        #interaction map
        I = torch.tanh(torch.mm(X,Y.t()))
        #contexts
        X_context = torch.mm(I,Y)
        Y_context = torch.mm(I.t(),X)
    #concatenate
    catX = torch.cat((X,X_context))
    catY = torch.cat((Y,Y_context))
    return [catX, catY]

#delfos with one input
class snet(nn.Module):
    def __init__(self, features=300, RNN_hidden=256, NN_hidden=1024, NN_depth=1, readout='max', dropout=0.1, activation='ReLU'):
        super(snet, self).__init__()
        self.features = features
        self.dim = RNN_hidden
        self.NN_hidden = NN_hidden
        self.NN_depth = NN_depth
        self.dropout = dropout
        self.activation = activation
        self.readout = readout
    
        self.biLSTM = nn.LSTM(self.features, self.dim, bidirectional=True)
        
        activation = c.get_activation_function(self.activation)
        
        # create NN layers
        if self.NN_depth == 1:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(2*self.dim, 1)
            ]
        else:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(2*self.dim, self.NN_hidden)
            ]
            for _ in range(self.NN_depth - 2):
                ffn.extend([
                    activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(self.NN_hidden, self.NN_hidden),
                ])
            ffn.extend([
                activation,
                nn.Dropout(self.dropout),
                nn.Linear(self.NN_hidden, 1),
            ])
        self.ffn = nn.Sequential(*ffn)
    
    def forward(self,X):
        #max sequence length
        N = X.shape[0]
        #batch size
        B = X.shape[1] 
        
        #biLSTM to get hidden states
        H, hcX = self.biLSTM(X, None) #NxBx2D
        
        H = torch.transpose(H,0,1) #BxNx2D
        
        #pool over atomic dimension (1)
        if self.readout == 'max':
            x = torch.max(H,1)[0]
        if self.readout == 'mean':
            x = torch.mean(H,1)
        if self.readout == 'sum':
            x = torch.sum(H,1)
        
        #feed forward neural network
        output = self.ffn(x) #Bx1
        return output
