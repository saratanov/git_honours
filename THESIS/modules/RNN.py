import torch
from torch import nn
from .utils import *

##############
### MODELS ###
##############

#delfos model
class double_RNN(nn.Module):
    def __init__(self, features=300, RNN_hidden=256, NN_hidden=1024, NN_depth=1, interaction=None,
                 readout='max', dropout=0.1, activation='ReLU'):
        super(double_RNN, self).__init__()
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
        
        activation = get_activation_function(self.activation)
        
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

#delfos with one input
class RNN(nn.Module):
    def __init__(self, features=300, RNN_hidden=256, NN_hidden=1024, NN_depth=1, readout='max',
                 dropout=0.1, activation='ReLU'):
        super(RNN, self).__init__()
        self.features = features
        self.dim = RNN_hidden
        self.NN_hidden = NN_hidden
        self.NN_depth = NN_depth
        self.dropout = dropout
        self.activation = activation
        self.readout = readout
    
        self.biLSTM = nn.LSTM(self.features, self.dim, bidirectional=True)
        
        activation = get_activation_function(self.activation)
        
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
