import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .MP_utils import *

class MPNN(nn.Module):
    """
    Model wrapper for message passing with a single molecule. Contains message passing featuriser + NN layers.
    
    Parameters
    ----------
    atom_messages : Bool
        True turns on message passing between atoms, False turns on message passing between bonds
    MP_depth : int
        Number of message passing steps
    MP_hidden : int
        Dimension of the MP hidden states
    readout : ['max','mean','sum']
        Readout function
    dropout : float between [0,1]
        Dropout probability for the NN layers
    NN_depth : int
        Number of NN layers
    NN_hidden : int
        Number of neurons per NN layer
    activation : ['ReLU','LeakyReLU','PReLU','tanh','SELU','ELU']
        Activation function for NN layers
    """
    def __init__(self, atom_messages=True, MP_hidden=128, MP_depth=3, readout='max', dropout=0.2, NN_depth=1, activation='ReLU', NN_hidden=64):
        super(MPNN, self).__init__()    
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=atom_messages)
        self.MP_depth = MP_depth
        self.MP_hidden = MP_hidden
        self.readout = readout
        self.dropout = dropout
        self.NN_depth = NN_depth
        self.NN_hidden = NN_hidden
        self.activation = activation
        
        self.encoder = MP(self.atom_messages, self.MP_hidden, self.MP_depth, self.readout, self.dropout, interaction=None)
        
        #activation
        activation = get_activation_function(self.activation)
        
        # create NN layers
        if self.NN_depth == 1:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(self.MP_hidden, 1)
            ]
        else:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(self.MP_hidden, self.NN_hidden)
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

    def forward(self, batch):
        """
        Parameters
        ----------
        smiles : list or BatchMolGraph
            List of SMILES strings or BatchMolGraph
        """
        encodings = self.encoder(batch)
        output = self.ffn(encodings)
        return output
    
class double_MPNN(nn.Module):
    """
    Model wrapper for message passing with two input molecules. Contains message passing featuriser + NN layers.
    
    Parameters
    ----------
    atom_messages : Bool
        True turns on message passing between atoms, False turns on message passing between bonds
    MP_depth : int
        Number of message passing steps
    MP_hidden : int
        Dimension of the MP hidden states
    interaction : ['exp','tanh',None]
        Type of interaction layer between solute and solvent hidden states
    readout : ['max','mean','sum']
        Readout function
    dropout : float between [0,1]
        Dropout probability
    NN_depth : int
        Number of NN layers
    NN_hidden : int
        Number of neurons per NN layer
    activation : ['ReLU','LeakyReLU','PReLU','tanh','SELU','ELU']
        Activation function for NN layers
    """
    def __init__(self, atom_messages=True, MP_hidden=128, MP_depth=3, readout='mean', dropout=0.2, interaction=None, NN_depth=1, activation='ReLU', NN_hidden=64):
        super(double_MPNN, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=atom_messages)
        self.MP_depth = MP_depth
        self.MP_hidden = MP_hidden
        self.interaction = interaction
        self.readout = readout
        self.dropout = dropout
        self.NN_depth = NN_depth
        self.NN_hidden = NN_hidden
        self.activation = activation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # separate encoders for solute and solvent
        self.encoder_sol = MP(self.atom_messages, self.MP_hidden, self.MP_depth, self.readout, self.dropout, self.interaction)
        self.encoder_solv = MP(self.atom_messages, self.MP_hidden, self.MP_depth, self.readout, self.dropout, self.interaction)
        
        #activation
        activation = get_activation_function(self.activation)
        
        # create NN layers
        if self.NN_depth == 1:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(self.MP_hidden*2, 1)
            ]
        else:
            ffn = [
                nn.Dropout(self.dropout),
                nn.Linear(self.MP_hidden*2, self.NN_hidden)
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

    def forward(self,sol,solv):
        """
        Parameters
        ----------
        sol : list or BatchMolGraph
            List of solute SMILES strings or BatchMolGraph
        solv : list 
            List of solvent SMILES strings or BatchMolGraph
        """
        
        # message passing (returns either a tensor of molecular or atomic feature vectors depending on interaction)
        sol = self.encoder_sol(sol)
        solv = self.encoder_solv(solv)
        
        num_pairs = len(sol)
        
        # interaction step
        if self.interaction in ['exp','tanh']:
            cats = [int_func(sol[x],solv[x],self.interaction) for x in range(num_pairs)]
            sol = [cats[x][0] for x in range(num_pairs)]
            solv = [cats[x][1] for x in range(num_pairs)] # num_pairs x (2*num_atoms x hidden_size)


            # pooling along the atomic dimension -> molecular feature vector
            if self.readout == 'max':
                sol = torch.stack([torch.max(mol, dim=0)[0] for mol in sol])
                solv = torch.stack([torch.max(mol, dim=0)[0] for mol in solv])
            else:
                sol = torch.stack([getattr(torch, self.readout)(mol, dim=0) for mol in sol])
                solv = torch.stack([getattr(torch, self.readout)(mol, dim=0) for mol in solv]) # num_pairs x hidden_size
        
        # concatenate solute / solvent feature vectors
        encodings = torch.cat([sol,solv], dim=1) # num_pairs x 2*hidden_size
        
        # NN
        output = self.ffn(encodings)
        return output
    
    def maps(self,sol,solv):
        """
        Parameters
        ----------
        sol : str
            SMILES string
        solv : str
            SMILES string
        """
        
        # message passing (returns either a tensor of molecular or atomic feature vectors depending on interaction)
        sol = self.encoder_sol([sol])
        solv = self.encoder_solv([solv])
        
        num_pairs = 1
        
        # interaction step
        if self.interaction in ['exp','tanh']:
            maps = int_func_map(sol[0],solv[0],self.interaction)

        return maps
    
    def feature_vecs(self,sol,solv):
        """
        Parameters
        ----------
        sol : list or BatchMolGraph
            List of solute SMILES strings or BatchMolGraph
        solv : list 
            List of solvent SMILES strings or BatchMolGraph
        """
        
        # message passing (returns either a tensor of molecular or atomic feature vectors depending on interaction)
        sol = self.encoder_sol(sol)
        solv = self.encoder_solv(solv)
        
        num_pairs = len(sol)
        
        # interaction step
        if self.interaction in ['exp','tanh']:
            cats = [int_func(sol[x],solv[x],self.interaction) for x in range(num_pairs)]
            sol = [cats[x][0] for x in range(num_pairs)]
            solv = [cats[x][1] for x in range(num_pairs)] # num_pairs x (2*num_atoms x hidden_size)


            # pooling along the atomic dimension -> molecular feature vector
            if self.readout == 'max':
                sol = torch.stack([torch.max(mol, dim=0)[0] for mol in sol])
                solv = torch.stack([torch.max(mol, dim=0)[0] for mol in solv])
            else:
                sol = torch.stack([getattr(torch, self.readout)(mol, dim=0) for mol in sol])
                solv = torch.stack([getattr(torch, self.readout)(mol, dim=0) for mol in solv]) # num_pairs x hidden_size
        
        # concatenate solute / solvent feature vectors
        encodings = torch.cat([sol,solv], dim=1) # num_pairs x 2*hidden_size
        
        return sol, solv, encodings
    
class MP(nn.Module):
    """
    Message passing featuriser based on chemprop. Output depends on if interaction is called for.
    
    Parameters
    ----------
    atom_messages : Bool
        True turns on message passing between atoms, False turns on message passing between bonds
    MP_depth : int
        Number of message passing steps
    MP_hidden : int
        Dimension of the MP hidden states
    interaction : ['exp','tanh',None]
        Type of interaction layer between solute and solvent hidden states
    readout : ['max','mean','sum']
        Readout function
    dropout : float between [0,1]
        Dropout probability
    """
    def __init__(self, atom_messages=True, MP_hidden=128, MP_depth=3, readout='max', dropout=0.2, interaction=None):
        super(MP, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.atom_messages = atom_messages
        self.bond_fdim = get_bond_fdim(atom_messages=self.atom_messages)
        self.hidden = MP_hidden
        self.depth = MP_depth
        self.readout = readout
        self.dropout = dropout
        self.interaction = interaction
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden)

        if self.atom_messages:
            w_h_input_size = self.hidden + self.bond_fdim
        else:
            w_h_input_size = self.hidden

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden, self.hidden)

    def forward(self, mol):
        """
        Parameters
        ----------
        mol : list or BatchMolGraph
            List of SMILES strings or BatchMolGraph
            
        Returns
        -------
        if interaction in ['exp','tanh']
            mol_tensors : list of torch.Tensors
                list of tensors containing the final hidden states (before readout) for each mol in the batch
        else:
            mol_vecs : torch.Tensor
                final feature vectors (after readout) for each mol in the batch
        """
        # convert list of smiles into BatchMolGraph
        if type(mol) == list:
            mol = mol2graph(mol)
        
        # get all feature vectors and connections
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        if self.atom_messages:
            a2a = mol.get_a2a().to(self.device)

        # hidden state initialisation
        if self.atom_messages:
            x = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            x = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = F.relu(x)  # num_bonds x hidden_size
        
        # message passing
        for depth in range(self.depth - 1):
            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = F.relu(x + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden
        
        # last message passing step (same for MP and D-MP)
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = F.relu(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        if self.interaction in ['exp','tanh']:
            # collate atomic feature vectors in a tensor for each molecule
            mol_tensors = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_tensors.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    mol_tensors.append(cur_hiddens)
            return mol_tensors # num_molecules x (num_atoms x hidden_size)
            
        else:
            # readout into a molecular feature vector
            mol_vecs = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                    if self.readout == 'mean':
                        mol_vec = mol_vec.sum(dim=0) / a_size
                    elif self.readout == 'sum':
                        mol_vec = mol_vec.sum(dim=0)
                    elif self.readout == 'max':
                        mol_vec = torch.max(mol_vec,dim=0)[0]
                    mol_vecs.append(mol_vec)
            mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
            return mol_vecs  # num_molecules x hidden