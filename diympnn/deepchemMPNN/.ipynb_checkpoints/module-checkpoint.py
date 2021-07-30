import math

import deepchem as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import numpy as np

import random
from collections import OrderedDict
from scipy.stats import pearsonr

#######################

class MPNN(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MPNN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        
        #self.encoder
        self.encoder = MP(args, atom_fdim, bond_fdim)
        self.NN = torch.nn.Linear(self.atom_fdim, 1)

    def forward(self, batch):
        """
        Params
        ------
        batch: list of SMILES
        """
        encodings = [self.encoder(b) for b in batch]
        output = self.NN(____)
        return output
    
class MP(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MP, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.directed = args.directed
        self.hidden_size = args.hidden_size
        
#       self.M = nn.Linear(self.atom_fdim, self.hidden_size)
        self.U = nn.Linear(2*self.atom_fdim, self.atom_fdim)
        
    def mol2graph(smile):
        """
        Constructs molecular graph from SMILES string

        Returns
        -------
        g: ordered dictionary
        bond features
        h: ordered dictionary
        atom features
        """
        g = OrderedDict({})
        h = OrderedDict({})
        molecule = Chem.MolFromSmiles(smile) #molecular graph type
        #collect atom features for each atom in the graph
        for i in range(0, molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i)
            h[i] = Variable(torch.FloatTensor(dc.feat.graph_features.atom_features(atom_i))).view(1, 75)

            #collect bond features for each bond connected to atom i (list of tuples of (features,atom))
            for j in range(0, molecule.GetNumAtoms()):
                e_ij = molecule.GetBondBetweenAtoms(i, j)
                if e_ij != None:
                    e_ij =  dc.feat.graph_features.bond_features(e_ij)
                    e_ij = Variable(torch.FloatTensor(e_ij).view(1, 6))
                    atom_j = molecule.GetAtomWithIdx(j)
                    if i not in g:
                        g[i] = []
                    g[i].append( (e_ij, j) )
        return g, h     
    
    def message_pass(g,h):
        m = OrderedDict({})
        for v in g.keys():
            neighbours = g[v]
            m_v = torch.zeroes((1,75))
            for neighbour in neighbors:
                e_vw = neighbour[0]
                w = neighbour[1]
                m_v = torch.add(h[w])
            m[v] = m_v
        return m 
    
    def update(h,m):
        for v in h.keys:
            h[v] = F.relu(self.U(torch.cat((h[v],m[v]))))
        return h
    
    def forward(self, smile):
        bonds, atoms = MP.mol2graph(smile)
        for step in range(self.depth):
            messages = message_pass(bonds, atoms)
            atoms = update(atoms,messages)
            
        # Readout
        mol_vec = []
        for v in atoms.keys():
            mol_vec.append(atoms[v])
        mol_vec = mol_vec.sum(dim=0)
        return mol_vec    
    

class TrainArgs:
    message_func = 'yes'
    update_func = 'yes'
    readout_func = 'yes'
    depth = 3
    directed = False
    activation = 'yes'
    hidden_size = 128
    