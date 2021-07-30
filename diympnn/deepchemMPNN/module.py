#Sara's homemade MPNN
#originally created 28th July

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

#TODO: initial atom / bond vector encoding
#TODO: double MPNN

class MPNN(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MPNN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.directed = args.directed
        
        if self.directed == False:
            self.encoder = MP(args, atom_fdim, bond_fdim)
            self.NN = torch.nn.Linear(self.atom_fdim, 1)
        else:
            self.encoder = D_MP(args, atom_fdim, bond_fdim)
            self.NN = torch.nn.Linear(___, 1)

    def forward(self, batch):
        """
        Params
        ------
        batch: list of SMILES
        """
        encodings = [self.encoder(b) for b in batch]
        output = [self.NN(e) for e in encodings]
        return output
    
class MP(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MP, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        
#       self.M = nn.Linear(self.atom_fdim, self.hidden_size)
        self.U = nn.Linear(2*self.atom_fdim, self.atom_fdim)   
    
    def message_pass(g,h):
        m = OrderedDict({})
        for v in g.keys():
            neighbours = g[v]
            m_v = torch.zeros((1,75))
            for neighbour in neighbours:
                e_vw = neighbour[0]
                w = neighbour[1]
                m_v = torch.add(h[w],m_v)
            m[v] = m_v
        return m 
    
    def forward(self, smile):
        bonds, atoms = MP.mol2graph(smile)
        for step in range(self.depth):
            messages = MP.message_pass(bonds, atoms)
            for v in atoms.keys():
                atoms[v] = F.relu(self.U(torch.cat((atoms[v],messages[v]),dim=1))) # update
            
        # Readout
        mol_vec = []
        for v in atoms.keys():
            mol_vec.append(atoms[v])
        mol_vec = sum(mol_vec)
        return mol_vec    

""" directed MPNN - too hard to implement with this representation
class D_MP(nn.Module):
    def __init__(self, args, atom_fdim, bond_fdim):
        super(MP, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        
        self.W_i = nn.Linear(self.atom_fdim + self.bond_fdim, self.hidden_size)
        self.W_m = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_a = nn.Linear(self.hidden_size + self.bond_fdim, self.hidden_size)
    
    def message_pass(g,h):
        m = OrderedDict({})
        for v in g.keys():
            neighbours = g[v]
            m_v = torch.zeros((1,75))
            for neighbour in neighbours:
                e_vw = neighbour[0]
                w = neighbour[1]
                m_v = torch.add(h[w],m_v)
            m[v] = m_v
        return m 
    
    def forward(self, smile):
        bonds, atoms = MP.mol2graph(smile)
        
        #initialise h_0
        h_0 = OrderedDict({})
        for v in bonds.keys():
            neighbours = bonds[v]
            for neighbour in neighbours:
                e_vw = neighbour[0]
                w = neighbour[1]
                h_vw = F.relu(self.W_i(torch.cat([atoms[v],e_vw],dim=1)))
                h_0[v].append( (h_vw, w) )
        
        #message pass + update
        h_b = h_0
        for step in range(self.depth):
            h_new = OrderedDict({})
            for v in h_b.keys():
                neighbours = [x[1] for x in h_b[v]]
                ### TODO: make truly directed message passing
                sum_h_n = torch.sum(torch.stack(h_n), dim=0)
                for x in range(len(neighbours)):
                    e_vw = neighbours[x][0]
                    w = neighbour[x][1]
                    m_vw = torch.sub(sum_h_n, e_vw)
                    h_vw = F.relu(torch.sum([h_0[v][x][0],self.W_m(m_vw)])
                    h_new[v].append( (h_vw, w) )
            h_b = h_new
        
        #atom vectors
        for v in atoms.keys():
            
                
                
                
                m_v = torch.zeros((1,75))
                for neighbour in neighbours:
                    e_vw = neighbour[0]
                    w = neighbour[1]
                    m_v = torch.add(h[w],m_v)
                m[v] = m_v
            messages = MP.message_pass(bonds, atoms)
            for v in atoms.keys():
                atoms[v] = F.relu(self.U(torch.cat((atoms[v],messages[v]),dim=1))) # update
            
        # Readout
        mol_vec = []
        for v in atoms.keys():
            mol_vec.append(atoms[v])
        mol_vec = sum(mol_vec)
        return mol_vec
        
"""

class TrainArgs:
    message_func = 'yes'
    update_func = 'yes'
    readout_func = 'yes'
    depth = 3
    directed = False
    activation = 'yes'
    hidden_size = 128
    
    
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