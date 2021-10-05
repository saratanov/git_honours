#basic functions for ML
#updated from basic.py 22nd September

#IMPORTS
from collections import defaultdict as ddict, OrderedDict as odict
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import PandasTools, AllChem as Chem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.Descriptors import MolWt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sklearn
import torch
import deepchem as dc
import copy
from tqdm import tqdm
import delfos as d
        
class Model:
    """
    Object containing a model and all its associated parameters.
    """
    def __init__(self, name, model, model_type, data_type, inputs,
                 lr=1e-3, optimiser=torch.optim.Adam, num_epochs=100, batch_size=32):
        self.name = name #e.g. "MPNN with attention"
        self.model = model #torch/sklearn regressor object
        self.model_type = model_type
        self.data_type = data_type #"SMILES" or "descriptors" or "ECFP" or "sentences"
        self.inputs = inputs #1 or 2
        
        self.experiments = []
        
        #torch specific variables
        if self.model_type == 'torch':
            self.lr = lr
            self.optimiser = optimiser
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            
def data_maker(solute, pka, solvent=None, ids=None):
    if ids == None:
        pass
    elif solvent == None:
        [solute,pka] = [[lis[x] for x in ids] for lis in (solute, pka)]
    else:
        [solute,solvent,pka] = [[lis[x] for x in ids] for lis in (solute, solvent, pka)]
    #SMILES
    if solvent == None:
        SMILES = solute
    else:
        SMILES = [(solute[i],solvent[i]) for i in range(len(solute))]
    SMILES_data = [SMILES, torch.Tensor(pka)]
    #sentences
    if solvent == None:
        sentences = d.delfos_data_1(solute)
    else:
        sentences = d.delfos_data(solute,solvent)
    sentence_data = [sentences, torch.Tensor(pka)]
    #collate data
    datasets = dict(SMILES=SMILES_data,
                    sentences=sentence_data)
    return datasets    

class pka_scaler:
    def __init__(self, pka):
        self.scaler = sklearn.preprocessing.StandardScaler()
        if type(pka) == np.ndarray:
            pka = pka.reshape(-1,1)
        else:
            pka = pka.detach().numpy().reshape(-1,1)
        self.scaler.fit(pka)
        
    def transform(self, targets):
        if type(targets) == np.ndarray:
            targets = targets.reshape(-1,1)
            transformed_targets = self.scaler.transform(targets)
            return transformed_targets.ravel()
        else:
            targets = targets.detach().numpy()
            transformed_targets = self.scaler.transform(targets)
            return torch.Tensor(transformed_targets)
    
    def inverse_transform(self, targets):
        if type(targets) == np.ndarray:
            targets = targets.reshape(-1,1)
            transformed_targets = self.scaler.inverse_transform(targets)
            return transformed_targets.ravel()
        else:
            targets = targets.detach().numpy()
            transformed_targets = self.scaler.inverse_transform(targets)
            return torch.Tensor(transformed_targets)

class Dataset(torch.utils.data.Dataset):
    """
    Creates universal dataset type for torch loaders and regressors.
    
    Parameters
    ----------
    list_IDs : list, np.array
        Indices to be used for training/testing
    datapoints: List
        for MP: List(Tuple(solute_smiles,solvent_smiles))
        for RNN: List(Tuple(solute_tensor,solvent_tensor))
        Datapoints, either in SMILES (str) or sentence (torch.Tensor) solute/solvent pairs
    labels: torch.Tensor
        Target values
    """
    def __init__(self, list_IDs, datapoints, labels):
        self.labels = labels
        self.datapoints = datapoints
        self.list_IDs = list_IDs
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        
        X = self.datapoints[ID]
        y = self.labels[ID]
        
        return X, y
    
def collate_double(batch):
    '''
    Collates double input batches for a torch loader.
        
    Parameters
    ----------
    batch: List = [(X,y)]
        List of (solute,solvent) pairs with their target value.
    
    Returns
    -------
    [sol_batch, solv_batch, targets]: List
        Type of output depends on if the original dataset contains SMILES or sentences.
        Each component is a list / torch.Tensor.
    '''
    if type(batch[0][0][0]) == str:
        sol_batch = [t[0][0] for t in batch]
        solv_batch = [t[0][1] for t in batch]
    else:
        sol_batch = [torch.Tensor(t[0][0]) for t in batch]
        sol_batch = torch.nn.utils.rnn.pad_sequence(sol_batch)
        solv_batch = [torch.Tensor(t[0][1]) for t in batch]
        solv_batch = torch.nn.utils.rnn.pad_sequence(solv_batch)
    targets = torch.Tensor([t[1].item() for t in batch])
    
    return [sol_batch, solv_batch, targets]

def collate_single(batch):
    '''
    Collates double input batches for a torch loader.
        
    Parameters
    ----------
    batch: List = [(X,y)]
        List of solutes with their target value.
    
    Returns
    -------
    [sol_batch, solv_batch, targets]: List
        Type of output depends on if the original dataset contains SMILES or sentences.
        Each component is a list / torch.Tensor.
    '''
    if type(batch[0][0]) == str:
        sol_batch = [t[0] for t in batch]
    else:
        sol_batch = [torch.Tensor(t[0]) for t in batch]
        sol_batch = torch.nn.utils.rnn.pad_sequence(sol_batch)
    targets = torch.Tensor([t[1].item() for t in batch])
    
    return [sol_batch, targets]

def loader_func(data, indices, inputs, batch_size=64):
    '''
    torch loader for double inputs.
        
    Parameters
    ----------
    indices : list, np.array
        Indices for selected samples.
    data : List = [(sol,solv),pka]
        Training data of (solute,solvent) pairs and target values.
    batch_size : int
        Size of selected batches
    
    Returns
    -------
    loader : torch.utils.data.DataLoader
        Batched dataloader for torch regressors
    '''
    dataset = Dataset(indices, data[0], data[1])
    if inputs == 1:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_single)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_double)
    return loader

def train(model, ids, data, scaler):
    """
    Train a model.

    Parameters
    ----------
    model : Model
        Regressor model
    ids : list, np.array
        Indices for training samples
    data : List = [(sol,solv),pka]
        Data of (solute,solvent) pairs and target values

    Returns
    -------
    model : Any
        Trained regressor model
    """
    #generate train set and val set (for early stopping)
    train_ids, val_ids, _, _ = train_test_split(ids, ids, test_size=0.1, random_state=1)
    train_loader = loader_func(data, train_ids, model.inputs, batch_size=model.batch_size)
    val_loader = loader_func(data, val_ids, model.inputs, batch_size=len(val_ids))

    regressor = copy.deepcopy(model.model)      
    optimiser = model.optimiser(regressor.parameters(), lr=model.lr)
    loss_function = torch.nn.MSELoss()
    early_stopping = EarlyStopping(model.name)
    
    if model.inputs == 2:
        for epoch in range(model.num_epochs):
            #train
            for (sol,solv,targets) in train_loader:
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                optimiser.zero_grad()
                outputs = regressor(sol,solv)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimiser.step()
            #evaluate
            for (sol,solv,targets) in val_loader:
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                outputs = regressor(sol,solv)
                loss = loss_function(outputs, targets)
                val_loss = loss.item()
            #early stopping
            early_stopping.store(val_loss, regressor)
            if early_stopping.stop:
                #print("Stopping at epoch "+str(epoch))
                break
                
    else:
        for epoch in range(model.num_epochs):
            #train
            for (mol,targets) in train_loader:
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                optimiser.zero_grad()
                outputs = regressor(mol)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimiser.step()
            #evaluate
            for (mol,targets) in val_loader:
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                outputs = regressor(mol)
                loss = loss_function(outputs, targets)
                val_loss = loss.item()
            #early stopping
            early_stopping.store(val_loss, regressor)
            if early_stopping.stop:
                #print("Stopping at epoch "+str(epoch))
                break
    
    regressor.load_state_dict(torch.load('checkpoints/'+model.name+'.pt'))
    return regressor

class EarlyStopping:
    def __init__(self, name, patience=10):
        self.patience = patience
        self.best_loss = 1e6
        self.steps = 0
        self.stop = False
        self.chk_name = name
    
    def store(self, loss, net):
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps = 0
            torch.save(net.state_dict(), 'checkpoints/'+self.chk_name+'.pt')
        else:
            self.steps += 1
            if self.steps > self.patience:
                self.stop = True

def test(model, regressor, ids, data, scaler):
    """
    Test a model.

    Parameters
    ----------
    model : Model
        Regressor model
    model_type : str = "sklearn" or "torch"
        Type of regressor model
    regressor :
        Specific regressor for testing
    ids : list, np.array
        Indices for training samples
    data : List = [(sol,solv),pka]
        Data of (solute,solvent) pairs and target values

    Returns
    -------
    Results: list
        List of MAE, RMSE.
    """
    loader = loader_func(data, ids, model.inputs, batch_size=len(ids))
    if model.inputs == 2:
        for (sol,solv,targets) in loader:
            outputs = regressor(sol,solv)
    else:
        for (mol,targets) in loader:
            outputs = regressor(mol)
    outputs = scaler.inverse_transform(outputs)
    targets= targets.detach().numpy()
    outputs = outputs.detach().numpy()
    results = [mae(targets, outputs),rmse(targets, outputs)]
    return results

def fit(model, data, test_ids, exp_name, datasets):
    """
    Fits a model according to the given test_ids and data.

    Parameters
    ----------
    model : torch_model / sklearn_model
        Regressor model.
    data : List = [(sol,solv),pka]
        Full dataset of (solute,solvent) pairs and target values.
    test_ids : list, np.array
        Selected test set indices.

    Returns
    -------
    trained_model : Any
        Trained torch regressor model.
    results : List
        MAE, RMSE, test set size
    """
    if model.model_type == 'torch':
        size = len(data[0])
    else:
        size = data[0].shape[0]
        
    train_ids = [i for i in range(size) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    trained_model = train(model, train_ids, data, scaler, datasets)
    results = test(model, trained_model, test_ids, data, scaler)
    model.experiments.append({'name':exp_name,'model':trained_model, 'results':results, 'scaler':scaler})
    return results

def fit_no_test(model, exp_name, datasets):
    """
    Fits a model to the whole dataset with no testing

    Parameters
    ----------
    model : torch_model / sklearn_model
        Regressor model.
    datasets : dict

    """
    data = datasets[model.data_type]
    ids = list(range(len(data[0])))
    scaler = pka_scaler(data[1])
    trained_model = train(model, ids, data, scaler)
    model.experiments.append({'name':exp_name,'model':trained_model, 'scaler':scaler})
    torch.save(trained_model.state_dict(), 'trained/'+model.name.replace(' ','_')+'_'+exp_name.replace(' ','_')+'.pt')

#RESULTS HELPERS
def rmse(y_true, y_pred):
    """Helper function"""
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    """Helper function"""
    return mean_absolute_error(y_true, y_pred)
    
