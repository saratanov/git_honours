from collections import defaultdict as ddict, OrderedDict as odict
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import torch
import deepchem as dc
import copy
from .data import *
from .fit import rmse, mae, EarlyStopping
import tqdm
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

class Model:
    """
    Object containing a torch model and all its associated parameters.
    
    Parameters
    ----------
    name : string
    model : torch regressor object
    data_type : ['SMILES','sentences']
    inputs : int [1,2]
        Number of inputs for torch models
    experiments : dict
        Collection of models, scalers, names of experiments
    lr : float
        Optimiser learning rate
    optimiser : torch.optim object
    batch_size : int
    num_epochs : int
        Maximum number of epochs for training
    """
    def __init__(self, name, model, data_type, inputs,
                 lr=1e-3, optimiser=torch.optim.Adam, num_epochs=100, batch_size=32):
        self.name = name #e.g. "MPNN with attention"
        self.model = model #torch/sklearn regressor object
        self.data_type = data_type #"SMILES" or "descriptors" or "ECFP" or "sentences"
        self.inputs = inputs #1 or 2
        
        self.experiments = ddict()
        
        self.lr = lr
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
            
def data_maker(solute, pka, solvent=None, ids=None):
    """
    Generate a dictionary containing solute/solvent data encoded in three ways:
            SMILES = list of tuples containing (solute,solvent) smiles strings
            graphs = list of tuples containing (solute,solvent) MolGraphs
            sentences = list of tuples conatining (solute,solvent) mol2vec embeddings
    
    Parameters
    ----------
    sol_smiles : list
        List of solute SMILES strings
    solv_smiles : list or None
        List of solvent SMILES strings
    pka : list
        List of pka values
    ids : list
        List of indices to be used to create the datasets
    Returns
    -------
    datasets : dict
        Three keys: SMILES, graphs, sentences
        Values contain a list, where data[0] = encodings and data[1] = torch.Tensor of pkas
    """
    if ids == None:
        pass
    elif solvent == None:
        [solute,pka] = [[lis[x] for x in ids] for lis in (solute, pka)]
    else:
        [solute,solvent,pka] = [[lis[x] for x in ids] for lis in (solute, solvent, pka)]
    #SMILES
    if solvent == None:
        SMILES = solute
        graphs = [MolGraph(solute[i]) for i in range(len(solute))]
    else:
        SMILES = [(solute[i],solvent[i]) for i in range(len(solute))]
        graphs = [(MolGraph(solute[i]),MolGraph(solvent[i])) for i in range(len(solute))]
    SMILES_data = [SMILES, torch.Tensor(pka)]
    #graphs
    graph_data = [graphs, torch.Tensor(pka)]
    #sentences
    sentences = sentence_dataset(solute,solvent)
    sentence_data = [sentences, torch.Tensor(pka)]
    #collate data
    datasets = dict(SMILES=SMILES_data,
                    graphs=graph_data,
                    sentences=sentence_data)
    return datasets

def data_maker_decon(solute, pka, data_type, solvent=None, ids=None):
    """
    Generate a dictionary containing solute/solvent data encoded in three ways:
            SMILES = list of tuples containing (solute,solvent) smiles strings
            graphs = list of tuples containing (solute,solvent) MolGraphs
            sentences = list of tuples conatining (solute,solvent) mol2vec embeddings
    
    Parameters
    ----------
    sol_smiles : list
        List of solute SMILES strings
    solv_smiles : list or None
        List of solvent SMILES strings
    pka : list
        List of pka values
    ids : list
        List of indices to be used to create the datasets
    Returns
    -------
    datasets : dict
        Three keys: SMILES, graphs, sentences
        Values contain a list, where data[0] = encodings and data[1] = torch.Tensor of pkas
    """
    if ids == None:
        pass
    elif solvent == None:
        [solute,pka] = [[lis[x] for x in ids] for lis in (solute, pka)]
    else:
        [solute,solvent,pka] = [[lis[x] for x in ids] for lis in (solute, solvent, pka)]

    if data_type == 'graphs':
        if solvent == None:
            graphs = [MolGraph(solute[i]) for i in range(len(solute))]
        else:
            graphs = [(MolGraph(solute[i]),MolGraph(solvent[i])) for i in range(len(solute))]
        data = [graphs, torch.Tensor(pka)]
    
    else:
        sentences = sentence_dataset(solute,solvent)
        data = [sentences, torch.Tensor(pka)]

    return data

def collate_single(batch):
    '''
    Collates single input batches for a torch loader.
        
    Parameters
    ----------
    batch : List = [(X,y)]
        List of solutes with their target value.
    
    Returns
    -------
    [sol_batch, targets] : List
        Type of output depends on if the original dataset contains SMILES or sentences.
        Each component is a list / torch.Tensor.
    '''
    if type(batch[0][0]) == str:
        sol_batch = [t[0] for t in batch]
    elif type(batch[0][0]) == MolGraph:
        sol_batch = BatchMolGraph([t[0] for t in batch])
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
    data : List = [(sol,solv),pka] or [sol,pka]
        list of (solute,solvent) pairs or solutes, and tensor of target values
    scaler : pka_scaler
        Standard scaler established on the training pka data.

    Returns
    -------
    model : nn.Module
        Trained regressor model
    """
    #generate train set and val set (for early stopping)
    train_ids, val_ids, _, _ = train_test_split(ids, ids, test_size=0.1, random_state=1)
    train_loader = loader_func(data, train_ids, model.inputs, batch_size=model.batch_size)
    val_loader = loader_func(data, val_ids, model.inputs, batch_size=model.batch_size)

    regressor = copy.deepcopy(model.model)      
    optimiser = model.optimiser(regressor.parameters(), lr=model.lr)
    loss_function = torch.nn.MSELoss()
    name = model.name.replace(' ','_')
    early_stopping = EarlyStopping(name, regressor)
    
    if model.inputs == 2:
        for epoch in range(model.num_epochs):
            for (sol,solv,targets) in tqdm.tqdm(train_loader):
                sol, solv = sol.to(device), solv.to(device)
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                optimiser.zero_grad()
                outputs = regressor(sol,solv).to(device)
                cuda_targets = targets.to(device)
                loss = loss_function(outputs, cuda_targets)
                loss.backward()
                optimiser.step()
            #evaluate
            val_loss = 0
            for (sol,solv,targets) in val_loader:
                sol, solv = sol.to(device), solv.to(device)
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                outputs = regressor(sol,solv).to(device)
                cuda_targets = targets.to(device)
                loss = loss_function(outputs, cuda_targets)
                val_loss += loss.item()
            #early stopping
            print(val_loss)
            early_stopping.store(val_loss, regressor)
            if early_stopping.stop:
                #print("Stopping at epoch "+str(epoch))
                break
                
    else:
        for epoch in range(model.num_epochs):
            #train
            for (mol,targets) in tqdm.tqdm(train_loader):
                print(mol)
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                optimiser.zero_grad()
                outputs = regressor(mol)
                cuda_targets = targets.to(device)
                loss = loss_function(outputs, cuda_targets)
                loss.backward()
                optimiser.step()
            #evaluate
            for (mol,targets) in val_loader:
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                outputs = regressor(mol).to(device)
                cuda_targets = targets.to(device)
                loss = loss_function(outputs, cuda_targets)
                val_loss = loss.item()
            #early stopping
            early_stopping.store(val_loss, regressor)
            if early_stopping.stop:
                #print("Stopping at epoch "+str(epoch))
                break
    
    regressor.load_state_dict(torch.load('checkpoints/'+name+'.pt'))
    return regressor

def test(model, regressor, ids, data, scaler):
    """
    Test a model.

    Parameters
    ----------
    model : Model
        Regressor model
    regressor :
        Specific regressor for testing
    ids : list, np.array
        Indices for training samples
    data : List = [(sol,solv),pka] or [sol,pka]
        list of (solute,solvent) pairs or solutes, and tensor of target values
    scaler : pka_scaler
        Standard scaler established on the training pka data.

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
    data : List = [(sol,solv),pka] or [sol,pka]
        list of (solute,solvent) pairs or solutes, and tensor of target values
    test_ids : list, np.array
        Selected test set indices.
    exp_name : string
        Name of experiment for saving.
    datasets : dict
        Dictionary of data for fitting.

    Returns
    -------
    trained_model : Any
        Trained torch regressor model.
    results : List
        MAE, RMSE, test set size
    """
    
    size = len(data[0])
    train_ids = [i for i in range(size) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    
    trained_model = train(model, train_ids, data, scaler)
    results = test(model, trained_model, test_ids, data, scaler)
    model.experiments[name+' tuned'] = {'model':trained_model, 'scaler':scaler, 'results':results}
    return results

def fit_no_test(model, exp_name, data):
    """
    Fits a model to the whole dataset with no testing

    Parameters
    ----------
    model : torch_model / sklearn_model
        Regressor model.
    exp_name : string
        Name of experiment for saving.
    datasets : dict
        Dictionary of data for fitting.

    """
    ids = list(range(len(data[0])))
    scaler = pka_scaler(data[1])
    trained_model = train(model, ids, data, scaler)
    model.experiments[exp_name] = {'model':trained_model, 'scaler':scaler}
    torch.save(trained_model.state_dict(), 'trained/'+model.name.replace(' ','_')+'_'+exp_name.replace(' ','_')+'.pt')

    
