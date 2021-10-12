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
    def __init__(self, name, model, model_type, data_type,
                 lr=1e-3, optimiser=torch.optim.Adam, num_epochs=100, batch_size=32):
        self.name = name #e.g. "MPNN with attention"
        self.model = model #torch/sklearn regressor object
        self.model_type = model_type
        self.data_type = data_type #"SMILES" or "descriptors" or "ECFP" or "sentences"
        
        self.experiments = []
        
        #torch specific variables
        if self.model_type == 'torch':
            self.lr = lr
            self.optimiser = optimiser
            self.batch_size = batch_size
            self.num_epochs = num_epochs
        
def data_maker(solute, solvent, pka, ids=None):
    if ids == None:
        pass
    else:
        [solute,solvent,pka] = [[lis[x] for x in ids] for lis in (solute, solvent, pka)]
    #ECFP
    featurizer = dc.feat.CircularFingerprint(size=512, radius=3)
    sol = featurizer.featurize(solute)
    solv = featurizer.featurize(solvent)
    ECFP_data = [np.concatenate((sol,solv),axis=1),np.array(pka)]
    #descriptors
    featurizer = dc.feat.RDKitDescriptors()
    sol = featurizer.featurize(solute)
    solv = featurizer.featurize(solvent)
    desc_data = [np.concatenate((sol,solv),axis=1),np.array(pka)]
    #SMILES
    SMILES_pairs = [(solute[i],solvent[i]) for i in range(len(solute))]
    SMILES_data = [SMILES_pairs, torch.Tensor(pka)]
    #sentences
    sentence_pairs = d.delfos_data(solute,solvent)
    sentence_data = [sentence_pairs, torch.Tensor(pka)]
    #collate data
    datasets = dict(ECFP=ECFP_data,
                    descriptors=desc_data,
                    SMILES=SMILES_data,
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

def double_loader(data, indices, batch_size=64):
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_double)
    return loader

def train(model, ids, data, scaler, datasets):
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
    if model.model_type == 'torch':
        solvent = [datasets['SMILES'][0][x][1] for x in ids]
        train_ids, val_ids, _, _ = train_test_split(ids, solvent, test_size=0.2, random_state=1)
        train_loader = double_loader(data, train_ids, batch_size=model.batch_size)
        val_loader = double_loader(data, val_ids, batch_size=len(val_ids))
        
        regressor = copy.deepcopy(model.model)      
        optimiser = model.optimiser(regressor.parameters(), lr=model.lr)
        loss_function = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=10)
        
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
        regressor.load_state_dict(torch.load('checkpoint.pt'))
    else:
        regressor = sklearn.base.clone(model.model)
        targets = scaler.transform(data[1][ids])
        regressor.fit(data[0][ids], targets)
    return regressor

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = 1e6
        self.steps = 0
        self.stop = False
    
    def store(self, loss, net):
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps = 0
            torch.save(net.state_dict(), 'checkpoint.pt')
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
    if model.model_type == 'torch':
        loader = double_loader(data, ids, batch_size=len(ids))
        for (sol,solv,targets) in loader:
            outputs = regressor(sol,solv)
            outputs = scaler.inverse_transform(outputs)
            targets= targets.detach().numpy()
            outputs = outputs.detach().numpy()
    else:
        outputs = regressor.predict(data[0][ids])
        outputs = scaler.inverse_transform(outputs)
        targets = data[1][ids]
        
    results = [mae(targets, outputs),rmse(targets, outputs)]
    return results

def predict(model, experiment, data):
    ids = list(range(len(data[0])))
    if model.model_type == 'torch':
        loader = double_loader(data, ids, batch_size=len(ids))
        for (sol,solv,targets) in loader:
            outputs = experiment['model'](sol,solv)
            outputs = experiment['scaler'].inverse_transform(outputs)
            targets= targets.detach().numpy()
            outputs = outputs.detach().numpy()
    else:
        outputs = experiment['model'].predict(data[0][ids])
        outputs = experiment['scaler'].inverse_transform(outputs)
        targets = data[1][ids]
        
    return targets, outputs

def CV_fit(model, data, datasets, folds=5, random_state: int=None):
    """
    Build a cross-validated regressor consisting of k-models.

    Parameters
    ----------
    model : torch_model / sklearn_model
        Regressor model. [stores trained CV models]
    data : List = [(sol,solv),pka]
        Full dataset of (solute,solvent) pairs and target values.
    folds : int
        Number of folds for cross validation.
    random_state : int
        Integer to use for seeding the k-fold split.

    Returns
    -------
    avg_result : List
        List of average MAE and RMSE.
    results : List
        List of lists of MAE and RMSE for each fold.
    """
    kf = KFold(n_splits=folds, shuffle=False, random_state=random_state)
    kf = kf.split(X=data[0])

    # Fit k models and store them
    results = []
    for train_ids, test_ids in kf:
        scaler = pka_scaler(data[1][train_ids])
        if model.data_type == 'descriptors':
            desc_scaler = StandardScaler()
            desc_scaler.fit(data[0][train_ids])
        data[0] = desc_scaler.transform(data[0])
        fold_model = train(model, train_ids, data, scaler, datasets)
        fold_result = test(model, fold_model, test_ids, data, scaler)

        results.append(fold_result)
    avg_result = np.mean(results, axis=0)
    return avg_result, results

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
    if model.data_type == 'descriptors':
        desc_scaler = StandardScaler()
        desc_scaler.fit(data[0][train_ids])
        data[0] = desc_scaler.transform(data[0])
        
    trained_model = train(model, train_ids, data, scaler, datasets)
    results = test(model, trained_model, test_ids, data, scaler)
    model.experiments.append({'name':exp_name,'model':trained_model, 'results':results, 'scaler':scaler})
    return results


#RESULTS HELPERS
def rmse(y_true, y_pred):
    """Helper function"""
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    """Helper function"""
    return mean_absolute_error(y_true, y_pred)

#HYPERPARAMETER OPTIMISATION
from timeit import default_timer as timer
from hyperopt import STATUS_OK, Trials, fmin, tpe

class fitness:
    """
    For conducting cross validation on a model with a given set of hyperparameters for optimisation.
    
    Parameters
    ----------
    model_dict : dict
        Key word arguments to be fed into a b.Model class.
    model_param_names : List
        Hyperparameter names specific to the regressor model.
    training_param_names : List
        Hyperparameter names specific to training.
    """
    def __init__(self, model_dict, model_param_names, training_param_names, datasets):
        self.m = model_dict
        self.model_param_names = model_param_names
        self.training_param_names = training_param_names
        self.datasets = datasets
    
    def objective(self, params):
        """
        Objective function for bayesian hyperparameter optimisation.
        
        Parameters
        ----------
        params : dict
            Specific set of model and training hyperparameters for testing.
        
        Returns
        -------
        dict
            Results of CV testing, including MAE loss, runtime and the original parameter list"""
        
        model_params = dict()
        training_params = dict()
        for param_name in self.model_param_names:
            model_params[param_name] = params[param_name]
        for param_name in self.training_param_names:
            training_params[param_name] = params[param_name]
        
        copy = self.m['model']
        self.m['model'] = self.m['model'](**model_params)
        self.m.update(training_params)
        
        model = Model(**self.m)
        data = self.datasets[model.data_type]

        start = timer()
        res, full_res = CV_fit(model, data, self.datasets)
        run_time = timer()-start

        loss = res[0]
        self.m['model'] = copy
        
        return {'loss': loss, 'params': params, 'run_time': run_time, 'status': STATUS_OK}
    
def hyperopt_func(model_dict, model_param_names, training_param_names, param_space, datasets, max_evals=30):
    """
    Bayesian hyperparameter optimisation function.
    
    Parameters
    ----------
    model_dict : dict
        Key word arguments to be fed into a b.Model class.
    model_param_names : List
        Hyperparameter names specific to the regressor model.
    training_param_names : List
        Hyperparameter names specific to training.
    param_space : dict
        Distribution of choices for each hyperparameter to be optimised.
    max_evals : int
        Maximum number of evaluations of hyperparameter sets.
        
    Returns
    -------
    results : list
        Results from each evaluation of the objective function, sorted from best to worst result.
    """
    tester = fitness(model_dict, model_param_names, training_param_names, datasets)
    trials = Trials()
    
    timer_start = timer()
    best = fmin(fn=tester.objective, 
                space=param_space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials, 
                rstate=np.random.RandomState(50))
    timer_end = timer()
    print('Total training time (min):',(timer_end-timer_start)/60)
    results = sorted(trials.results, key = lambda x: x['loss'])
    return results