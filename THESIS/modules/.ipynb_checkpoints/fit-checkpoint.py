import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn
import torch
import copy
from .data import *
from collections import defaultdict as ddict
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
    """
    Object containing a model and all its associated parameters.
    
    Parameters
    ----------
    name : string
    model : torch/sklearn regressor object
    model_type : ['torch','sklearn']
    data_type : ['SMILES','descriptors','ECFP','sentences']
    experiments : dict
        Collection of models, scalers, names of experiments
    lr : float
        Optimiser learning rate
    optimiser : torch.optim object
    batch_size : int
    num_epochs : int
        Maximum number of epochs for training
    """
    def __init__(self, name, model, model_type, data_type,
                 lr=1e-3, optimiser=torch.optim.Adam, num_epochs=100, batch_size=32):
        self.name = name 
        self.model = model 
        self.model_type = model_type 
        self.data_type = data_type 
        
        self.experiments = ddict()
        
        #torch specific variables
        if self.model_type == 'torch':
            self.lr = lr
            self.optimiser = optimiser
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

def train(model, ids, data, scaler):
    """
    Train a model using early stopping.

    Parameters
    ----------
    model : Model
        Regressor model
    ids : list, np.ndarray
        Training ids for the given data
    data : list
        if model_type == torch: list of (solute,solvent) pairs and tensor of target values
        if model_type == sklearn: list of solute;solvent vectors and array of target values
    scaler : pka_scaler
        Standard scaler established on the training pka data.
        
    Returns
    -------
    model : torch/sklearn regressor model
        Trained regressor model
    """
    if model.model_type == 'torch':
        train_ids, val_ids, _, _ = train_test_split(ids, ids, test_size=0.1, random_state=1)
        train_loader = double_loader(data, train_ids, batch_size=model.batch_size)
        val_loader = double_loader(data, val_ids, batch_size=len(val_ids))
        
        regressor = copy.deepcopy(model.model)      
        optimiser = model.optimiser(regressor.parameters(), lr=model.lr)
        loss_function = torch.nn.MSELoss()
        name = model.name.replace(' ','_')
        early_stopping = EarlyStopping(name,regressor)
        
        for epoch in range(model.num_epochs):
            #train
            for (sol,solv,targets) in train_loader:
                if model.data_type == 'sentences':
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
            for (sol,solv,targets) in val_loader:
                if model.data_type == 'sentences':
                    sol, solv = sol.to(device), solv.to(device)
                targets = targets.view(-1,1)
                targets = scaler.transform(targets)
                outputs = regressor(sol,solv).to(device)
                cuda_targets = targets.to(device)
                loss = loss_function(outputs, cuda_targets)
                val_loss = loss.item()
            #early stopping
            early_stopping.store(val_loss, regressor)
            if early_stopping.stop:
                #print("Stopping at epoch "+str(epoch))
                break
        regressor.load_state_dict(torch.load('checkpoints/'+name+'.pt'))
    else:
        regressor = sklearn.base.clone(model.model)
        targets = scaler.transform(data[1][ids])
        regressor.fit(data[0][ids], targets)
    return regressor

class EarlyStopping:
    """
    Module for tracking early stopping for torch training.
    Stops training after #patience steps of no improvement in the best validation loss.
    Saves the best validation loss and the trained module.
    
    Parameters
    ----------
    name : string
        Name of model for checkpoint saving.
    patience : int
        Number of steps to wait after the best loss before terminating training.
    loss : float
        Validation loss.
    net : torch model
        Trained model corresponding to loss.
    """
    def __init__(self, name, net, patience=10):
        self.patience = patience
        self.best_loss = 1e6
        self.steps = 0
        self.stop = False
        self.chk_name = 'checkpoints/'+name+'.pt'
        torch.save(net.state_dict(), self.chk_name)
    
    def store(self, loss, net):
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps = 0
            torch.save(net.state_dict(), self.chk_name)
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
    regressor : torch / sklearn regressor
        Specific trained regressor for testing
    ids : list, np.array
        Indices for testing samples
    data : List = [(sol,solv),pka]
        if model_type == torch: list of (solute,solvent) pairs and tensor of target values
        if model_type == sklearn: list of solute;solvent vectors and array of target values
    scaler : pka_scaler
        Standard scaler established on the training pka data.
        
    Returns
    -------
    Results: list
        List of MAE, RMSE.
    """
    if model.model_type == 'torch':
        loader = double_loader(data, ids, batch_size=len(ids))
        for (sol,solv,targets) in loader:
            if model.data_type == 'sentences':
                sol, solv = sol.to(device), solv.to(device)
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

def predict(model, exp_name, data, ids=None):
    """
    Use a trained model to predict for comparison with target values.

    Parameters
    ----------
    model : Model
        Regressor model
    exp_name : string
        Name of the experiment containing the trained model and target value scaler (and descriptor scaler)
    ids : list, np.array
        Indices for testing samples
    data : List = [(sol,solv),pka]
        if model_type == torch: list of (solute,solvent) pairs and tensor of target values
        if model_type == sklearn: list of solute;solvent vectors and array of target values

    Returns
    -------
    Results: list
        List of MAE, RMSE.
    """
    experiment = model.experiments[exp_name] 
    if ids==None:
        ids = list(range(len(data[0])))
    if model.model_type == 'torch':
        loader = double_loader(data, ids, batch_size=32)
        target_list = []
        output_list = []
        for (sol,solv,targets) in loader:
            outputs = experiment['model'](sol,solv)
            outputs = experiment['scaler'].inverse_transform(outputs)
            targets = targets.detach().numpy()
            outputs = outputs.detach().numpy()
            target_list.append(targets)
            output_list.append(outputs)
        targets = np.concatenate(target_list)
        outputs = np.concatenate(output_list)
    else:
        if model.data_type == 'descriptors':
            desc_scaler = StandardScaler()
            desc_scaler.fit(experiment['desc scaling data'])
            x_data = desc_scaler.transform(data[0][ids])
        else:
            x_data = data[0][ids]
        outputs = experiment['model'].predict(x_data)
        outputs = experiment['scaler'].inverse_transform(outputs)
        targets = data[1][ids]
        
    return targets, outputs

def CV_fit(model, data, folds=5, random_state: int=None):
    """
    Build a cross-validated regressor consisting of k-models.

    Parameters
    ----------
    model : Model
        Regressor model. 
    data : List = [(sol,solv),pka]
        if model_type == torch: list of (solute,solvent) pairs and tensor of target values
        if model_type == sklearn: list of solute;solvent vectors and array of target values
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
        fold_model = train(model, train_ids, data, scaler)
        fold_result = test(model, fold_model, test_ids, data, scaler)

        results.append(fold_result)
    avg_result = np.mean(results, axis=0)
    return avg_result, results

def fit(model, data, test_ids, exp_name, train_ids=None):
    """
    Fits a model according to the given test_ids and data.

    Parameters
    ----------
    model : Model
        Regressor model.
    data : List = [(sol,solv),pka]
        if model_type == torch: list of (solute,solvent) pairs and tensor of target values
        if model_type == sklearn: list of solute;solvent vectors and array of target values
    test_ids : list, np.array
        Selected test set indices.
    exp_name : string
        Name of experiment for saving.
    train_ids : list, np.array
        Selected training set indices.

    Returns
    -------
    results : List
        MAE, RMSE
    """
    if model.model_type == 'torch':
        size = len(data[0])
    else:
        size = data[0].shape[0]
        
    if train_ids == None:
        train_ids = [i for i in range(size) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    
    if model.data_type == 'descriptors':
        desc_scaler = StandardScaler()
        scaling_data = data[0][train_ids]
        desc_scaler.fit(scaling_data)
        data[0] = desc_scaler.transform(data[0])
    else:
        scaling_data = None
        
    trained_model = train(model, train_ids, data, scaler)
    results = test(model, trained_model, test_ids, data, scaler)
    model.experiments[exp_name] = {'model':trained_model, 'results':results, 'scaler':scaler, 'desc scaling data':scaling_data}
    return results

def save_model(model, exp_name):
    m = model.experiments[exp_name]['model']
    filename = 'trained/'+model.name+'_'+exp_name
    filename = filename.replace(' ','_')
    if model.model_type == 'sklearn':
        pickle.dump(m, open(filename+'.pkl', 'wb'))
    else:
        torch.save(m.state_dict(), filename+'.pt')
        
def load_model(model, exp_name):
    filename = 'trained/'+model.name+'_'+exp_name
    filename = filename.replace(' ','_')
    if model.model_type == 'sklearn':
        loaded_model = pickle.load(open(filename+'.pkl', 'rb'))
    else:
        loaded_model = copy.deepcopy(model.model) 
        loaded_model.load_state_dict(torch.load(filename+'.pt'))
    model.experiments[exp_name] = ddict()
    model.experiments[exp_name]['model'] = loaded_model
        
def load_exp(model, exp_name, data, train_ids):
    load_model(model, exp_name)
    scaler = pka_scaler(data[1][train_ids])
    model.experiments[exp_name]['scaler'] = scaler
    if model.data_type == 'descriptors':
        scaling_data = data[0][train_ids]
        model.experiments[exp_name]['desc scaling data'] = scaling_data
        
#RESULTS HELPERS
def rmse(y_true, y_pred):
    """Helper function"""
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    """Helper function"""
    return mean_absolute_error(y_true, y_pred)