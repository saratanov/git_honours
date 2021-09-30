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
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import deepchem as dc
import copy
from tqdm import tqdm

class torch_model:
    """
    Object containing a torch model and all its associated parameters.
    """
    model_type = "torch"
    
    def __init__(self, name, model, params, data_type):
        self.name = name #e.g. "MPNN with attention"
        self.model = model #torch/sklearn regressor object
        self.params = params #dictionary of parameters
        self.data_type = data_type #"SMILES" or "descriptors" or "ECFP" or "sentences"

#FEATURISERS
not_used_desc = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']
desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc])

#DATASET - returns (solute,solvent) pair
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

#CROSS-VAL
class torch_testing:
    """
    Regressor that predicts based on predictions of k models from k-fold CV.
    Accepts any torch-like regressor as base regressor. It trains k models
    by doing k-fold CV and stores the individual models. Predictions
    on new samples are done by calculating mean predictions from all models.
    
    Parameters
    ----------
    est : Any
        torch (-like) regressor model
    params : 
        Regressor parameters
    n_folds : int
        Number of folds for k-fold
    shuffle : bool
        Shuffling of data for CV
    """
    def __init__(self, est: Any, n_folds: int = 2, params: Any = None, shuffle: bool = True, num_epochs: int = 2):
        self.est = est
        self.params = params
        
        self.models = []
        
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        
        self.cv_scores = ddict(list)
        
    def train_func(self, model, indices, data):
        """
        Train a torch model.
        
        Parameters
        ----------
        model : Any
            torch regressor model
        indices : list, np.array
            Indices for training samples
        data : List = [(sol,solv),pka]
            Data of (solute,solvent) pairs and target values
            
        Returns
        -------
        model : Any
            trained torch regressor model
        """
        
        trainloader = double_loader(data, indices, batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_function = torch.nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            for (sol,solv,targets) in trainloader:
                targets = targets.view(-1,1)
                optimizer.zero_grad()
                outputs = model(sol,solv)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
        return model
    
    def test_func(self, model, indices, data):
        """
        Test a torch model.
        
        Parameters
        ----------
        model : Any
            Trained torch regressor model.
        indices : list, np.array
            Indices for testing samples.
        data : List = [(sol,solv),pka]
            Data of (solute,solvent) pairs and target values.
            
        Returns
        -------
        Results: list
            List of MAE, RMSE.
        """
        testloader = double_loader(data, indices, batch_size=len(indices))
        for (sol,solv,targets) in testloader:
            outputs = est_trained(sol,solv)
            targets= targets.detach().numpy()
            outputs = outputs.detach().numpy()
            results = [mae(targets, outputs),rmse(targets, outputs)]
        return results
            
    def CV_fit(self, data, random_state: int=None):
        """
        Build a cross-validated regressor consisting of k-models.
        
        Parameters
        ----------
        data : List = [(sol,solv),pka]
            Full dataset of (solute,solvent) pairs and target values.
        random_state : int
            Integer to use for seeding the k-fold split.
            
        Returns
        -------
        trained_models : list
            List of trained torch regressor model.
        avg_result : List
            List of average MAE and RMSE.
        results : List
            List of lists of MAE and RMSE for each fold.
        """
        
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=random_state)
        
        kf = kf.split(X=data[0])

        # Fit k models and store them
        results = []
        trained_models = []
        for train_index, test_index in kf:
            model = self.est(**self.params) #create a fresh copy of the initial model
            est_trained = self.train_func(model, train_index, data) #train
            fold_result = test_func(est_trained, test_index, data) #test
            
            results.append(fold_result)
            trained_models.append(est_trained)
            self.models.append(est_trained)
        avg_result = np.mean(results, axis=0)
        return trained_models, avg_result, results

    def fit(self, data, test_ids):
        """
        Fits a model according to the given test_ids and data.

        Parameters
        ----------
        data : List = [(sol,solv),pka]
            Full dataset of (solute,solvent) pairs and target values.
        test_ids : list, np.array
            Selected test set indices.
            
        Returns
        -------
        est_trained : Any
            Trained torch regressor model.
        results : List
            MAE, RMSE, test set size
        """
        train_ids = [i for i in range(len(data[0])) if i not in test_ids]
        
        model = self.est(**self.params)
        est_trained = self.train_func(model, train_ids, data)
        
        results = test_func(est_trained, test_ids, data)
        results.append(len(test_ids))
        return est_trained, results

    def predict(self, x_data: torch.Tensor) -> np.ndarray:
        """
        Predict using prediction mean from ensemble of k CV models.
        
        Parameters
        ----------
        x_data : torch.Tensor
            Samples to predict
        
        Returns
        -------
        numpy.ndarray
            Predicted values
        """

        return np.mean([m(x_data) for m in self.models], axis=0)
    
class CV_scikit:
    """
    Regressor that predicts based on predictions of k models from k-fold CV.
    Accepts any Scikit-learn-like regressor as base regressor. It trains k models
    by doing k-fold CV and stores the individual models. Predictions
    on new samples are done by calculating mean predictions from all models.
    
    Parameters
    ----------
    est : Any
        Scikit-learn (-like) regressor object. Must contain .fit() and .predict() methods.
    params : Dict[str, Any]
        Regressor parameters
    n_folds : int
        Number of folds for k-fold
    shuffle : bool
        Shuffling of data for CV
    """
    __slots__ = ('est', 'params', 'models', 'n_folds', 'shuffle', 'cv_scores')

    def __init__(self, est: Any, params: Dict[str, Any], n_folds: int = 5, shuffle: bool = True):
        self.est = est
        self.params = params
        self.models = []
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_scores = ddict(list)

    def CV_fit(self, data, random_state: int=None) -> None:
        """
        Build a cross-validated regressor consisting of k-models.
        
        Parameters
        ----------
        data : List = [(sol,solv),pka]
            Full dataset of (solute,solvent) pairs and target values.
        random_state : int
            Integer to use for seeding the k-fold split.
            
        Returns
        -------
        trained_models : list
            List of trained sklearn regressor model.
        avg_result : List
            List of average MAE and RMSE.
        results : List
            List of lists of MAE and RMSE for each fold.
        """

        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=random_state)
        kf = kf.split(X=x_data, y=y_data)

        # Fit k models and store them
        results = []
        trained_models = []
        for train_index, test_index in kf:
            est_tmp = self.est(**self.params)
            est_tmp.fit(data[0][train_index], data[1][train_index])
            
            outputs = est_tmp.predict(data[0][test_index])
            targets = data[1][test_index]
            results.append([mae(targets, outputs),rmse(targets, outputs)])
            trained_models.append(est_tmp)
            self.models.append(est_tmp)
        avg_result = np.mean(results, axis=0)
        return trained_models, avg_result, results    
        
    def fit(self, data, test_ids) -> None:
        """
        Fits a model according to the given test_ids and data.

        Parameters
        ----------
        data : List = [(sol,solv),pka]
            Full dataset of (solute,solvent) pairs and target values.
        test_ids : list, np.array
            Selected test set indices.
            
        Returns
        -------
        est_trained : Any
            Trained sklearn regressor model.
        results : List
            MAE, RMSE, test set size
        """
        size = data[0].shape[0]
        train_ids = [i for i in range(size) if i not in test_ids]
        
        est_tmp = self.est(**self.params)
        est_tmp.fit(data[0][train_ids], data[1][train_ids])

        outputs = est_tmp.predict(data[0][test_ids])
        targets = data[1][test_ids]
        results = [mae(targets, outputs),rmse(targets, outputs), len(test_ids)]
        return est_tmp, results

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Predict using prediction mean from k models.
        
        Parameters
        ----------
        x_data : numpy.ndarray
            Samples to predict
        
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        return np.mean([m.predict(x_data) for m in self.models], axis=0)
    
def rmse(y_true, y_pred):
    """Helper function"""
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    """Helper function"""
    return mean_absolute_error(y_true, y_pred)

def train_cv_model(est_cls, data, params, random_state,
                   cv=5, shuffle=True, torch_model=True):
    """Trains a cross-validated model"""
    if torch_model:
        cvr = CV_torch(est=est_cls, params=params, n_folds=cv, shuffle=shuffle)
        cvr.CV_fit(data, random_state=random_state)
    else:
        cvr = CV_scikit(est=est_cls, params=params, n_folds=cv, shuffle=shuffle)
        cvr.CV_fit(data, random_state=random_state)
    return cvr

def generate_score_board(name):
    print(f'{name} CV Scores:')
    for ts, m in models[name].items():
        print(f'\t{ts}')
        for k, v in m.cv_scores.items():
            print(f'\t\t- {k}: {np.mean(v):.3f} ± {np.std(v):.3f}')
            
def calc_xy_data(mols):
    """Calculates descriptors and fingerprints for an iterable of pairs of RDKit molecules"""
    descs = []  # 196/200 RDKit descriptors
    fmorgan3 = []  # 4096 bit
    def fmorgan3_func(mol):
        return Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True)
    for mol in mols:
        descs.append(desc_calc.CalcDescriptors(mol[0])+desc_calc.CalcDescriptors(mol[1]))
        fmorgan3.append(fmorgan3_func(mol[0])+fmorgan3_func(mol[1]))
    descs = np.array(descs)
    fmorgan3 = np.array(fmorgan3)
    return descs, fmorgan3, np.concatenate([descs, fmorgan3], axis=1)

def calc_x_data(mols):
    """Calculates descriptors and fingerprints for an iterable of RDKit molecules"""
    descs = []  # 196/200 RDKit descriptors
    fmorgan3 = []  # 4096 bit
    for mol in mols:
        descs.append(desc_calc.CalcDescriptors(mol))
        fmorgan3.append(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True))
    descs = np.array(descs)
    fmorgan3 = np.array(fmorgan3)
    return descs, fmorgan3, np.concatenate([descs, fmorgan3], axis=1)
