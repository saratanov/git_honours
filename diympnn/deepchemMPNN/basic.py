#basic functions for ML

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

#FEATURISERS
not_used_desc = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']
desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc])

#DATASET
class Dataset(torch.utils.data.Dataset):
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

#CROSS-VAL
class CV_torch:
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
#    __slots__ = ('est', 'models', 'n_folds', 'params', 'shuffle', 'num_epochs')

    def __init__(self, est: Any, n_folds: int = 2, params: Any = None, shuffle: bool = True, num_epochs: int = 2):
        self.est = est
        self.params = params
        
        self.models = []
        self.LOSO_models = ddict(list)
        self.LOEO_models = ddict(list)
        self.LOMO_models = ddict(list)
        
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        
        self.cv_scores = ddict(list)
        self.LOSO_scores = ddict(list)
        self.LOEO_scores = ddict(list)
        self.LOMO_scores = ddict(list)
        
    def train_func(self, model, indices, x_data, y_data):
        """
        Train a torch model.
        
        Parameters
        ----------
        model : Any
            torch regressor model
        indices :
            Indices for training samples
        x_data : 
            Training data
        y_data :
            Target values
        """
        
        dataset = Dataset(indices, x_data, y_data)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_function = torch.nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in trainloader:
                inputs = x_batch
                targets = y_batch.view(-1,1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
        return model
            
    def fit(self, x_data: torch.Tensor, y_data: torch.Tensor, random_state: int=None) -> None:
        """
        Build a regressor consisting of k-models.
        
        Parameters
        ----------
        x_data : torch.tensor
            Training data
        y_data : torch.tensor
            Target values
        random_state : int
            Integer to use for seeding the k-fold split
        """
        
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=random_state)
        kf = kf.split(X=x_data, y=y_data)

        # Fit k models and store them
#        pbar = tqdm(total=self.n_folds, position=0, leave=True, desc="Cross val")
        for train_index, test_index in kf:
            model = self.est(**self.params) #create a fresh copy of the initial model
            est_trained = self.train_func(model, train_index, x_data, y_data)
            test_set = Dataset(test_index, x_data, y_data)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_index))
            for test_x, test_y in testloader:
                test_pred = est_trained(test_x)
                test_y = test_y.detach().numpy()
                test_pred = test_pred.detach().numpy()
                for sf in (mae, rmse, r2_score):
                    self.cv_scores[str(sf).split(' ')[1]].append(sf(test_y, test_pred))
            self.models.append(est_trained)
#            pbar.update(1)
#        pbar.close()
        
    def special_fit(self, x_data: torch.Tensor, y_data: torch.Tensor, special: str) -> None:
        """
        Creates a leave-one-x-out cross validation model.
        special = 'LOSO' or 'LOEO' or 'LOMO'
        The LOxO_scores dictionary contains [MAE, RMSE, test size].
        The models are stored in the LOxO_models dictionary.
        """
        total_sol_list = [pair[0] for pair in x_data]
        total_solv_list = [pair[1] for pair in x_data]
        size = len(total_sol_list)

        if special == 'LOSO':
            spec_list = list(set(total_solv_list)) #all unique solvents in dataset
        elif special == 'LOEO':
            spec_list = ['F','N'] #elements
        elif special == 'LOMO':
            mass_list = [MolWt(Chem.MolFromSmiles(mol)) for mol in total_sol_list]
            spec_list = [50,100,150] #mass cutoffs
        
        for spec in spec_list:
            if special == 'LOSO':
                test_index = [i for i, x in enumerate(total_solv_list) if x == spec]
            elif special == 'LOEO':
                test_index = [i for i, x in enumerate(total_sol_list) if spec in x]
            elif special == 'LOMO':
                test_index = [i for i, x in enumerate(mass_list) if x > spec]
            train_index = [i for i in range(size) if i not in test_index]
            
            model = self.est(**self.params)
            est_trained = self.train_func(model, train_index, x_data, y_data)
            test_set = Dataset(test_index, x_data, y_data)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_index))
            
            for test_x, test_y in testloader:
                test_pred = est_trained(test_x)
                test_y = test_y.detach().numpy()
                test_pred = test_pred.detach().numpy()
                results = [mae(test_y, test_pred),rmse(test_y, test_pred),len(test_index)]
                
                if special == 'LOSO':
                    self.LOSO_scores[spec].append(results)
                    self.LOSO_models[spec].append(est_trained)
                elif special == 'LOEO':
                    self.LOEO_scores[spec].append(results)
                    self.LOEO_models[spec].append(est_trained)
                elif special == 'LOMO':
                    self.LOMO_scores[str(spec)].append(results)
                    self.LOMO_models[str(spec)].append(est_trained)

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

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, random_state: int=None) -> None:
        """
        Build a regressor consisting of k-models.
        
        Parameters
        ----------
        x_data : numpy.ndarray
            Training data
        y_data : numpy.ndarray
            Target values
        random_state : int
            Integer to use for seeding the k-fold split
        """

        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=random_state)
        kf = kf.split(X=x_data, y=y_data)

        # Fit k models and store them
        for train_index, test_index in kf:
            est_tmp = self.est(**self.params)
            est_tmp.fit(x_data[train_index], y_data[train_index])
            test_pred = est_tmp.predict(x_data[test_index])
            for sf in (mae, rmse, r2_score):
                self.cv_scores[str(sf).split(' ')[1]].append(sf(y_data[test_index], test_pred))
            self.models.append(est_tmp)

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

def calc_stats_str(pka1, pka2):
    """Calculates R², MAE and RMSE for two iterables of floats or integers"""
    assert len(pka1) == len(pka2), "Both iterables must have the same length"
    return f'R²: {r2_score(pka1, pka2):.3f}\n' \
           f'MAE: {mae(pka1, pka2):.3f}\n' \
           f'RMSE: {rmse(pka1, pka2):.3f}'

def train_cv_model(est_cls, x_data, y_data, params, random_state,
                   cv=2, shuffle=True, torch_model=True):
    """Trains a cross-validated model"""
    if torch_model == True:
        cvr = CV_torch(est=est_cls, params=params, n_folds=cv, shuffle=shuffle)
        cvr.fit(x_data, y_data, random_state=random_state)
    else:
        cvr = CV_scikit(est=est_cls, params=params, n_folds=cv, shuffle=shuffle)
        cvr.fit(x_data, y_data, random_state=random_state)
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
