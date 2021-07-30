# 2017 DeepCrystal Technologies - Patrick Hop
#
# Data loading a splitting file
#
# MIT License - have fun!!
# ===========================================================

import os
import random
from collections import OrderedDict

import deepchem as dc
from deepchem.splits import RandomSplitter
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

def split(dataset,
          frac_train=.80,
          frac_valid=.10,
          frac_test=.10,
          log_every_n=1000):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    list_idx = list(range(dataset.size))
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=frac_test)
    train_idx, valid_idx = train_test_split(train_idx, test_size=frac_valid)
    return train_idx, valid_idx, test_idx

def load_dataset(filename, whiten=False):
    f = open(filename, 'r')
    features = []
    labels = []
    tracer = 0
    for line in f:
        if tracer ==0:
            tracer += 1
            continue
        print(tracer)
        splits =  line[:-1].split(',')
        labels.append(float(splits[-1]))
        features.append(splits[-2])
    features = np.array(features)
    labels = np.array(labels, dtype='float32').reshape(-1, 1)

    train_ind, val_ind, test_ins = split(features)

    train_features = np.take(features, train_ind)
    train_labels = np.take(labels, train_ind)
    val_features = np.take(features, val_ind)
    val_labels = np.take(labels, val_ind)

    return train_features, train_labels, val_features, val_labels