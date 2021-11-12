import torch
from torch import nn
from mol2vec.features import mol2alt_sentence
from gensim.models import word2vec
from rdkit import Chem
import numpy as np
import deepchem as dc
import sklearn
from .MP_utils import BatchMolGraph, MolGraph

#modified sentence2vec function to return lists of word vectors
def sentences2vecs(sentences, model, unseen=None):
    """
    Generate vectors for each word in a sentence (list) in a list of sentences.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032
    Returns
    -------
    list of arrays, each sentence -> tensor of word vectors
    """
    keys = set(model.wv.key_to_index)
    bigveclist = []
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        veclist = []
        if unseen:
            veclist.append([model.wv.get_vector(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence])
        else:
            veclist.append([model.wv.get_vector(y) for y in sentence 
                            if y in set(sentence) & keys])
        vecarray = np.concatenate(veclist, axis=1)
        vectensor = torch.Tensor(vecarray)
        bigveclist.append(vectensor)
    return bigveclist

#load mol2vec model
mol2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')

def sentence_generator(smiles):
    """
    Generate a list of substructures at radii r=0 and r=1.
    """
    mol = Chem.MolFromSmiles(smiles)
    sentence = mol2alt_sentence(mol,1)
    return sentence

#return dataset of vector sentences for each solute solvent pair
def sentence_dataset(sol_smiles, solv_smiles=None):
    """
    Return list of sentences (tensors) for each input SMILES string according to the mol2vec encoder.
    
    Parameters
    ----------
    sol_smiles : list
        List of solute SMILES strings
    solv_smiles : list or None
        List of solvent SMILES strings
        if None then only the solute SMILES are encoded
    Returns
    -------
    if solv_smiles == None
        list of tuples containing the (solute,solvent) sentences
    else
        list of solute sentences (tensors)
    """
    if solv_smiles == None:
        sentences = [sentence_generator(x) for x in sol_smiles]
        vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
        return vecs 
    else:
        data = []
        size = len(sol_smiles)
        for mols in [sol_smiles,solv_smiles]:
            sentences = [sentence_generator(x) for x in mols]
            vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
            data.append(vecs)
        pairs = [(data[0][i],data[1][i]) for i in range(size)]
        return pairs

def data_maker(solute, solvent, pka, ids=None):
    """
    Generate a dictionary containing solute/solvent data encoded in five ways:
            ECFP = concatenated solute;solvent 2048 bit ECFP encodings
            descriptors = concatenated solute;solvent vectors, each with 200 features calculated by RDKit
            SMILES = list of tuples containing (solute,solvent) smiles strings
            graphs = list of tuples containing (solute,solvent) MolGraphs
            sentences = list of tuples conatining (solute,solvent) mol2vec embeddings
    
    Parameters
    ----------
    sol_smiles : list
        List of solute SMILES strings
    solv_smiles : list
        List of solvent SMILES strings
    pka : list
        List of pka values
    ids : list
        List of indices to be used to create the datasets
    Returns
    -------
    datasets : dict
        Five keys: ECFP, descriptors, graphs, SMILES, sentences
        Values contain a list, where data[0] = paired encodings and data[1] = pka as either an array or tensor
    """
    if ids == None:
        pass
    else:
        [solute,solvent,pka] = [[lis[x] for x in ids] for lis in (solute, solvent, pka)]
    #ECFP
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=3)
    sol = featurizer.featurize(solute)
    solv = featurizer.featurize(solvent)
    ECFP_data = [np.concatenate((sol,solv),axis=1),np.array(pka)]
    #descriptors
    featurizer = dc.feat.RDKitDescriptors()
    sol = featurizer.featurize(solute)
    sol = np.delete(sol, (17,18,19,20,21,22,23,24), axis=1)
    solv = featurizer.featurize(solvent)
    solv = np.delete(solv, (17,18,19,20,21,22,23,24), axis=1)
    desc_data = [np.concatenate((sol,solv),axis=1),np.array(pka)]
    #SMILES
    SMILES_pairs = [(solute[i],solvent[i]) for i in range(len(solute))]
    SMILES_data = [SMILES_pairs, torch.Tensor(pka)]
    #molgraph
    graph_pairs = [(MolGraph(solute[i]),MolGraph(solvent[i])) for i in range(len(solute))]
    graph_data = [graph_pairs, torch.Tensor(pka)]
    #sentences
    sentence_pairs = sentence_dataset(solute,solvent)
    sentence_data = [sentence_pairs, torch.Tensor(pka)]
    #collate data
    datasets = dict(ECFP=ECFP_data,
                    descriptors=desc_data,
                    graphs=graph_data,
                    SMILES=SMILES_data,
                    sentences=sentence_data)
    return datasets

class pka_scaler:
    """Uses training pka data to scale the output predictions.
       Can take either an ndarray or tensor as input.
       Must be initialised on the training data prior to training.
           transform: to be used on target values during training
           inverse_transform: to be used on test predictions during testing
    """
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
            targets = targets.cpu().detach().numpy()
            transformed_targets = self.scaler.transform(targets)
            return torch.Tensor(transformed_targets)
    
    def inverse_transform(self, targets):
        if type(targets) == np.ndarray:
            targets = targets.reshape(-1,1)
            transformed_targets = self.scaler.inverse_transform(targets)
            return transformed_targets.ravel()
        else:
            targets = targets.cpu().detach().numpy()
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
        Each component is a BatchMolGraph / torch.Tensor.
    '''
    if type(batch[0][0][0]) == MolGraph:
        sol_batch = BatchMolGraph([t[0][0] for t in batch])
        solv_batch = BatchMolGraph([t[0][1] for t in batch])
    elif type(batch[0][0][0]) == str:
        sol_batch = [t[0][0] for t in batch]
        solv_batch = [t[0][1] for t in batch]
    else:
        sol_batch = [torch.Tensor(t[0][0]) for t in batch]
        sol_batch = nn.utils.rnn.pad_sequence(sol_batch)
        solv_batch = [torch.Tensor(t[0][1]) for t in batch]
        solv_batch = nn.utils.rnn.pad_sequence(solv_batch)
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
