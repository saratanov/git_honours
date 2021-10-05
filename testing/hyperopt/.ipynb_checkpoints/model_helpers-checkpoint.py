import torch
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
import numpy as np
import torch.nn as nn

#MODEL HELPERS

def int_func(X,Y,func):
    if func == 'exp':
        #interaction map
        I = torch.exp(torch.mm(X,Y.t()))
        #inverse of sum of columns for I and transpose I
        x_norm = torch.pow(torch.sum(I, dim=1),-1)
        y_norm = torch.pow(torch.sum(I.t(), dim=1),-1)
        #interaction maps with normalised columns
        I_x = I * x_norm[:,None] 
        I_y = I.t() * y_norm[:,None]
        #contexts
        X_context = torch.mm(I_x,Y)
        Y_context = torch.mm(I_y,X)
    if func == 'tanh':
        #interaction map
        I = torch.tanh(torch.mm(X,Y.t()))
        #contexts
        X_context = torch.mm(I,Y)
        Y_context = torch.mm(I.t(),X)
    #concatenate
    catX = torch.cat((X,X_context))
    catY = torch.cat((Y,Y_context))
    return [catX, catY]

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
        
#modified sentence2vec function to return lists of word vectors
def sentences2vecs(sentences, model, unseen=None):
    """Generate vectors for each word in a sentence sentence (list) in a list of sentences.
    
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
    list of arrays, each sentence -> array of word vectors
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
    mol = Chem.MolFromSmiles(smiles)
    sentence = mol2alt_sentence(mol,1)
    return sentence

#return dataset of vector sentences for each solute solvent pair
def delfos_data(sol_smiles, solv_smiles):
    data = []
    size = len(sol_smiles)
    for mols in [sol_smiles,solv_smiles]:
        sentences = [sentence_generator(x) for x in mols]
        vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
        data.append(vecs)
    pairs = [(data[0][i],data[1][i]) for i in range(size)]
    return pairs

def delfos_data_1(smiles):
    sentences = [sentence_generator(x) for x in smiles]
    vecs = sentences2vecs(sentences, mol2vec_model, unseen='UNK')
    return vecs 