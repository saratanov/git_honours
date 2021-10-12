import torch
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