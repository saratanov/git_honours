from collections import defaultdict as ddict, OrderedDict as odict
import torch
from .fit import fit, train, test
from .data import pka_scaler

def transfer_weights(model, chk_name):
    #double input
    if 'Gsolv' in chk_name:
        #load state
        model.model.load_state_dict(torch.load('trained/'+chk_name,map_location=torch.device('cpu')))
        #freeze all parameters except NN
        for name, param in model.model.named_parameters():
            if 'ffn' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    #single input
    else:
        #load state
        pre_dict = torch.load('trained/'+chk_name,map_location=torch.device('cpu'))
        new_dict = odict()
        #load parameters
        for name in pre_dict.keys():
            if 'biLSTM' in name:
                new_name = name.replace('biLSTM','biLSTM_X')
                new_dict[new_name] = pre_dict[name]
            elif 'encoder' in name:
                new_name = name.replace('encoder','encoder_sol')
                new_dict[new_name] = pre_dict[name]
        model_dict = model.model.state_dict()
        model_dict.update(new_dict)
        model.model.load_state_dict(model_dict)
        #freeze solute encoder
        for name, param in model.model.named_parameters():
            if 'encoder_sol' in name:
                param.requires_grad = False
            if 'biLSTM_X' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
def finetune(model, data, test_ids, exp_name, new_lr):
    #unfreeze all weights
    model.model = model.experiments[exp_name]['model']
    for param in model.model.parameters():
        param.requires_grad = True
        
    #adjust learning rate
    old_lr = model.lr
    model.lr = new_lr
    
    #retrain
    exp_name = exp_name+' finetuned'
    results = fit(model, data, test_ids, exp_name)
    model.lr = old_lr
    return results

def transfer_finetune(model, file, data, test_ids, exp_name, new_lr, train_ids=None):
    #initialise
    old_lr = model.lr
    if train_ids == None:
        train_ids = [i for i in range(len(data[0])) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    #transfer weights
    transfer_weights(model, file)
    #retrain random layers
    model.model = train(model, train_ids, data, scaler)
    #unfreeze weights and finetune with new learning rate
    for param in model.model.parameters():
        param.requires_grad = True
    model.lr = new_lr
    results = fit(model, data, test_ids, exp_name, train_ids)
    #restore old learning rate
    model.lr = old_lr
    return results

def frankenstein(model, data, test_ids, exp_name, new_lr, train_ids=None):
    name = model.name.replace(' ','_')
    #transfer solvent + solute weights
    transfer_weights(model, name+'_Gsolv.pt')
    transfer_weights(model, name+'_Water_pka.pt')
    #freeze encoders
    for name, param in model.model.named_parameters():
        if 'ffn' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    #train ffn
    old_lr = model.lr
    if train_ids == None:
        train_ids = [i for i in range(len(data[0])) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    model.model = train(model, train_ids, data, scaler)
    #unfreeze encoders
    for param in model.model.parameters():
        param.requires_grad = True
    #finetune with new learning rate
    model.lr = new_lr
    results = fit(model, data, test_ids, exp_name, train_ids)
    #restore old learning rate
    model.lr = old_lr
    return results    
    
def frankenstein2(model, data, test_ids, exp_name, new_lr, train_ids=None):
    mname = model.name.replace(' ','_')
    #transfer solvent + solute weights

    pre_dict = torch.load('trained/'+mname+'_Gsolv.pt',map_location=torch.device('cpu'))
    new_dict = odict()
    for name in pre_dict.keys():
        if 'biLSTM' in name:
            new_dict[name] = pre_dict[name]
        elif 'encoder' in name:
            new_dict[name] = pre_dict[name]
    model_dict = model.model.state_dict()
    model_dict.update(new_dict)
    model.model.load_state_dict(model_dict)    

    transfer_weights(model, mname+'_Water_pka.pt')
    #freeze encoders
    for name, param in model.model.named_parameters():
        if 'ffn' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    #train ffn
    old_lr = model.lr
    if train_ids == None:
        train_ids = [i for i in range(len(data[0])) if i not in test_ids]
    scaler = pka_scaler(data[1][train_ids])
    model.model = train(model, train_ids, data, scaler)
    #unfreeze encoders
    for param in model.model.parameters():
        param.requires_grad = True
    #finetune with new learning rate
    model.lr = new_lr
    results = fit(model, data, test_ids, exp_name, train_ids)
    #restore old learning rate
    model.lr = old_lr
    return results    
    