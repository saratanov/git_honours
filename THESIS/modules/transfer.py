from collections import defaultdict as ddict, OrderedDict as odict
import torch
from .fit import fit

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