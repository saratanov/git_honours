from collections import defaultdict as ddict, OrderedDict as odict
import torch

def transfer_weights(model, chk_name):
    #double input
    if 'Gsolv' in chk_name:
        #load state
        model.model.load_state_dict(torch.load('trained/'+chk_name))
        #freeze all parameters except NN
        for name, param in model.model.named_parameters():
            if 'ffn' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    #single input
    else:
        #load state
        pre_dict = torch.load('trained/'+chk_name)
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