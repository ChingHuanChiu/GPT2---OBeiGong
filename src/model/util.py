from collections import OrderedDict

import torch
import dill




def load_model_from_checkpoint(device: str, model: torch.nn.Module, model_ckpt: str, is_ddp_model: bool, optimizer = None):
    """reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    ckpt = torch.load(model_ckpt, map_location=device, pickle_module=dill)
    state_dict = ckpt['model_state_dict']
    
    if is_ddp_model:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
                # remove 'module.' of DataParallel/DistributedDataParallel
                name = k[7:] 
                new_state_dict[name] = v
    else:
        new_state_dict = state_dict

        
    model.load_state_dict(new_state_dict)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
        
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return model, optimizer
    
    return model









