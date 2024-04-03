from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn
import dill
import hydra
from omegaconf import OmegaConf
from consistency_policy.base_workspace import BaseWorkspace

"""Next 2 Utils from the original CM implementation"""

@torch.no_grad()
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def reduce_dims(x, target_dims):
    """Reduces dimensions from the end of a tensor until it has target_dims dimensions."""
    dims_to_reduce = x.ndim - target_dims
    if dims_to_reduce < 0:
         raise ValueError(
             f"input has {x.ndim} dims but target_dims is {target_dims}, which is greater"
         )
    for _ in range(dims_to_reduce):
        x = x.squeeze(-1)
    
    return x


def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[6:]  
            new_state_dict[new_k] = v

    return new_state_dict



def get_policy(ckpt_path, cfg = None):
    """
    Returns loaded policy from checkpoint
    If cfg is None, the ckpt's saved cfg will be used
    """
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg'] if cfg is None else cfg

    cfg.training.inference_mode = True
    cfg.training.online_rollouts = False
    

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    return policy
