from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn
import dill
import hydra
from omegaconf import OmegaConf
from consistency_policy.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
import re
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as pt
import numpy as np

NORMALIZER_PREFIX_LENGTH = 11
MODEL_PREFIX_LENGTH = 6

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

def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()

def rot6d_to_rmat(rot_6d: torch.Tensor) -> torch.Tensor:
    return pt.rotation_6d_to_matrix(rot_6d)

def rmat_to_euler(rot_mat: np.ndarray, degrees=False) -> np.ndarray:
    if isinstance(rot_mat, torch.Tensor):
        rot_mat = rot_mat.cpu().numpy()
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat

def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[MODEL_PREFIX_LENGTH:]  
            new_state_dict[new_k] = v

    return new_state_dict

def load_normalizer(workspace_state_dict):
    keys = workspace_state_dict['state_dicts']['model'].keys()
    normalizer_keys = [key for key in keys if 'normalizer' in key]
    normalizer_dict = {key[NORMALIZER_PREFIX_LENGTH:]: workspace_state_dict['state_dicts']['model'][key] for key in normalizer_keys}

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_dict)

    return normalizer

def get_policy(ckpt_path, cfg = None, dataset_path = None):
    """
    Returns loaded policy from checkpoint
    If cfg is None, the ckpt's saved cfg will be used
    """
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg'] if cfg is None else cfg

    cfg.training.inference_mode = True
    cfg.training.online_rollouts = False

    if dataset_path is not None:
        cfg.task.dataset.dataset_path = dataset_path
        cfg.task.envrunner.dataset_path = dataset_path

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_checkpoint(path=ckpt_path, exclude_keys=['optimizer'])
    workspace_state_dict = torch.load(ckpt_path)
    normalizer = load_normalizer(workspace_state_dict)

    policy = workspace.model
    policy.set_normalizer(normalizer)

    return policy

def get_cfg(ckpt_path):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    return cfg
