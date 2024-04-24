from typing import Dict
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import numpy as np
import random
from sklearn.neighbors import KernelDensity

from consistency_policy.ctm_unet import CTMConditionalUnet1D
from consistency_policy.diffusion import CTM_Scheduler, Huber_Loss
from consistency_policy.utils import state_dict_to_model
from consistency_policy.diffusion_unet_with_dropout import ConditionalUnet1D

class CTMPPUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: CTM_Scheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=128,
            down_dims=(256,512,1024),
            dropout_rate=.0,
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            initial_ema_decay: float = 0.9,
            delta = .0,
            special_skip = True,
            #teacher
            teacher_path = None,
            #KDE
            use_kde = False, 
            kde_samples = 0,
            #warm start
            edm = None,
            #CTM
            losses = None,
            dsm_weights = "none",
            ctm_sampler = "ctm",
            #chaining args
            chaining_times = ['D', 27, 54],
            inference_mode = False,
            ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps



        model = CTMConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            dropout_rate=dropout_rate,
        )

        self.obs_encoder = obs_encoder
        self.model = model # Warm starting is done in the workspace

        self.model_ema = copy.deepcopy(model)
        self.model_ema.requires_grad_(False)
        self.use_ema = False

        self.model.prepare_drop_generators()
        self.model_ema.prepare_drop_generators()

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.noise_scheduler = noise_scheduler

        #extra args
        self.ema_decay = initial_ema_decay
        self.delta = delta
        self.special_skip = special_skip
        self.use_kde = use_kde
        self.kde_samples = kde_samples

        teacher = ConditionalUnet1D(
                    input_dim=input_dim,
                    local_cond_dim=None,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                )   
        
        if inference_mode == True:
            print("You should be doing inference only!")
        else:
            state_dict = state_dict_to_model(torch.load(teacher_path))
            teacher.load_state_dict(state_dict)
            teacher.eval()
            teacher.requires_grad_(False)
            print("Using teacher: ", teacher_path)
        self.teacher = teacher
        

        self.chaining_steps = 1
        self.debug = False

        self.losses = {}
        for loss, weight in zip(losses[0], losses[1]):
            self.losses[loss] = weight
        
        self.dsm_weights = dsm_weights
        self.ctm_sampler = ctm_sampler

        self.chaining_times = chaining_times
        self.chain = False #DEFAULT is False, you have to enable this yourself when you want it

        print("Using losses: ", self.losses)


        print("CM params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Teacher params: %e" % sum(p.numel() for p in self.teacher.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    def drop_teacher(self):
        # When we are in inference mode, we have to load the teacher in the first place for the state dict to resolve
        # but we don't want to actually keep it around during inference
        # TODO: we shouldn't load the teacher in the first place
        self.teacher = None
    
    # ========= forward  ============
    def _forward(self, model,
            sample: torch.Tensor, 
            timestep: torch.Tensor,
            stop_time: torch.Tensor,
            local_cond=None, global_cond=None, clamp=False):
        
        denoise = lambda x, t, s: model(x, t, s, local_cond=local_cond, global_cond=global_cond)
        return self.noise_scheduler.CTM_calc_out(denoise, sample, timestep, stop_time, clamp=clamp)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            ):

        trajectory = self.noise_scheduler.sample_inital_position(condition_data, generator=generator)
        
        t = torch.tensor([self.noise_scheduler.time_max], device = condition_data.device)
        s = torch.tensor([self.noise_scheduler.time_min], device = condition_data.device)

        # 1. apply conditioning
        trajectory[condition_mask] = condition_data[condition_mask]

        # 2. predict model output, WHICH IS NOW THE ACTUAL PREDICTION
        out = self._forward(self.model,
                            trajectory, t, s, local_cond=local_cond, 
                            global_cond=global_cond, clamp=True) #clamp at inference time

        # finally make sure conditioning is enforced
        out[condition_mask] = condition_data[condition_mask]


        if self.chain == False:
            return out
        
        for t in self.chaining_times[1:]:
            t = torch.tensor([float(t)], device = condition_data.device)
            if self.chaining_times[0] == "C":
                t = self.noise_scheduler.timesteps_to_times(t)
            s = torch.tensor([self.noise_scheduler.time_min], device = condition_data.device)

            trajectory = self.noise_scheduler.add_noise(out, t)
            # trajectory = self.noise_scheduler.trajectory_time_product(out, t)

            out = self._forward(self.model, trajectory, t, s, 
                                    local_cond=local_cond, global_cond=global_cond, clamp=True)

        return out


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps


        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        if self.use_kde and B == 1:
            cond_data = cond_data.repeat(self.kde_samples, 1, 1)
            cond_mask = cond_mask.repeat(self.kde_samples, 1, 1)
            global_cond = global_cond.repeat(self.kde_samples, 1)
            nsample = self.conditional_sample(
                cond_data,
                cond_mask, 
                local_cond=local_cond,
                global_cond=global_cond,)
            
            nsample = nsample.reshape(self.kde_samples, B, T, -1)
            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred).cpu().numpy()

            action_pred = action_pred.reshape(self.kde_samples, -1)
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(action_pred)


            log_dens = kde.score_samples(action_pred)
            idx = np.argmax(log_dens)
            action_pred = action_pred[idx][None,...]

            action_pred = action_pred.reshape(B, T, -1)
            action_pred = torch.tensor(action_pred, device=device, dtype=dtype)

            

        else:
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,)
            
            # unnormalize prediction
            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        total_loss = {}

        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]


        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        if "ctm" in self.losses.keys():

        
            #t, s, and u are all given as absolute bins
            t, s, u = self.noise_scheduler.sample_times(trajectory, time_sampler=self.ctm_sampler)
            times = self.noise_scheduler.timesteps_to_times(t)
            stops = self.noise_scheduler.timesteps_to_times(s)
            u_times = self.noise_scheduler.timesteps_to_times(u)

            ###### NOT IMPLEMENTED YET
            # weights = self.noise_scheduler.get_weights(t, s, u, "ctm")
            weights = None
            
            noise_traj = self.noise_scheduler.add_noise(trajectory, times)

            denoise = lambda x, t: self.teacher(x, t, local_cond=local_cond, global_cond=global_cond)
            u_noise_traj = noise_traj
            #like this doesn't work b/c its a batch dimension but also not having this vectorized is going to make it soo slow
            distances = u - t
            max_d = torch.max(distances)

            # TODO: shape error when we use max_d, doesn't matter when we have small ode max steps
            for d in range(self.noise_scheduler.ode_steps_max):
                ct = torch.stack([(t_i + d).clamp(int(t_i.item()), int(u_i.item())) for t_i, u_i in zip(t, u)])
                nt = torch.stack([(t_i + d + 1).clamp(int(t_i.item()), int(u_i.item())) for t_i, u_i in zip(t, u)])

                current_times = self.noise_scheduler.timesteps_to_times(ct)
                next_times = self.noise_scheduler.timesteps_to_times(nt)

                u_noise_traj = self.noise_scheduler.step(denoise, u_noise_traj, current_times, next_times, clamp=False)


            ### current times is > next times! this means the ema model runs on next times
            
            # t -> s
            pred = self._forward(self.model, noise_traj, times, stops, 
                                    local_cond=local_cond, global_cond=global_cond)

            # u -> s
            with torch.no_grad():
                target = self._forward(self.model_ema, u_noise_traj, u_times, stops,
                                    local_cond=local_cond, global_cond=global_cond)

            # now we take both back to 0
            with torch.no_grad():
                start = torch.tensor([self.noise_scheduler.time_min], device = trajectory.device).expand(times.shape)

                pred = self._forward(self.model_ema, pred, stops, start, 
                                    local_cond=local_cond, global_cond=global_cond)
                
                target = self._forward(self.model_ema, target, stops, start,
                                    local_cond=local_cond, global_cond=global_cond)


            loss = Huber_Loss(pred, target, delta = self.delta, weights=weights)

            total_loss["ctm"] = loss * self.losses["ctm"]

        if "dsm" in self.losses.keys():
            times, _ = self.noise_scheduler.sample_times(trajectory, time_sampler='ctm_dsm')
            weights = self.noise_scheduler.get_weights(times, None, self.dsm_weights)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, times)
            
            # Predict the initial state
            stop = torch.tensor([self.noise_scheduler.time_min], device = trajectory.device).expand(times.shape)
            pred = self._forward(self.model, noisy_trajectory, times, stop,
                                    local_cond=local_cond, global_cond=global_cond, clamp=False)
            
            target = trajectory



            loss = Huber_Loss(pred, target, delta = self.delta, weights=weights)

            total_loss["dsm"] = loss * self.losses["dsm"]


        return total_loss

    # ========= consistency_utils  ============

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        ema_decay = self.ema_decay

        torch._foreach_mul_(param_ema, ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - ema_decay)

    def enable_chaining(self):
        if self.chaining_times is not None or self.chaining_times == "None":
            self.chain = True
            print("Chaining enabled with times: ", self.chaining_times)

        else:
            raise ValueError("Chaining times not set")
    
    def disable_chaining(self):
        self.chain = False


    # ========= testing teacher ==========
    
    def conditional_sample_teacher(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            ):

        model = self.teacher
        scheduler = self.noise_scheduler

        trajectory = scheduler.sample_inital_position(condition_data, generator=generator)
    
        timesteps = torch.arange(0, self.noise_scheduler.bins, device=condition_data.device)
        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            trajectory[condition_mask] = condition_data[condition_mask]

            t = scheduler.timesteps_to_times(b)
            next_t = scheduler.timesteps_to_times(next_b)

            denoise = lambda traj, t: model(traj, t, local_cond=local_cond, global_cond=global_cond)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(denoise, trajectory, t, next_t)
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]      

        return trajectory


    @torch.no_grad()
    def predict_action_teacher(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample_teacher(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        start = To - 1
        end = start + self.n_action_steps

        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action

        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
        
