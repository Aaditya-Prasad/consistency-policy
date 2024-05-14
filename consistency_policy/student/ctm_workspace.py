if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
import math
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import sklearn
from contextlib import contextmanager
import time

from consistency_policy.student.ctm_policy import CTMPPUnetHybridImagePolicy
from consistency_policy.base_workspace import BaseWorkspace
from consistency_policy.utils import load_normalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)




class CTMWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.policy.inference_mode = cfg.training.inference_mode
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: CTMPPUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        if cfg.training.debug:
            self.model.debug = True

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.p_epochs = cfg.training.p_epochs

    def run(self):

        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            self.model.noise_scheduler.ode_steps_max = 1
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.training.output_dir = "outputs/temp"

            cfg.task.env_runner.n_envs = 1
            cfg.task.env_runner.n_test = 1
            cfg.task.env_runner.n_train = 1
            cfg.task.env_runner.n_test_vis = 1
            cfg.task.env_runner.n_train_vis = 1
            
        if cfg.policy.edm != "None" and cfg.training.inference_mode == False:
            print(f"Warm starting from {cfg.policy.edm}")
            self.load_checkpoint(path=cfg.policy.edm, exclude_keys=['ema_model', 'optimizer', 'epoch', 'global_step', '_output_dir'], 
                                 update_dict_dim=cfg.policy.diffusion_step_embed_dim, strict=False)
            self.model.obs_encoder.eval()
            self.model.obs_encoder.requires_grad_(False)
        else:
            print("No warm start provided, assuming inference mode")

        # resume training
        if cfg.training.resume:
            if cfg.training.resume_path != "None":
                print(f"Resuming from checkpoint {cfg.training.resume_path}")
                self.load_checkpoint(path=cfg.training.resume_path, exclude_keys=['optimizer'])
                workspace_state_dict = torch.load(cfg.training.resume_path)
                normalizer = load_normalizer(workspace_state_dict)
                self.model.set_normalizer(normalizer)

            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file() and cfg.training.resume_path == "None":
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, exclude_keys=['optimizer'])
                workspace_state_dict = torch.load(lastest_ckpt_path)
                normalizer = load_normalizer(workspace_state_dict)
                self.model.set_normalizer(normalizer)
        

        print("EPOCH", self.epoch)

        if not cfg.training.inference_mode:
            # configure dataset
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseImageDataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            steps_per_epoch = len(train_dataloader)
            normalizer = dataset.get_normalizer()

            # configure validation dataset
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            steps_per_val_epoch = len(val_dataloader)

            self.model.set_normalizer(normalizer)   

            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

            self.global_step = 0
            self.epoch = 0
            
            # configure lr scheduler
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    steps_per_epoch * self.p_epochs) \
                        // cfg.training.gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
            )

        # configure env
        self.output_dir = cfg.training.output_dir
        if cfg.training.online_rollouts:
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, 
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        wandb.run.log_code(".")
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )


        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        if not cfg.training.inference_mode:
            optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None



        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        training_sample_every = cfg.training.sample_every
        val_sample_every = cfg.training.val_sample_every

        if cfg.training.inference_mode:
            self.model.drop_teacher()
        
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                if not cfg.training.inference_mode:
                    # ========= train for this epoch ==========
                    train_losses = list()
                    ctm_losses = list()
                    dsm_losses = list()

                    steps_per_epoch = len(train_dataloader)
                    stepping_batches = math.ceil(steps_per_epoch / cfg.training.gradient_accumulate_every) * self.p_epochs # 160 IS ESTIMATED NUMBER OF EPOCHS

                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # compute loss
                            raw_loss = self.model.compute_loss(batch)

                            loss = 0
                            loss_logs = dict()
                            for k, v in raw_loss.items():
                                loss_logs[k] = v.item()
                                loss += v
                                
                                if k == 'ctm':
                                    ctm_losses.append(v.item())
                                elif k == 'dsm':
                                    dsm_losses.append(v.item())

                            nloss = loss / cfg.training.gradient_accumulate_every
                            nloss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                                self.model.ema_update()
                            


                            # logging
                            raw_loss_cpu = nloss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0],
                            }


                            step_log.update(loss_logs)


                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                    if len(ctm_losses) > 0:
                        ctm_loss = np.mean(ctm_losses)
                        step_log['ctm'] = ctm_loss

                    if len(dsm_losses) > 0:
                        dsm_loss = np.mean(dsm_losses)
                        step_log['dsm'] = dsm_loss


                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy.use_ema = True
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and cfg.training.online_rollouts:
                    policy.chaining_steps = cfg.training.val_chaining_steps
                    runner_log = env_runner.run(policy)
                    policy.chaining_steps = 1
                    # log all
                    step_log.update(runner_log)

                if not cfg.training.inference_mode:
                    # run validation
                    t_time = 0
                    count = 0
                    if (self.epoch % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            val_mse_error = list()
                            val_mse_teacher_error = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = sum(self.model.compute_loss(batch).values())
                                    val_losses.append(loss)
                                    
                                    if (self.epoch % val_sample_every) == 0:
                                        obs_dict = batch['obs']
                                        gt_action = batch['action']

                                        policy.chaining_steps = cfg.training.val_chaining_steps
                                        start_time = time.time()
                                        result = policy.predict_action(obs_dict)
                                        t = time.time() - start_time
                                        policy.chaining_steps = 1

                                        result_teacher = policy.predict_action_teacher(obs_dict)

                                        t_time += t
                                        count += 1

                                        pred_action = result['action_pred']
                                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)

                                        pred_action_teacher = result_teacher['action_pred']
                                        mse_teacher = torch.nn.functional.mse_loss(pred_action_teacher, gt_action)


                                        val_mse_teacher_error.append(mse_teacher.item())
                                        val_mse_error.append(mse.item())

                                        del pred_action_teacher
                                        del result_teacher
                                        del mse_teacher 
                                        del obs_dict
                                        del gt_action
                                        del result
                                        del pred_action
                                        del mse

                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                step_log['val_loss'] = val_loss
                                
                            if len(val_mse_error) > 0:
                                val_mse_error = torch.mean(torch.tensor(val_mse_error)).item()
                                step_log['val_mse_error'] = val_mse_error

                                val_avg_inference_time = t_time / count
                                step_log['val_avg_inference_time'] = val_avg_inference_time

                                val_mse_teacher_error = torch.mean(torch.tensor(val_mse_teacher_error)).item()
                                step_log['val_mse_teacher_error'] = val_mse_teacher_error
                                
                    # run diffusion sampling on a training batch
                    t_time = 0
                    count = 0
                    if (self.epoch % training_sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            
                            
                            start_time = time.time()
                            result = policy.predict_action(obs_dict)
                            t = time.time() - start_time
                            
                            t_time += t
                            count += 1
                            

                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            step_log['train_avg_inference_time'] = t_time / count

                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = CTMWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
