# Consistency Policy

[[Project page]](TODO)
[[Paper]](TODO)
[[Data]](TODO)


[Aaditya Prasad](https://www.linkedin.com/in/aaditya-prasad/)<sup>1</sup>,
[Kevin Lin](https://kevin-thankyou-lin.github.io/)<sup>1</sup>,
[Linqi Zhou](https://alexzhou907.github.io/)<sup>1</sup>,
[Jeannette Bohg](https://web.stanford.edu/~bohg/)<sup>1</sup>,


<sup>1</sup>Stanford University

<img src="media/teaser.png" alt="drawing" width="100%"/>
<img src="media/multimodal_sim.png" alt="drawing" width="100%"/>

# Basic Usage

## Installation
The below instructions are copied from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), though our conda_environment.yaml is different than theirs. 

To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```

## Training
Training is done similarly to [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). The user defines a config yaml file with their desired parameters and then runs the following command:
```console
(consistency-policy)[Consistency_Policy]$ python train.py --config-dir=configs/ --config-name=edm_square.yaml
```
Example configs for the robomimic square task are provided in the configs/ directory. Three types of networks are supported in this implementation and are referred to by their diffusion framework: EDM, CTMP, DDIM. 

To train a policy, pick a desired framework and update ```task```, ```dataset```, and ```shape_meta keys```, as well as any other hyperparamaters. If you are training without an online sim, set ```training.online_rollouts```=false. You should always set training.inference_mode=false while training. 

Below are specific instructions for the different networks. 

### Teacher Network (EDM)
Before distilling a few-step Consistency Policy, a teacher network needs to be trained. The implemented teacher network follows the [EDM](https://arxiv.org/abs/2206.00364) diffusion framework. ```policy.noise_scheduler``` holds the key introduced hyperparamaters; this controls the noise scheduler and diffusion process. If you have not worked with diffusion models before, ```policy.noise_scheduler.bins```, which is 80 in the example configs, is the main paramater you might wish to change. This is the number of discretization steps in the backwards diffusion process. This number does not affect training speed (and can be changed after/during training). More bins leads to more accurate policy inference (which will show up in mse_error and eval scores) and longer inference times. 

### Student Network (CTMP)
Once you have a trained teacher checkpoint, you are ready to distill a Consistency Policy. Set ```policy.teacher_path``` in the config to the desired ckpt path. It is heavily recommended to warm-start your CP with the teacher checkpoint, which only requires setting ```policy.edm``` to the same path as ```policy.teacher_path```. 

Your student and teacher must have the same ```policy.diffusion_step_embed_dim```! Ensure that these are the same and check your EDM cfg for the correct value if you are not sure.

As with the teacher network, ```policy.noise_scheduler``` contains most of the specialized hyperparamaters. Increasing the number of bins increases training convergence time as well as the accuracy of the converged model, though both of these effects vary in size. Additionally, ```policy.losses``` lists the objectives that are taken into account (```dsm``` and ```ctm```) as well as their multipliers: these correspond to $\alpha$ and $\beta$ in Eq. 8 of the paper and allow you to adjust the relative weighting of the losses. 

By default, you are training and evaluating a single-step network. For multi-step inference, see the Deploying section below.  

### Baseline Network (DDiM)
The baseline network is largely the same as in Diffusion Policy's implementation, and uses the Hugging Face DDiM noise scheduler. ```policy.num_inference``` steps plays a similar role to the teacher network's ```policy.noise_scheduler.bins```. The number of inference steps can be changed at test time and increases both accuracy and inference time. The baseline network cannot be used for distilation but can be useful to check your setup with, since it doesn't require the training of both a teacher and student network. 

## Deploying

Once you have trained a policy, you can use the ```get_policy``` function in ```consistency_policy.utils``` to load an inference-ready version of the policy from a checkpoint. If you wish to change any of the test-time hyperparamaters of the policy, you can pass in a new config file with your desired changes. By default, ```get_policy``` loads the config that the model was trained with, activates inference mode, and deactivates online rollouts. 

We also include a ```PolicyWrapper``` that wraps a provided policy with action and observation chunking. ```example.ipynb``` shows an example of loading a policy, wrapping it, and generating new actions. 

As mentioned earlier, a Consistency Policy can complete multi-step inference at test time. Before chaining is enabled, you must define the timesteps that you wish to chain at under ```policy.chaining_times```. We found that even partitions of discretized time work well as a heurstic: thus, our default setting is ```policy.chaining_times = ['D',27,54]``` for three-step inference that chains from 0, 27, and 54 bins. Once you have set this paramater, you must call ```policy.enable_chaining()``` (the ```PolicyWrapper``` supports this method as well). More details and an explanation of chaining can be found in our paper. 

## 🧾 Checkout our experiment logs!

## 🛠️ Installation

### 🦾 Real Robot

## 🖥️ Reproducing Simulation Benchmark Results 
### Download Training Data
Under the repo root, create data subdirectory:
```console
[diffusion_policy]$ mkdir data && cd data
```

Download the corresponding zip file from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/)
```console
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```

Extract training data:
```console
[data]$ unzip pusht.zip && rm -f pusht.zip && cd ..
```

Grab config file for the corresponding experiment:
```console
[diffusion_policy]$ wget -O image_pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
```

### Running for a single seed
Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).
```console
[diffusion_policy]$ conda activate robodiff
(robodiff)[diffusion_policy]$ wandb login
```

Launch training with seed 42 on GPU 0.
```console
(robodiff)[diffusion_policy]$ python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```



### 🆕 Evaluate Pre-trained Checkpoints
TODO

## 🦾 Demo, Training and Eval on a Real Robot
TODO

## 🔩 Key Components
TODO


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## 🙏 Acknowledgement
TODO
