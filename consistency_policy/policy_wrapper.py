
import torch 
import numpy as np
from collections import deque
from consistency_policy.utils import euler_to_quat, rot6d_to_rmat, rmat_to_euler
from torchvision import transforms as T
from diffusion_policy.common.pytorch_util import dict_apply


class PolicyWrapper:
    def __init__(self, policy, n_obs, n_acts, d_pos, d_rot, cfg=None, device="cpu"):
        self.policy = policy

        self.obs_chunker = ObsChunker(n_obs_steps=n_obs)
        self.obs_chunker.reset()

        self.action_chunker = ActionChunker(n_act_steps=n_acts)
        self.action_chunker.reset()

        self.cfg = cfg
        self.device = device

        self.c_obs = 0
        self.c_acts = 0

        self.d_pos = d_pos
        self.d_rot = d_rot

        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.to(self.device)),
            # add an unsqueeze to make it a batch of 1
            T.Lambda(lambda x: x.unsqueeze(0)) #.unsqueeze(0))
        ])


    def get_action(self, observation):
        action = self.action_chunker.get_action()
        if action is None:
            #we need to calculate actions

            # TODO: load this from shape_meta cfg rather than hardcoding
            obs_dict = {
                "base_pose": observation["base_pose"],
                "arm_pos": observation["arm_pos"],
                "arm_quat": observation["arm_quat"],
                "gripper_pos": observation["gripper_pos"],
                "base_image": observation["base_image"],
                "wrist_image": observation["wrist_image"],
            }

            # transform image data
            for key in obs_dict:
                if "image" in key:
                    obs_dict[key] = self.transform(obs_dict[key])
                else:
                    # add an unsqueeze to make it a batch of 1
                    obs_dict[key] = torch.tensor(obs_dict[key]).to(self.device).unsqueeze(0) #.unsqueeze(0)


            assert not torch.any(obs_dict['arm_quat'][0] < 0), 'quaternion with negative value on first entry found' \
                                                            'policy learning assumes non-negative quat representation'

            # convert all values in obs_dict to numpy
            obs_dict = dict_apply(obs_dict, lambda x: x.cpu().numpy())
            self.obs_chunker.add_obs(obs_dict)
            obs = self.obs_chunker.get_obs_history()


            obs_dict_torch = dict_apply(obs,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )

            result = self.policy.predict_action(obs_dict_torch)

            # Hardcoded action shapes
            actions = result["action"]
            pos = actions[..., :self.d_pos].cpu().numpy()
            gripper = actions[..., [-1]].cpu().numpy()

            rot = actions[..., self.d_pos: self.d_pos + self.d_rot]
            # rot = rot6d_to_rmat(rot)
            rot = rot[0].cpu().numpy()
            # rot = rmat_to_euler(rot)
            rot = rot[None, :]

            # action passed to env.step(action) is a numpy array
            action = np.concatenate([pos, rot, gripper], axis=-1)[0]
            self.action_chunker.add_action(action)
            action = self.action_chunker.get_action()

        return action


    def reset(self):
        self.obs_chunker.reset()
        self.action_chunker.reset()

    def enable_chaining(self):
        if hasattr(self.policy, "enable_chaining"):
            self.policy.enable_chaining()
        else:
            raise NotImplementedError("Chosen policy does not support chaining.")

class ActionChunker:
    """Wrapper for chunking actions. Takes in an action sequence; returns one action when queried.
    Returns None if already popped out all actions.
    """

    def __init__(self, n_act_steps):
        """
        Args:
            n_act_steps (int): number of actions to buffer before requiring a new action sequence to be added.
        """
        self.n_act_steps = n_act_steps
        self.actions = deque()

    def reset(self):
        self.action_history = None

    def add_action(self, action):
        """Add a sequence of actions to the chunker.

        Args:
            action (np.ndarray): An array of actions, shape (N, action_dim).
        """
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array.")
        if len(action.shape) != 2:
            raise ValueError("Action array must have shape (N, action_dim).")

        # slice the actions into chunks of size n_act_steps
        action = action[:self.n_act_steps]

        # Extend the deque with the new actions
        self.actions.extend(action)

    def get_action(self):
        """Get the next action from the chunker.

        Returns:
            np.ndarray or None: The next action, or None if no actions are left.
        """
        if self.actions:
            return self.actions.popleft()
        else:
            return None


class ObsChunker:
    """
    Wrapper for chunking observations. Builds up a buffer of n_obs_steps observations and releases them all at once.
    """
    def __init__(self, n_obs_steps):
        """
        Args:
            n_obs_steps (int): number of observations to buffer before releasing them all at once.
        """
        self.n_obs_steps = n_obs_steps
        self.obs_history = None

    def reset(self):
        self.obs_history = None

    def add_obs(self, obs):
        if self.obs_history is None:
            self.obs_history = {}
            for k in obs:
                self.obs_history[k] = deque(maxlen=self.n_obs_steps)
        for k in obs:
            self.obs_history[k].append(obs[k])

    def get_obs_history(self):
        current_obs = {k: v[-1] for k, v in self.obs_history.items()} # Get the most recent observation
        while self.obs_history is None or len(next(iter(self.obs_history.values()))) < self.n_obs_steps:
            for k in current_obs:
                # add the current obs to the history
                self.obs_history[k].append(current_obs[k])

        obs_to_return = {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
        return obs_to_return
