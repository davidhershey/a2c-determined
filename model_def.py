from typing import Any, Dict, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical

from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, LRScheduler
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from model import init, FeatureEncoderNet
from storage import RolloutStorage

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

from torch.utils.tensorboard import SummaryWriter


class A2CTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        # self.logger = TorchWriter()
        self.n_stack = self.context.get_hparam("n_stack")
        self.env_name = self.context.get_hparam("env_name")
        self.num_envs = self.context.get_hparam("num_envs")
        self.rollout_size = self.context.get_hparam("rollout_size")
        self.curiousity = self.context.get_hparam("curiousity")
        self.lr = self.context.get_hparam("lr")
        self.icm_beta = self.context.get_hparam("icm_beta")
        self.value_coeff = self.context.get_hparam("value_coeff")
        self.entropy_coeff = self.context.get_hparam("entropy_coeff")
        self.max_grad_norm = self.context.get_hparam("max_grad_norm")

        env = make_atari_env(self.env_name, num_env=self.num_envs, seed=42)
        self.env = VecFrameStack(env, n_stack=self.n_stack)
        eval_env = make_atari_env(self.env_name, num_env=1, seed=42)
        self.eval_env = VecFrameStack(eval_env, n_stack=self.n_stack)

        # constants
        self.in_size = self.context.get_hparam("in_size")  # in_size
        self.num_actions = env.action_space.n

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.feat_enc_net = self.context.Model(FeatureEncoderNet(self.n_stack, self.in_size))
        self.actor = self.context.Model(init_(nn.Linear(self.feat_enc_net.hidden_size, self.num_actions)))
        self.critic = self.context.Model(init_(nn.Linear(self.feat_enc_net.hidden_size, 1)))
        self.set_recurrent_buffers(self.num_envs)

        params = list(self.feat_enc_net.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = self.context.Optimizer(torch.optim.Adam(params, self.lr))

        self.is_cuda = torch.cuda.is_available()
        self.storage = RolloutStorage(self.rollout_size, self.num_envs, self.env.observation_space.shape[0:-1],
                                      self.n_stack, is_cuda=self.is_cuda, value_coeff=self.value_coeff,
                                      entropy_coeff=self.entropy_coeff)

        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))

        self.writer = SummaryWriter(log_dir="/tmp/tensorboard")
        self.global_eval_count = 0

    def set_recurrent_buffers(self, buf_size):
        self.feat_enc_net.reset_lstm(buf_size=buf_size)

    def reset_recurrent_buffers(self, reset_indices):
        self.feat_enc_net.reset_lstm(reset_indices=reset_indices)

    def build_training_data_loader(self) -> DataLoader:
        ds = torchvision.datasets.MNIST(
            self.download_directory,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # These are the precomputed mean and standard deviation of the
                    # MNIST data; this normalizes the data to have zero mean and unit
                    # standard deviation.
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]),
            download=True
        )
        return DataLoader(ds, batch_size=1)

    def build_validation_data_loader(self) -> DataLoader:
        ds = torchvision.datasets.MNIST(
            self.download_directory,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # These are the precomputed mean and standard deviation of the
                    # MNIST data; this normalizes the data to have zero mean and unit
                    # standard deviation.
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]),
            download=True
        )
        return DataLoader(ds, batch_size=1)

    def train_batch(
        self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        final_value, entropy = self.episode_rollout()
        self.opt.zero_grad()
        total_loss, value_loss, policy_loss, entropy_loss = self.storage.a2c_loss(final_value, entropy)
        self.context.backward(total_loss)

        def clip_grads(parameters):
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        self.context.step_optimizer(self.opt, clip_grads)
        self.storage.after_update()

        return {
            'loss': total_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss
        }

    def get_action(self, state, deterministic=False):
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        policy = self.actor(feature)
        value = torch.squeeze(self.critic(feature))

        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        if not deterministic:
            action = cat.sample()
            return (action, cat.log_prob(action), cat.entropy().mean(), value, feature)
        else:
            action = np.argmax(action_prob.detach().cpu().numpy(), axis=1)
            return (action, [], [], value, feature)

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.rollout_size):
            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy, value, a2c_features = self.get_action(self.storage.get_state(step))
            # accumulate episode entropy
            episode_entropy += entropy

            # interact
            obs, rewards, dones, infos = self.env.step(a_t.cpu().numpy())

            # save episode reward
            self.storage.log_episode_rewards(infos)

            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones)
            self.reset_recurrent_buffers(reset_indices=dones)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            state = self.storage.get_state(step + 1)
            final_features = self.feat_enc_net(state)
            final_value = torch.squeeze(self.critic(final_features))

        return final_value, episode_entropy

    def evaluate_full_dataset(self, data_loader, model) -> Dict[str, Any]:
        self.global_eval_count += 1
        episode_rewards, episode_lengths = [], []
        n_eval_episodes = 10
        self.set_recurrent_buffers(1)
        frames = []
        with torch.no_grad():
            for episode in range(n_eval_episodes):
                obs = self.eval_env.reset()
                done, state = False, None
                episode_reward = 0.0
                episode_length = 0
                while not done:
                    state = self.storage.obs2tensor(obs)
                    if episode == 0:
                        frame = torch.unsqueeze(torch.squeeze(state)[0], 0).detach()
                        frames.append(frame)
                    action, _, _, _, _ = self.get_action(state, deterministic=True)
                    obs, reward, done, _info = self.eval_env.step(action)
                    reward = reward[0]
                    done = done[0]
                    episode_reward += reward
                    episode_length += 1
                if episode == 0:
                    video = torch.unsqueeze(torch.stack(frames), 0)
                    self.writer.add_video('policy', video, global_step=self.global_eval_count, fps=20)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        self.set_recurrent_buffers(self.num_envs)
        return {'mean_reward': mean_reward}
