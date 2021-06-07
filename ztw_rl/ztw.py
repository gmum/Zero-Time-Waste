import math as m
import random
import warnings
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Generator, Iterable, Optional, Tuple, Type, Union

import gym
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import (BernoulliDistribution, CategoricalDistribution,
                                                    DiagGaussianDistribution, Distribution,
                                                    MultiCategoricalDistribution, StateDependentNoiseDistribution,
                                                    make_proba_distribution)
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import (get_obs_shape, preprocess_obs)
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped
from torch.utils.data import TensorDataset

from utils import set_lr


def gather_data(model: BaseAlgorithm, env: VecEnv, n: int) -> Dict[str, torch.Tensor]:
    observations = []
    actions = []
    obs = env.reset()
    for _ in range(n):
        observations.append(torch.from_numpy(obs))
        action, _ = model.predict(obs, deterministic=False)
        actions.append(torch.from_numpy(action))
        obs, _, _, _ = env.step(action)
    return {'observations': torch.cat(observations), 'actions': torch.cat(actions)}


def get_bc_data(results_dir: Path,
                model: BaseAlgorithm,
                env: VecEnv,
                timesteps: int,
                neptune_run: neptune.run.Run = None) -> Dict[str, torch.Tensor]:
    save_path = results_dir / 'bc_data.pth'
    if save_path.exists():
        bc_data_dict = torch.load(save_path)
    else:
        bc_data_dict = gather_data(model, env, timesteps)
        torch.save(bc_data_dict, save_path)
        if neptune_run is not None:
            neptune_run['bc_data'].upload(str(save_path))
    return bc_data_dict


class SDNPool(nn.Module):
    def __init__(self, input_channels: int, pool_size: int = 4):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.after_pool_dim = input_channels * pool_size * pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        mixed = mixed.view(mixed.size(0), -1)
        return mixed


class InternalClassifier(nn.Module):
    def __init__(self, input_channels: int, inner_channels: int, output_dim: int, prev_dim: int = 0):
        super().__init__()

    def forward(self, x: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RunningEnsemble(nn.Module):
    def __init__(self, head_index: int, output_dim: int):
        super().__init__()
        self.head_index = head_index
        self.num_heads = self.head_index + 1
        self.output_dim = output_dim
        self.input_dim = (self.head_index + 1) * self.output_dim
        # weights are per-head
        self.weight = nn.Parameter(torch.normal(0, 0.01, size=(self.num_heads, ), requires_grad=True))
        # biases are per-output
        self.bias = nn.Parameter(torch.zeros(size=(self.output_dim, )))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape is [batch_size, head_index, input_dim]
        # x is IC logit outputs
        x = torch.log_softmax(x, dim=-1)
        resized_weight = torch.exp(self.weight.view(1, -1, 1))
        x = x * resized_weight
        return x.mean(1) + self.bias


class EEModel(nn.Module):
    def __init__(self,
                 orig_model: BaseAlgorithm,
                 heads_at: Iterable[int],
                 ic_class: Type[InternalClassifier],
                 mode: str,
                 ic_width_multiplier: int = 1):
        super().__init__()
        self.orig_model = orig_model
        self.device = self.orig_model.device
        self.extractor = self.orig_model.policy.features_extractor
        self.policy = self.orig_model.policy
        self.mlp_extractor = self.policy.mlp_extractor
        self.heads_at = heads_at
        self.stacking = True if mode != 'sdn' else False
        self.ics = nn.ModuleList()
        self.num_actions = self.orig_model.action_space.n
        self.ic_width_multiplier = ic_width_multiplier
        prev_layer, layer_counter, prev_dim = None, 0, 0
        for i, layer in enumerate(self.extractor.cnn):
            if isinstance(layer, torch.nn.ReLU):
                if layer_counter in self.heads_at:
                    self.ics.append(
                        ic_class(prev_layer.out_channels, int(self.ic_width_multiplier * prev_layer.out_channels),
                                 self.num_actions, prev_dim))
                    prev_dim = self.num_actions if self.stacking else 0
                layer_counter += 1
            prev_layer = layer
        self.to(self.device)
        #==================================================================
        print(f'Extractor CNN: {self.extractor.cnn}')
        print(f'Extractor FC: {self.extractor.linear}')
        print(f'MLP extractor shared: {self.mlp_extractor.shared_net}')
        print(f'MLP extractor policy: {self.mlp_extractor.policy_net}')
        print(f'Policy action FC: {self.policy.action_net}')
        if len(self.ics) > 0:
            print(f'IC: {self.ics[-1]}')
        # =================================================================
        # needed to save the model without errors
        self.orig_model = None

    def extractor_forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_counter, prev_output, output_logits = 0, None, []
        for layer in self.extractor.cnn:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                if layer_counter in self.heads_at:
                    logits = self.ics[self.heads_at.index(layer_counter)](x, prev_output)
                    output_logits.append(logits)
                    prev_output = logits if self.stacking else None
                layer_counter += 1
        extracted_features = self.extractor.linear(x)
        return extracted_features, output_logits

    def logits_to_action(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(dist.probs, dim=1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        preprocessed_obs = preprocess_obs(obs,
                                          self.policy.observation_space,
                                          normalize_images=self.policy.normalize_images)
        # print(f'input size: {preprocessed_obs.size()}')
        x, logits_l = self.extractor_forward(preprocessed_obs)
        # TODO optionally break that into seprate layers and add an IC
        x = self.mlp_extractor.shared_net(x)
        x = self.mlp_extractor.policy_net(x)
        x = self.policy.action_net(x)
        logits_l.append(x)
        action_l, log_prob_l = [], []
        for logits in logits_l:
            actions, log_probs = self.logits_to_action(logits, deterministic=deterministic)
            action_l.append(actions)
            log_prob_l.append(log_prob_l)
        return logits_l, action_l, log_prob_l

    def modules(self) -> Iterable[nn.Module]:
        # it is necessary to overwrite this as code in profiler.py depends on the order of items in this iterator
        ord_modules = []
        layer_counter = 0
        for layer in self.extractor.cnn:
            ord_modules.append(layer)
            if isinstance(layer, torch.nn.ReLU):
                if layer_counter in self.heads_at:
                    ord_modules.append(self.ics[self.heads_at.index(layer_counter)])
                layer_counter += 1
        ord_modules.append(self.extractor.linear)
        ord_modules.append(self.mlp_extractor.shared_net)
        ord_modules.append(self.mlp_extractor.policy_net)
        ord_modules.append(self.policy.action_net)
        return chain.from_iterable(m.modules() for m in ord_modules)


class REModel(nn.Module):
    def __init__(self, ee_model: EEModel):
        super().__init__()
        self.ee_model = ee_model
        self.device = self.ee_model.device
        self.extractor = self.ee_model.extractor
        self.policy = self.ee_model.policy
        self.mlp_extractor = self.ee_model.policy.mlp_extractor
        self.heads_at = self.ee_model.heads_at
        self.stacking = self.ee_model.stacking
        self.ics = self.ee_model.ics
        self.ensembles = nn.ModuleList()
        self.num_actions = self.ee_model.num_actions
        for i in range(len(self.ics)):
            self.ensembles.append(RunningEnsemble(i, self.num_actions))
        self.to(self.device)

    def extractor_forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_counter, prev_output, ee_output_logits, re_output_logits = 0, None, [], []
        for layer in self.extractor.cnn:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                if layer_counter in self.heads_at:
                    logits = self.ics[self.heads_at.index(layer_counter)](x, prev_output)
                    ee_output_logits.append(logits)
                    prev_output = logits if self.stacking else None
                    up_to_now = torch.log_softmax(torch.stack(ee_output_logits, dim=1), dim=-1)
                    re_logits = self.ensembles[self.heads_at.index(layer_counter)](up_to_now)
                    re_output_logits.append(re_logits)
                layer_counter += 1
        extracted_features = self.extractor.linear(x)
        return extracted_features, re_output_logits

    def logits_to_action(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ee_model.logits_to_action(logits, deterministic)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        preprocessed_obs = preprocess_obs(obs,
                                          self.policy.observation_space,
                                          normalize_images=self.policy.normalize_images)
        # print(f'input size: {preprocessed_obs.size()}')
        x, logits_l = self.extractor_forward(preprocessed_obs)
        x = self.mlp_extractor.shared_net(x)
        x = self.mlp_extractor.policy_net(x)
        x = self.policy.action_net(x)
        logits_l.append(x)
        action_l, log_prob_l = [], []
        for logits in logits_l:
            actions, log_probs = self.logits_to_action(logits, deterministic=deterministic)
            action_l.append(actions)
            log_prob_l.append(log_prob_l)
        return logits_l, action_l, log_prob_l

    def modules(self) -> Iterable[nn.Module]:
        # it is necessary to overwrite this as code in profiler.py depends on the order of items in this iterator
        ord_modules = []
        layer_counter = 0
        for layer in self.extractor.cnn:
            ord_modules.append(layer)
            if isinstance(layer, torch.nn.ReLU):
                if layer_counter in self.heads_at:
                    ord_modules.append(self.ics[self.heads_at.index(layer_counter)])
                    ord_modules.append(self.ensembles[self.heads_at.index(layer_counter)])
                layer_counter += 1
        ord_modules.append(self.extractor.linear)
        ord_modules.append(self.mlp_extractor.shared_net)
        ord_modules.append(self.mlp_extractor.policy_net)
        ord_modules.append(self.policy.action_net)
        return chain.from_iterable(m.modules() for m in ord_modules)


def train_bc(s: SimpleNamespace, model: EEModel, env: VecEnv, neptune_run: neptune.run.Run = None):
    lamb = s.lamb
    data = data['observations']
    dataset = TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=s.batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=8)
    max_batch = len(data) * s.epochs
    current_batch = 0
    optimizer = s.optimizer_class(model.ics.parameters(), lr=s.ic_lr(current_batch, max_batch))
    for epoch in range(s.epochs):
        for obs in loader:
            logits_l, _, _ = model(obs[0])
            orig_log_probs = torch.log_softmax(logits_l[-1], dim=-1).detach()
            ic_losses = []
            for i, logits in enumerate(logits_l[:-1]):
                ic_log_probs = torch.log_softmax(logits, dim=-1)
                labels = orig_log_probs.argmax(dim=-1)
                ce_loss = F.nll_loss(ic_log_probs, labels)
                kl_loss = F.kl_div(ic_log_probs, orig_log_probs, log_target=True)
                loss = s.lamb * ce_loss + (1 - s.lamb) * kl_loss
                ic_losses.append(loss)
                if neptune_run is not None:
                    neptune_run[f'bc_train/ic_{i}_loss'].log(loss.item(), current_batch)
            set_lr(optimizer, s.ic_lr(current_batch, max_batch))
            optimizer.zero_grad()
            loss = torch.stack(ic_losses).sum()
            loss.backward()
            optimizer.step()
            if neptune_run is not None:
                neptune_run[f'bc_train/ics_loss'].log(loss.item(), current_batch)
            else:
                print(f'batch {current_batch} (epoch {epoch + 1}) ics loss: {loss.item()}')
            current_batch += 1


class ObservationBuffer:
    def __init__(self, buffer_size: int, env: VecEnv, device: Union[torch.device, str] = 'cpu'):
        self.device = device
        self.buf_size = buffer_size
        self.obs_shape = get_obs_shape(env.observation_space)
        dtype = torch.from_numpy(np.array([0.0], dtype=env.observation_space.dtype)).dtype
        self.observations = torch.zeros((self.buf_size, *self.obs_shape),
                                        dtype=dtype,
                                        device=self.device,
                                        pin_memory=torch.cuda.is_available())
        self.full = False
        self.first_available = 0

    def insert_single(self, obs: torch.Tensor):
        self.observations[self.first_available].copy_(obs, non_blocking=True)
        if self.first_available == self.buf_size - 1:
            self.first_available = 0
            self.full = True
        else:
            self.first_available += 1

    def insert(self, obs: torch.Tensor):
        # TODO possibly optimize
        for i in range(obs.size(0)):
            self.insert_single(obs[i])

    def reset(self):
        self.full = False
        self.first_available = 0

    def sample(self, batch_size: int) -> torch.Tensor:
        assert batch_size > self.first_available or self.full
        if self.full:
            indices = torch.randint(low=0, high=self.first_available, size=(batch_size, ))
        else:
            indices = torch.randint(low=0, high=self.buf_size, size=(batch_size, ))
        return self.observations[indices]

    def get(self, batch_size: Optional[int] = None) -> Generator[torch.Tensor, None, None]:
        assert self.full
        perm = torch.randperm(self.buf_size, device=self.device)
        start_idx = 0
        while start_idx < self.buf_size:
            indices = perm[start_idx:start_idx + batch_size]
            yield self.observations[indices]
            start_idx += batch_size


def train_on_policy(s: SimpleNamespace, model: EEModel, env: VecEnv, neptune_run: neptune.run.Run = None):
    lamb = s.lamb
    n_steps = s.algo_args['n_steps']
    buffer_size = n_steps * env.num_envs
    buffer = ObservationBuffer(buffer_size=buffer_size, env=env, device='cpu')
    num_timesteps = 0
    current_batch = 0
    max_batch = int(m.ceil(buffer_size / s.batch_size) * s.epochs * m.ceil(s.timesteps // buffer_size))
    parameters = model.ensembles.parameters() if isinstance(model, REModel) else model.ics.parameters()
    optimizer = s.optimizer_class(parameters, lr=s.ic_lr(current_batch, max_batch))
    obs = env.reset()
    while num_timesteps < s.timesteps:
        # collect phase
        for _ in range(n_steps):
            obs = torch.from_numpy(obs)
            buffer.insert(obs)
            obs = obs.to(model.device)
            _, action_l, _ = model(obs, deterministic=s.deterministic)
            # pick a random IC for the next action
            ic_index = torch.randint(low=0, high=len(action_l), size=(env.num_envs, ), device=model.device)
            action_tensor = torch.stack(action_l, dim=0).gather(dim=0, index=ic_index.unsqueeze(1))
            obs, _, _, _ = env.step(action_tensor)
            num_timesteps += env.num_envs
        # gradient descent phase
        for epoch in range(s.epochs):
            for i, gd_obs in enumerate(buffer.get(s.batch_size)):
                gd_obs = gd_obs.to(model.device)
                logits_l, _, _ = model(gd_obs)
                orig_log_probs = torch.log_softmax(logits_l[-1], dim=-1).detach()
                ic_losses = []
                for i, logits in enumerate(logits_l[:-1]):
                    ic_log_probs = torch.log_softmax(logits, dim=-1)
                    labels = orig_log_probs.argmax(dim=-1)
                    ce_loss = F.nll_loss(ic_log_probs, labels)
                    kl_loss = F.kl_div(ic_log_probs, orig_log_probs, log_target=True)
                    loss = lamb * ce_loss + (1 - lamb) * kl_loss
                    ic_losses.append(loss)
                    if neptune_run is not None:
                        neptune_run[f'bc_train/ic_{i}_loss'].log(loss.item(), current_batch)
                set_lr(optimizer, s.ic_lr(current_batch, max_batch))
                optimizer.zero_grad()
                loss = torch.stack(ic_losses).sum()
                loss.backward()
                optimizer.step()
                if neptune_run is not None:
                    neptune_run[f'bc_train/ics_loss'].log(loss.item(), current_batch)
                else:
                    print(
                        f'batch {current_batch} (timestep: {num_timesteps} epoch {epoch + 1}) ics loss: {loss.item()}')
                current_batch += 1


def eval_ztw(model: EEModel,
             env: VecEnv,
             n_eval_episodes: int = 10,
             deterministic: bool = True,
             conf_threshold: float = 1.0,
             stop_on_ic: Optional[int] = None) -> Tuple[float, float]:
    is_monitor_wrapped = False
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
        is_monitor_wrapped = is_vecenv_wrapped(env, VecEnv) or env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)
    if not is_monitor_wrapped:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    episode_rewards, episode_lengths = [], []
    chosen_ics = {i: 0 for i in range(len(model.ics) + 1)}
    not_reseted = True
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            obs = env.reset()
            not_reseted = False
        # done, state = False, None
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            obs = torch.from_numpy(obs)
            obs = obs.to(model.device)
            logits_l, action_l, log_prob_l = model(obs, deterministic=deterministic)
            for i in range(len(logits_l)):
                if stop_on_ic is not None:
                    if i == stop_on_ic:
                        action = action_l[i]
                        chosen_ics[i] += 1
                        break
                else:
                    ic_confidence = torch.softmax(logits_l[i], dim=-1).max(dim=-1)[0]
                    if ic_confidence.item() > conf_threshold:
                        action = action_l[i]
                        chosen_ics[i] += 1
                        break
            else:
                action = action_l[-1]
                chosen_ics[len(model.ics)] += 1
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if is_monitor_wrapped:
                # Do not trust "done" with episode endings.
                # Remove vecenv stacking (if any)
                if isinstance(env, VecEnv):
                    info = info[0]
                if "episode" in info.keys():
                    # Monitor wrapper includes "episode" key in info if environment
                    # has been wrapped with it. Use those rewards instead.
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
            else:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward, chosen_ics
