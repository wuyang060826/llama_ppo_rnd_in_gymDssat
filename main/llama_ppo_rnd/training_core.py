"""Numerical training components that can be tested without DSSAT or model weights."""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch


class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step < self.warmup_steps:
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                progress = min(1.0, (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)
            group['lr'] = lr

    def get_lr(self): return [group['lr'] for group in self.optimizer.param_groups]


@dataclass
class Trajectory:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    state_str: str
    intrinsic_reward: float = 0.0


class PPOBuffer:
    def __init__(self, gamma, gae_lambda, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.trajectories: List[Trajectory] = []
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def add(self, state, action, reward, next_state, done, log_prob, value, state_str, intrinsic_reward=0.0):
        self.trajectories.append(Trajectory(state, action, reward, next_state, done, log_prob, value, state_str, intrinsic_reward))

    def clear(self): self.trajectories = []
    def __len__(self): return len(self.trajectories)

    def compute_gae_and_normalize(self, last_value, use_intrinsic=True, intrinsic_coef=0.01):
        n = len(self.trajectories)
        if n == 0:
            raise ValueError("Cannot compute advantages for an empty rollout")
        ext_rewards = np.array([t.reward for t in self.trajectories])
        int_rewards = np.array([t.intrinsic_reward for t in self.trajectories])

        self.reward_mean = ext_rewards.mean()
        self.reward_std = ext_rewards.std() + 1e-8
        norm_ext_rewards = (ext_rewards - self.reward_mean) / self.reward_std

        ext_rewards_t = torch.tensor(norm_ext_rewards, dtype=torch.float32, device=self.device)
        int_rewards_t = torch.tensor(int_rewards, dtype=torch.float32, device=self.device)
        values_t = torch.tensor([t.value for t in self.trajectories], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor([t.done for t in self.trajectories], dtype=torch.float32, device=self.device)

        advantages_ext = torch.zeros(n, dtype=torch.float32, device=self.device)
        gae_ext = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value if not dones_t[t] else 0.0
                next_non_terminal = 1.0 - dones_t[t]
            else:
                next_value = values_t[t + 1]
                next_non_terminal = 1.0 - dones_t[t]

            delta_ext = ext_rewards_t[t] + self.gamma * next_value * next_non_terminal - values_t[t]
            gae_ext = delta_ext + self.gamma * self.gae_lambda * next_non_terminal * gae_ext
            advantages_ext[t] = gae_ext

        returns = advantages_ext + values_t

        if use_intrinsic and intrinsic_coef > 0 and int_rewards.sum() != 0:
            advantages_int = torch.zeros(n, dtype=torch.float32, device=self.device)
            gae_int = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    next_non_terminal = 1.0 - dones_t[t]
                else:
                    next_non_terminal = 1.0 - dones_t[t]

                delta_int = int_rewards_t[t] + self.gamma * next_non_terminal * 0.0
                gae_int = delta_int + self.gamma * self.gae_lambda * next_non_terminal * gae_int
                advantages_int[t] = gae_int

            advantages = advantages_ext + intrinsic_coef * advantages_int
        else:
            advantages = advantages_ext

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        return advantages, returns

    def get_mini_batches(self, batch_size):
        indices = np.random.permutation(len(self.trajectories))
        return [indices[start:start + batch_size] for start in range(0, len(self.trajectories), batch_size)]

