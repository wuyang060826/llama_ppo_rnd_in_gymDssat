#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DreamerV3 + gym-dssat 深度强化学习 - 优化版本
基于论文 "Mastering Diverse Domains through World Models" (Danijar Hafner, 2023)

主要修复:
1. RSSM架构 - 正确实现posterior计算，保持序列连续性
2. KL散度 - 使用正确的KL平衡策略 (free bits + two-sided KL)
3. Symlog归一化 - 处理不同尺度的奖励和状态
4. Two-hot编码 - 用于奖励和价值预测的离散化表示
5. Actor-Critic - 使用正确的lambda return和TD(lambda)
6. 超参数优化 - 基于论文推荐值调整

输出指标:
- 农学指标: 最终产量、灌溉/施肥量、水分利用率(WUE)、氮肥利用率(NUE)
- AI指标: 平均回报、样本效率、收敛速度
"""

import numpy as np
import pandas as pd
import random
from collections import deque, OrderedDict
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent, kl_divergence, Distribution
from torch.utils.data import DataLoader, TensorDataset
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings
import json
import traceback
warnings.filterwarnings('ignore')

# 显存碎片优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"   TF32 Acceleration: Enabled")


# ============================================================================
#                              超参数配置 (优化版)
# ============================================================================

@dataclass
class DreamerV3Config:
    """DreamerV3 Hyperparameter Configuration - Optimized"""
    
    # === Training Parameters ===
    n_episodes: int = 500
    max_steps_per_episode: int = 200
    validation_frequency: int = 50
    validation_episodes: int = 10
    
    # === World Model Parameters (增大容量) ===
    state_size: int = 25
    action_size: int = 25
    hidden_size: int = 512              # 增大: 256 -> 512
    latent_size: int = 128              # 增大: 64 -> 128
    stochastic_size: int = 64           # 增大: 32 -> 64
    deterministic_size: int = 512       # 增大: 64 -> 512
    
    # === RSSM Parameters ===
    rssm_hidden: int = 256              # 增大: 128 -> 256
    gru_layers: int = 1
    
    # === World Model Training (优化学习率和序列长度) ===
    world_model_lr: float = 1e-4        # 降低: 3e-4 -> 1e-4 (更稳定)
    world_model_batch_size: int = 32
    world_model_seq_len: int = 30       # 优化: 50 -> 30 (平衡效果和速度)
    world_model_train_steps: int = 5    # 优化: 10 -> 5 (减少训练时间)
    free_nats: float = 1.0              # 修正: 3.0 -> 1.0 (正确的free bits)
    kl_balance: float = 0.8             # KL平衡因子
    
    # === Actor-Critic Parameters (优化) ===
    actor_lr: float = 3e-5              # 调整: 8e-5 -> 3e-5
    critic_lr: float = 1e-4             # 调整: 8e-5 -> 1e-4
    gamma: float = 0.997                # 增大: 0.99 -> 0.997
    gae_lambda: float = 0.95
    entropy_coef: float = 0.003         # 增大: 0.001 -> 0.003 (更多探索)
    imagination_horizon: int = 10       # 优化: 15 -> 10 (减少想象步数)
    target_critic_tau: float = 0.02     # 目标网络更新率
    
    # === Network Parameters ===
    num_layers: int = 3
    activation: str = 'elu'
    layer_norm: bool = True
    
    # === Two-hot Encoding Parameters (新增) ===
    two_hot_bins: int = 255             # Two-hot编码的bin数量
    two_hot_min: float = -20.0          # Symlog空间最小值
    two_hot_max: float = 20.0           # Symlog空间最大值
    
    # === Optimization ===
    weight_decay: float = 1e-5
    max_grad_norm: float = 100.0
    use_bf16: bool = True
    
    # === Replay Buffer ===
    buffer_size: int = 100000
    warmup_steps: int = 2000            # 增加预热步数
    
    # === Reward Function Parameters ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === Metrics ===
    target_performance: float = 500.0
    convergence_window: int = 50
    convergence_threshold: float = 0.05
    
    # === Output ===
    output_dir: str = '/home/gymusr/only_dreamV3_results/results_dreamerv3_optimized'
    checkpoint_dir: str = '/home/gymusr/only_dreamV3_checkpoints/checkpoints_dreamerv3_optimized'
    log_interval: int = 10


config = DreamerV3Config()


# ============================================================================
#                              辅助函数
# ============================================================================

def dict2array(state: dict) -> np.ndarray:
    """Convert state dictionary to numpy array"""
    if state is None:
        raise ValueError("State cannot be None")
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)


def action_to_dict(action: int, state: np.ndarray) -> Dict[str, float]:
    """将离散动作转换为gym-dssat动作字典
    
    动作空间：25个离散动作 (5氮肥等级 × 5灌溉等级)
    - 氮肥：0, 40, 80, 120, 160 kg/ha
    - 灌溉：0, 6, 12, 18, 24 mm
    """
    n_level = action % 5  # 0-4
    w_level = action // 5  # 0-4
    
    anfer = n_level * 40  # 氮肥量 kg/ha
    amir = w_level * 6    # 灌溉量 mm
    
    # 约束条件
    if len(state) > 0 and state[0] >= 10000:
        anfer = 0
    if len(state) > 21 and state[21] >= 1600:
        amir = 0
    
    return {'anfer': anfer, 'amir': amir}


def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3):
    """Calculate reward based on state and actions"""
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action
    return -k2 * n_action - k3 * w_action


def compute_wue(yield_kg_ha, irrigation_mm):
    """Water Use Efficiency: kg yield per mm irrigation"""
    if irrigation_mm <= 0:
        return 0.0
    return yield_kg_ha / irrigation_mm


def compute_nue(yield_kg_ha, nitrogen_kg_ha):
    """Nitrogen Use Efficiency: kg yield per kg nitrogen"""
    if nitrogen_kg_ha <= 0:
        return 0.0
    return yield_kg_ha / nitrogen_kg_ha


# ============================================================================
#                              Symlog和Two-hot工具函数 (新增)
# ============================================================================

def symlog(x):
    """Symmetric logarithm transformation"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


def two_hot_encode(x, bins, min_val, max_val):
    """
    Two-hot encode a continuous value.
    Returns (indices, weights) for two adjacent bins.
    
    Args:
        x: (batch,) tensor of values to encode
        bins: number of bins
        min_val: minimum value of the range
        max_val: maximum value of the range
    
    Returns:
        (indices_low, indices_high, weights_low, weights_high)
    """
    # Scale to [0, bins-1]
    scale = (bins - 1) / (max_val - min_val)
    scaled = (x - min_val) * scale
    
    # Get lower and upper bin indices
    indices_low = torch.floor(scaled).long().clamp(0, bins - 1)
    indices_high = (indices_low + 1).clamp(0, bins - 1)
    
    # Compute weights (linear interpolation)
    weights_high = scaled - indices_low.float()
    weights_low = 1.0 - weights_high
    
    return indices_low, indices_high, weights_low, weights_high


def two_hot_decode(probs, min_val, max_val):
    """
    Decode two-hot probabilities to expected value.
    
    Args:
        probs: (batch, bins) probability distribution
        min_val: minimum value of the range
        max_val: maximum value of the range
    
    Returns:
        (batch,) expected values
    """
    bins = probs.shape[-1]
    # Create bin centers
    bin_centers = torch.linspace(min_val, max_val, bins, device=probs.device)
    # Expected value
    return (probs * bin_centers).sum(dim=-1)


class RunningMeanStd:
    """Running mean and standard deviation for normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# ============================================================================
#                              网络定义 (优化版)
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron with optional layer normalization"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=3,
                 activation='elu', layer_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        
        act_fn = {'elu': nn.ELU, 'relu': nn.ReLU, 'silu': nn.SiLU}[activation]
        
        in_size = input_size
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(in_size, hidden_size))
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_size))
            self.layers.append(act_fn())
            in_size = hidden_size
        
        self.layers.append(nn.Linear(in_size, output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """Encode observation to latent distribution"""
    def __init__(self, state_size, hidden_size, stochastic_size):
        super().__init__()
        self.net = MLP(state_size, hidden_size, hidden_size, num_layers=3)
        self.mean_head = nn.Linear(hidden_size, stochastic_size)
        self.log_std_head = nn.Linear(hidden_size, stochastic_size)
        self.stochastic_size = stochastic_size
    
    def forward(self, obs):
        """Encode observation to stochastic state distribution"""
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # 使用tanh限制log_std范围，提高稳定性
        log_std = torch.tanh(log_std) * 2  # 范围约 [-2, 2]
        std = F.softplus(log_std) + 0.1
        return mean, std
    
    def sample(self, mean, std, deterministic=False):
        """Sample from the distribution"""
        if deterministic:
            return mean
        dist = Normal(mean, std)
        return dist.rsample()
    
    def get_distribution(self, mean, std):
        """Get Independent Normal distribution"""
        return Independent(Normal(mean, std), 1)


class Decoder(nn.Module):
    """Decode latent state to observation reconstruction"""
    def __init__(self, latent_size, hidden_size, state_size):
        super().__init__()
        self.net = MLP(latent_size, hidden_size, state_size, num_layers=3)
    
    def forward(self, latent):
        return self.net(latent)


class TwoHotRewardPredictor(nn.Module):
    """
    Reward predictor using two-hot encoding (DreamerV3 style).
    Outputs logits for discrete bins, then decodes to expected value.
    """
    def __init__(self, latent_size, hidden_size, bins=255, min_val=-20.0, max_val=20.0):
        super().__init__()
        self.net = MLP(latent_size, hidden_size, bins, num_layers=3)
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        # Pre-compute bin centers
        self.register_buffer('bin_centers', torch.linspace(min_val, max_val, bins))
    
    def forward(self, latent):
        """Get logits for two-hot distribution"""
        return self.net(latent)
    
    def get_probs(self, logits):
        """Get softmax probabilities"""
        return F.softmax(logits, dim=-1)
    
    def decode(self, probs):
        """Decode probabilities to expected value in symlog space"""
        return (probs * self.bin_centers).sum(dim=-1)
    
    def loss(self, logits, targets):
        """
        Compute two-hot cross-entropy loss.
        targets should be in symlog space.
        """
        # Two-hot encode targets
        indices_low, indices_high, weights_low, weights_high = two_hot_encode(
            targets, self.bins, self.min_val, self.max_val
        )
        
        # Create target distribution
        batch_size = targets.shape[0]
        target_dist = torch.zeros(batch_size, self.bins, device=targets.device)
        target_dist.scatter_add_(1, indices_low.unsqueeze(1), weights_low.unsqueeze(1))
        target_dist.scatter_add_(1, indices_high.unsqueeze(1), weights_high.unsqueeze(1))
        
        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        
        return loss


class ContinuePredictor(nn.Module):
    """Predict episode continuation (1 - done)"""
    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.net = MLP(latent_size, hidden_size, 1, num_layers=3)
    
    def forward(self, latent):
        return torch.sigmoid(self.net(latent)).squeeze(-1)


class RSSM(nn.Module):
    """
    Recurrent State Space Model - Optimized Version
    正确实现DreamerV3的RSSM架构
    
    关键改进:
    1. Prior仅依赖于deterministic state
    2. Posterior依赖于deterministic state和编码后的观测
    3. GRU正确更新deterministic state
    """
    def __init__(self, stochastic_size, deterministic_size, action_size, hidden_size):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.action_size = action_size
        
        # Prior网络: z_t = f(h_t)
        self.prior_net = MLP(deterministic_size, hidden_size, hidden_size, num_layers=2)
        self.prior_mean = nn.Linear(hidden_size, stochastic_size)
        self.prior_log_std = nn.Linear(hidden_size, stochastic_size)
        
        # Posterior网络: z_t = f(h_t, o_t_encoded)
        # 输入是deterministic state和编码后观测的拼接
        self.posterior_net = MLP(
            deterministic_size + stochastic_size, hidden_size, hidden_size, num_layers=2
        )
        self.posterior_mean = nn.Linear(hidden_size, stochastic_size)
        self.posterior_log_std = nn.Linear(hidden_size, stochastic_size)
        
        # GRU用于更新deterministic state
        # 输入: 上一时刻的stochastic state和action的拼接
        self.gru = nn.GRUCell(stochastic_size + action_size, deterministic_size)
    
    def initial_state(self, batch_size, device):
        """Get initial state"""
        return {
            'stochastic': torch.zeros(batch_size, self.stochastic_size, device=device),
            'deterministic': torch.zeros(batch_size, self.deterministic_size, device=device)
        }
    
    def get_prior(self, deterministic):
        """Get prior distribution from deterministic state"""
        prior_features = self.prior_net(deterministic)
        prior_mean = self.prior_mean(prior_features)
        prior_log_std = self.prior_log_std(prior_features)
        prior_log_std = torch.tanh(prior_log_std) * 2  # 限制范围
        prior_std = F.softplus(prior_log_std) + 0.1
        return prior_mean, prior_std
    
    def get_posterior(self, deterministic, obs_encoded):
        """
        Get posterior distribution from deterministic state and encoded observation.
        obs_encoded: (batch, stochastic_size) - 编码后的观测均值
        """
        posterior_input = torch.cat([deterministic, obs_encoded], dim=-1)
        posterior_features = self.posterior_net(posterior_input)
        posterior_mean = self.posterior_mean(posterior_features)
        posterior_log_std = self.posterior_log_std(posterior_features)
        posterior_log_std = torch.tanh(posterior_log_std) * 2
        posterior_std = F.softplus(posterior_log_std) + 0.1
        return posterior_mean, posterior_std
    
    def forward_step(self, prev_state, prev_action, obs_encoded=None):
        """
        One step of RSSM.
        
        Args:
            prev_state: dict with 'stochastic' and 'deterministic'
            prev_action: (batch, action_size) one-hot encoded
            obs_encoded: (batch, stochastic_size) encoded observation mean (for posterior)
        
        Returns:
            new_state: dict with updated 'stochastic' and 'deterministic'
            distributions: dict with prior and posterior distribution parameters
        """
        # Step 1: 更新deterministic state通过GRU
        gru_input = torch.cat([prev_state['stochastic'], prev_action], dim=-1)
        deter = self.gru(gru_input, prev_state['deterministic'])
        
        # Step 2: 计算prior分布
        prior_mean, prior_std = self.get_prior(deter)
        
        # Step 3: 计算posterior分布（如果有观测）
        if obs_encoded is not None:
            post_mean, post_std = self.get_posterior(deter, obs_encoded)
            # 从posterior采样
            posterior_dist = Independent(Normal(post_mean, post_std), 1)
            stochastic = posterior_dist.rsample()
        else:
            # 没有观测时使用prior（想象阶段）
            post_mean, post_std = prior_mean, prior_std
            prior_dist = Independent(Normal(prior_mean, prior_std), 1)
            stochastic = prior_dist.rsample()
        
        new_state = {
            'stochastic': stochastic,
            'deterministic': deter
        }
        
        distributions = {
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'posterior_mean': post_mean,
            'posterior_std': post_std
        }
        
        return new_state, distributions
    
    def forward_sequence(self, obs_encoded_sequence, actions_sequence, prev_state=None):
        """
        Forward pass through a sequence (for world model training).
        
        Args:
            obs_encoded_sequence: (batch, seq_len, stochastic_size)
            actions_sequence: (batch, seq_len, action_size) one-hot
            prev_state: initial state (optional)
        
        Returns:
            states: list of state dicts
            distributions: list of distribution dicts
        """
        batch_size, seq_len = obs_encoded_sequence.shape[:2]
        
        if prev_state is None:
            prev_state = self.initial_state(batch_size, obs_encoded_sequence.device)
        
        states = []
        distributions = []
        
        for t in range(seq_len):
            new_state, dists = self.forward_step(
                prev_state,
                actions_sequence[:, t],
                obs_encoded_sequence[:, t]
            )
            states.append(new_state)
            distributions.append(dists)
            prev_state = new_state
        
        return states, distributions
    
    def imagine(self, prev_state, actions_sequence):
        """
        Imagine trajectory using prior only (for actor-critic training).
        
        Args:
            prev_state: initial state dict
            actions_sequence: (batch, horizon, action_size) one-hot
        
        Returns:
            states: list of state dicts
            distributions: list of prior distributions
        """
        batch_size, horizon = actions_sequence.shape[:2]
        
        states = []
        distributions = []
        
        for t in range(horizon):
            new_state, dists = self.forward_step(
                prev_state,
                actions_sequence[:, t],
                obs_encoded=None  # 使用prior
            )
            states.append(new_state)
            distributions.append(dists)
            prev_state = new_state
        
        return states, distributions


class Actor(nn.Module):
    """Actor network for policy"""
    def __init__(self, latent_size, hidden_size, action_size):
        super().__init__()
        self.net = MLP(latent_size, hidden_size, hidden_size, num_layers=3)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.action_size = action_size
    
    def forward(self, latent):
        """Get action logits"""
        features = self.net(latent)
        logits = self.action_head(features)
        return logits
    
    def get_action(self, latent, deterministic=False):
        """Sample action from policy"""
        logits = self.forward(latent)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        # One-hot encode
        action_onehot = F.one_hot(action, num_classes=self.action_size).float()
        return action, action_onehot, probs
    
    def get_log_prob_and_entropy(self, latent):
        """Get log probabilities and entropy for all actions"""
        logits = self.forward(latent)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_onehot = F.one_hot(action, num_classes=self.action_size).float()
        
        return action, action_onehot, dist.log_prob(action), dist.entropy()


class TwoHotCritic(nn.Module):
    """
    Critic using two-hot encoding (DreamerV3 style).
    Predicts value distribution instead of single value.
    """
    def __init__(self, latent_size, hidden_size, bins=255, min_val=-20.0, max_val=20.0):
        super().__init__()
        self.net = MLP(latent_size, hidden_size, bins, num_layers=3)
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.register_buffer('bin_centers', torch.linspace(min_val, max_val, bins))
    
    def forward(self, latent):
        """Get logits for value distribution"""
        return self.net(latent)
    
    def get_probs(self, logits):
        """Get softmax probabilities"""
        return F.softmax(logits, dim=-1)
    
    def decode(self, probs):
        """Decode probabilities to expected value in symlog space"""
        return (probs * self.bin_centers).sum(dim=-1)
    
    def loss(self, logits, targets):
        """Two-hot cross-entropy loss"""
        indices_low, indices_high, weights_low, weights_high = two_hot_encode(
            targets, self.bins, self.min_val, self.max_val
        )
        
        batch_size = targets.shape[0]
        target_dist = torch.zeros(batch_size, self.bins, device=targets.device)
        target_dist.scatter_add_(1, indices_low.unsqueeze(1), weights_low.unsqueeze(1))
        target_dist.scatter_add_(1, indices_high.unsqueeze(1), weights_high.unsqueeze(1))
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        
        return loss


# ============================================================================
#                              Replay Buffer (优化版)
# ============================================================================

class ReplayBuffer:
    """Replay buffer with episode-aware sequence sampling"""
    
    def __init__(self, buffer_size, state_size, action_size):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.episode_boundaries = []  # 记录episode边界
        self.current_episode_start = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        if done:
            self.episode_boundaries.append(self.ptr)
            self.current_episode_start = (self.ptr + 1) % self.buffer_size
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample_sequences(self, batch_size, seq_len):
        """
        Sample sequences for world model training.
        确保序列不跨越episode边界。
        """
        valid_starts = []
        
        # 找到所有有效的起始位置（不会跨越episode边界）
        for i in range(max(0, self.size - seq_len)):
            # 检查从i开始的seq_len长度内是否有episode结束
            end_idx = min(i + seq_len, self.size)
            has_boundary = False
            for boundary in self.episode_boundaries:
                if i <= boundary < end_idx:
                    has_boundary = True
                    break
            if not has_boundary:
                valid_starts.append(i)
        
        if len(valid_starts) < batch_size:
            # 如果没有足够的有效起始位置，使用随机采样
            max_start = max(0, self.size - seq_len)
            indices = np.random.choice(max_start, size=batch_size, replace=True)
        else:
            indices = np.random.choice(valid_starts, size=batch_size, replace=False)
        
        # 预分配数组
        batch_states = np.zeros((batch_size, seq_len, self.state_size), dtype=np.float32)
        batch_actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        batch_rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        batch_dones = np.zeros((batch_size, seq_len), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            end_idx = min(idx + seq_len, self.size)
            actual_len = end_idx - idx
            
            batch_states[i, :actual_len] = self.states[idx:end_idx]
            batch_actions[i, :actual_len] = self.actions[idx:end_idx]
            batch_rewards[i, :actual_len] = self.rewards[idx:end_idx]
            batch_dones[i, :actual_len] = self.dones[idx:end_idx]
        
        return (
            torch.from_numpy(batch_states),
            torch.from_numpy(batch_actions),
            torch.from_numpy(batch_rewards),
            torch.from_numpy(batch_dones)
        )
    
    def __len__(self):
        return self.size


# ============================================================================
#                              DreamerV3 Agent (优化版)
# ============================================================================

class DreamerV3Agent:
    """DreamerV3 Agent with World Model and Actor-Critic - Optimized"""
    
    def __init__(self, config):
        self.config = config
        
        # World Model components
        self.encoder = Encoder(
            config.state_size, config.hidden_size, config.stochastic_size
        ).to(device)
        
        self.rssm = RSSM(
            config.stochastic_size, config.deterministic_size,
            config.action_size, config.rssm_hidden
        ).to(device)
        
        self.decoder = Decoder(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size, config.state_size
        ).to(device)
        
        # 使用Two-hot奖励预测器
        self.reward_predictor = TwoHotRewardPredictor(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size,
            bins=config.two_hot_bins,
            min_val=config.two_hot_min,
            max_val=config.two_hot_max
        ).to(device)
        
        self.continue_predictor = ContinuePredictor(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size
        ).to(device)
        
        # Actor-Critic (使用Two-hot Critic)
        self.actor = Actor(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size, config.action_size
        ).to(device)
        
        self.critic = TwoHotCritic(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size,
            bins=config.two_hot_bins,
            min_val=config.two_hot_min,
            max_val=config.two_hot_max
        ).to(device)
        
        # Target critic
        self.target_critic = TwoHotCritic(
            config.stochastic_size + config.deterministic_size,
            config.hidden_size,
            bins=config.two_hot_bins,
            min_val=config.two_hot_min,
            max_val=config.two_hot_max
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.world_model_optimizer = optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters()),
            lr=config.world_model_lr,
            weight_decay=config.weight_decay
        )
        
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_lr,
            weight_decay=config.weight_decay
        )
        
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, config.state_size, config.action_size
        )
        
        # Normalizers
        self.state_normalizer = RunningMeanStd(shape=(config.state_size,))
        
        # Metrics tracking
        self.training_metrics = {
            'episodes': [],
            'episode_rewards': [],
            'episode_yields': [],
            'episode_irrigation': [],
            'episode_fertilizer': [],
            'wue': [],
            'nue': [],
            'model_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'kl_divergence': [],
            'reconstruction_loss': [],
            'reward_loss': [],
            'total_steps': 0,
            'sample_efficiency_step': None,
            'convergence_step': None
        }
        
        self.best_avg_reward = float('-inf')
        self.recent_rewards = deque(maxlen=config.convergence_window)
    
    def normalize_state(self, state):
        """Normalize state for training"""
        self.state_normalizer.update(state.reshape(1, -1))
        return self.state_normalizer.normalize(state)
    
    def act(self, state, deterministic=False):
        """Select action given state"""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Encode state
            mean, std = self.encoder(state_t)
            stochastic = self.encoder.sample(mean, std, deterministic=True)
            
            # Get deterministic state from RSSM (initial)
            deter = torch.zeros(1, self.config.deterministic_size, device=device)
            latent = torch.cat([stochastic, deter], dim=-1)
            
            # Get action
            action, action_onehot, probs = self.actor.get_action(latent, deterministic)
            
            return action.item(), probs.cpu().numpy()
    
    def train_world_model(self, batch_size, seq_len):
        """
        Train world model on sampled sequences - Optimized Version
        
        关键改进:
        1. 正确的序列训练（保持状态连续性）
        2. 正确的KL平衡策略
        3. Symlog奖励变换
        4. Two-hot奖励预测
        """
        if len(self.buffer) < batch_size:
            return None
        
        states, actions, rewards, dones = self.buffer.sample_sequences(batch_size, seq_len)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        batch_size_actual = states.shape[0]
        seq_len_actual = states.shape[1]
        
        # Symlog变换奖励
        rewards_symlog = symlog(rewards)
        
        # 编码所有观测
        obs_encoded = []
        for t in range(seq_len_actual):
            mean, std = self.encoder(states[:, t])
            obs_encoded.append(mean)  # 使用均值作为编码
        obs_encoded = torch.stack(obs_encoded, dim=1)  # (batch, seq, stochastic_size)
        
        # One-hot编码动作
        actions_onehot = F.one_hot(actions, num_classes=self.config.action_size).float()
        
        # 初始化状态
        prev_state = self.rssm.initial_state(batch_size_actual, device)
        
        # 收集损失
        recon_losses = []
        reward_losses = []
        continue_losses = []
        kl_losses_prior = []
        kl_losses_posterior = []
        
        # 序列前向传播
        for t in range(seq_len_actual):
            # RSSM前向步骤
            new_state, dists = self.rssm.forward_step(
                prev_state,
                actions_onehot[:, t],
                obs_encoded[:, t]
            )
            
            # Latent state
            latent = torch.cat([new_state['stochastic'], new_state['deterministic']], dim=-1)
            
            # 重构损失
            recon = self.decoder(latent)
            recon_loss = F.mse_loss(recon, states[:, t], reduction='mean')
            recon_losses.append(recon_loss)
            
            # 奖励预测损失 (Two-hot)
            reward_logits = self.reward_predictor(latent)
            reward_loss = self.reward_predictor.loss(reward_logits, rewards_symlog[:, t])
            reward_losses.append(reward_loss)
            
            # 继续预测损失
            pred_continue = self.continue_predictor(latent)
            continue_loss = F.binary_cross_entropy(pred_continue, 1 - dones[:, t], reduction='mean')
            continue_losses.append(continue_loss)
            
            # KL散度 - 双向计算
            prior_dist = Independent(Normal(dists['prior_mean'], dists['prior_std']), 1)
            post_dist = Independent(Normal(dists['posterior_mean'], dists['posterior_std']), 1)
            
            # KL(q||p) - 用于优化posterior
            kl_qp = kl_divergence(post_dist, prior_dist)
            # KL(p||q) - 用于优化prior
            kl_pq = kl_divergence(prior_dist, post_dist)
            
            # Free bits: max(kl - free_nats, 0)
            kl_qp_free = torch.clamp(kl_qp - self.config.free_nats, min=0)
            kl_pq_free = torch.clamp(kl_pq - self.config.free_nats, min=0)
            
            kl_losses_prior.append(kl_qp_free.mean())
            kl_losses_posterior.append(kl_pq_free.mean())
            
            # 更新状态
            prev_state = new_state
        
        # 总损失
        # KL平衡: 80% KL(q||p) + 20% KL(p||q)
        kl_loss = (
            self.config.kl_balance * sum(kl_losses_prior) / len(kl_losses_prior) +
            (1 - self.config.kl_balance) * sum(kl_losses_posterior) / len(kl_losses_posterior)
        )
        
        total_loss = (
            sum(recon_losses) / len(recon_losses) +
            sum(reward_losses) / len(reward_losses) +
            sum(continue_losses) / len(continue_losses) +
            kl_loss
        )
        
        # 优化
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters()),
            self.config.max_grad_norm
        )
        self.world_model_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': sum(l.item() for l in recon_losses) / len(recon_losses),
            'reward_loss': sum(l.item() for l in reward_losses) / len(reward_losses),
            'continue_loss': sum(l.item() for l in continue_losses) / len(continue_losses),
            'kl_loss': kl_loss.item()
        }
    
    def train_actor_critic(self, batch_size, imagination_horizon):
        """
        Train actor and critic in imagination - Optimized Version
        
        关键改进:
        1. 正确的Lambda Return计算
        2. 使用目标网络计算价值估计
        3. 正确的优势函数标准化
        """
        if len(self.buffer) < batch_size:
            return None, None
        
        # 采样初始状态
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        initial_states = torch.tensor(
            self.buffer.states[indices], dtype=torch.float32, device=device
        )
        
        # 编码初始状态
        with torch.no_grad():
            mean, std = self.encoder(initial_states)
            stochastic = self.encoder.sample(mean, std)
            deter = torch.zeros(batch_size, self.config.deterministic_size, device=device)
        
        # 想象轨迹
        state = {'stochastic': stochastic, 'deterministic': deter}
        
        imagined_rewards = []
        imagined_log_probs = []
        entropies = []
        imagined_values = []
        imagined_continues = []
        latents_list = []
        
        for t in range(imagination_horizon):
            latent = torch.cat([state['stochastic'], state['deterministic']], dim=-1)
            
            # 获取动作
            action, action_onehot, log_prob, entropy = self.actor.get_log_prob_and_entropy(latent)
            
            # RSSM想象步骤（使用prior）
            new_state, _ = self.rssm.forward_step(state, action_onehot, obs_encoded=None)
            state = new_state
            
            new_latent = torch.cat([state['stochastic'], state['deterministic']], dim=-1)
            
            # 预测奖励 (symlog空间)
            reward_logits = self.reward_predictor(new_latent.detach())
            reward_probs = self.reward_predictor.get_probs(reward_logits)
            reward = self.reward_predictor.decode(reward_probs)
            
            # 预测继续概率
            continue_prob = self.continue_predictor(new_latent.detach())
            
            # 获取价值估计（使用目标网络）
            value_logits = self.target_critic(new_latent.detach())
            value_probs = self.target_critic.get_probs(value_logits)
            value = self.target_critic.decode(value_probs)
            
            imagined_rewards.append(reward)
            imagined_log_probs.append(log_prob)
            entropies.append(entropy)
            imagined_values.append(value)
            imagined_continues.append(continue_prob)
            latents_list.append(new_latent.detach())
        
        # 堆叠数据
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_log_probs = torch.stack(imagined_log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        imagined_values = torch.stack(imagined_values, dim=1)
        imagined_continues = torch.stack(imagined_continues, dim=1)
        latents_tensor = torch.stack(latents_list, dim=1)
        
        # Lambda Returns (正确的实现)
        # R_t = r_t + gamma * continue_t * (lambda * R_{t+1} + (1-lambda) * V_{t+1})
        returns = torch.zeros_like(imagined_rewards)
        returns[:, -1] = imagined_rewards[:, -1] + self.config.gamma * imagined_continues[:, -1] * imagined_values[:, -1]
        
        for t in reversed(range(imagination_horizon - 1)):
            next_return = self.config.gae_lambda * returns[:, t + 1] + (1 - self.config.gae_lambda) * imagined_values[:, t + 1]
            returns[:, t] = imagined_rewards[:, t] + self.config.gamma * imagined_continues[:, t] * next_return
        
        # 优势函数
        advantages = returns - imagined_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor损失
        actor_loss = -(imagined_log_probs * advantages.detach()).mean()
        actor_loss -= self.config.entropy_coef * entropies.mean()
        
        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        # Critic损失 (Two-hot)
        # 重新计算critic输出
        critic_logits = self.critic(latents_tensor.view(-1, latents_tensor.shape[-1]))
        critic_loss = self.critic.loss(critic_logits, returns.view(-1))
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                (1 - self.config.target_critic_tau) * target_param.data +
                self.config.target_critic_tau * param.data
            )
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path, episode):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'continue_predictor': self.continue_predictor.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode
        }, os.path.join(path, f'model_ep{episode}.pth'))
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.continue_predictor.load_state_dict(checkpoint['continue_predictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        if 'world_model_optimizer' in checkpoint:
            self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return checkpoint.get('episode', 0)


# ============================================================================
#                              指标记录器
# ============================================================================

class MetricsLogger:
    """Logger for tracking and saving metrics"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        self.metrics = {
            'episode': [],
            'yield': [],
            'irrigation': [],
            'fertilizer': [],
            'wue': [],
            'nue': [],
            'reward': [],
            'episode_length': [],
            'training_time': [],
        }
        
        self.ai_metrics = {
            'episode': [],
            'avg_reward': [],
            'total_steps': [],
            'model_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'sample_efficiency_step': None,
            'convergence_step': None
        }
        
        self.training_history = []
    
    def log_episode(self, episode, yield_kg, irrigation_mm, fertilizer_kg,
                   reward, episode_length, training_time):
        """Log episode metrics"""
        wue = compute_wue(yield_kg, irrigation_mm)
        nue = compute_nue(yield_kg, fertilizer_kg)
        
        self.metrics['episode'].append(episode)
        self.metrics['yield'].append(yield_kg)
        self.metrics['irrigation'].append(irrigation_mm)
        self.metrics['fertilizer'].append(fertilizer_kg)
        self.metrics['wue'].append(wue)
        self.metrics['nue'].append(nue)
        self.metrics['reward'].append(reward)
        self.metrics['episode_length'].append(episode_length)
        self.metrics['training_time'].append(training_time)
    
    def log_training(self, episode, avg_reward, total_steps,
                    model_loss=None, actor_loss=None, critic_loss=None):
        """Log training metrics"""
        self.ai_metrics['episode'].append(episode)
        self.ai_metrics['avg_reward'].append(avg_reward)
        self.ai_metrics['total_steps'].append(total_steps)
        self.ai_metrics['model_loss'].append(model_loss)
        self.ai_metrics['actor_loss'].append(actor_loss)
        self.ai_metrics['critic_loss'].append(critic_loss)
    
    def set_sample_efficiency(self, step):
        """Set sample efficiency step"""
        self.ai_metrics['sample_efficiency_step'] = step
    
    def set_convergence(self, step):
        """Set convergence step"""
        self.ai_metrics['convergence_step'] = step
    
    def save_results(self):
        """Save all results to files"""
        # Save agronomic metrics
        df_agro = pd.DataFrame(self.metrics)
        df_agro.to_excel(os.path.join(self.output_dir, 'data', 'agronomic_metrics.xlsx'), index=False)
        df_agro.to_json(os.path.join(self.output_dir, 'data', 'agronomic_metrics.json'), orient='records')
        
        # Save AI metrics
        df_ai = pd.DataFrame({k: v for k, v in self.ai_metrics.items() 
                             if isinstance(v, list)})
        df_ai.to_excel(os.path.join(self.output_dir, 'data', 'ai_metrics.xlsx'), index=False)
        df_ai.to_json(os.path.join(self.output_dir, 'data', 'ai_metrics.json'), orient='records')
        
        # Generate plots
        self._generate_plots()
        
        print(f"Results saved to {self.output_dir}")
    
    def _generate_plots(self):
        """Generate training plots"""
        fig_dir = os.path.join(self.output_dir, 'figures')
        
        # Plot 1: Agronomic Metrics
        if len(self.metrics['episode']) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Yield
            ax = axes[0, 0]
            ax.plot(self.metrics['episode'], self.metrics['yield'], 'g-', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Yield (kg/ha)')
            ax.set_title('Crop Yield over Training')
            ax.grid(True, alpha=0.3)
            
            # Irrigation
            ax = axes[0, 1]
            ax.plot(self.metrics['episode'], self.metrics['irrigation'], 'b-', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Irrigation (mm)')
            ax.set_title('Total Irrigation over Training')
            ax.grid(True, alpha=0.3)
            
            # WUE
            ax = axes[1, 0]
            ax.plot(self.metrics['episode'], self.metrics['wue'], 'purple', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('WUE (kg/mm)')
            ax.set_title('Water Use Efficiency')
            ax.grid(True, alpha=0.3)
            
            # NUE
            ax = axes[1, 1]
            ax.plot(self.metrics['episode'], self.metrics['nue'], 'orange', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('NUE (kg/kg)')
            ax.set_title('Nitrogen Use Efficiency')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'agronomic_metrics.png'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(fig_dir, 'agronomic_metrics.pdf'), bbox_inches='tight')
            plt.close()
        
        # Plot 2: AI Metrics
        if len(self.ai_metrics['episode']) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss curves
            ax = axes[0]
            if self.ai_metrics['model_loss'][0] is not None:
                model_losses = [l for l in self.ai_metrics['model_loss'] if l is not None]
                episodes = [e for e, l in zip(self.ai_metrics['episode'], self.ai_metrics['model_loss']) if l is not None]
                ax.plot(episodes, model_losses, 'b-', label='Model Loss')
            if self.ai_metrics['actor_loss'][0] is not None:
                actor_losses = [l for l in self.ai_metrics['actor_loss'] if l is not None]
                episodes = [e for e, l in zip(self.ai_metrics['episode'], self.ai_metrics['actor_loss']) if l is not None]
                ax.plot(episodes, actor_losses, 'r-', label='Actor Loss')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Reward curve
            ax = axes[1]
            ax.plot(self.ai_metrics['episode'], self.ai_metrics['avg_reward'], 'g-', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Reward')
            ax.set_title('Average Reward over Training')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(fig_dir, 'training_curves.pdf'), bbox_inches='tight')
            plt.close()


# ============================================================================
#                              训练函数
# ============================================================================

def train_dreamerv3():
    """Main training function for DreamerV3 - Optimized"""
    print("=" * 70)
    print("DreamerV3 + gym-dssat Training (Optimized Version)")
    print("World Model (RSSM) -> Latent Actor-Critic")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_dir, 'best'), exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_dir, 'periodic'), exist_ok=True)
    
    # Initialize environment
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/gymusr/logs/dssat-pdi.log',
        'mode': 'all', 'seed': 123456, 'random_weather': True
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    # Initialize agent
    agent = DreamerV3Agent(config)
    
    # Initialize logger
    logger = MetricsLogger(config.output_dir)
    
    # Training tracking
    total_steps = 0
    best_reward = float('-inf')
    start_time = time.time()
    
    episode_times = deque(maxlen=20)
    
    print(f"\nStarting training ({config.n_episodes} episodes)")
    print("-" * 70)
    
    pbar = tqdm(range(1, config.n_episodes + 1), desc="Training", unit="ep")
    
    for episode in pbar:
        ep_start_time = time.time()
        
        # Reset environment
        state = env.reset()
        state_array = dict2array(state)
        
        episode_reward = 0
        episode_irrigation = 0
        episode_fertilizer = 0
        episode_steps = 0
        
        done = False
        
        while not done and episode_steps < config.max_steps_per_episode:
            # Normalize state
            norm_state = agent.normalize_state(state_array)
            
            # Select action
            action, probs = agent.act(norm_state)
            
            # Convert action to dictionary format for gym-dssat
            action_dict = action_to_dict(action, state_array)
            
            # Step environment
            next_state, env_reward, done, info = env.step(action_dict)
            next_state_array = dict2array(next_state) if next_state is not None else state_array
            
            # Calculate reward using custom reward function
            reward = get_reward(
                state_array, action_dict['anfer'], action_dict['amir'],
                next_state_array, done, config.k1, config.k2, config.k3
            )
            
            # Track irrigation and fertilizer
            episode_irrigation += action_dict['amir']
            episode_fertilizer += action_dict['anfer']
            
            # Store transition
            agent.buffer.add(
                norm_state, action, reward,
                agent.normalize_state(next_state_array),
                done
            )
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state_array = next_state_array
        
        # Training after warmup
        if total_steps >= config.warmup_steps:
            # Train world model
            for _ in range(config.world_model_train_steps):
                wm_metrics = agent.train_world_model(
                    config.world_model_batch_size,
                    config.world_model_seq_len
                )
            
            # Train actor-critic
            actor_loss, critic_loss = agent.train_actor_critic(
                config.world_model_batch_size,
                config.imagination_horizon
            )
        else:
            wm_metrics = None
            actor_loss, critic_loss = None, None
        
        # Track metrics
        ep_time = time.time() - ep_start_time
        episode_times.append(ep_time)
        
        # Get yield from final state
        # Note: state[4] appears to be in an unusual scale
        # Based on observation, it might already be in kg/ha or need different conversion
        if done and len(state_array) > 4:
            raw_yield = state_array[4]
            # Debug: print raw value to understand the scale
            # Based on typical maize yield: 3000-15000 kg/ha
            # If raw_yield is around 5-15, it's probably tons/ha
            # If raw_yield is around 3000-15000, it's probably kg/ha already
            if raw_yield <= 0:
                yield_kg = 0
            elif raw_yield < 100:
                # Likely in tons/ha, convert to kg/ha
                yield_kg = raw_yield * 1000
            elif raw_yield < 20000:
                # Likely already in kg/ha
                yield_kg = raw_yield
            else:
                # Unreasonable value, set to 0
                yield_kg = 0
        else:
            yield_kg = 0
        
        logger.log_episode(
            episode, yield_kg, episode_irrigation, episode_fertilizer,
            episode_reward, episode_steps, ep_time
        )
        
        logger.log_training(
            episode, episode_reward, total_steps,
            wm_metrics['total_loss'] if wm_metrics else None,
            actor_loss, critic_loss
        )
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config.checkpoint_dir, 'best'), episode)
        
        # Periodic saving
        if episode % 100 == 0:
            agent.save(os.path.join(config.checkpoint_dir, 'periodic'), episode)
        
        # Check sample efficiency
        if (logger.ai_metrics['sample_efficiency_step'] is None and 
            episode_reward >= config.target_performance):
            logger.set_sample_efficiency(total_steps)
        
        # Check convergence
        agent.recent_rewards.append(episode_reward)
        if len(agent.recent_rewards) >= config.convergence_window:
            recent_var = np.var(list(agent.recent_rewards))
            recent_mean = np.mean(list(agent.recent_rewards))
            if recent_var < config.convergence_threshold * recent_mean:
                if logger.ai_metrics['convergence_step'] is None:
                    logger.set_convergence(total_steps)
        
        # Update progress bar
        avg_time = np.mean(list(episode_times))
        pbar.set_postfix({
            'reward': f'{episode_reward:.1f}',
            'yield': f'{yield_kg:.0f}',
            'steps': total_steps,
            'avg_time': f'{avg_time:.1f}s'
        })
        
        # Log progress
        if episode % config.log_interval == 0:
            avg_reward = np.mean(logger.metrics['reward'][-config.log_interval:])
            print(f"\nEpisode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Yield = {yield_kg:.0f} kg/ha, Steps = {total_steps}")
    
    # Save final model
    agent.save(os.path.join(config.checkpoint_dir, 'final'), config.n_episodes)
    
    # Save results
    logger.save_results()
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total Episodes: {config.n_episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Reward: {best_reward:.2f}")
    
    if logger.ai_metrics['sample_efficiency_step']:
        print(f"Sample Efficiency Step: {logger.ai_metrics['sample_efficiency_step']} "
              f"(target: {config.target_performance})")
    
    if logger.ai_metrics['convergence_step']:
        print(f"Convergence Step: {logger.ai_metrics['convergence_step']}")
    else:
        print("Convergence: Not detected")
    
    print(f"\nResults saved to: {config.output_dir}")
    print(f"Models saved to: {config.checkpoint_dir}")
    
    env.close()
    
    return agent, logger


def main():
    """Main entry point"""
    os.makedirs('/home/gymusr/logs', exist_ok=True)
    
    agent, logger = train_dreamerv3()
    
    print("\n" + "=" * 70)
    print("DreamerV3 Training Summary (Optimized)")
    print("=" * 70)
    print(f"\nModel Architecture:")
    print(f"  - Input: Pure numerical state ({config.state_size} dimensions)")
    print(f"  - World Model: RSSM with stochastic ({config.stochastic_size}) + deterministic ({config.deterministic_size}) states")
    print(f"  - Actor-Critic: Operating in latent space")
    print(f"  - Imagination Horizon: {config.imagination_horizon} steps")
    print(f"\nKey Optimizations:")
    print(f"  - Symlog normalization for rewards/values")
    print(f"  - Two-hot encoding for discrete prediction")
    print(f"  - KL balancing with free bits")
    print(f"  - Lambda returns for critic training")
    print(f"\nOutputs:")
    print(f"  - Excel: {config.output_dir}/data/agronomic_metrics.xlsx")
    print(f"  - JSON: {config.output_dir}/data/ai_metrics.json")
    print(f"  - Plots (PNG/PDF): {config.output_dir}/figures/")


if __name__ == "__main__":
    main()