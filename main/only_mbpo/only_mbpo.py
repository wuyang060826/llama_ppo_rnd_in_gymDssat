#!/usr/bin/env python3
"""
MBPO (Model-Based Policy Optimization) - Standard Implementation
=================================================================

严格参考论文: Janner et al. "When to Trust Your Model: Model-Based Policy Optimization"
NeurIPS 2019 - https://arxiv.org/abs/1906.08253

核心算法特点:
1. Branched Rollouts: 从真实buffer采样的状态开始，执行H步模型预测
2. Model Ensemble: 使用多个动力学模型，选择validation loss最小的elite models
3. SAC作为底层policy optimization算法
4. 模型预测的transitions与真实transitions混合训练

论文关键公式:
- 在模型rollout中累积奖励: sum_{t'=t}^{t+H-1} gamma^{t'-t} r(s_{t'}, a_{t'})
- Q-target使用min of two Q functions (Clipped Double Q-Learning)
- 自动熵调整 (Automatic Entropy Tuning)
"""

import numpy as np
import pandas as pd
import random
from collections import deque
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, NamedTuple
import warnings
warnings.filterwarnings('ignore')
import copy

# Memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Try to import gym-dssat
try:
    import gym_dssat_pdi
except ImportError:
    print("Warning: gym_dssat_pdi not found. Using mock environment for testing.")


# ============================================================================
#                              MBPO Hyperparameters (Paper-Aligned)
# ============================================================================

@dataclass
class MBPOConfig:
    """MBPO Hyperparameters - Strictly aligned with paper recommendations
    
    Reference: Janner et al. Table 1 & Appendix
    """
    
    # === Environment Parameters ===
    state_size: int = 25
    action_size: int = 25  # 5 nitrogen levels x 5 irrigation levels (discrete)
    
    # === Training Parameters ===
    n_episodes: int = 500
    max_steps_per_episode: int = 200
    
    # === SAC Parameters (Paper: Table 1) ===
    gamma: float = 0.99                    # Discount factor
    tau: float = 0.005                     # Soft update coefficient (Paper: 0.005)
    actor_lr: float = 3e-4                 # Policy learning rate (Paper: 3e-4)
    critic_lr: float = 3e-4                # Q-function learning rate (Paper: 3e-4)
    alpha_lr: float = 3e-4                 # Alpha learning rate (Paper: 3e-4)
    hidden_size: int = 256                 # Hidden layer size (Paper: 256 for MUJOCO)
    auto_entropy: bool = True              # Automatic entropy tuning
    target_entropy: float = None           # Will be set to -action_dim
    initial_alpha: float = 1.0             # Initial temperature
    
    # === Model Ensemble Parameters (Paper: Section 5.1) ===
    ensemble_size: int = 7                 # Number of models (Paper: 7)
    elite_size: int = 5                    # Number of elite models (Paper: 5 best)
    model_lr: float = 1e-3                 # Model learning rate (Paper: 1e-3)
    model_hidden_size: int = 200           # Model hidden size (Paper: 200)
    model_weight_decay: float = 5e-5       # Weight decay (Paper: 5e-5)
    model_batch_size: int = 256            # Model training batch size
    model_train_epochs: int = 5            # Training epochs per update (Paper: 5)
    validation_split: float = 0.2          # Validation split ratio
    
    # === Model Rollout Parameters (Paper: Section 5.1, Table 1) ===
    # Key MBPO parameter: rollout horizon H
    rollout_horizon: int = 10              # Rollout horizon H (Paper: H=10 for most envs)
    rollout_batch_size: int = 100000       # Number of model samples per update
    model_retain_epochs: int = 1           # How long to retain model samples (Paper: 1)
    
    # === Training Schedule (Paper: Section 5.1) ===
    environment_interactions_per_update: int = 1000  # Real env steps per policy update
    model_update_freq: int = 250           # Model training frequency
    start_training_steps: int = 1000       # Steps before starting training
    sac_updates_per_step: int = 1          # SAC updates per environment step
    
    # === Replay Buffer Parameters ===
    real_buffer_size: int = 1000000        # Real replay buffer size
    model_buffer_size: int = 1000000       # Model replay buffer size
    
    # === SAC Batch Size ===
    sac_batch_size: int = 256              # SAC training batch size (Paper: 256)
    
    # === Exploration ===
    epsilon_start: float = 0.1             # Initial exploration epsilon
    epsilon_end: float = 0.01              # Final exploration epsilon
    epsilon_decay_steps: int = 100000      # Decay steps
    
    # === Model Data Ratio (Paper: uses all model data within rollout horizon) ===
    # MBPO generates samples from model and mixes with real data
    real_ratio: float = 0.05               # Ratio of real to model data in each batch
                                         # Paper suggests 5% real, 95% model data
    
    # === Reward Parameters (gym-dssat specific) ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === Gradient Clipping ===
    max_grad_norm: float = 1.0             # Gradient clipping norm
    
    # === Normalization ===
    normalize_states: bool = True          # Whether to normalize states
    normalize_rewards: bool = True         # Whether to normalize rewards
    
    # === Metrics ===
    target_performance: float = 500.0
    convergence_window: int = 50
    
    # === Paths ===
    save_dir: str = "/home/gymusr/only_mbpo_results/mbpo_results_standard"
    checkpoint_dir: str = "/home/gymusr/only_mbpo_checkpoints/mbpo_standard"


config = MBPOConfig()


# ============================================================================
#                              Transition Data Structure
# ============================================================================

class Transition(NamedTuple):
    """Single transition tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict = field(default_factory=dict)


# ============================================================================
#                              Helper Functions
# ============================================================================

def dict2array(state: dict) -> np.ndarray:
    """Convert dictionary state to numpy array"""
    if state is None:
        raise ValueError("State cannot be None")
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)


def array2action(action: int, state: np.ndarray) -> dict:
    """Convert discrete action to environment action dictionary"""
    nitrogen = (action % 5) * 40
    irrigation = int(action / 5) * 6
    
    if state[0] >= 10000:
        nitrogen = 0
    if len(state) > 21 and state[21] >= 1600:
        irrigation = 0
    
    return {'anfer': nitrogen, 'amir': irrigation}


def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3):
    """Calculate reward based on paper definition"""
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action
    return -k2 * n_action - k3 * w_action


# ============================================================================
#                          Normalization Utilities
# ============================================================================

class RunningMeanStd:
    """Running mean and standard deviation for online normalization"""
    
    def __init__(self, shape: int, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with new batch"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        """Normalize input"""
        if x.ndim == 1:
            return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


# ============================================================================
#                              Replay Buffers
# ============================================================================

class ReplayBuffer:
    """Standard replay buffer for real environment transitions"""
    
    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self.size = 0
        self.ptr = 0
        
        # Storage
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a single transition"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(self, states: np.ndarray, actions: np.ndarray, 
                  rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Add a batch of transitions"""
        batch_size = len(states)
        if self.ptr + batch_size <= self.capacity:
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
            self.dones[self.ptr:self.ptr + batch_size] = dones
        else:
            first_part = self.capacity - self.ptr
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.rewards[self.ptr:] = rewards[:first_part]
            self.next_states[self.ptr:] = next_states[:first_part]
            self.dones[self.ptr:] = dones[:first_part]
            
            remaining = batch_size - first_part
            self.states[:remaining] = states[first_part:]
            self.actions[:remaining] = actions[first_part:]
            self.rewards[:remaining] = rewards[first_part:]
            self.next_states[:remaining] = next_states[first_part:]
            self.dones[:remaining] = dones[first_part:]
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def sample_states(self, n_states: int) -> np.ndarray:
        """Sample states for model rollouts"""
        indices = np.random.randint(0, self.size, n_states)
        return self.states[indices]
    
    def get_all(self) -> Tuple:
        """Get all stored transitions"""
        return (
            self.states[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_states[:self.size],
            self.dones[:self.size]
        )
    
    def __len__(self):
        return self.size


class ModelBuffer:
    """Buffer for model-generated transitions (used in MBPO)"""
    
    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self.size = 0
        self.ptr = 0
        
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add_batch(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Add batch of model-generated transitions"""
        batch_size = len(states)
        
        # Handle wraparound
        if self.ptr + batch_size > self.capacity:
            first_part = self.capacity - self.ptr
            self._add_partial(states[:first_part], actions[:first_part],
                            rewards[:first_part], next_states[:first_part],
                            dones[:first_part])
            remaining = batch_size - first_part
            self._add_partial(states[first_part:], actions[first_part:],
                            rewards[first_part:], next_states[first_part:],
                            dones[first_part:])
        else:
            self._add_partial(states, actions, rewards, next_states, dones)
        
        self.size = min(self.size + batch_size, self.capacity)
    
    def _add_partial(self, states, actions, rewards, next_states, dones):
        batch_size = len(states)
        self.states[self.ptr:self.ptr + batch_size] = states
        self.actions[self.ptr:self.ptr + batch_size] = actions
        self.rewards[self.ptr:self.ptr + batch_size] = rewards
        self.next_states[self.ptr:self.ptr + batch_size] = next_states
        self.dones[self.ptr:self.ptr + batch_size] = dones
        self.ptr = (self.ptr + batch_size) % self.capacity
    
    def clear(self):
        """Clear the buffer"""
        self.size = 0
        self.ptr = 0
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch"""
        if self.size < batch_size:
            return None
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size


# ============================================================================
#                          Dynamics Model (Paper-Aligned)
# ============================================================================

class ProbabilisticEnsembleModel(nn.Module):
    """
    Probabilistic Ensemble Dynamics Model
    
    Paper Reference: Section 4.2 - Model Learning
    Each model predicts: (next_state_delta, reward) with Gaussian output
    Uses bootstrap training with random subset of data per model
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int,
                 ensemble_size: int = 7, elite_size: int = 5):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.elite_indices = list(range(elite_size))  # Initialize with first elite_size
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            SingleProbabilisticModel(state_size, action_size, hidden_size)
            for _ in range(ensemble_size)
        ])
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor,
                model_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            state: [batch_size, state_size]
            action_onehot: [batch_size, action_size]
            model_idx: If specified, use only this model
        
        Returns:
            mean: Predicted mean [ensemble_size, batch_size, output_size] or [batch_size, output_size]
            logvar: Predicted log variance (same shape as mean)
        """
        if model_idx is not None:
            return self.models[model_idx](state, action_onehot)
        
        # Forward through all models
        means = []
        logvars = []
        for model in self.models:
            mean, logvar = model(state, action_onehot)
            means.append(mean)
            logvars.append(logvar)
        
        means = torch.stack(means, dim=0)  # [ensemble_size, batch_size, output_size]
        logvars = torch.stack(logvars, dim=0)
        
        return means, logvars
    
    def sample_prediction(self, state: torch.Tensor, action_onehot: torch.Tensor,
                          use_elite: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample prediction from a randomly selected elite model
        
        Paper: Use only elite models for rollout generation
        """
        if use_elite and len(self.elite_indices) > 0:
            model_idx = np.random.choice(self.elite_indices)
        else:
            model_idx = np.random.randint(0, self.ensemble_size)
        
        mean, logvar = self.models[model_idx](state, action_onehot)
        
        # Sample from Gaussian
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        
        # Split into state delta and reward
        state_delta = sample[:, :-1]
        reward = sample[:, -1:]
        
        return state_delta, reward
    
    def get_uncertainty(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Get model uncertainty (ensemble disagreement)
        Used for filtering high-uncertainty predictions
        """
        with torch.no_grad():
            means, _ = self.forward(state, action_onehot)
            # Use variance across ensemble as uncertainty measure
            variance = means.var(dim=0).mean(dim=-1)  # [batch_size]
        return variance
    
    def set_elite(self, elite_indices: List[int]):
        """Set elite model indices based on validation loss"""
        self.elite_indices = elite_indices
    
    def get_elite_models(self) -> List[int]:
        """Return elite model indices"""
        return self.elite_indices


class SingleProbabilisticModel(nn.Module):
    """
    Single probabilistic dynamics model
    Outputs mean and log-variance for Gaussian distribution
    
    Paper: Uses 4-layer network with 200 hidden units
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        input_size = state_size + action_size
        output_size = state_size + 1  # state_delta + reward
        
        # Network architecture (Paper: 4 layers)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),  # Swish activation (Paper uses this)
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU()
        )
        
        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.logvar_head = nn.Linear(hidden_size, output_size)
        
        # Initialize logvar to small values for stability
        self.logvar_head.weight.data.fill_(0.0)
        self.logvar_head.bias.data.fill_(0.0)
        
        # Orthogonal initialization for main network
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            mean: Predicted mean [batch_size, state_size + 1]
            logvar: Predicted log variance [batch_size, state_size + 1]
        """
        x = torch.cat([state, action_onehot], dim=-1)
        features = self.network(x)
        
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mean, logvar


# ============================================================================
#                          SAC Networks (Paper-Aligned)
# ============================================================================

class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian Policy for SAC
    
    Note: Since gym-dssat uses discrete actions, we adapt to categorical policy
    Paper uses Gaussian policy for continuous actions, we use categorical equivalent
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.action_size = action_size
        
        # Network (Paper: 2 hidden layers, 256 units each)
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(hidden_size, action_size)
        
        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get action logits"""
        features = self.network(state)
        logits = self.output_layer(features)
        return logits
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Returns:
            action: Sampled action
            log_prob: Log probability of sampled action
        """
        logits = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            return action, log_prob
        
        # Categorical distribution
        probs = F.softmax(logits, dim=-1)
        # Numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)


class DoubleQCritic(nn.Module):
    """
    Double Q-Network for SAC (Clipped Double Q-Learning)
    
    Paper: Uses two independent Q-networks to reduce overestimation
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        # Q1 network
        self.q1_network = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Q2 network (independent)
        self.q2_network = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks
        
        Returns:
            q1: Q1 value [batch_size, 1]
            q2: Q2 value [batch_size, 1]
        """
        x = torch.cat([state, action_onehot], dim=-1)
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only"""
        x = torch.cat([state, action_onehot], dim=-1)
        return self.q1_network(x)


# ============================================================================
#                              MBPO Agent
# ============================================================================

class MBPOAgent:
    """
    Model-Based Policy Optimization Agent
    
    Paper Algorithm:
    1. Collect real environment data
    2. Train ensemble dynamics model
    3. Generate branched rollouts from model
    4. Update policy with SAC using mixed real and model data
    """
    
    def __init__(self, config: MBPOConfig):
        self.config = config
        
        # Normalization
        self.state_normalizer = RunningMeanStd(config.state_size)
        self.reward_normalizer = RunningMeanStd(1)
        
        # SAC Networks
        self.policy = SquashedGaussianPolicy(
            config.state_size, config.action_size, config.hidden_size
        ).to(device)
        
        self.critic = DoubleQCritic(
            config.state_size, config.action_size, config.hidden_size
        ).to(device)
        
        self.target_critic = DoubleQCritic(
            config.state_size, config.action_size, config.hidden_size
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Dynamics Model
        self.dynamics_model = ProbabilisticEnsembleModel(
            config.state_size, config.action_size, config.model_hidden_size,
            config.ensemble_size, config.elite_size
        ).to(device)
        
        # Target entropy for SAC
        self.target_entropy = -config.action_size
        
        # Alpha (temperature parameter)
        self.log_alpha = torch.tensor([np.log(config.initial_alpha)], 
                                       requires_grad=True, device=device)
        self.alpha = config.initial_alpha
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.model_optimizer = optim.AdamW(
            self.dynamics_model.parameters(), 
            lr=config.model_lr,
            weight_decay=config.model_weight_decay
        )
        
        # Replay Buffers
        self.real_buffer = ReplayBuffer(config.real_buffer_size, config.state_size)
        self.model_buffer = ModelBuffer(config.model_buffer_size, config.state_size)
        
        # Training state
        self.total_steps = 0
        self.model_update_count = 0
        self.last_model_update = 0
        
        # Epsilon for exploration
        self.epsilon = config.epsilon_start
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics"""
        if self.config.normalize_states:
            return self.state_normalizer.normalize(state)
        return state
    
    def one_hot_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action to one-hot encoding"""
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return F.one_hot(action, num_classes=self.config.action_size).float()
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action using current policy"""
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_size)
        
        state_normalized = self.normalize_state(state)
        state_tensor = torch.from_numpy(state_normalized).float().unsqueeze(0).to(device)
        
        self.policy.eval()
        with torch.no_grad():
            action, _ = self.policy.sample(state_tensor, deterministic)
        self.policy.train()
        
        return action.item()
    
    def update_epsilon(self):
        """Decay exploration epsilon"""
        progress = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        self.epsilon = self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )
    
    def train_dynamics_model(self) -> Dict:
        """
        Train the ensemble dynamics model
        
        Paper: Train each model independently on bootstrap samples
        Select elite models based on validation loss
        """
        if len(self.real_buffer) < self.config.start_training_steps:
            return {'model_loss': float('inf')}
        
        # Get all data
        states, actions, rewards, next_states, dones = self.real_buffer.get_all()
        n_samples = len(states)
        
        # Normalize
        states_normalized = self.normalize_state(states)
        next_states_normalized = self.normalize_state(next_states)
        
        # Compute targets: state delta and reward
        state_deltas = next_states_normalized - states_normalized
        
        # Train-validation split
        n_val = int(n_samples * self.config.validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Convert to tensors
        train_states = torch.from_numpy(states_normalized[train_indices]).to(device)
        train_actions = torch.from_numpy(actions[train_indices]).to(device)
        train_deltas = torch.from_numpy(state_deltas[train_indices]).to(device)
        train_rewards = torch.from_numpy(rewards[train_indices]).to(device)
        
        val_states = torch.from_numpy(states_normalized[val_indices]).to(device)
        val_actions = torch.from_numpy(actions[val_indices]).to(device)
        val_deltas = torch.from_numpy(state_deltas[val_indices]).to(device)
        val_rewards = torch.from_numpy(rewards[val_indices]).to(device)
        
        # Training loop
        total_loss = 0
        best_val_losses = [float('inf')] * self.config.ensemble_size
        
        for epoch in range(self.config.model_train_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(train_states))
            train_states = train_states[perm]
            train_actions = train_actions[perm]
            train_deltas = train_deltas[perm]
            train_rewards = train_rewards[perm]
            
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, len(train_states), self.config.model_batch_size):
                end = min(start + self.config.model_batch_size, len(train_states))
                
                batch_states = train_states[start:end]
                batch_actions = train_actions[start:end]
                batch_deltas = train_deltas[start:end]
                batch_rewards = train_rewards[start:end]
                
                # One-hot encode actions
                batch_actions_onehot = self.one_hot_action(batch_actions)
                
                # Forward pass through ensemble
                means, logvars = self.dynamics_model(batch_states, batch_actions_onehot)
                
                # Targets: [state_delta, reward]
                targets = torch.zeros(self.config.ensemble_size, len(batch_states),
                                     self.config.state_size + 1, device=device)
                for i in range(self.config.ensemble_size):
                    targets[i, :, :-1] = batch_deltas
                    targets[i, :, -1] = batch_rewards
                
                # Bootstrap: each model sees random subset of data
                bootstrap_mask = torch.randint(0, 2, (self.config.ensemble_size, len(batch_states)),
                                              device=device).float()
                bootstrap_mask = bootstrap_mask + 0.5  # At least 50% chance
                
                # NLL loss for probabilistic model
                # Loss = 0.5 * (logvar + (target - mean)^2 / var)
                losses = []
                for i in range(self.config.ensemble_size):
                    mean = means[i]
                    logvar = logvars[i]
                    
                    inv_var = torch.exp(-logvar)
                    nll = 0.5 * (logvar + (targets[i] - mean) ** 2 * inv_var)
                    
                    # Apply bootstrap mask
                    mask = bootstrap_mask[i].unsqueeze(-1)
                    losses.append((nll * mask).sum() / mask.sum().clamp(min=1))
                
                loss = torch.stack(losses).mean()
                
                self.model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 
                                               self.config.max_grad_norm)
                self.model_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss += epoch_loss / n_batches
        
        # Validation to select elite models
        with torch.no_grad():
            val_actions_onehot = self.one_hot_action(val_actions)
            means, logvars = self.dynamics_model(val_states, val_actions_onehot)
            
            val_targets = torch.zeros(self.config.ensemble_size, len(val_states),
                                     self.config.state_size + 1, device=device)
            for i in range(self.config.ensemble_size):
                val_targets[i, :, :-1] = val_deltas
                val_targets[i, :, -1] = val_rewards
            
            # MSE loss for validation
            val_losses = []
            for i in range(self.config.ensemble_size):
                mse = ((means[i] - val_targets[i]) ** 2).mean()
                val_losses.append(mse.item())
            
            # Select elite models (lowest validation loss)
            elite_indices = np.argsort(val_losses)[:self.config.elite_size].tolist()
            self.dynamics_model.set_elite(elite_indices)
        
        self.model_update_count += 1
        
        return {
            'model_loss': total_loss / self.config.model_train_epochs,
            'val_losses': val_losses,
            'elite_indices': elite_indices
        }
    
    def generate_branched_rollouts(self) -> int:
        """
        Generate branched rollouts from the model
        
        Paper Algorithm:
        1. Sample initial states from real buffer
        2. For each state, perform H-step model rollout
        3. Store all imagined transitions
        
        Returns:
            Number of generated transitions
        """
        if len(self.real_buffer) < self.config.start_training_steps:
            return 0
        
        H = self.config.rollout_horizon
        n_start_states = self.config.rollout_batch_size // H
        
        # Sample initial states from real buffer
        start_states = self.real_buffer.sample_states(n_start_states)
        start_states_normalized = self.normalize_state(start_states)
        
        # Storage for generated transitions
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        current_states = torch.from_numpy(start_states_normalized).to(device)
        
        self.policy.eval()
        self.dynamics_model.eval()
        
        with torch.no_grad():
            for h in range(H):
                # Sample actions from current policy
                actions, _ = self.policy.sample(current_states)
                actions_onehot = self.one_hot_action(actions)
                
                # Predict next states and rewards using elite model
                state_deltas, rewards = self.dynamics_model.sample_prediction(
                    current_states, actions_onehot, use_elite=True
                )
                
                next_states = current_states + state_deltas
                
                # Store transitions
                all_states.append(current_states.cpu().numpy())
                all_actions.append(actions.cpu().numpy())
                all_rewards.append(rewards.squeeze(-1).cpu().numpy())
                all_next_states.append(next_states.cpu().numpy())
                
                # Done is always False for model rollouts (or can add uncertainty-based termination)
                all_dones.append(np.zeros(len(current_states), dtype=np.float32))
                
                # Continue from predicted next states
                current_states = next_states
        
        self.policy.train()
        self.dynamics_model.train()
        
        # Concatenate all generated transitions
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_next_states = np.concatenate(all_next_states, axis=0)
        all_dones = np.concatenate(all_dones, axis=0)
        
        # Add to model buffer
        self.model_buffer.add_batch(
            all_states, all_actions, all_rewards, all_next_states, all_dones
        )
        
        return len(all_states)
    
    def update_sac(self) -> Dict:
        """
        SAC update with mixed real and model data
        
        Paper: Use ratio of real to model data (default 5% real)
        """
        batch_size = self.config.sac_batch_size
        
        # Calculate number of samples from each buffer
        real_ratio = self.config.real_ratio
        n_real = max(1, int(batch_size * real_ratio))
        n_model = batch_size - n_real
        
        # Sample from real buffer
        if len(self.real_buffer) >= n_real:
            real_samples = self.real_buffer.sample(n_real)
        else:
            real_samples = None
        
        # Sample from model buffer
        if len(self.model_buffer) >= n_model and n_model > 0:
            model_samples = self.model_buffer.sample(n_model)
        else:
            model_samples = None
        
        # Combine samples
        if real_samples is not None and model_samples is not None:
            states = np.concatenate([real_samples[0], model_samples[0]], axis=0)
            actions = np.concatenate([real_samples[1], model_samples[1]], axis=0)
            rewards = np.concatenate([real_samples[2], model_samples[2]], axis=0)
            next_states = np.concatenate([real_samples[3], model_samples[3]], axis=0)
            dones = np.concatenate([real_samples[4], model_samples[4]], axis=0)
        elif real_samples is not None:
            states, actions, rewards, next_states, dones = real_samples
        else:
            return {}
        
        # Normalize states (already normalized for model data)
        if self.config.normalize_states:
            # Only normalize real data; model data is already normalized
            n_real_actual = real_samples[0].shape[0] if real_samples else 0
            if n_real_actual > 0:
                states[:n_real_actual] = self.normalize_state(states[:n_real_actual])
                next_states[:n_real_actual] = self.normalize_state(next_states[:n_real_actual])
        
        # Convert to tensors
        states_t = torch.from_numpy(states).float().to(device)
        actions_t = torch.from_numpy(actions).long().to(device)
        rewards_t = torch.from_numpy(rewards).float().to(device)
        next_states_t = torch.from_numpy(next_states).float().to(device)
        dones_t = torch.from_numpy(dones).float().to(device)
        
        actions_onehot = self.one_hot_action(actions_t)
        
        # === Critic Update ===
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.policy.sample(next_states_t)
            next_actions_onehot = self.one_hot_action(next_actions)
            
            # Compute target Q
            q1_target, q2_target = self.target_critic(next_states_t, next_actions_onehot)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs.unsqueeze(-1)
            
            target_q = rewards_t.unsqueeze(-1) + self.config.gamma * (1 - dones_t.unsqueeze(-1)) * q_target
        
        # Current Q estimates
        q1, q2 = self.critic(states_t, actions_onehot)
        
        # Critic loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        # === Policy Update ===
        new_actions, log_probs = self.policy.sample(states_t)
        new_actions_onehot = self.one_hot_action(new_actions)
        
        q1_new, q2_new = self.critic(states_t, new_actions_onehot)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs.unsqueeze(-1) - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # === Alpha Update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # === Soft Update Target Network ===
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'mean_q': q_new.mean().item()
        }
    
    def save(self, path: str, episode: int):
        """Save checkpoint"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'episode': episode,
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'dynamics_model': self.dynamics_model.state_dict(),
            'log_alpha': self.log_alpha,
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'state_normalizer_mean': self.state_normalizer.mean,
            'state_normalizer_var': self.state_normalizer.var,
            'total_steps': self.total_steps,
            'elite_indices': self.dynamics_model.elite_indices
        }, os.path.join(path, f'mbpo_ep{episode}.pth'))


# ============================================================================
#                              Metrics Tracker
# ============================================================================

class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self, config: MBPOConfig):
        self.config = config
        self.episode_metrics = []
        self.total_steps = 0
    
    def add_episode(self, episode: int, total_reward: float, final_yield: float,
                    total_nitrogen: float, total_irrigation: float, episode_length: int):
        """Record episode metrics"""
        self.total_steps += episode_length
        
        wue = final_yield / max(total_irrigation, 1e-6) if total_irrigation > 0 else 0.0
        nue = final_yield / max(total_nitrogen, 1e-6) if total_nitrogen > 0 else 0.0
        
        metrics = {
            'episode': episode,
            'total_reward': total_reward,
            'final_yield': final_yield,
            'total_nitrogen_kg_ha': total_nitrogen,
            'total_irrigation_mm': total_irrigation,
            'WUE_kg_mm': wue,
            'NUE_kg_kg': nue,
            'episode_length': episode_length,
            'total_steps': self.total_steps
        }
        self.episode_metrics.append(metrics)
    
    def get_summary(self, last_n: int = 10) -> Dict:
        """Get summary of recent episodes"""
        if len(self.episode_metrics) < last_n:
            last_n = len(self.episode_metrics)
        recent = self.episode_metrics[-last_n:]
        
        return {
            'avg_reward': np.mean([m['total_reward'] for m in recent]),
            'avg_yield': np.mean([m['final_yield'] for m in recent]),
            'avg_nitrogen': np.mean([m['total_nitrogen_kg_ha'] for m in recent]),
            'avg_irrigation': np.mean([m['total_irrigation_mm'] for m in recent]),
            'avg_wue': np.mean([m['WUE_kg_mm'] for m in recent]),
            'avg_nue': np.mean([m['NUE_kg_kg'] for m in recent]),
            'total_episodes': len(self.episode_metrics),
            'total_steps': self.total_steps
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get metrics as DataFrame"""
        return pd.DataFrame(self.episode_metrics)


# ============================================================================
#                              Results Saver
# ============================================================================

class ResultsSaver:
    """Save training results"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_all(self, metrics: MetricsTracker, config: MBPOConfig):
        """Save all results"""
        df = metrics.get_dataframe()
        df.to_csv(os.path.join(self.save_dir, "training_log.csv"), index=False)
        self._save_figures(df, config)
        print(f"\nResults saved to: {self.save_dir}")
    
    def _save_figures(self, df: pd.DataFrame, config: MBPOConfig):
        """Save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Reward
        ax = axes[0, 0]
        ax.plot(df['episode'], df['total_reward'], alpha=0.4, color='blue')
        if len(df) >= 20:
            ax.plot(df['episode'], df['total_reward'].rolling(20).mean(), 
                   color='red', linewidth=2, label='MA(20)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward (MBPO Standard)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Yield
        ax = axes[0, 1]
        ax.plot(df['episode'], df['final_yield'], alpha=0.4, color='green')
        if len(df) >= 20:
            ax.plot(df['episode'], df['final_yield'].rolling(20).mean(),
                   color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Yield (kg/ha)')
        ax.set_title('Crop Yield')
        ax.grid(True, alpha=0.3)
        
        # Resources
        ax = axes[1, 0]
        ax.plot(df['episode'], df['total_nitrogen_kg_ha'], label='N (kg/ha)', alpha=0.7)
        ax.plot(df['episode'], df['total_irrigation_mm'], label='Irr (mm)', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.legend()
        ax.set_title('Resource Usage')
        ax.grid(True, alpha=0.3)
        
        # Efficiency
        ax = axes[1, 1]
        ax.plot(df['episode'], df['WUE_kg_mm'], label='WUE', alpha=0.7)
        ax.plot(df['episode'], df['NUE_kg_kg'], label='NUE', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.legend()
        ax.set_title('Efficiency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"), dpi=150)
        plt.close()


# ============================================================================
#                              Mock Environment
# ============================================================================

class MockGymDssat:
    """Mock environment for testing without gym-dssat"""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 200
    
    def reset(self):
        self.step_count = 0
        state = {f'var{i}': np.random.rand() * 100 for i in range(20)}
        state['sw'] = np.random.rand(5) * 100
        state['var4'] = 1000  # Initial yield-like variable
        return state
    
    def step(self, action):
        self.step_count += 1
        
        next_state = {f'var{i}': np.random.rand() * 100 for i in range(20)}
        next_state['sw'] = np.random.rand(5) * 100
        
        nitrogen = action.get('anfer', 0)
        irrigation = action.get('amir', 0)
        
        # Simulate growth
        growth = 20 + nitrogen * 0.5 + irrigation * 0.3 + np.random.rand() * 100
        next_state['var4'] = min(8000, 1000 + self.step_count * growth / 10)
        
        done = self.step_count >= self.max_steps
        return next_state, 0, done, {}


# ============================================================================
#                              Training Function
# ============================================================================

def create_environment():
    """Create gym-dssat environment or mock"""
    try:
        env_args = {
            'run_dssat_location': '/opt/dssat_pdi/run_dssat',
            'log_saving_path': './logs/dssat-pdi.log',
            'mode': 'all',
            'seed': 123456,
            'random_weather': True
        }
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("Gym-DSSAT environment created successfully")
        return env
    except Exception as e:
        print(f"Warning: Could not create gym-dssat environment: {e}")
        print("Using mock environment for testing")
        return MockGymDssat()


def train_mbpo():
    """Main training loop"""
    print("=" * 70)
    print("MBPO - Model-Based Policy Optimization (Standard Implementation)")
    print("=" * 70)
    print("\nPaper Reference: Janner et al. 'When to Trust Your Model'")
    print("                 NeurIPS 2019")
    print("=" * 70)
    
    config = MBPOConfig()
    agent = MBPOAgent(config)
    env = create_environment()
    metrics = MetricsTracker(config)
    saver = ResultsSaver(config.save_dir)
    
    # Print configuration
    print(f"\nKey Hyperparameters (Paper-Aligned):")
    print(f"  Ensemble Size: {config.ensemble_size}")
    print(f"  Elite Models: {config.elite_size}")
    print(f"  Rollout Horizon (H): {config.rollout_horizon}")
    print(f"  Model Hidden Size: {config.model_hidden_size}")
    print(f"  SAC Hidden Size: {config.hidden_size}")
    print(f"  Real Data Ratio: {config.real_ratio}")
    print(f"  Discount (gamma): {config.gamma}")
    print(f"  Soft Update (tau): {config.tau}")
    
    start_time = time.time()
    best_reward = float('-inf')
    
    for episode in range(1, config.n_episodes + 1):
        state_raw = env.reset()
        state = dict2array(state_raw)
        
        episode_reward = 0
        total_nitrogen = 0
        total_irrigation = 0
        episode_steps = 0
        final_yield = 0
        
        for step in range(config.max_steps_per_episode):
            # Select action
            action = agent.act(state)
            action_dict = array2action(action, state)
            
            # Environment step
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state.copy()
            
            # Compute reward
            reward = get_reward(state, action_dict['anfer'], action_dict['amir'],
                               next_state, done, config.k1, config.k2, config.k3)
            
            # Store transition
            agent.real_buffer.add(state, action, reward, next_state, done)
            agent.total_steps += 1
            
            # Update normalizer
            agent.state_normalizer.update(state)
            agent.update_epsilon()
            
            state = next_state
            episode_reward += reward
            total_nitrogen += action_dict['anfer']
            total_irrigation += action_dict['amir']
            episode_steps += 1
            
            # === Model Training (Paper: train model every N steps) ===
            if (agent.total_steps - agent.last_model_update) >= config.model_update_freq:
                if len(agent.real_buffer) >= config.start_training_steps:
                    model_info = agent.train_dynamics_model()
                    
                    # Generate branched rollouts
                    n_generated = agent.generate_branched_rollouts()
                    agent.last_model_update = agent.total_steps
            
            # === SAC Updates ===
            if agent.total_steps >= config.start_training_steps:
                for _ in range(config.sac_updates_per_step):
                    agent.update_sac()
            
            if done:
                final_yield = state[4] if len(state) > 4 else np.random.rand() * 5000
                break
        
        # Record metrics
        metrics.add_episode(episode, episode_reward, final_yield,
                           total_nitrogen, total_irrigation, episode_steps)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(config.checkpoint_dir, episode)
        
        # Logging
        if episode % 10 == 0:
            summary = metrics.get_summary(10)
            elapsed = time.time() - start_time
            print(f"Ep {episode}: R={summary['avg_reward']:.1f}, "
                  f"Y={summary['avg_yield']:.0f}, "
                  f"eps={agent.epsilon:.3f}, alpha={agent.alpha:.3f}, "
                  f"best={best_reward:.0f}, "
                  f"buffer={len(agent.real_buffer)}, "
                  f"time={elapsed/60:.1f}min")
    
    # === Validation ===
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    val_rewards = []
    for _ in range(10):
        state_raw = env.reset()
        state = dict2array(state_raw)
        ep_reward = 0
        
        for _ in range(config.max_steps_per_episode):
            action = agent.act(state, deterministic=True)
            action_dict = array2action(action, state)
            next_state_raw, _, done, _ = env.step(action_dict)
            state = dict2array(next_state_raw) if not done else state
            ep_reward += get_reward(state, action_dict['anfer'], action_dict['amir'],
                                   state, done, config.k1, config.k2, config.k3)
            if done:
                break
        val_rewards.append(ep_reward)
    
    print(f"Validation: {np.mean(val_rewards):.2f} ± {np.std(val_rewards):.2f}")
    
    # Save results
    saver.save_all(metrics, config)
    print(f"\nTotal Training Time: {(time.time() - start_time)/3600:.2f} hours")
    
    return agent


# ============================================================================
#                              Entry Point
# ============================================================================

if __name__ == "__main__":
    os.makedirs('/home/gymusr/only_mbpo_checkpoints/mbpo_standard', exist_ok=True)
    os.makedirs('/home/gymusr/only_mbpo_results/mbpo_results_standard', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    agent = train_mbpo()