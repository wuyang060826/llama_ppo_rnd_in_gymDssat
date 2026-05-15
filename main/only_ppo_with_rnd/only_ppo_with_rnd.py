#!/usr/bin/env python3
"""
Pure PPO with RND Baseline for Gym-DSSAT Crop Management
=========================================================
A strong baseline for agricultural reinforcement learning research.

Features:
- Pure PPO implementation (no LLM, numerical state input only)
- Standard RND (Random Network Distillation) for exploration
- Comprehensive agricultural and AI metrics tracking
- Optimized for A100 40G GPU

References:
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Burda et al. (2018) "Exploration by Random Network Distillation"
- Schulman et al. (2016) "High-Dimensional Continuous Control Using GAE"
"""

import numpy as np
import pandas as pd
import random
from collections import deque, OrderedDict
import time
import math
import json
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

import gym
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

# Memory optimization
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
#                              Device Configuration
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("PPO + RND Baseline - Gym-DSSAT Crop Management")
print("=" * 70)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32 Acceleration: Enabled")


# ============================================================================
#                              Configuration
# ============================================================================

@dataclass
class PPORNDConfig:
    """PPO + RND Hyperparameter Configuration"""
    
    # Training parameters
    n_episodes: int = 1000
    max_steps_per_episode: int = 200
    
    # PPO core parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Optimizer parameters
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    
    # PPO update parameters
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    update_frequency: int = 10
    
    # Network parameters
    state_size: int = 25
    action_size: int = 25
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    
    # Reward function parameters
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # Exploration parameters
    entropy_decay: float = 0.999
    entropy_min: float = 0.001
    
    # RND parameters
    rnd_lr: float = 1e-4
    rnd_hidden_size: int = 256
    rnd_output_size: int = 128
    rnd_update_proportion: float = 0.25
    intrinsic_reward_weight: float = 0.01
    intrinsic_reward_scale: float = 1.0
    intrinsic_gamma: float = 0.99
    
    # Optimization options
    use_bf16: bool = True
    
    # Evaluation parameters
    eval_frequency: int = 100
    n_eval_episodes: int = 10
    
    # Logging parameters
    log_frequency: int = 10
    save_frequency: int = 200
    
    # Expert threshold for sample efficiency
    expert_performance_threshold: float = 8000.0
    
    # Convergence parameters
    convergence_window: int = 100
    convergence_threshold: float = 0.1
    
    # Output directory
    output_dir: str = '/home/gymusr/only_ppo_with_rnd_results'


config = PPORNDConfig()


# ============================================================================
#                              Helper Functions
# ============================================================================

def dict2array(state: dict) -> np.ndarray:
    """Convert gym-dssat dictionary state to numpy array"""
    if state is None:
        raise ValueError("State cannot be None")
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)


def get_reward(state: np.ndarray, n_action: float, w_action: float, 
               next_state: np.ndarray, done: bool, 
               k1: float, k2: float, k3: float) -> float:
    """Calculate reward value"""
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action
    return -k2 * n_action - k3 * w_action


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def state_hash(state: np.ndarray) -> int:
    """Generate unique hash for state (for exploration coverage)"""
    discretized = (state / 10).astype(int)
    return hash(discretized.tobytes())


# ============================================================================
#                           Learning Rate Scheduler
# ============================================================================

class CosineAnnealingWarmup:
    """Cosine annealing learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step < self.warmup_steps:
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 
                               (1 + math.cos(math.pi * progress)) / 2)
            group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
#                              Network Definitions
# ============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network with separate critics for extrinsic and intrinsic rewards"""
    
    def __init__(self, state_size: int, action_size: int, 
                 hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        
        # Shared feature extraction layers
        shared_layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes[:-1]:
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            shared_layers.extend([
                linear,
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)
        
        # Critic head for extrinsic rewards
        self.critic_ext = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        nn.init.orthogonal_(self.critic_ext[-1].weight, gain=1.0)
        nn.init.constant_(self.critic_ext[-1].bias, 0.0)
        
        # Critic head for intrinsic rewards (RND)
        self.critic_int = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        nn.init.orthogonal_(self.critic_int[-1].weight, gain=1.0)
        nn.init.constant_(self.critic_int[-1].bias, 0.0)
        
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value_ext = self.critic_ext(features)
        value_int = self.critic_int(features)
        return logits, value_ext, value_int
    
    def get_action_value(self, x):
        logits, value_ext, value_int = self.forward(x)
        dist = Categorical(logits=logits)
        return dist, value_ext, value_int
    
    def get_value(self, x):
        """Get value estimates for given states"""
        features = self.shared(x)
        value_ext = self.critic_ext(features)
        value_int = self.critic_int(features)
        return value_ext, value_int


# ============================================================================
#                              RND Network
# ============================================================================

class RNDNetwork(nn.Module):
    """Random Network Distillation (RND) for exploration
    
    Reference: Burda et al. (2018) "Exploration by Random Network Distillation"
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 output_size: int = 128):
        super().__init__()
        
        # Target network (fixed, not trained)
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False
        for layer in self.target:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_output = self.target(x)
        predict_output = self.predictor(x)
        return target_output, predict_output
    
    def get_intrinsic_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate intrinsic reward (prediction error)"""
        with torch.no_grad():
            target_output = self.target(x)
            predict_output = self.predictor(x)
            error = F.mse_loss(predict_output, target_output, reduction='none').mean(dim=-1)
        return error


class RunningMeanStd:
    """Running mean and standard deviation for RND reward normalization"""
    
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.epsilon = epsilon
        
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
        
    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -clip, clip)


# ============================================================================
#                              PPO Buffer
# ============================================================================

class RolloutBuffer:
    """PPO experience buffer with support for intrinsic rewards"""
    
    def __init__(self, gamma: float, gae_lambda: float, device: torch.device,
                 intrinsic_gamma: float = 0.99):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.intrinsic_gamma = intrinsic_gamma
        self.device = device
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values_ext = []
        self.values_int = []
        self.n_actions = []
        self.w_actions = []
        
        self.reward_normalizer = RunningMeanStd()
        self.int_reward_normalizer = RunningMeanStd()
        
    def add(self, state, action, reward, intrinsic_reward, next_state, done, 
            log_prob, value_ext, value_int, n_action, w_action):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.intrinsic_rewards.append(intrinsic_reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values_ext.append(value_ext)
        self.values_int.append(value_int)
        self.n_actions.append(n_action)
        self.w_actions.append(w_action)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.intrinsic_rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values_ext.clear()
        self.values_int.clear()
        self.n_actions.clear()
        self.w_actions.clear()
    
    def __len__(self):
        return len(self.states)
    
    def compute_gae(self, last_value_ext: float, last_value_int: float):
        """Compute GAE for both extrinsic and intrinsic rewards"""
        n = len(self.states)
        
        # Normalize rewards
        rewards = np.array(self.rewards)
        int_rewards = np.array(self.intrinsic_rewards)
        
        self.reward_normalizer.update(rewards)
        self.int_reward_normalizer.update(int_rewards)
        
        norm_rewards = self.reward_normalizer.normalize(rewards)
        norm_int_rewards = self.int_reward_normalizer.normalize(int_rewards)
        
        # Convert to tensors
        rewards_t = torch.tensor(norm_rewards, dtype=torch.float32, device=self.device)
        int_rewards_t = torch.tensor(norm_int_rewards, dtype=torch.float32, device=self.device)
        values_ext_t = torch.tensor(self.values_ext, dtype=torch.float32, device=self.device)
        values_int_t = torch.tensor(self.values_int, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        # Compute extrinsic GAE
        advantages_ext = torch.zeros(n, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0 if dones_t[t] else last_value_ext
            else:
                next_value = values_ext_t[t + 1]
            next_non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_value * next_non_terminal - values_ext_t[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages_ext[t] = gae
        
        # Compute intrinsic GAE
        advantages_int = torch.zeros(n, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0 if dones_t[t] else last_value_int
            else:
                next_value = values_int_t[t + 1]
            next_non_terminal = 1.0 - dones_t[t]
            delta = int_rewards_t[t] + self.intrinsic_gamma * next_value * next_non_terminal - values_int_t[t]
            gae = delta + self.intrinsic_gamma * self.gae_lambda * next_non_terminal * gae
            advantages_int[t] = gae
        
        # Compute returns
        returns_ext = advantages_ext + values_ext_t
        returns_int = advantages_int + values_int_t
        
        # Normalize advantages
        advantages_ext = (advantages_ext - advantages_ext.mean()) / (advantages_ext.std() + 1e-8)
        advantages_int = (advantages_int - advantages_int.mean()) / (advantages_int.std() + 1e-8)
        
        return advantages_ext, advantages_int, returns_ext, returns_int
    
    def get_mini_batches(self, batch_size: int) -> List[np.ndarray]:
        indices = np.random.permutation(len(self.states))
        batches = []
        for start in range(0, len(self.states), batch_size):
            batches.append(indices[start:start + batch_size])
        return batches


# ============================================================================
#                              Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Calculator for agricultural and AI metrics"""
    
    def __init__(self, config: PPORNDConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        # Agricultural metrics
        self.yields = []
        self.total_irrigation = []
        self.total_fertilizer = []
        self.wue_list = []
        self.nue_list = []
        
        # AI metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.intrinsic_rewards = []
        
        # Time tracking
        self.start_time = time.time()
        self.episode_times = []
        
        # Convergence tracking
        self.best_avg_reward = float('-inf')
        self.convergence_episode = None
        self.convergence_step = None
        self.rewards_window = deque(maxlen=self.config.convergence_window)
        
        # Sample efficiency tracking
        self.total_steps = 0
        self.sample_efficiency_step = None
        self.expert_threshold_reached = False
        
        # Exploration coverage tracking
        self.visited_states = set()
        self.rnd_errors = []
        self.exploration_coverage = 0.0
        
        # Episode details for saving
        self.episode_details = []
        
    def add_episode(self, episode: int, reward: float, yield_val: float, 
                    irrigation: float, fertilizer: float,
                    episode_length: int, intrinsic_reward_sum: float,
                    episode_time: float):
        """Add episode data"""
        self.episode_rewards.append(reward)
        self.yields.append(yield_val)
        self.total_irrigation.append(irrigation)
        self.total_fertilizer.append(fertilizer)
        self.episode_lengths.append(episode_length)
        self.intrinsic_rewards.append(intrinsic_reward_sum)
        self.episode_times.append(episode_time)
        
        # Calculate WUE and NUE
        if irrigation > 0:
            wue = yield_val / irrigation
            self.wue_list.append(wue)
        if fertilizer > 0:
            nue = yield_val / fertilizer
            self.nue_list.append(nue)
        
        # Update convergence tracking
        self.rewards_window.append(reward)
        current_avg = np.mean(list(self.rewards_window))
        
        if current_avg > self.best_avg_reward:
            self.best_avg_reward = current_avg
        
        # Check convergence
        if self.convergence_episode is None and len(self.rewards_window) >= self.config.convergence_window:
            window_std = np.std(list(self.rewards_window))
            window_mean = np.mean(list(self.rewards_window))
            if window_std < self.config.convergence_threshold * abs(window_mean) and window_mean > 0:
                self.convergence_episode = episode
                self.convergence_step = self.total_steps
        
        # Check sample efficiency (expert level)
        if not self.expert_threshold_reached and yield_val >= self.config.expert_performance_threshold:
            self.sample_efficiency_step = self.total_steps
            self.expert_threshold_reached = True
        
        # Store episode details
        detail = {
            'episode': episode,
            'reward': reward,
            'yield_kg_ha': yield_val,
            'irrigation_mm': irrigation,
            'fertilizer_kg_ha': fertilizer,
            'wue_kg_mm': self.wue_list[-1] if self.wue_list else 0,
            'nue_kg_kg': self.nue_list[-1] if self.nue_list else 0,
            'episode_length': episode_length,
            'intrinsic_reward': intrinsic_reward_sum,
            'time_s': episode_time,
            'total_steps': self.total_steps
        }
        self.episode_details.append(detail)
        
    def add_update(self, policy_loss: float, value_loss: float, entropy: float):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        
    def add_step(self, state: np.ndarray, rnd_error: float):
        """Add step data for exploration tracking"""
        self.total_steps += 1
        state_key = state_hash(state)
        self.visited_states.add(state_key)
        self.rnd_errors.append(rnd_error)
        
    def update_exploration_coverage(self):
        """Update exploration coverage based on RND error decay"""
        if len(self.rnd_errors) < 100:
            return
        # States with low RND error are considered "explored"
        recent_errors = self.rnd_errors[-1000:]
        threshold = np.percentile(recent_errors, 20)
        explored_count = sum(1 for e in recent_errors if e < threshold)
        self.exploration_coverage = min(1.0, explored_count / 100.0)
        
    def get_agricultural_metrics(self, last_n: int = None) -> Dict[str, Dict[str, float]]:
        """Get agricultural metrics"""
        def get_stats(data):
            if not data:
                return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
            arr = np.array(data)
            return {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'max': float(arr.max()),
                'min': float(arr.min())
            }
        
        if last_n:
            yields = self.yields[-last_n:]
            irrigation = self.total_irrigation[-last_n:]
            fertilizer = self.total_fertilizer[-last_n:]
            wue = self.wue_list[-last_n:] if len(self.wue_list) >= last_n else self.wue_list
            nue = self.nue_list[-last_n:] if len(self.nue_list) >= last_n else self.nue_list
        else:
            yields = self.yields
            irrigation = self.total_irrigation
            fertilizer = self.total_fertilizer
            wue = self.wue_list
            nue = self.nue_list
        
        return {
            'yield_kg_ha': get_stats(yields),
            'irrigation_mm': get_stats(irrigation),
            'fertilizer_kg_ha': get_stats(fertilizer),
            'WUE_kg_mm': get_stats(wue),
            'NUE_kg_kg': get_stats(nue)
        }
    
    def get_ai_metrics(self) -> Dict[str, Any]:
        """Get AI metrics"""
        return {
            'avg_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'best_avg_reward': float(self.best_avg_reward),
            'total_steps': self.total_steps,
            'sample_efficiency_step': self.sample_efficiency_step,
            'convergence_episode': self.convergence_episode,
            'convergence_step': self.convergence_step,
            'training_time_s': time.time() - self.start_time,
            'avg_episode_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            'exploration_coverage_pct': self.exploration_coverage * 100,
            'visited_states_count': len(self.visited_states),
            'avg_intrinsic_reward': float(np.mean(self.intrinsic_rewards)) if self.intrinsic_rewards else 0
        }
    
    def get_summary(self, last_n: int = 10) -> str:
        """Get summary string"""
        agri = self.get_agricultural_metrics(last_n)
        ai = self.get_ai_metrics()
        
        summary = f"""
================================================================================
                    Training Metrics Summary (Last {last_n} Episodes)
================================================================================
Agricultural Metrics:
  Yield (kg/ha):        {agri['yield_kg_ha']['mean']:.1f} +/- {agri['yield_kg_ha']['std']:.1f}
  Irrigation (mm):      {agri['irrigation_mm']['mean']:.1f} +/- {agri['irrigation_mm']['std']:.1f}
  Fertilizer (kg/ha):   {agri['fertilizer_kg_ha']['mean']:.1f} +/- {agri['fertilizer_kg_ha']['std']:.1f}
  WUE (kg/mm):          {agri['WUE_kg_mm']['mean']:.2f} +/- {agri['WUE_kg_mm']['std']:.2f}
  NUE (kg/kg):          {agri['NUE_kg_kg']['mean']:.2f} +/- {agri['NUE_kg_kg']['std']:.2f}
--------------------------------------------------------------------------------
AI Metrics:
  Average Reward:       {ai['avg_reward']:.2f}
  Best Average Reward:  {ai['best_avg_reward']:.2f}
  Total Steps:          {ai['total_steps']}
  Sample Efficiency:    {ai['sample_efficiency_step'] if ai['sample_efficiency_step'] else 'Not reached'}
  Convergence Episode:  {ai['convergence_episode'] if ai['convergence_episode'] else 'Not converged'}
  Exploration Coverage: {ai['exploration_coverage_pct']:.1f}%
  Training Time:        {ai['training_time_s']:.1f}s
================================================================================
"""
        return summary


# ============================================================================
#                              PPO+RND Agent
# ============================================================================

class PPORNDAgent:
    """PPO Agent with RND exploration"""
    
    def __init__(self, state_size: int, action_size: int, config: PPORNDConfig):
        self.config = config
        self.device = device
        
        # Actor-Critic network
        self.network = ActorCriticNetwork(
            state_size, action_size, config.hidden_sizes
        ).to(self.device)
        
        # RND network
        self.rnd = RNDNetwork(
            state_size, config.rnd_hidden_size, config.rnd_output_size
        ).to(self.device)
        
        # Optimizer for actor-critic
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Optimizer for RND predictor
        self.rnd_optimizer = optim.Adam(
            self.rnd.predictor.parameters(),
            lr=config.rnd_lr
        )
        
        # Learning rate scheduler
        if config.use_lr_schedule:
            total_steps = config.n_episodes * config.max_steps_per_episode // config.update_frequency
            self.lr_scheduler = CosineAnnealingWarmup(
                self.optimizer, config.warmup_steps, total_steps, config.min_lr_ratio
            )
        else:
            self.lr_scheduler = None
        
        # Buffer
        self.buffer = RolloutBuffer(config.gamma, config.gae_lambda, self.device)
        
        # Metrics
        self.metrics = MetricsCalculator(config)
        
        # Entropy coefficient decay
        self.current_entropy_coef = config.entropy_coef
        
        # State normalization
        self.state_mean = np.zeros(state_size)
        self.state_std = np.ones(state_size)
        self.state_count = 0
        
    def normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize state"""
        if update_stats:
            self.state_count += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.state_count
            delta2 = state - self.state_mean
            self.state_std = np.sqrt(
                (self.state_std**2 * (self.state_count - 1) + delta * delta2) / self.state_count + 1e-8
            )
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float, float]:
        """Select action"""
        self.network.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                dist, value_ext, value_int = self.network.get_action_value(state_t)
            
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value_ext.item(), value_int.item()
    
    def get_intrinsic_reward(self, state: np.ndarray) -> float:
        """Get intrinsic reward from RND"""
        self.rnd.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            error = self.rnd.get_intrinsic_reward(state_t)
        return error.item() * self.config.intrinsic_reward_scale
    
    def update_rnd(self, states: torch.Tensor) -> float:
        """Update RND predictor network"""
        self.rnd.train()
        
        # Only use a portion of states for RND update
        n_samples = int(len(states) * self.config.rnd_update_proportion)
        indices = np.random.choice(len(states), n_samples, replace=False)
        batch_states = states[indices]
        
        target_output, predict_output = self.rnd(batch_states)
        rnd_loss = F.mse_loss(predict_output, target_output)
        
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd.predictor.parameters(), self.config.max_grad_norm)
        self.rnd_optimizer.step()
        
        return rnd_loss.item()
    
    def update(self, last_value_ext: float, last_value_int: float) -> Dict[str, float]:
        """Execute PPO update"""
        self.network.train()
        
        # Compute GAE
        advantages_ext, advantages_int, returns_ext, returns_int = \
            self.buffer.compute_gae(last_value_ext, last_value_int)
        
        # Prepare data
        all_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        all_actions = torch.LongTensor(self.buffer.actions).to(self.device)
        all_old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # PPO iterations
        for epoch in range(self.config.ppo_epochs):
            mini_batches = self.buffer.get_mini_batches(self.config.mini_batch_size)
            
            for indices in mini_batches:
                self.optimizer.zero_grad()
                
                batch_states = all_states[indices]
                batch_actions = all_actions[indices]
                batch_old_log_probs = all_old_log_probs[indices]
                batch_advantages_ext = advantages_ext[indices]
                batch_advantages_int = advantages_int[indices]
                batch_returns_ext = returns_ext[indices]
                batch_returns_int = returns_int[indices]
                
                # Combine advantages
                batch_advantages = batch_advantages_ext + self.config.intrinsic_reward_weight * batch_advantages_int
                
                with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                    dist, new_values_ext, new_values_int = self.network.get_action_value(batch_states)
                    
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Policy ratio
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # PPO clipped objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 
                                       1 + self.config.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value losses
                    value_loss_ext = F.mse_loss(new_values_ext.squeeze(), batch_returns_ext)
                    value_loss_int = F.mse_loss(new_values_int.squeeze(), batch_returns_int)
                    value_loss = value_loss_ext + value_loss_int
                    
                    # Total loss
                    loss = (policy_loss + 
                           self.config.value_coef * value_loss - 
                           self.current_entropy_coef * entropy)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Update RND
        rnd_loss = self.update_rnd(all_states)
        
        # Clear buffer
        self.buffer.clear()
        
        # Entropy decay
        self.current_entropy_coef = max(
            self.config.entropy_min,
            self.current_entropy_coef * self.config.entropy_decay
        )
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'rnd_loss': rnd_loss,
            'entropy_coef': self.current_entropy_coef
        }
    
    def save(self, path: str, episode: int, extra_info: Dict = None):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        save_dict = {
            'episode': episode,
            'network': self.network.state_dict(),
            'rnd_predictor': self.rnd.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rnd_optimizer': self.rnd_optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count,
            'current_entropy_coef': self.current_entropy_coef,
            'config': self.config.__dict__,
        }
        if extra_info:
            save_dict.update(extra_info)
        torch.save(save_dict, os.path.join(path, f'model_ep{episode}.pth'))
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.rnd.predictor.load_state_dict(checkpoint['rnd_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer'])
        self.state_mean = checkpoint.get('state_mean', np.zeros(self.config.state_size))
        self.state_std = checkpoint.get('state_std', np.ones(self.config.state_size))
        self.state_count = checkpoint.get('state_count', 0)
        self.current_entropy_coef = checkpoint.get('current_entropy_coef', self.config.entropy_coef)
        return checkpoint


# ============================================================================
#                              Action Mapping
# ============================================================================

def action_to_dict(action: int, state: np.ndarray) -> Dict[str, float]:
    """Convert discrete action to gym-dssat action dictionary
    
    Action space: 25 discrete actions (5 nitrogen levels x 5 irrigation levels)
    - Nitrogen: 0, 40, 80, 120, 160 kg/ha
    - Irrigation: 0, 6, 12, 18, 24 mm
    """
    n_level = action % 5
    w_level = action // 5
    
    anfer = n_level * 40
    amir = w_level * 6
    
    # Constraints
    if state[0] >= 10000:
        anfer = 0
    if state[21] >= 1600:
        amir = 0
    
    return {'anfer': anfer, 'amir': amir}


# ============================================================================
#                              Result Saver
# ============================================================================

class ResultSaver:
    """Save results in multiple formats (PNG, PDF, Excel)"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
    def save_episode_results(self, metrics: MetricsCalculator, episode: int, 
                            is_final: bool = False):
        """Save results for current episode"""
        if not metrics.episode_details:
            return
            
        # Save detailed episode data to Excel
        df = pd.DataFrame(metrics.episode_details)
        
        if is_final:
            excel_path = os.path.join(self.output_dir, 'data', 'all_episodes.xlsx')
        else:
            excel_path = os.path.join(self.output_dir, 'data', f'episodes_up_to_{episode}.xlsx')
        
        df.to_excel(excel_path, index=False)
        
        # Save summary metrics
        summary_data = {
            'Agricultural Metrics': metrics.get_agricultural_metrics(),
            'AI Metrics': metrics.get_ai_metrics()
        }
        
        with open(os.path.join(self.output_dir, 'data', 'metrics_summary.json'), 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
    
    def save_plots(self, metrics: MetricsCalculator, episode: int, 
                   is_final: bool = False):
        """Generate and save plots"""
        if not metrics.episode_details:
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        episodes = [d['episode'] for d in metrics.episode_details]
        
        # Plot 1: Yield
        ax = axes[0, 0]
        ax.plot(episodes, [d['yield_kg_ha'] for d in metrics.episode_details], 'b-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Yield (kg/ha)')
        ax.set_title('Final Yield')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reward
        ax = axes[0, 1]
        ax.plot(episodes, [d['reward'] for d in metrics.episode_details], 'g-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Irrigation
        ax = axes[0, 2]
        ax.plot(episodes, [d['irrigation_mm'] for d in metrics.episode_details], 'c-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Irrigation (mm)')
        ax.set_title('Total Irrigation')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Fertilizer
        ax = axes[1, 0]
        ax.plot(episodes, [d['fertilizer_kg_ha'] for d in metrics.episode_details], 'm-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fertilizer (kg/ha)')
        ax.set_title('Total Fertilizer')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: WUE
        ax = axes[1, 1]
        ax.plot(episodes, [d['wue_kg_mm'] for d in metrics.episode_details], 'orange', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('WUE (kg/mm)')
        ax.set_title('Water Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: NUE
        ax = axes[1, 2]
        ax.plot(episodes, [d['nue_kg_kg'] for d in metrics.episode_details], 'purple', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('NUE (kg/kg)')
        ax.set_title('Nitrogen Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Intrinsic Reward
        ax = axes[2, 0]
        ax.plot(episodes, [d['intrinsic_reward'] for d in metrics.episode_details], 'r-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Intrinsic Reward')
        ax.set_title('RND Intrinsic Reward')
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Episode Length
        ax = axes[2, 1]
        ax.plot(episodes, [d['episode_length'] for d in metrics.episode_details], 'brown', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Exploration Coverage (moving window)
        ax = axes[2, 2]
        coverage_data = []
        for i, d in enumerate(metrics.episode_details):
            if i >= 10:
                coverage = len(set(state_hash(np.zeros(25)) for _ in range(i))) / max(1, i) * 100
            coverage_data.append(min(100, (i + 1) * 0.5))
        ax.plot(episodes, coverage_data, 'teal', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Exploration Coverage (estimated)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save PNG
        if is_final:
            png_path = os.path.join(self.output_dir, 'figures', 'training_curves_final.png')
        else:
            png_path = os.path.join(self.output_dir, 'figures', f'training_curves_ep{episode}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        
        # Save PDF
        if is_final:
            pdf_path = os.path.join(self.output_dir, 'figures', 'training_curves_final.pdf')
        else:
            pdf_path = os.path.join(self.output_dir, 'figures', f'training_curves_ep{episode}.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close()
        
    def save_final_results(self, metrics: MetricsCalculator, agent: PPORNDAgent):
        """Save all final results"""
        # Save all episode details
        self.save_episode_results(metrics, metrics.episode_details[-1]['episode'], is_final=True)
        
        # Save all plots
        self.save_plots(metrics, metrics.episode_details[-1]['episode'], is_final=True)
        
        # Save detailed metrics summary to Excel
        agri_metrics = metrics.get_agricultural_metrics()
        ai_metrics = metrics.get_ai_metrics()
        
        # Create summary DataFrame
        summary_rows = []
        
        # Agricultural metrics
        for key, stats in agri_metrics.items():
            for stat_name, value in stats.items():
                summary_rows.append({
                    'Category': 'Agricultural',
                    'Metric': key,
                    'Statistic': stat_name,
                    'Value': value
                })
        
        # AI metrics
        for key, value in ai_metrics.items():
            summary_rows.append({
                'Category': 'AI',
                'Metric': key,
                'Statistic': 'value',
                'Value': value
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(os.path.join(self.output_dir, 'data', 'final_summary.xlsx'), index=False)
        
        # Save training curves as multi-page PDF
        with PdfPages(os.path.join(self.output_dir, 'figures', 'all_training_curves.pdf')) as pdf:
            # Page 1: Main metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            episodes = [d['episode'] for d in metrics.episode_details]
            
            axes[0, 0].plot(episodes, [d['yield_kg_ha'] for d in metrics.episode_details], 'b-')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Yield (kg/ha)')
            axes[0, 0].set_title('Yield Over Training')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(episodes, [d['reward'] for d in metrics.episode_details], 'g-')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Reward Over Training')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(episodes, [d['wue_kg_mm'] for d in metrics.episode_details], 'orange')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('WUE (kg/mm)')
            axes[1, 0].set_title('Water Use Efficiency')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(episodes, [d['nue_kg_kg'] for d in metrics.episode_details], 'purple')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('NUE (kg/kg)')
            axes[1, 1].set_title('Nitrogen Use Efficiency')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Page 2: Resource usage
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].plot(episodes, [d['irrigation_mm'] for d in metrics.episode_details], 'c-')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Irrigation (mm)')
            axes[0, 0].set_title('Total Irrigation')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(episodes, [d['fertilizer_kg_ha'] for d in metrics.episode_details], 'm-')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Fertilizer (kg/ha)')
            axes[0, 1].set_title('Total Fertilizer')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(episodes, [d['intrinsic_reward'] for d in metrics.episode_details], 'r-')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Intrinsic Reward')
            axes[1, 0].set_title('RND Intrinsic Reward')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(episodes, [d['episode_length'] for d in metrics.episode_details], 'brown')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        print(f"\nResults saved to: {self.output_dir}")


# ============================================================================
#                              Training Function
# ============================================================================

def train_ppo_rnd():
    """Main training function for PPO+RND"""
    print("\n" + "=" * 70)
    print("Starting PPO+RND Training")
    print("=" * 70)
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    config.output_dir = '/home/gymusr/only_ppo_with_rnd_results'
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create environment
    print("\nInitializing environment...")
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("   Environment created successfully")
    except Exception as e:
        print(f"   Environment creation failed: {e}")
        print("   Please ensure gym-dssat is properly installed")
        return None
    
    # Get state dimension
    sample_state = dict2array(env.reset())
    state_size = len(sample_state)
    print(f"   State dimension: {state_size}")
    
    # Update config
    config.state_size = state_size
    
    # Create agent
    print("\nCreating PPO+RND agent...")
    agent = PPORNDAgent(state_size, config.action_size, config)
    print(f"   Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"   RND parameters: {sum(p.numel() for p in agent.rnd.parameters()):,}")
    
    # Create result saver
    saver = ResultSaver(config.output_dir)
    
    # Create checkpoint directory
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    
    print(f"\nStarting training ({config.n_episodes} episodes)")
    print("-" * 70)
    
    best_avg_reward = float('-inf')
    episode_pbar = tqdm(range(1, config.n_episodes + 1), desc="Training Progress")
    
    for episode in episode_pbar:
        episode_start_time = time.time()
        
        # Reset environment
        state_raw = dict2array(env.reset())
        state = agent.normalize_state(state_raw, update_stats=True)
        
        # Episode tracking
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_n_amount = 0
        episode_w_amount = 0
        episode_yield = 0
        episode_length = 0
        
        for step in range(config.max_steps_per_episode):
            # Select action
            action, log_prob, value_ext, value_int = agent.act(state)
            
            # Get intrinsic reward
            intrinsic_reward = agent.get_intrinsic_reward(state)
            
            # Convert to environment action
            action_dict = action_to_dict(action, state_raw)
            
            # Execute action
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state_raw = dict2array(next_state_raw) if not done else state_raw
            
            # Calculate extrinsic reward
            reward = get_reward(
                state_raw, action_dict['anfer'], action_dict['amir'],
                next_state_raw, done, config.k1, config.k2, config.k3
            )
            
            # Store in buffer
            agent.buffer.add(
                state, action, reward, intrinsic_reward, 
                agent.normalize_state(next_state_raw, update_stats=False),
                done, log_prob, value_ext, value_int,
                action_dict['anfer'], action_dict['amir']
            )
            
            # Update metrics
            agent.metrics.add_step(state_raw, intrinsic_reward)
            
            # Update state
            state_raw = next_state_raw
            state = agent.normalize_state(state_raw, update_stats=True)
            
            episode_reward += reward
            episode_intrinsic_reward += intrinsic_reward
            episode_n_amount += action_dict['anfer']
            episode_w_amount += action_dict['amir']
            episode_length += 1
            
            if done:
                episode_yield = state_raw[4]
                break
        
        episode_time = time.time() - episode_start_time
        
        # Record episode data
        agent.metrics.add_episode(
            episode, episode_reward, episode_yield,
            episode_w_amount, episode_n_amount,
            episode_length, episode_intrinsic_reward,
            episode_time
        )
        
        # Update exploration coverage
        agent.metrics.update_exploration_coverage()
        
        # Update policy
        if episode % config.update_frequency == 0 and len(agent.buffer) >= config.mini_batch_size:
            with torch.no_grad():
                last_value_ext, last_value_int = agent.network.get_value(
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                )
                last_value_ext = last_value_ext.item()
                last_value_int = last_value_int.item()
                if done:
                    last_value_ext = 0.0
                    last_value_int = 0.0
            
            update_info = agent.update(last_value_ext, last_value_int)
            agent.metrics.add_update(
                update_info['policy_loss'],
                update_info['value_loss'],
                update_info['entropy']
            )
        
        # Update progress bar
        episode_pbar.set_postfix({
            'Reward': f'{episode_reward:.0f}',
            'Yield': f'{episode_yield:.0f}',
            'WUE': f'{agent.metrics.wue_list[-1]:.1f}' if agent.metrics.wue_list else 'N/A'
        })
        
        # Log every 10 episodes
        if episode % config.log_frequency == 0:
            agri_metrics = agent.metrics.get_agricultural_metrics(last_n=10)
            ai_metrics = agent.metrics.get_ai_metrics()
            lr = agent.lr_scheduler.get_lr()[0] if agent.lr_scheduler else config.learning_rate
            
            print(f"\nEp {episode:4d} | "
                  f"Reward: {episode_reward:8.1f} | "
                  f"Yield: {episode_yield:7.1f} kg/ha | "
                  f"N: {episode_n_amount:5.0f} kg/ha | "
                  f"W: {episode_w_amount:5.0f} mm | "
                  f"Intrinsic: {episode_intrinsic_reward:6.2f} | "
                  f"LR: {lr:.2e}")
            
            print(f"         | "
                  f"Avg(10) Yield: {agri_metrics['yield_kg_ha']['mean']:.1f} +/- {agri_metrics['yield_kg_ha']['std']:.1f} | "
                  f"WUE: {agri_metrics['WUE_kg_mm']['mean']:.2f} | "
                  f"NUE: {agri_metrics['NUE_kg_kg']['mean']:.2f} | "
                  f"Coverage: {ai_metrics['exploration_coverage_pct']:.1f}%")
        
        # Save best model
        current_avg = np.mean(agent.metrics.episode_rewards[-100:]) if len(agent.metrics.episode_rewards) >= 100 else np.mean(agent.metrics.episode_rewards)
        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            agent.save(os.path.join(config.output_dir, 'checkpoints', 'best'), episode, {
                'avg_reward': current_avg,
                'best_yield': episode_yield
            })
        
        # Periodic saving
        if episode % config.save_frequency == 0:
            agent.save(os.path.join(config.output_dir, 'checkpoints'), episode)
            saver.save_episode_results(agent.metrics, episode)
            saver.save_plots(agent.metrics, episode)
    
    # Training complete
    env.close()
    episode_pbar.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Print final summary
    print(agent.metrics.get_summary(last_n=10))
    
    # Save final model and results
    agent.save(os.path.join(config.output_dir, 'checkpoints', 'final'), config.n_episodes)
    saver.save_final_results(agent.metrics, agent)
    
    return agent


def evaluate_agent(agent: PPORNDAgent, n_episodes: int = 10):
    """Evaluate trained agent"""
    print("\n" + "=" * 70)
    print(f"Model Evaluation ({n_episodes} episodes)")
    print("=" * 70)
    
    # Create environment
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 999,
        'random_weather': True
    }
    
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    # Evaluation metrics
    eval_metrics = MetricsCalculator(config)
    
    for ep in range(n_episodes):
        state_raw = dict2array(env.reset())
        state = agent.normalize_state(state_raw, update_stats=False)
        
        episode_reward = 0
        episode_n_amount = 0
        episode_w_amount = 0
        episode_yield = 0
        episode_length = 0
        episode_intrinsic = 0
        
        done = False
        while not done and episode_length < config.max_steps_per_episode:
            # Deterministic policy
            action, _, _, _ = agent.act(state, deterministic=True)
            
            intrinsic_reward = agent.get_intrinsic_reward(state)
            
            action_dict = action_to_dict(action, state_raw)
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state_raw = dict2array(next_state_raw) if not done else state_raw
            
            reward = get_reward(
                state_raw, action_dict['anfer'], action_dict['amir'],
                next_state_raw, done, config.k1, config.k2, config.k3
            )
            
            state_raw = next_state_raw if not done else state_raw
            state = agent.normalize_state(state_raw, update_stats=False)
            
            episode_reward += reward
            episode_intrinsic += intrinsic_reward
            episode_n_amount += action_dict['anfer']
            episode_w_amount += action_dict['amir']
            episode_length += 1
            
            if done:
                episode_yield = state_raw[4]
        
        eval_metrics.add_episode(
            ep + 1, episode_reward, episode_yield,
            episode_w_amount, episode_n_amount,
            episode_length, episode_intrinsic, 0
        )
        
        print(f"Eval {ep+1:2d} | "
              f"Reward: {episode_reward:8.1f} | "
              f"Yield: {episode_yield:7.1f} kg/ha | "
              f"N: {episode_n_amount:5.0f} | W: {episode_w_amount:5.0f}")
    
    env.close()
    
    # Print evaluation summary
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70)
    print(eval_metrics.get_summary(last_n=n_episodes))
    
    return eval_metrics


# ============================================================================
#                              Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Train
    agent = train_ppo_rnd()
    
    # Evaluate
    if agent:
        evaluate_agent(agent, n_episodes=config.n_eval_episodes)
    
    print("\nAll tasks completed!")