#!/usr/bin/env python3
"""
LLaMA + PPO Baseline Version for gym-dssat
============================================
Standard LLaMA Encoder + Standard PPO Algorithm + Original Paper Reward Function

Features:
- Standard PPO implementation with clipped objective
- LLaMA-based state encoding
- Original reward function from the DQN+BERT paper
- Comprehensive metrics tracking (Agronomic + AI metrics)
- Result saving in multiple formats (PNG, PDF, Excel)
- Training progress bar with time tracking

Author: Baseline Implementation
Date: 2024
"""

import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#                              Device Configuration
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import transformers
from transformers import LlamaModel, LlamaTokenizerFast

# ============================================================================
#                              Hyperparameter Configuration
# ============================================================================

@dataclass
class PPOConfig:
    """PPO Hyperparameter Configuration - Baseline Version"""
    
    # === Training Parameters ===
    n_episodes: int = 2000
    max_steps_per_episode: int = 200
    
    # === PPO Core Parameters ===
    gamma: float = 0.99              # Discount factor (same as original paper)
    gae_lambda: float = 0.95         # GAE lambda
    clip_ratio: float = 0.2          # PPO clip ratio
    entropy_coef: float = 0.01       # Entropy coefficient
    value_coef: float = 0.5          # Value loss coefficient
    max_grad_norm: float = 0.5       # Gradient clipping
    
    # === Optimizer Parameters ===
    learning_rate: float = 3e-5      # Standard learning rate
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # === PPO Update Parameters ===
    ppo_epochs: int = 4              # PPO update epochs
    mini_batch_size: int = 64
    update_frequency: int = 10       # Update every N episodes
    
    # === Network Parameters ===
    token_size: int = 128            # Max token length
    state_size: int = 25             # State dimension
    action_size: int = 25            # Action space size (5x5)
    hidden_size: int = 256           # Hidden layer size
    projection_size: int = 256       # Projection layer size
    
    # === Reward Function Parameters (from original paper) ===
    k1: float = 0.158                # Yield reward coefficient
    k2: float = 0.79                 # Nitrogen penalty coefficient
    k3: float = 1.1                  # Water penalty coefficient
    k4: float = 0.0                  # Unused parameter
    
    # === Evaluation Parameters ===
    eval_frequency: int = 10         # Evaluate every N episodes
    target_performance: float = 1400 # Target score for sample efficiency
    convergence_window: int = 50     # Window for convergence detection
    convergence_threshold: float = 0.1  # Threshold for convergence

config = PPOConfig()


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


def array2str(state: np.ndarray) -> str:
    """Convert state array to string for LLaMA tokenization (same as original paper)"""
    state_str = ""
    for i, num in enumerate(state):
        if i == 0:
            state_str += str(round(num / 40)) + " "
        elif i == 4:
            state_str += str(round(num / 100)) + " "
        elif i == 7:
            state_str += str(round(num / 10)) + " "
        elif i == 20:
            state_str += str(round(num / 100)) + " "
        elif i == 21:
            state_str += str(round(num / 6)) + " "
        elif i == 23:
            state_str += str(round(num)) + " "
        elif i >= 9 and i <= 17:
            state_str += str(round(num * 1000)) + " "
        elif i == 18:
            state_str += str(round(num * 100)) + " "
        else:
            state_str += str(round(num)) + " "
    return state_str


def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3):
    """
    Original reward function from the DQN+BERT paper
    
    Args:
        state: Current state array
        n_action: Nitrogen fertilizer application (kg/ha)
        w_action: Irrigation amount (mm)
        next_state: Next state array
        done: Episode termination flag
        k1, k2, k3: Reward coefficients
    
    Returns:
        reward: Calculated reward value
    """
    if done:
        # Terminal reward includes yield
        return k1 * state[4] - k2 * n_action - k3 * w_action
    else:
        # Intermediate reward only penalizes actions
        return -k2 * n_action - k3 * w_action


def print_gpu_memory(prefix: str = ""):
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


# ============================================================================
#                              Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """
    Calculate and track agronomic and AI metrics
    
    Agronomic Metrics:
    - Final Yield (kg/ha)
    - Irrigation Amount (mm)
    - Fertilizer Amount (kg/ha)
    - Water Use Efficiency (WUE, kg/mm)
    - Nitrogen Use Efficiency (NUE, kg/kg)
    
    AI Metrics:
    - Average Return
    - Sample Efficiency (steps to reach target performance)
    - Convergence Speed (steps to converge)
    """
    
    def __init__(self, target_performance: float = 1400, 
                 convergence_window: int = 50, 
                 convergence_threshold: float = 0.1):
        self.target_performance = target_performance
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        # Episode-level metrics
        self.episode_yields = []
        self.episode_irrigation = []
        self.episode_fertilizer = []
        self.episode_returns = []
        self.episode_wue = []
        self.episode_nue = []
        
        # Step-level tracking
        self.total_steps = 0
        self.target_reached_step = None
        self.convergence_step = None
        
        # Running statistics
        self.running_returns = deque(maxlen=convergence_window)
        
    def add_episode(self, yield_val: float, irrigation: float, 
                    fertilizer: float, episode_return: float):
        """Add episode metrics"""
        self.episode_yields.append(yield_val)
        self.episode_irrigation.append(irrigation)
        self.episode_fertilizer.append(fertilizer)
        self.episode_returns.append(episode_return)
        
        # Calculate WUE and NUE
        if irrigation > 0:
            wue = yield_val / irrigation
        else:
            wue = 0.0
        self.episode_wue.append(wue)
        
        if fertilizer > 0:
            nue = yield_val / fertilizer
        else:
            nue = 0.0
        self.episode_nue.append(nue)
        
        # Update running returns for convergence detection
        self.running_returns.append(episode_return)
        
    def add_steps(self, n_steps: int):
        """Add environment interaction steps"""
        self.total_steps += n_steps
        
    def check_sample_efficiency(self):
        """Check if target performance is reached"""
        if self.target_reached_step is None:
            if len(self.episode_returns) >= 10:
                recent_avg = np.mean(self.episode_returns[-10:])
                if recent_avg >= self.target_performance:
                    self.target_reached_step = self.total_steps
                    return True
        return False
    
    def check_convergence(self):
        """Check if training has converged"""
        if self.convergence_step is None and len(self.running_returns) >= self.convergence_window:
            returns = list(self.running_returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Check if variance is below threshold relative to mean
            if mean_return > 0 and std_return / mean_return < self.convergence_threshold:
                self.convergence_step = self.total_steps
                return True
        return False
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {
            # Agronomic metrics
            'final_yield_mean': np.mean(self.episode_yields) if self.episode_yields else 0,
            'final_yield_std': np.std(self.episode_yields) if self.episode_yields else 0,
            'final_yield_max': np.max(self.episode_yields) if self.episode_yields else 0,
            'irrigation_mean': np.mean(self.episode_irrigation) if self.episode_irrigation else 0,
            'irrigation_std': np.std(self.episode_irrigation) if self.episode_irrigation else 0,
            'fertilizer_mean': np.mean(self.episode_fertilizer) if self.episode_fertilizer else 0,
            'fertilizer_std': np.std(self.episode_fertilizer) if self.episode_fertilizer else 0,
            'wue_mean': np.mean(self.episode_wue) if self.episode_wue else 0,
            'wue_std': np.std(self.episode_wue) if self.episode_wue else 0,
            'nue_mean': np.mean(self.episode_nue) if self.episode_nue else 0,
            'nue_std': np.std(self.episode_nue) if self.episode_nue else 0,
            
            # AI metrics
            'average_return': np.mean(self.episode_returns) if self.episode_returns else 0,
            'return_std': np.std(self.episode_returns) if self.episode_returns else 0,
            'total_steps': self.total_steps,
            'sample_efficiency': self.target_reached_step if self.target_reached_step else -1,
            'convergence_speed': self.convergence_step if self.convergence_step else -1,
        }
        return summary
    
    def get_recent_metrics(self, n: int = 10) -> Dict:
        """Get metrics for recent n episodes"""
        n = min(n, len(self.episode_yields))
        if n == 0:
            return {}
        
        return {
            'recent_yield_mean': np.mean(self.episode_yields[-n:]),
            'recent_irrigation_mean': np.mean(self.episode_irrigation[-n:]),
            'recent_fertilizer_mean': np.mean(self.episode_fertilizer[-n:]),
            'recent_return_mean': np.mean(self.episode_returns[-n:]),
            'recent_wue_mean': np.mean(self.episode_wue[-n:]),
            'recent_nue_mean': np.mean(self.episode_nue[-n:]),
        }


# ============================================================================
#                              Network Definitions
# ============================================================================

class ProjectionLayer(nn.Module):
    """Projection layer to map LLaMA hidden states to policy space"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.projection(x)


class LLaMAEncoder(nn.Module):
    """LLaMA-based state encoder"""
    def __init__(self, llama_model, projection_size: int = 256):
        super().__init__()
        self.llama = llama_model
        self.hidden_size = llama_model.config.hidden_size
        self.projection_size = projection_size
        self.projection = ProjectionLayer(self.hidden_size, projection_size)
        
    def forward(self, input_ids, attention_mask):
        # Get LLaMA outputs
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return self.projection(embeddings)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        logits, values = self.forward(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ============================================================================
#                              PPO Buffer
# ============================================================================

class PPOBuffer:
    """Buffer for storing PPO rollout data"""
    
    def __init__(self, gamma: float, gae_lambda: float, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.state_strs = []
        
    def add(self, state, action, reward, value, log_prob, done, state_str):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.state_strs.append(state_str)
        
    def clear(self):
        """Clear the buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.state_strs.clear()
        
    def __len__(self):
        return len(self.states)
    
    def compute_gae(self, last_value: float):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            last_value: Value estimate of the last state
            
        Returns:
            returns: Computed returns
            advantages: Computed advantages
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        last_gae = 0
        last_return = last_value
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # GAE calculation
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
            # Returns calculation
            last_return = self.rewards[t] + self.gamma * last_return * next_non_terminal
            returns[t] = last_return
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages


# ============================================================================
#                              PPO Agent
# ============================================================================

class PPOAgent:
    """PPO Agent with LLaMA encoder"""
    
    def __init__(self, llama_model, tokenizer, config: PPOConfig):
        self.config = config
        self.tokenizer = tokenizer
        
        # Networks
        self.encoder = LLaMAEncoder(llama_model, config.projection_size).to(device)
        self.actor_critic = ActorCritic(config.projection_size, config.action_size, config.hidden_size).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.actor_critic.parameters()),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Buffer
        self.buffer = PPOBuffer(config.gamma, config.gae_lambda, device)
        
        # Training statistics
        self.training_step = 0
        
    def tokenize(self, texts):
        """Tokenize text inputs"""
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.config.token_size
        ).to(device)
    
    @torch.no_grad()
    def act(self, state: np.ndarray):
        """Select action given state"""
        self.encoder.eval()
        self.actor_critic.eval()
        
        state_str = array2str(state)
        inputs = self.tokenize(state_str)
        
        embedding = self.encoder(inputs['input_ids'], inputs['attention_mask'])
        action, log_prob, value = self.actor_critic.get_action(embedding)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done, state_str):
        """Store transition in buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done, state_str)
    
    def update(self, last_value: float):
        """Update policy using PPO"""
        self.encoder.train()
        self.actor_critic.train()
        
        # Compute GAE
        returns, advantages = self.buffer.compute_gae(last_value)
        
        # Convert to tensors
        states_str = self.buffer.state_strs
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        total_loss = 0
        n_updates = 0
        
        # PPO update epochs
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(self.buffer))
            
            for start in range(0, len(self.buffer), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch data
                batch_strs = [states_str[i] for i in batch_indices]
                inputs = self.tokenize(batch_strs)
                
                # Forward pass
                embeddings = self.encoder(inputs['input_ids'], inputs['attention_mask'])
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    embeddings, actions[batch_indices]
                )
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, returns[batch_indices])
                
                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        self.training_step += 1
        
        return total_loss / max(n_updates, 1)
    
    def save(self, path: str, episode: int):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
        }, os.path.join(path, f'model_episode_{episode}.pth'))
        print(f"Model saved to {path}/model_episode_{episode}.pth")


# ============================================================================
#                              Result Saver
# ============================================================================

class ResultSaver:
    """Save training results in multiple formats"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.episode_data = []
        
    def add_episode(self, episode: int, score: float, yield_val: float, 
                    irrigation: float, fertilizer: float, wue: float, nue: float,
                    loss: float = 0.0):
        """Add episode data"""
        self.episode_data.append({
            'episode': episode,
            'score': score,
            'yield': yield_val,
            'irrigation': irrigation,
            'fertilizer': fertilizer,
            'wue': wue,
            'nue': nue,
            'loss': loss,
        })
    
    def save_excel(self, filename: str = 'training_results.xlsx'):
        """Save results to Excel"""
        df = pd.DataFrame(self.episode_data)
        filepath = os.path.join(self.output_dir, filename)
        df.to_excel(filepath, index=False)
        print(f"Results saved to {filepath}")
        return filepath
    
    def save_plots_png(self, filename: str = 'training_plots.png'):
        """Save training plots as PNG"""
        if not self.episode_data:
            return None
            
        df = pd.DataFrame(self.episode_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Score
        axes[0, 0].plot(df['episode'], df['score'], 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Training Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Yield
        axes[0, 1].plot(df['episode'], df['yield'], 'g-', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Yield (kg/ha)')
        axes[0, 1].set_title('Final Yield')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Irrigation
        axes[0, 2].plot(df['episode'], df['irrigation'], 'c-', alpha=0.7)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Irrigation (mm)')
        axes[0, 2].set_title('Irrigation Amount')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Fertilizer
        axes[1, 0].plot(df['episode'], df['fertilizer'], 'm-', alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Fertilizer (kg/ha)')
        axes[1, 0].set_title('Nitrogen Fertilizer')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: WUE
        axes[1, 1].plot(df['episode'], df['wue'], 'r-', alpha=0.7)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('WUE (kg/mm)')
        axes[1, 1].set_title('Water Use Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: NUE
        axes[1, 2].plot(df['episode'], df['nue'], 'orange', alpha=0.7)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('NUE (kg/kg)')
        axes[1, 2].set_title('Nitrogen Use Efficiency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to {filepath}")
        return filepath
    
    def save_plots_pdf(self, filename: str = 'training_plots.pdf'):
        """Save training plots as PDF"""
        if not self.episode_data:
            return None
            
        df = pd.DataFrame(self.episode_data)
        filepath = os.path.join(self.output_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Page 1: Overview
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].plot(df['episode'], df['score'], 'b-', alpha=0.7)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Training Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(df['episode'], df['yield'], 'g-', alpha=0.7)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Yield (kg/ha)')
            axes[0, 1].set_title('Final Yield')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(df['episode'], df['irrigation'], 'c-', alpha=0.7, label='Irrigation')
            axes[1, 0].plot(df['episode'], df['fertilizer'], 'm-', alpha=0.7, label='Fertilizer')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Amount')
            axes[1, 0].set_title('Resource Usage')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(df['episode'], df['wue'], 'r-', alpha=0.7, label='WUE')
            axes[1, 1].plot(df['episode'], df['nue'], 'orange', alpha=0.7, label='NUE')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].set_title('Resource Use Efficiency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Moving averages
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            window = 50
            
            if len(df) >= window:
                df_ma = df.copy()
                df_ma['score_ma'] = df['score'].rolling(window=window).mean()
                df_ma['yield_ma'] = df['yield'].rolling(window=window).mean()
                df_ma['wue_ma'] = df['wue'].rolling(window=window).mean()
                df_ma['nue_ma'] = df['nue'].rolling(window=window).mean()
                
                axes[0, 0].plot(df['episode'], df['score'], 'b-', alpha=0.3)
                axes[0, 0].plot(df_ma['episode'], df_ma['score_ma'], 'b-', linewidth=2, label=f'MA({window})')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_title('Training Score with Moving Average')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(df['episode'], df['yield'], 'g-', alpha=0.3)
                axes[0, 1].plot(df_ma['episode'], df_ma['yield_ma'], 'g-', linewidth=2, label=f'MA({window})')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Yield (kg/ha)')
                axes[0, 1].set_title('Yield with Moving Average')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(df['episode'], df['irrigation'], 'c-', alpha=0.3, label='Irrigation')
                axes[1, 0].plot(df['episode'], df['fertilizer'], 'm-', alpha=0.3, label='Fertilizer')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Amount')
                axes[1, 0].set_title('Resource Usage')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].plot(df['episode'], df['wue'], 'r-', alpha=0.3, label='WUE')
                axes[1, 1].plot(df['episode'], df['nue'], 'orange', alpha=0.3, label='NUE')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Efficiency')
                axes[1, 1].set_title('Resource Use Efficiency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"PDF saved to {filepath}")
        return filepath
    
    def save_summary(self, metrics: MetricsCalculator, filename: str = 'training_summary.xlsx'):
        """Save training summary"""
        summary = metrics.get_summary()
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([summary])
        
        # Transpose for better readability
        summary_df = summary_df.T.reset_index()
        summary_df.columns = ['Metric', 'Value']
        
        filepath = os.path.join(self.output_dir, filename)
        summary_df.to_excel(filepath, index=False)
        print(f"Summary saved to {filepath}")
        return filepath


# ============================================================================
#                              Training Function
# ============================================================================

def initialize_llama(model_path: str):
    """Initialize LLaMA model and tokenizer"""
    print(f"Initializing LLaMA model: {model_path}")
    
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        use_cache=False
    ).to(device)
    
    print(f"   Model loaded successfully")
    return model, tokenizer


def train_ppo():
    """Main training function"""
    print("=" * 70)
    print("LLaMA + PPO Baseline Training")
    print("=" * 70)
    
    # Initialize paths
    model_path = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'
    output_dir = '/home/gymusr/llama_ppo_results'
    checkpoint_dir = '/home/gymusr/llama_ppo_checkpoints'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize LLaMA
    try:
        llama_model, tokenizer = initialize_llama(model_path)
    except Exception as e:
        print(f"Failed to load LLaMA model: {e}")
        print("Please ensure the model path is correct.")
        return
    
    # Initialize environment
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True,
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("Environment initialized successfully")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return
    
    # Initialize agent
    agent = PPOAgent(llama_model, tokenizer, config)
    print("Agent initialized successfully")
    
    # Initialize metrics calculator and result saver
    metrics = MetricsCalculator(
        target_performance=config.target_performance,
        convergence_window=config.convergence_window,
        convergence_threshold=config.convergence_threshold
    )
    result_saver = ResultSaver(output_dir)
    
    # Training statistics
    best_score = float('-inf')
    total_start_time = time.time()
    
    print(f"\nStarting training ({config.n_episodes} episodes)")
    print("-" * 70)
    
    # Main training loop with progress bar
    pbar = tqdm(range(1, config.n_episodes + 1), desc="Training Progress", 
                unit="ep", ncols=100)
    
    for episode in pbar:
        episode_start_time = time.time()
        
        # Reset environment
        state = dict2array(env.reset())
        episode_reward = 0
        n_amount = 0  # Nitrogen fertilizer amount
        w_amount = 0  # Irrigation amount
        episode_yield = 0
        episode_steps = 0
        
        # Episode loop
        for step in range(config.max_steps_per_episode):
            # Select action
            action, log_prob, value = agent.act(state)
            
            # Map action to environment action
            action_dict = {
                'anfer': (action % 5) * 40,  # Nitrogen: 0, 40, 80, 120, 160 kg/ha
                'amir': int(action / 5) * 6,  # Irrigation: 0, 6, 12, 18, 24 mm
            }
            
            # Apply constraints
            if state[0] >= 10000:  # Limit based on state
                action_dict['anfer'] = 0
            if state[21] >= 1600:
                action_dict['amir'] = 0
            
            # Execute action
            next_state_raw, _, done, _ = env.step(action_dict)
            
            if done:
                episode_yield = state[4]  # Get yield at termination
                next_state = state
            else:
                next_state = dict2array(next_state_raw)
            
            # Calculate reward (using original paper's reward function)
            reward = get_reward(
                state, action_dict['anfer'], action_dict['amir'],
                next_state, done, config.k1, config.k2, config.k3
            )
            
            # Store transition
            agent.store_transition(
                state, action, reward, value, log_prob, done, array2str(state)
            )
            
            # Update state and accumulators
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            episode_steps += 1
            
            if done:
                break
        
        # Update metrics
        metrics.add_episode(episode_yield, w_amount, n_amount, episode_reward)
        metrics.add_steps(episode_steps)
        
        # Calculate WUE and NUE
        wue = episode_yield / w_amount if w_amount > 0 else 0
        nue = episode_yield / n_amount if n_amount > 0 else 0
        
        # Store episode data
        result_saver.add_episode(
            episode=episode,
            score=episode_reward,
            yield_val=episode_yield,
            irrigation=w_amount,
            fertilizer=n_amount,
            wue=wue,
            nue=nue
        )
        
        # Update progress bar
        episode_time = time.time() - episode_start_time
        pbar.set_postfix({
            'score': f'{episode_reward:.0f}',
            'yield': f'{episode_yield:.0f}',
            'time': f'{episode_time:.1f}s'
        })
        
        # PPO update
        loss = 0
        if episode % config.update_frequency == 0 and len(agent.buffer) >= config.mini_batch_size:
            # Get last value for GAE
            with torch.no_grad():
                _, _, last_value = agent.act(state)
                if done:
                    last_value = 0.0
            
            loss = agent.update(last_value)
        
        # Check sample efficiency
        if metrics.check_sample_efficiency():
            print(f"\nTarget performance reached at step {metrics.target_reached_step}")
        
        # Check convergence
        if metrics.check_convergence():
            print(f"\nConvergence detected at step {metrics.convergence_step}")
        
        # Save best model
        if episode_reward > best_score:
            best_score = episode_reward
            agent.save(checkpoint_dir, episode)
        
        # Print detailed metrics every 10 episodes
        if episode % 10 == 0:
            recent = metrics.get_recent_metrics(10)
            summary = metrics.get_summary()
            
            elapsed_time = time.time() - total_start_time
            avg_time_per_ep = elapsed_time / episode
            
            print(f"\n{'='*60}")
            print(f"Episode {episode}/{config.n_episodes}")
            print(f"{'='*60}")
            print(f"[Agronomic Metrics - Last 10 Episodes]")
            print(f"  Yield:        {recent.get('recent_yield_mean', 0):.2f} kg/ha")
            print(f"  Irrigation:   {recent.get('recent_irrigation_mean', 0):.2f} mm")
            print(f"  Fertilizer:   {recent.get('recent_fertilizer_mean', 0):.2f} kg/ha")
            print(f"  WUE:          {recent.get('recent_wue_mean', 0):.4f} kg/mm")
            print(f"  NUE:          {recent.get('recent_nue_mean', 0):.4f} kg/kg")
            print(f"\n[AI Metrics]")
            print(f"  Avg Return:   {recent.get('recent_return_mean', 0):.2f}")
            print(f"  Total Steps:  {metrics.total_steps}")
            print(f"  Sample Eff:   {summary['sample_efficiency'] if summary['sample_efficiency'] > 0 else 'Not reached'}")
            print(f"  Conv Speed:   {summary['convergence_speed'] if summary['convergence_speed'] > 0 else 'Not converged'}")
            print(f"\n[Training Info]")
            print(f"  Time:         {elapsed_time/60:.1f} min (avg {avg_time_per_ep:.1f}s/ep)")
            print(f"{'='*60}\n")
    
    # Training complete
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Total training time: {total_time/60:.1f} minutes")
    
    # Final metrics
    final_summary = metrics.get_summary()
    print(f"\n[Final Summary]")
    print(f"  Average Yield:     {final_summary['final_yield_mean']:.2f} +/- {final_summary['final_yield_std']:.2f} kg/ha")
    print(f"  Average Return:    {final_summary['average_return']:.2f}")
    print(f"  Sample Efficiency: {final_summary['sample_efficiency'] if final_summary['sample_efficiency'] > 0 else 'Not reached'}")
    print(f"  Convergence Speed: {final_summary['convergence_speed'] if final_summary['convergence_speed'] > 0 else 'Not converged'}")
    
    # Save all results
    print(f"\nSaving results...")
    result_saver.save_excel('llama_ppo_baseline_results.xlsx')
    result_saver.save_plots_png('llama_ppo_baseline_plots.png')
    result_saver.save_plots_pdf('llama_ppo_baseline_plots.pdf')
    result_saver.save_summary(metrics, 'llama_ppo_baseline_summary.xlsx')
    
    # Close environment
    env.close()
    print("\nAll done!")
    
    return metrics, result_saver


# ============================================================================
#                              Validation Function
# ============================================================================

def validate_agent(model_path: str, checkpoint_path: str, n_episodes: int = 100):
    """
    Validate trained agent
    
    Args:
        model_path: Path to LLaMA model
        checkpoint_path: Path to saved checkpoint
        n_episodes: Number of validation episodes
    """
    print("=" * 70)
    print("Validation Mode")
    print("=" * 70)
    
    # Initialize model and environment
    llama_model, tokenizer = initialize_llama(model_path)
    
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi-val.log',
        'mode': 'all',
        'seed': 42,  # Different seed for validation
        'random_weather': True,
    }
    
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    # Load agent
    agent = PPOAgent(llama_model, tokenizer, config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.encoder.load_state_dict(checkpoint['encoder'])
    agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Initialize metrics
    metrics = MetricsCalculator(target_performance=config.target_performance)
    
    # Validation loop
    print(f"\nRunning validation ({n_episodes} episodes)...")
    
    for episode in tqdm(range(1, n_episodes + 1), desc="Validation"):
        state = dict2array(env.reset())
        episode_reward = 0
        n_amount = 0
        w_amount = 0
        episode_yield = 0
        
        for step in range(config.max_steps_per_episode):
            # Deterministic action selection (no sampling)
            with torch.no_grad():
                state_str = array2str(state)
                inputs = agent.tokenize(state_str)
                embedding = agent.encoder(inputs['input_ids'], inputs['attention_mask'])
                logits, _ = agent.actor_critic(embedding)
                action = torch.argmax(logits, dim=-1).item()
            
            action_dict = {
                'anfer': (action % 5) * 40,
                'amir': int(action / 5) * 6,
            }
            
            if state[0] >= 10000:
                action_dict['anfer'] = 0
            if state[21] >= 1600:
                action_dict['amir'] = 0
            
            next_state_raw, _, done, _ = env.step(action_dict)
            
            if done:
                episode_yield = state[4]
                next_state = state
            else:
                next_state = dict2array(next_state_raw)
            
            reward = get_reward(
                state, action_dict['anfer'], action_dict['amir'],
                next_state, done, config.k1, config.k2, config.k3
            )
            
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            
            if done:
                break
        
        metrics.add_episode(episode_yield, w_amount, n_amount, episode_reward)
        
        # Print every 10 episodes
        if episode % 10 == 0:
            recent = metrics.get_recent_metrics(10)
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"  Yield: {recent.get('recent_yield_mean', 0):.2f} kg/ha")
            print(f"  Return: {recent.get('recent_return_mean', 0):.2f}")
            print(f"  WUE: {recent.get('recent_wue_mean', 0):.4f} kg/mm")
            print(f"  NUE: {recent.get('recent_nue_mean', 0):.4f} kg/kg")
    
    # Final validation results
    summary = metrics.get_summary()
    print(f"\n{'='*70}")
    print("Validation Complete!")
    print(f"{'='*70}")
    print(f"[Validation Results]")
    print(f"  Average Yield:     {summary['final_yield_mean']:.2f} +/- {summary['final_yield_std']:.2f} kg/ha")
    print(f"  Average Return:    {summary['average_return']:.2f}")
    print(f"  Average WUE:       {summary['wue_mean']:.4f} kg/mm")
    print(f"  Average NUE:       {summary['nue_mean']:.4f} kg/kg")
    
    env.close()
    return metrics


# ============================================================================
#                              Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA + PPO Baseline for gym-dssat')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'],
                        help='Running mode: train or validate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for validation')
    parser.add_argument('--n_episodes', type=int, default=None,
                        help='Number of episodes (overrides config)')
    
    args = parser.parse_args()
    
    if args.n_episodes:
        config.n_episodes = args.n_episodes
    
    if args.mode == 'train':
        train_ppo()
    elif args.mode == 'validate':
        if args.checkpoint is None:
            print("Error: --checkpoint is required for validation mode")
        else:
            model_path = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'
            validate_agent(model_path, args.checkpoint, n_episodes=100)