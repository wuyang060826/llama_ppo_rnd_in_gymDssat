#!/usr/bin/env python3
"""
PPO + Chinese-LLaMA-2 + gym-dssat 渐进式泛化增强版
解决原版本过度正则化导致无法收敛到最优的问题

核心改进：
1. 渐进式增强策略 - 训练早期弱增强，后期逐渐增强
2. 精简trick - 只保留最有效的测量噪声注入
3. 增强强度衰减 - 根据训练进度动态调整
4. 智能验证 - 使用集成模型在最佳时机验证

参考论文：
《The New Agronomists: Language Models are Experts in Crop Management》
- 论文Table 6证明：温度噪声(±2°C)影响最大(-11.9%)，其他噪声影响较小
- 关键发现：适度噪声反而能提升泛化，但过度噪声会损害性能
"""
from reportlab.lib.units import inch
import numpy as np
import pandas as pd
import random
from collections import deque, OrderedDict
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')
import copy

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"   TF32 Acceleration: Enabled")

from transformers import LlamaModel, LlamaTokenizerFast


# ============================================================================
#                              超参数配置
# ============================================================================

@dataclass
class PPOConfig:
    """PPO超参数配置 - 渐进式泛化增强版"""
    
    # === 训练参数 ===
    n_episodes: int = 2000              
    max_steps_per_episode: int = 200    
    
    # === PPO核心参数 ===
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    
    # === 优化器参数 ===
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    embedder_lr: float = 1e-5
    projection_lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # === 学习率调度 ===
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05
    
    # === PPO更新参数 ===
    ppo_epochs: int = 5
    mini_batch_size: int = 64
    update_frequency: int = 10
    
    # === 网络参数 ===
    token_size: int = 128
    state_size: int = 25
    action_size: int = 25
    hidden_size: int = 256
    projection_size: int = 256
    
    # === 缓存与内存 ===
    embedding_cache_size: int = 100
    
    # === 奖励函数参数 ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === 奖励归一化 ===
    reward_norm: bool = True
    
    # === 探索参数 ===
    entropy_decay: float = 0.999
    entropy_min: float = 0.001
    
    # === 优化选项 ===
    use_bf16: bool = True
    
    # === 指标计算参数 ===
    expert_performance_threshold: float = 500.0
    convergence_window: int = 50
    convergence_threshold: float = 0.05
    rnd_threshold: float = 0.1
    
    # === RND内在奖励参数 ===
    use_rnd_intrinsic: bool = True
    intrinsic_coef: float = 0.01
    intrinsic_decay_start: float = 0.3
    intrinsic_decay_end: float = 0.7
    intrinsic_reward_scale: float = 1.0
    
    # ============ 渐进式泛化增强参数 ============
    # 渐进式增强开关
    use_progressive_augmentation: bool = True
    
    # 增强开始和完全激活的训练进度
    aug_start_progress: float = 0.4      # 30%进度后开始增强
    aug_full_progress: float = 0.6       # 70%进度时增强完全激活
    
    # 测量噪声参数 (论文Table 6证明最有效)
    temperature_noise_max: float = 2.0       # 温度噪声最大值(°C)
    rainfall_noise_accuracy: float = 0.95    # 降雨准确率(提高以减少噪声)
    soil_moisture_noise_max: float = 0.02    # 土壤湿度噪声最大值
    
    # 状态掩码参数 (保守设置)
    state_dropout_prob: float = 0.05         # 降低掩码概率
    state_dropout_max_features: int = 1      # 每次最多掩盖1个特征
    
    # 早停机制 (更宽松的设置)
    early_stopping_patience: int = 300       # 增加耐心值
    early_stopping_min_delta: float = 20.0   # 增加最小改善阈值
    
    # 策略集成
    ensemble_size: int = 5
    ensemble_save_interval: int = 100
    
    # 验证设置
    n_validation_seeds: int = 3              # 减少验证种子数
    n_validation_episodes_per_seed: int = 5  # 增加每种子验证轮数
    
    # === 结果保存路径 ===
    output_prefix: str = "llama_ppo_progressive"

config = PPOConfig()


# ============================================================================
#                          渐进式泛化增强模块
# ============================================================================

class ProgressiveAugmentor:
    """
    渐进式泛化增强模块
    
    核心思想：
    1. 训练早期(0-30%): 几乎不增强，让模型充分学习基本策略
    2. 训练中期(30-70%): 逐渐增加增强强度
    3. 训练后期(70-100%): 完全增强，提升泛化能力
    """
    
    def __init__(self, config):
        self.config = config
        self.training_mode = True
        self.current_progress = 0.0  # 0.0 ~ 1.0
        
        # 当前增强强度
        self.current_aug_strength = 0.0
        
        # 统计
        self.augmentation_stats = {
            'dropout_applied': 0,
            'noise_applied': 0,
            'total_steps': 0
        }
    
    def set_mode(self, training: bool):
        """设置增强模式"""
        self.training_mode = training
    
    def update_progress(self, episode: int, total_episodes: int):
        """更新训练进度和增强强度"""
        self.current_progress = episode / total_episodes
        
        if self.config.use_progressive_augmentation:
            if self.current_progress < self.config.aug_start_progress:
                # 早期：不增强
                self.current_aug_strength = 0.0
            elif self.current_progress < self.config.aug_full_progress:
                # 中期：线性增加增强强度
                progress_in_phase = (self.current_progress - self.config.aug_start_progress) / \
                                   (self.config.aug_full_progress - self.config.aug_start_progress)
                self.current_aug_strength = progress_in_phase
            else:
                # 后期：完全增强
                self.current_aug_strength = 1.0
        else:
            # 非渐进模式：始终完全增强
            self.current_aug_strength = 1.0
    
    def augment_state(self, state: np.ndarray) -> np.ndarray:
        """
        对状态进行渐进式增强
        """
        if not self.training_mode:
            return state
        
        self.augmentation_stats['total_steps'] += 1
        augmented_state = state.copy()
        
        # 根据当前增强强度决定是否应用增强
        if random.random() > self.current_aug_strength:
            return augmented_state
        
        # 1. 测量噪声注入 (论文证明最有效)
        augmented_state = self._apply_measurement_noise(augmented_state)
        
        # 2. 状态掩码 (概率很低)
        if random.random() < self.config.state_dropout_prob * self.current_aug_strength:
            augmented_state = self._apply_state_dropout(augmented_state)
        
        return augmented_state
    
    def _apply_measurement_noise(self, state: np.ndarray) -> np.ndarray:
        """
        测量噪声注入 - 根据论文Table 6的发现
        温度噪声影响最大，其他噪声影响较小
        """
        # 温度噪声 (论文中-11.9%影响，最关键)
        if 1 < len(state) and self.current_aug_strength > 0.3:
            # 噪声范围随增强强度增加
            noise_range = self.config.temperature_noise_max * self.current_aug_strength
            temp_noise = random.uniform(-noise_range, noise_range)
            state[1] += temp_noise
            self.augmentation_stats['noise_applied'] += 1
        
        # 降雨噪声 (论文中-3.2%影响，次要)
        if 2 < len(state) and self.current_aug_strength > 0.5:
            if random.random() > self.config.rainfall_noise_accuracy:
                rain_noise = random.gauss(0, state[2] * 0.05) if state[2] > 0 else 0
                state[2] = max(0, state[2] + rain_noise)
        
        # 土壤湿度噪声 (论文中影响很小，仅在后期应用)
        if 20 < len(state) and self.current_aug_strength > 0.7:
            sw_noise = random.uniform(
                -self.config.soil_moisture_noise_max * self.current_aug_strength,
                self.config.soil_moisture_noise_max * self.current_aug_strength
            )
            state[20] = np.clip(state[20] + sw_noise, 0, 1)
        
        return state
    
    def _apply_state_dropout(self, state: np.ndarray) -> np.ndarray:
        """状态掩码 - 保守使用"""
        # 保护关键决策特征
        protected_indices = {20, 21, 22, 23, 24}
        droppable_indices = [i for i in range(len(state)) if i not in protected_indices]
        
        if droppable_indices:
            # 只掩盖1个特征
            drop_idx = random.choice(droppable_indices)
            state[drop_idx] = 0.0
            self.augmentation_stats['dropout_applied'] += 1
        
        return state
    
    def get_aug_strength_info(self) -> dict:
        """获取当前增强状态信息"""
        return {
            'progress': self.current_progress,
            'aug_strength': self.current_aug_strength,
            'dropout_rate': self.augmentation_stats['dropout_applied'] / max(1, self.augmentation_stats['total_steps']),
            'noise_rate': self.augmentation_stats['noise_applied'] / max(1, self.augmentation_stats['total_steps'])
        }


# ============================================================================
#                              辅助函数
# ============================================================================

def dict2array(state: dict) -> np.ndarray:
    if state is None: raise ValueError("State cannot be None")
    new_state = []
    for key in state.keys():
        if key != 'sw': new_state.append(state[key])
        else: new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)

def array2str(state: np.ndarray) -> str:
    """状态转文本 - 保持原始方法以获得最佳性能"""
    parts = []
    for i, num in enumerate(state):
        if i == 0: parts.append(f"day{int(num/40)}")
        elif i == 4: parts.append(f"yield{int(num/100)}")
        elif i == 7: parts.append(f"wt{int(num/10)}")
        elif i == 20: parts.append(f"sm{int(num/100)}")
        elif i == 21: parts.append(f"ir{int(num/6)}")
        elif i == 23: parts.append(f"ac{int(num)}")
        elif 9 <= i <= 17: parts.append(f"n{i-8}:{int(num*1000)}")
        elif i == 18: parts.append(f"pct{int(num*100)}")
        else: parts.append(f"{int(num)}")
    return " ".join(parts)

def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3):
    if done: return k1 * state[4] - k2 * n_action - k3 * w_action
    return -k2 * n_action - k3 * w_action

def print_gpu_memory(prefix: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory {prefix}: Used={allocated:.2f}GB, Reserved={reserved:.2f}GB")


# ============================================================================
#                           学习率调度器
# ============================================================================

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
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2)
            group['lr'] = lr
    
    def get_lr(self): return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
#                              网络定义
# ============================================================================

class ProjectionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, output_size * 2),
            nn.LayerNorm(output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size),
            nn.LayerNorm(output_size),
        )
    def forward(self, x): return self.projection(x)

class LLaMAEmbedder(nn.Module):
    def __init__(self, llama_model, projection_size=256):
        super().__init__()
        self.llama = llama_model
        self.hidden_size = llama_model.config.hidden_size
        self.projection_size = projection_size
        self.projection = ProjectionLayer(self.hidden_size, projection_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return self.projection(embeddings)

class ActorCriticHead(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # 保持原始dropout
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size // 2, action_size)
        self.critic_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        features = self.shared(x)
        return self.actor_head(features), self.critic_head(features)
    
    def get_action_value(self, x):
        logits, value = self.forward(x)
        return Categorical(logits=logits), value


# ============================================================================
#                         RND (Random Network Distillation)
# ============================================================================

class RNDNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x):
        return self.network(x)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        x = np.array(x)
        if x.size == 0:
            return
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x, clip_range=5.0):
        normalized = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normalized, -clip_range, clip_range)


class RNDModule:
    def __init__(self, state_size, hidden_size=128, lr=1e-4):
        self.target_network = RNDNetwork(state_size, hidden_size).to(device)
        self.predictor_network = RNDNetwork(state_size, hidden_size).to(device)
        
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)
        
        self.state_visit_counts = {}
        self.total_states_explored = 0
        self.exploration_threshold = config.rnd_threshold
        self.intrinsic_rms = RunningMeanStd()
        
    def discretize_state(self, state):
        key_dims = [0, 4, 20, 21]
        discretized = tuple(int(state[i] / 100) for i in key_dims if i < len(state))
        return discretized
    
    def get_intrinsic_reward(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predictor_output = self.predictor_network(state_tensor)
        prediction_error = F.mse_loss(predictor_output, target_output, reduction='none').mean().item()
        
        return prediction_error
    
    def update_predictor(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predictor_output = self.predictor_network(state_tensor)
        loss = F.mse_loss(predictor_output, target_output)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_exploration_stats(self, state, intrinsic_reward):
        discretized = self.discretize_state(state)
        
        if discretized not in self.state_visit_counts:
            self.state_visit_counts[discretized] = 0
        
        self.state_visit_counts[discretized] += 1
        
        if intrinsic_reward < self.exploration_threshold:
            if self.state_visit_counts[discretized] == 1:
                self.total_states_explored += 1
    
    def get_exploration_coverage(self, estimated_total_states=1000):
        coverage = min(100.0, (self.total_states_explored / estimated_total_states) * 100)
        return coverage
    
    def get_scaled_intrinsic_reward(self, state, current_episode, total_episodes):
        raw_intrinsic = self.get_intrinsic_reward(state)
        self.intrinsic_rms.update(np.array([raw_intrinsic]))
        normalized_intrinsic = self.intrinsic_rms.normalize(raw_intrinsic)
        
        progress = current_episode / total_episodes
        if progress < config.intrinsic_decay_start:
            decay_coef = 1.0
        elif progress > config.intrinsic_decay_end:
            decay_coef = 0.0
        else:
            decay_progress = (progress - config.intrinsic_decay_start) / (config.intrinsic_decay_end - config.intrinsic_decay_start)
            decay_coef = 1.0 - decay_progress
        
        scaled_intrinsic = normalized_intrinsic * config.intrinsic_coef * decay_coef * config.intrinsic_reward_scale
        
        return scaled_intrinsic, normalized_intrinsic, decay_coef


# ============================================================================
#                              策略集成
# ============================================================================

class PolicyEnsemble:
    def __init__(self, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.checkpoints = []
        self.checkpoint_scores = []
        self.best_checkpoint = None
        self.best_score = float('-inf')
    
    def add_checkpoint(self, model_state: dict, score: float):
        if score > self.best_score:
            self.best_score = score
            # 使用state_dict的浅拷贝而不是深度复制
            self.best_checkpoint = {k: v.cpu().clone() if torch.is_tensor(v) else v 
                                   for k, v in model_state.items()}
        
        # 只保存必要的信息，避免深度复制
        checkpoint = {
            'score': score,
            'state': {k: v.cpu().clone() if torch.is_tensor(v) else v 
                     for k, v in model_state.items()}
        }
        self.checkpoints.append(checkpoint)
        self.checkpoint_scores.append(score)
        
        if len(self.checkpoints) > self.ensemble_size:
            min_idx = np.argmin(self.checkpoint_scores)
            self.checkpoints.pop(min_idx)
            self.checkpoint_scores.pop(min_idx)
    
    def get_best_state(self) -> Optional[dict]:
        return self.best_checkpoint


# ============================================================================
#                              早停机制
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 300, min_delta: float = 20.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_scores = []
    
    def __call__(self, val_score: float, episode: int) -> bool:
        self.val_scores.append(val_score)
        
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"\n[Early Stopping] Triggered at episode {episode}")
            print(f"   Best validation score: {self.best_score:.2f}")
            self.early_stop = True
            return True
        
        return False


# ============================================================================
#                              指标记录器
# ============================================================================

class MetricsLogger:
    def __init__(self, output_prefix="llama_ppo_progressive"):
        self.output_prefix = output_prefix
        self.output_dir = f"/home/wuyang/results/llama_ppo_rnd_results/0417"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.final_yields = []
        self.irrigation_amounts = []
        self.fertilizer_amounts = []
        self.wue_values = []
        self.nue_values = []
        
        self.episode_rewards = []
        self.avg_rewards = []
        self.val_rewards = []
        self.generalization_gap = []
        self.exploration_coverages = []
        
        self.episode_steps = []
        self.total_steps = 0
        self.losses = []
        self.learning_rates = []
        
        # 渐进增强记录
        self.aug_strengths = []
        
        self.expert_threshold_reached = False
        self.sample_efficiency = None
        self.sample_efficiency_step = None
        
        self.convergence_reached = False
        self.convergence_step = None
        self.recent_rewards = deque(maxlen=config.convergence_window)
        
        self.metrics_every_10 = []
        
    def record_episode(self, episode, yield_val, irrigation, fertilizer, 
                       reward, steps, exploration_coverage=None, aug_strength=None):
        self.final_yields.append(yield_val)
        self.irrigation_amounts.append(irrigation)
        self.fertilizer_amounts.append(fertilizer)
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.total_steps += steps
        
        wue = yield_val / irrigation if irrigation > 0 else 0
        nue = yield_val / fertilizer if fertilizer > 0 else 0
        self.wue_values.append(wue)
        self.nue_values.append(nue)
        
        if exploration_coverage is not None:
            self.exploration_coverages.append(exploration_coverage)
        
        if aug_strength is not None:
            self.aug_strengths.append(aug_strength)
        
        avg_reward = np.mean(self.episode_rewards[-min(10, len(self.episode_rewards)):])
        self.avg_rewards.append(avg_reward)
        
        self.recent_rewards.append(reward)
        
        if not self.expert_threshold_reached and avg_reward >= config.expert_performance_threshold:
            self.expert_threshold_reached = True
            self.sample_efficiency = self.total_steps
            self.sample_efficiency_step = episode
        
        if not self.convergence_reached and len(self.recent_rewards) >= config.convergence_window:
            reward_std = np.std(list(self.recent_rewards))
            reward_mean = np.mean(list(self.recent_rewards))
            if reward_mean > 0 and reward_std / reward_mean < config.convergence_threshold:
                self.convergence_reached = True
                self.convergence_step = episode
    
    def record_validation(self, val_reward):
        self.val_rewards.append(val_reward)
        if len(self.episode_rewards) > 0:
            train_reward = np.mean(self.episode_rewards[-10:])
            gap = train_reward - val_reward
            self.generalization_gap.append(gap)
    
    def record_update(self, loss, lr):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        
    def get_current_metrics(self, episode):
        return {
            'episode': episode,
            'final_yield': self.final_yields[-1] if self.final_yields else 0,
            'irrigation': self.irrigation_amounts[-1] if self.irrigation_amounts else 0,
            'fertilizer': self.fertilizer_amounts[-1] if self.fertilizer_amounts else 0,
            'wue': self.wue_values[-1] if self.wue_values else 0,
            'nue': self.nue_values[-1] if self.nue_values else 0,
            'episode_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'avg_reward': self.avg_rewards[-1] if self.avg_rewards else 0,
            'val_reward': self.val_rewards[-1] if self.val_rewards else 0,
            'sample_efficiency': self.sample_efficiency if self.sample_efficiency else 'N/A',
            'convergence_step': self.convergence_step if self.convergence_step else 'N/A',
            'exploration_coverage': self.exploration_coverages[-1] if self.exploration_coverages else 0,
            'total_steps': self.total_steps,
            'loss': self.losses[-1] if self.losses else 0,
            'learning_rate': self.learning_rates[-1] if self.learning_rates else 0,
            'generalization_gap': self.generalization_gap[-1] if self.generalization_gap else 0,
            'aug_strength': self.aug_strengths[-1] if self.aug_strengths else 0
        }
    
    def print_metrics(self, episode, phase="Training"):
        metrics = self.get_current_metrics(episode)
        
        print(f"\n{'='*70}")
        print(f"  {phase} Metrics - Episode {episode}")
        print(f"{'='*70}")
        
        print(f"\n  [Agronomic Indicators]")
        print(f"    Final Yield:        {metrics['final_yield']:.2f} kg/ha")
        print(f"    Irrigation:         {metrics['irrigation']:.2f} mm")
        print(f"    Fertilizer:         {metrics['fertilizer']:.2f} kg/ha")
        print(f"    WUE:                {metrics['wue']:.4f} kg/mm")
        print(f"    NUE:                {metrics['nue']:.4f} kg/kg")
        
        print(f"\n  [AI Indicators]")
        print(f"    Episode Reward:     {metrics['episode_reward']:.2f}")
        print(f"    Avg Reward:         {metrics['avg_reward']:.2f}")
        print(f"    Val Reward:         {metrics['val_reward']:.2f}")
        print(f"    Generalization Gap: {metrics['generalization_gap']:.2f}")
        print(f"    Aug Strength:       {metrics['aug_strength']:.2%}")
        print(f"    Exploration:        {metrics['exploration_coverage']:.2f}%")
        
        print(f"\n  [Training Info]")
        print(f"    Total Steps:        {metrics['total_steps']}")
        print(f"    Loss:               {metrics['loss']:.4f}")
        print(f"    Learning Rate:      {metrics['learning_rate']:.2e}")
        print(f"{'='*70}\n")
    
    def save_metrics_every_10(self, episode):
        self.metrics_every_10.append(self.get_current_metrics(episode))
    
    def save_all_results(self):
        self._save_to_excel()
        self._save_to_pdf()
        self._save_plots()
        print(f"\nAll results saved to: {self.output_dir}")
    
    def _save_to_excel(self):
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_metrics.xlsx")
        
        data = {
            'Episode': list(range(1, len(self.final_yields) + 1)),
            'Final_Yield_kg_ha': self.final_yields,
            'Irrigation_mm': self.irrigation_amounts,
            'Fertilizer_kg_ha': self.fertilizer_amounts,
            'WUE_kg_mm': self.wue_values,
            'NUE_kg_kg': self.nue_values,
            'Episode_Reward': self.episode_rewards,
            'Avg_Reward': self.avg_rewards,
            'Val_Reward': self.val_rewards + [0]*(len(self.final_yields) - len(self.val_rewards)),
            'Generalization_Gap': self.generalization_gap + [0]*(len(self.final_yields) - len(self.generalization_gap)),
            'Aug_Strength': self.aug_strengths + [0]*(len(self.final_yields) - len(self.aug_strengths)),
            'Total_Steps': np.cumsum(self.episode_steps),
            'Exploration_Coverage_pct': self.exploration_coverages + [0]*(len(self.final_yields) - len(self.exploration_coverages))
        }
        
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Episodes', index=False)
            if self.metrics_every_10:
                pd.DataFrame(self.metrics_every_10).to_excel(writer, sheet_name='Every_10_Episodes', index=False)
            
            summary_data = {
                'Metric': ['Total_Episodes', 'Total_Steps', 'Avg_Final_Yield', 'Max_Final_Yield',
                          'Avg_Train_Reward', 'Max_Train_Reward', 'Avg_Val_Reward', 'Avg_Generalization_Gap'],
                'Value': [len(self.final_yields), self.total_steps, np.mean(self.final_yields), 
                         np.max(self.final_yields), np.mean(self.episode_rewards), np.max(self.episode_rewards),
                         np.mean(self.val_rewards) if self.val_rewards else 0,
                         np.mean(self.generalization_gap) if self.generalization_gap else 0]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Excel saved: {filepath}")
    
    def _save_to_pdf(self):
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_report.pdf")
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("PPO + LLaMA Training Report (Progressive Augmentation)", styles['Heading1']))
        elements.append(Spacer(1, 20))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Episodes', str(len(self.final_yields))],
            ['Avg Final Yield (kg/ha)', f"{np.mean(self.final_yields):.2f}"],
            ['Max Final Yield (kg/ha)', f"{np.max(self.final_yields):.2f}"],
            ['Avg Train Reward', f"{np.mean(self.episode_rewards):.2f}"],
            ['Max Train Reward', f"{np.max(self.episode_rewards):.2f}"],
            ['Avg Val Reward', f"{np.mean(self.val_rewards) if self.val_rewards else 0:.2f}"],
            ['Avg Generalization Gap', f"{np.mean(self.generalization_gap) if self.generalization_gap else 0:.2f}"]
        ]
        
        table = Table(summary_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        doc.build(elements)
        print(f"PDF saved: {filepath}")
    
    def _save_plots(self):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        episodes = list(range(1, len(self.final_yields) + 1))
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('PPO + LLaMA Training Metrics (Progressive Augmentation)', fontsize=16)
        
        # 1. 产量
        axes[0, 0].plot(episodes, self.final_yields, 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Final Yield (kg/ha)')
        axes[0, 0].set_title('Final Yield')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 回报曲线
        axes[0, 1].plot(episodes, self.episode_rewards, 'b-', alpha=0.5, label='Train')
        axes[0, 1].plot(episodes, self.avg_rewards, 'r-', linewidth=2, label='Avg Train')
        if self.val_rewards:
            val_ep = list(range(config.update_frequency, len(self.final_yields)+1, config.update_frequency))[:len(self.val_rewards)]
            axes[0, 1].plot(val_ep, self.val_rewards, 'g-', linewidth=2, label='Val')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 增强强度
        if self.aug_strengths:
            axes[0, 2].plot(episodes[:len(self.aug_strengths)], self.aug_strengths, 'purple', alpha=0.7)
            axes[0, 2].axhline(y=0.3, color='orange', linestyle='--', label='Aug Start')
            axes[0, 2].axhline(y=0.7, color='red', linestyle='--', label='Aug Full')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Aug Strength')
        axes[0, 2].set_title('Progressive Augmentation Strength')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-9. 其他指标...
        axes[1, 0].plot(episodes, self.wue_values, 'c-', alpha=0.7)
        axes[1, 0].set_title('WUE')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(episodes, self.nue_values, 'm-', alpha=0.7)
        axes[1, 1].set_title('NUE')
        axes[1, 1].grid(True, alpha=0.3)
        
        if self.generalization_gap:
            gap_ep = list(range(config.update_frequency, len(self.final_yields)+1, config.update_frequency))[:len(self.generalization_gap)]
            axes[1, 2].plot(gap_ep, self.generalization_gap, 'r-', alpha=0.7)
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Generalization Gap')
        axes[1, 2].grid(True, alpha=0.3)
        
        if self.exploration_coverages:
            axes[2, 0].plot(episodes[:len(self.exploration_coverages)], self.exploration_coverages, 'orange')
        axes[2, 0].set_title('Exploration Coverage')
        axes[2, 0].grid(True, alpha=0.3)
        
        if self.losses:
            axes[2, 1].plot(self.losses, 'purple', alpha=0.7)
        axes[2, 1].set_title('Loss')
        axes[2, 1].grid(True, alpha=0.3)
        
        if self.learning_rates:
            axes[2, 2].plot(self.learning_rates, 'brown', alpha=0.7)
        axes[2, 2].set_title('Learning Rate')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_metrics_plot.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filepath}")


# ============================================================================
#                              PPO Buffer
# ============================================================================

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
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def get_mini_batches(self, batch_size):
        indices = np.random.permutation(len(self.trajectories))
        return [indices[start:start + batch_size] for start in range(0, len(self.trajectories), batch_size)]


# ============================================================================
#                              PPO Agent
# ============================================================================

class PPOAgent:
    def __init__(self, llama_model, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        
        self.augmentor = ProgressiveAugmentor(config)
        self.ensemble = PolicyEnsemble(config.ensemble_size)
        
        self.embedder = LLaMAEmbedder(llama_model, config.projection_size).to(device)
        self.actor_critic = ActorCriticHead(config.projection_size, config.action_size, config.hidden_size).to(device)
        
        self.optimizer = optim.AdamW([
            {'params': self.embedder.llama.parameters(), 'lr': config.embedder_lr},
            {'params': self.embedder.projection.parameters(), 'lr': config.projection_lr},
            {'params': self.actor_critic.parameters(), 'lr': config.actor_lr},
        ], betas=config.betas, weight_decay=config.weight_decay)
        
        total_steps = config.n_episodes * config.max_steps_per_episode // config.update_frequency
        self.lr_scheduler = CosineAnnealingWarmup(self.optimizer, config.warmup_steps, total_steps, config.min_lr_ratio)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=not config.use_bf16)
        self.buffer = PPOBuffer(config.gamma, config.gae_lambda, device)
        self.inference_cache = {}
        self.rnd = RNDModule(config.state_size)
        
    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding='max_length', 
                              truncation=True, max_length=self.config.token_size).to(device)
    
    @torch.no_grad()
    def get_cached_embedding(self, state_str):
        if state_str in self.inference_cache:
            return self.inference_cache[state_str]
        
        inputs = self.tokenize([state_str])
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16 or self.scaler.is_enabled()):
            embed = self.embedder(inputs['input_ids'], inputs['attention_mask'])
        
        if len(self.inference_cache) > self.config.embedding_cache_size:
            self.inference_cache.popitem()
        self.inference_cache[state_str] = embed
        return embed

    def act(self, state, use_augmentation=False, episode=0, total_episodes=1):
        self.embedder.eval()
        self.actor_critic.eval()
        
        if use_augmentation:
            self.augmentor.set_mode(True)
            self.augmentor.update_progress(episode, total_episodes)
            augmented_state = self.augmentor.augment_state(state)
        else:
            self.augmentor.set_mode(False)
            augmented_state = state
        
        state_str = array2str(augmented_state)
        embedding = self.get_cached_embedding(state_str)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            dist, value = self.actor_critic.get_action_value(embedding)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item(), augmented_state
    
    def update(self, last_value, current_episode=0, total_episodes=None):
        self.embedder.train()
        self.actor_critic.train()
        
        use_intrinsic = self.config.use_rnd_intrinsic
        intrinsic_coef = self.config.intrinsic_coef
        
        if use_intrinsic and total_episodes is not None:
            progress = current_episode / total_episodes
            if progress > config.intrinsic_decay_end:
                use_intrinsic = False
        
        advantages, returns = self.buffer.compute_gae_and_normalize(
            last_value, use_intrinsic=use_intrinsic, intrinsic_coef=intrinsic_coef
        )
        
        all_state_strs = [t.state_str for t in self.buffer.trajectories]
        all_actions = torch.tensor([t.action for t in self.buffer.trajectories], dtype=torch.long, device=device)
        all_old_log_probs = torch.tensor([t.log_prob for t in self.buffer.trajectories], dtype=torch.float32, device=device)
        
        total_loss = 0
        n_updates = 0
        
        for epoch in range(self.config.ppo_epochs):
            mini_batches = self.buffer.get_mini_batches(self.config.mini_batch_size)
            
            for indices in mini_batches:
                self.optimizer.zero_grad()
                
                batch_strs = [all_state_strs[i] for i in indices]
                inputs = self.tokenize(batch_strs)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                    embeddings = self.embedder(inputs['input_ids'], inputs['attention_mask'])
                    dist, new_values = self.actor_critic.get_action_value(embeddings)
                    
                    new_log_probs = dist.log_prob(all_actions[indices])
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - all_old_log_probs[indices])
                    surr1 = ratio * advantages[indices]
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages[indices]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(new_values.squeeze(), returns[indices])
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                if self.config.use_bf16:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()
                
                if self.config.use_bf16:
                    torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), self.config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                else:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), self.config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                self.lr_scheduler.step()
                total_loss += loss.item()
                n_updates += 1
        
        self.buffer.clear()
        self.inference_cache.clear()
        
        return total_loss / max(n_updates, 1)
    
    def save_checkpoint(self, path, episode, score):
        os.makedirs(path, exist_ok=True)
        state_dict = {
            'embedder': self.embedder.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(path, f'model_ep{episode}.pth'))
        self.ensemble.add_checkpoint(state_dict, score)
    
    def load_best_model(self):
        best_state = self.ensemble.get_best_state()
        if best_state is not None:
            self.embedder.load_state_dict(best_state['embedder'])
            self.actor_critic.load_state_dict(best_state['actor_critic'])
            return True
        return False


# ============================================================================
#                              验证函数
# ============================================================================

def validate(agent, env, config, logger, episode):
    print(f"\n{'='*50}")
    print(f"  Validation Phase - Episode {episode}")
    print(f"{'='*50}")
    
    agent.augmentor.set_mode(False)
    agent.load_best_model()
    
    all_yields = []
    all_irrigations = []
    all_fertilizers = []
    all_rewards = []
    all_wues = []
    all_nues = []
    
    validation_seeds = [123456 + i * 1000 for i in range(config.n_validation_seeds)]
    
    for val_seed in validation_seeds:
        env_args = {
            'run_dssat_location': '/opt/dssat_pdi/run_dssat',
            'log_saving_path': f'/home/wuyang/results/llama_ppo_rnd_results/logs/dssat-pdi-val-{val_seed}.log',
            'mode': 'all', 'seed': val_seed, 'random_weather': True
        }
        
        try:
            val_env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        except Exception as e:
            print(f"Warning: Could not create validation env: {e}")
            continue
        
        for val_ep in range(config.n_validation_episodes_per_seed):
            state = dict2array(val_env.reset())
            episode_reward, n_amount, w_amount, ep_yield = 0, 0, 0, 0
            done = False
            steps = 0
            
            while not done and steps < config.max_steps_per_episode:
                action, log_prob, value, _ = agent.act(state, use_augmentation=False)
                
                action_dict = {
                    'anfer': (action % 5) * 40,
                    'amir': int(action / 5) * 6
                }
                
                if state[0] >= 10000: action_dict['anfer'] = 0
                if state[21] >= 1600: action_dict['amir'] = 0
                
                next_state_raw, _, done, _ = val_env.step(action_dict)
                next_state = dict2array(next_state_raw) if not done else state
                
                reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                                    next_state, done, config.k1, config.k2, config.k3)
                
                state = next_state
                episode_reward += reward
                n_amount += action_dict['anfer']
                w_amount += action_dict['amir']
                steps += 1
                
                if done:
                    ep_yield = state[4]
            
            all_yields.append(ep_yield)
            all_irrigations.append(w_amount)
            all_fertilizers.append(n_amount)
            all_rewards.append(episode_reward)
            all_wues.append(ep_yield / w_amount if w_amount > 0 else 0)
            all_nues.append(ep_yield / n_amount if n_amount > 0 else 0)
        
        val_env.close()
    
    if not all_yields:
        return None
    
    val_metrics = {
        'episode': episode,
        'final_yield': np.mean(all_yields),
        'final_yield_std': np.std(all_yields),
        'irrigation': np.mean(all_irrigations),
        'fertilizer': np.mean(all_fertilizers),
        'wue': np.mean(all_wues),
        'nue': np.mean(all_nues),
        'episode_reward': np.mean(all_rewards),
        'episode_reward_std': np.std(all_rewards),
        'n_validation_episodes': len(all_yields)
    }
    
    logger.record_validation(val_metrics['episode_reward'])
    
    print(f"\n  [Validation Results - {val_metrics['n_validation_episodes']} episodes]")
    print(f"    Avg Yield:         {val_metrics['final_yield']:.2f} ± {val_metrics['final_yield_std']:.2f} kg/ha")
    print(f"    Avg Reward:        {val_metrics['episode_reward']:.2f} ± {val_metrics['episode_reward_std']:.2f}")
    print(f"{'='*50}\n")
    
    return val_metrics


# ============================================================================
#                              训练主函数
# ============================================================================

def initialize_llama(model_path, use_bf16=True):
    print(f"Initializing LLaMA model: {model_path}")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = LlamaModel.from_pretrained(model_path, torch_dtype=torch_dtype, use_cache=False).to(device)
    model.gradient_checkpointing_enable()
    print(f"   Precision: {torch_dtype}")
    return model, tokenizer

def train_ppo():
    print("=" * 70)
    print("PPO + LLaMA Training Started (Progressive Augmentation Version)")
    print("=" * 70)
    
    print("\n[Progressive Augmentation Strategy]")
    print("  Phase 1 (0-30%):    No augmentation - learn basic policy")
    print("  Phase 2 (30-70%):   Gradually increase augmentation")
    print("  Phase 3 (70-100%):  Full augmentation - maximize generalization")

    model_path = '/home/wuyang/models/chinese-llama-2-1.3b'
    llama_model, tokenizer = initialize_llama(model_path, config.use_bf16)
    
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/llama_ppo_rnd_results/logs/dssat-pdi.log',
        'mode': 'all', 'seed': 123456, 'random_weather': True
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    agent = PPOAgent(llama_model, tokenizer, config)
    logger = MetricsLogger(config.output_prefix)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    scores, yields, best_score = [], [], float('-inf')
    
    print(f"\nStarting training ({config.n_episodes} episodes)")
    pbar = tqdm(range(1, config.n_episodes + 1), desc="Training", unit="episode", ncols=100)
    
    start_time = time.time()
    
    for episode in pbar:
        state = dict2array(env.reset())
        episode_reward, n_amount, w_amount, ep_yield = 0, 0, 0, 0
        steps = 0
        
        for step in range(config.max_steps_per_episode):
            action, log_prob, value, augmented_state = agent.act(
                state, use_augmentation=True, episode=episode, total_episodes=config.n_episodes
            )
            
            action_dict = {
                'anfer': (action % 5) * 40,
                'amir': int(action / 5) * 6
            }
            
            if state[0] >= 10000: action_dict['anfer'] = 0
            if state[21] >= 1600: action_dict['amir'] = 0
            
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            
            reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                                next_state, done, config.k1, config.k2, config.k3)
            
            # RND
            raw_intrinsic = agent.rnd.get_intrinsic_reward(augmented_state)
            agent.rnd.update_predictor(augmented_state)
            agent.rnd.update_exploration_stats(augmented_state, raw_intrinsic)
            
            if config.use_rnd_intrinsic:
                scaled_intrinsic, norm_intrinsic, _ = agent.rnd.get_scaled_intrinsic_reward(
                    augmented_state, episode, config.n_episodes
                )
                agent.buffer.add(augmented_state, action, reward, next_state, done, log_prob, value, 
                               array2str(augmented_state), intrinsic_reward=norm_intrinsic)
            else:
                agent.buffer.add(augmented_state, action, reward, next_state, done, log_prob, value, 
                               array2str(augmented_state))
            
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            steps += 1
            
            if done:
                ep_yield = state[4]
                break
        
        exploration_coverage = agent.rnd.get_exploration_coverage()
        aug_info = agent.augmentor.get_aug_strength_info()
        
        logger.record_episode(episode, ep_yield, w_amount, n_amount, 
                             episode_reward, steps, exploration_coverage, aug_info['aug_strength'])
        
        scores.append(episode_reward)
        yields.append(ep_yield)
        
        pbar.set_postfix({
            'reward': f'{episode_reward:.0f}',
            'yield': f'{ep_yield:.0f}',
            'aug': f'{aug_info["aug_strength"]:.0%}'
        })
        
        # 更新
        if episode % config.update_frequency == 0 and len(agent.buffer) >= config.mini_batch_size:
            with torch.no_grad():
                last_val = agent.actor_critic(agent.get_cached_embedding(array2str(state)))[1].item()
                if done: last_val = 0.0
            
            loss = agent.update(last_val, current_episode=episode, total_episodes=config.n_episodes)
            lr = agent.lr_scheduler.get_lr()[0]
            logger.record_update(loss, lr)
        
        # 每10轮打印
        if episode % 10 == 0:
            logger.print_metrics(episode, phase="Training")
            logger.save_metrics_every_10(episode)
        
        # 每100轮验证
        if episode % 100 == 0:
            val_metrics = validate(agent, env, config, logger, episode)
            if val_metrics and early_stopping(val_metrics['episode_reward'], episode):
                print("\n[Early Stopping] Training stopped")
                break
        
        # 保存最佳
        if  episode > 880 and episode_reward > 1100 and ep_yield > 11000:
            best_score = episode_reward
            agent.save_checkpoint(
                f'/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best_0417', 
                episode, episode_reward
            )
    
    pbar.close()
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  Training Completed! Total Time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    # 最终验证
    print("\n[Final Validation]")
    validate(agent, env, config, logger, config.n_episodes)
    
    logger.print_metrics(config.n_episodes, phase="Final")
    logger.save_all_results()
    
    env.close()
    print("\nTraining finished successfully!")
    return logger


if __name__ == "__main__":
    os.makedirs(f'/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints', exist_ok=True)
    os.makedirs('/home/wuyang/results/llama_ppo_rnd_results/logs', exist_ok=True)
    train_ppo()