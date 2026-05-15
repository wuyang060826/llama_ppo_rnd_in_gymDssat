#!/usr/bin/env python3
"""
PPO + Chinese-LLaMA-2 + gym-dssat 深度优化修复版 v3
针对 A100 40G 优化

修复核心:
1. 梯度流修复 - 训练时强制重新计算嵌入，确保LLaMA和投影层参与反向传播
2. 缓存一致性 - 训练阶段禁用缓存，避免旧参数嵌入干扰策略更新
3. BF16支持 - 利用A100的BF16加速并提升数值稳定性

新增功能:
4. 农学指标输出 - 最终产量、灌溉/施肥量、WUE、NUE
5. AI指标输出 - 平均回报、样本效率、收敛速度、探索覆盖率
6. 结果保存 - PNG/PDF/Excel格式
7. 时间进度条 - 训练过程可视化

v2修复（根据修改意见）:
- 验证使用greedy策略（与基线一致）
- 修复PPO last_value处理（不再强制设为0）
- 取消外部奖励batch-level标准化（与基线/论文一致）
- 验证触发条件对齐基线

v3修复（RND内在奖励双重缩放问题）:
- 在RNDModule中完成"归一化 + 系数缩放 + 衰减"，返回scaled_intrinsic
- 训练循环中直接把scaled_intrinsic写入buffer
- PPOBuffer.compute_gae_and_normalize中不再乘intrinsic_coef，直接相加
"""

import numpy as np
import pandas as pd
import random
from collections import deque, OrderedDict
import time
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# For PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 显存碎片优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # 开启TF32加速 (A100支持)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"   TF32 Acceleration: Enabled")

# 导入transformers
from transformers import LlamaModel, LlamaTokenizerFast

# ============================================================================
#                              超参数配置
# ============================================================================

@dataclass
class PPOConfig:
    """PPO超参数配置 - 修复版v3"""
    
    # === 训练参数 ===
    n_episodes: int = 2000              
    max_steps_per_episode: int = 200    
    
    # === PPO核心参数 ===
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.02             # 适当放宽KL限制
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0          # 放宽梯度裁剪
    
    # === 优化器参数 ===
    actor_lr: float = 5e-5              # 提高Actor学习率
    critic_lr: float = 1e-4             # Critic学习率
    embedder_lr: float = 1e-5           # LLaMA微调学习率
    projection_lr: float = 1e-4         # 投影层学习率
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # === 学习率调度 ===
    warmup_steps: int = 200             # 增加预热步数
    min_lr_ratio: float = 0.05
    
    # === PPO更新参数 ===
    ppo_epochs: int = 5                 # 增加更新轮数
    mini_batch_size: int = 64           
    update_frequency: int = 10         
    gradient_accumulation_steps: int = 1  # 修复：暂不使用累积，简化逻辑
    
    # === 网络参数 ===
    token_size: int = 128
    state_size: int = 25
    action_size: int = 25
    hidden_size: int = 256
    projection_size: int = 256
    
    # === 缓存与内存 ===
    embedding_cache_size: int = 100     # 缓存仅用于推理加速
    
    # === 奖励函数参数 ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === 奖励归一化 ===
    reward_norm: bool = True           # v2修复：取消batch-level外部奖励归一化，与基线一致
    
    # === 探索参数 ===
    entropy_decay: float = 0.999
    entropy_min: float = 0.001
    
    # === 优化选项 ===
    use_bf16: bool = True               # A100使用BF16更稳定
    
    # === 指标计算参数 ===
    expert_performance_threshold: float = 1000.0  # v2修复：与基线一致，样本效率阈值
    convergence_window: int = 50                 # 收敛判定窗口大小
    convergence_threshold: float = 0.1           # v2修复：与基线一致，收敛判定阈值
    rnd_threshold: float = 0.1                   # RND内在奖励衰减阈值，与新文档3一致
    
    # === RND内在奖励参数（保守探索配置）===
    use_rnd_intrinsic: bool = True               # 是否启用RND内在奖励参与策略更新
    intrinsic_coef: float = 0.01                 # 内在奖励系数（非常保守，防止干扰外部奖励）
    intrinsic_decay_start: float = 0.3           # 开始衰减的训练进度比例（30%后开始衰减）
    intrinsic_decay_end: float = 0.7             # 完全衰减的训练进度比例（70%后完全关闭）
    intrinsic_reward_scale: float = 1.0          # 内在奖励缩放因子（归一化后的缩放）
    
    # === 验证参数（与基线一致）===
    validation_interval: int = 100               # 验证间隔
    validation_episodes: int = 10                # 验证episode数
    validation_min_episode: int = 600            # 验证触发最小episode
    validation_min_reward: float = 1000.0        # 验证触发最小reward
    validation_min_yield: float = 10000.0        # 验证触发最小yield
    
    # === 结果保存路径 ===
    output_prefix: str = "llama_ppo_rnd"

config = PPOConfig()


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
    parts = []
    for i, num in enumerate(state):
        # 优化文本描述，使其更符合LLM预训练分布
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
            nn.Dropout(0.1),
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
#                    用于探索覆盖率计算
# ============================================================================

class RNDNetwork(nn.Module):
    """RND预测网络"""
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
    """运行时均值和标准差计算，用于内在奖励归一化"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        """更新统计量"""
        x = np.array(x)
        if x.size == 0:
            return
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size
        
        # 合并统计量
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
        """归一化"""
        normalized = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normalized, -clip_range, clip_range)


class RNDModule:
    """
    RND模块用于计算探索覆盖率和内在奖励
    
    v3修复：内在奖励缩放在此模块中统一完成
    - 归一化：使用RunningMeanStd
    - 系数缩放：乘以intrinsic_coef
    - 衰减：基于训练进度的线性衰减
    """
    def __init__(self, state_size, hidden_size=128, lr=1e-4):
        self.target_network = RNDNetwork(state_size, hidden_size).to(device)
        self.predictor_network = RNDNetwork(state_size, hidden_size).to(device)
        
        # 冻结目标网络
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)
        
        # 状态访问记录
        self.state_visit_counts = {}  # 离散化状态 -> 访问次数
        self.total_states_explored = 0
        self.exploration_threshold = config.rnd_threshold
        
        # 状态离散化参数
        self.state_bins = 10
        
        # 内在奖励归一化（关键：独立的归一化器，不影响外部奖励）
        self.intrinsic_rms = RunningMeanStd()
        
        # 训练进度追踪（用于衰减）
        self.current_episode = 0
        self.total_episodes = config.n_episodes
        
    def discretize_state(self, state):
        """将连续状态离散化用于探索统计（与已收敛新文档3一致：只取4个关键维度）"""
        key_dims = [0, 4, 20, 21]  # day, yield, soil moisture, irrigation
        discretized = tuple(int(state[i] / 100) for i in key_dims if i < len(state))
        return discretized
    
    def get_intrinsic_reward(self, state):
        """计算内在奖励（预测误差）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predictor_output = self.predictor_network(state_tensor)
        prediction_error = F.mse_loss(predictor_output, target_output, reduction='none').mean().item()
        
        return prediction_error
    
    def update_predictor(self, state):
        """更新预测网络"""
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
        """更新探索统计"""
        discretized = self.discretize_state(state)
        
        if discretized not in self.state_visit_counts:
            self.state_visit_counts[discretized] = 0
        
        self.state_visit_counts[discretized] += 1
        
        # 如果内在奖励低于阈值，认为该状态已被充分探索
        if intrinsic_reward < self.exploration_threshold:
            if self.state_visit_counts[discretized] == 1:  # 首次被充分探索
                self.total_states_explored += 1
    
    def get_exploration_coverage(self, estimated_total_states=1000):
        """计算探索覆盖率（与已收敛新文档3一致）"""
        coverage = min(100.0, (self.total_states_explored / estimated_total_states) * 100)
        return coverage
    
    def compute_intrinsic_decay(self, current_episode, total_episodes):
        """
        计算内在奖励的衰减系数
        基于训练进度：在intrinsic_decay_start之前为1.0，之后线性衰减到0
        
        Args:
            current_episode: 当前episode
            total_episodes: 总episode数
        
        Returns:
            衰减系数 [0, 1]
        """
        progress = current_episode / total_episodes
        
        if progress < config.intrinsic_decay_start:
            # 在衰减开始前，系数为1.0
            return 1.0
        elif progress > config.intrinsic_decay_end:
            # 在衰减结束后，系数为0.0
            return 0.0
        else:
            # 线性衰减
            decay_progress = (progress - config.intrinsic_decay_start) / (config.intrinsic_decay_end - config.intrinsic_decay_start)
            return 1.0 - decay_progress
    
    def get_scaled_intrinsic_reward(self, state, current_episode, total_episodes):
        """
        v3修复：在此方法中完成所有缩放，避免双重缩放问题
        
        整合了：
        1. 内在奖励计算（预测误差）
        2. 独立归一化（RunningMeanStd）
        3. 系数缩放（intrinsic_coef）
        4. 衰减系数（decay_coef）
        
        返回的scaled_intrinsic可以直接写入buffer，无需在GAE中再次缩放
        
        Returns:
            scaled_intrinsic: 最终用于策略更新的内在奖励值（已包含所有缩放）
            raw_intrinsic: 原始内在奖励（用于日志记录和探索统计）
            decay_coef: 当前衰减系数（用于日志记录）
        """
        # 1. 计算原始内在奖励（预测误差）
        raw_intrinsic = self.get_intrinsic_reward(state)
        
        # 2. 更新归一化统计量并归一化
        self.intrinsic_rms.update(np.array([raw_intrinsic]))
        normalized_intrinsic = self.intrinsic_rms.normalize(raw_intrinsic)
        
        # 3. 计算衰减系数
        decay_coef = self.compute_intrinsic_decay(current_episode, total_episodes)
        
        # 4. v3修复：在此处完成所有缩放
        # scaled_intrinsic = 归一化值 × 系数 × 衰减 × 额外缩放
        scaled_intrinsic = normalized_intrinsic * config.intrinsic_coef * decay_coef * config.intrinsic_reward_scale
        
        return scaled_intrinsic, raw_intrinsic, decay_coef


# ============================================================================
#                              指标记录器
# ============================================================================

class MetricsLogger:
    """指标记录器 - 记录并保存所有训练指标"""
    
    def __init__(self, output_prefix="llama_ppo_rnd"):
        self.output_prefix = output_prefix
        self.output_dir = f"/home/wuyang/results/llama_ppo_rnd_results/0415_v2"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 农学指标
        self.final_yields = []          # 最终产量 (kg/ha)
        self.irrigation_amounts = []    # 灌溉量 (mm)
        self.fertilizer_amounts = []    # 施肥量 (kg/ha)
        self.wue_values = []            # 水分利用率 (kg/mm)
        self.nue_values = []            # 氮肥利用率 (kg/kg)
        
        # AI指标
        self.episode_rewards = []       # 每轮回报
        self.avg_rewards = []           # 平均回报
        self.sample_efficiency = None   # 样本效率 (步)
        self.convergence_speed = None   # 收敛速度 (步)
        self.exploration_coverages = [] # 探索覆盖率 (%)
        
        # 训练过程记录
        self.episode_steps = []         # 每轮步数
        self.total_steps = 0            # 总交互步数
        self.losses = []                # 损失值
        self.learning_rates = []        # 学习率
        
        # 样本效率追踪
        self.expert_threshold_reached = False
        self.sample_efficiency_step = None
        
        # 收敛速度追踪
        self.convergence_reached = False
        self.convergence_step = None
        self.recent_rewards = deque(maxlen=config.convergence_window)
        
        # 每10轮记录
        self.metrics_every_10 = []
        
    def record_episode(self, episode, yield_val, irrigation, fertilizer, 
                       reward, steps, exploration_coverage=None):
        """记录单轮训练数据"""
        self.final_yields.append(yield_val)
        self.irrigation_amounts.append(irrigation)
        self.fertilizer_amounts.append(fertilizer)
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.total_steps += steps
        
        # 计算WUE和NUE
        wue = yield_val / irrigation if irrigation > 0 else 0
        nue = yield_val / fertilizer if fertilizer > 0 else 0
        self.wue_values.append(wue)
        self.nue_values.append(nue)
        
        if exploration_coverage is not None:
            self.exploration_coverages.append(exploration_coverage)
        
        # 计算平均回报
        avg_reward = np.mean(self.episode_rewards[-min(10, len(self.episode_rewards)):])
        self.avg_rewards.append(avg_reward)
        
        # 更新最近回报队列
        self.recent_rewards.append(reward)
        
        # 检查样本效率
        if not self.expert_threshold_reached and avg_reward >= config.expert_performance_threshold:
            self.expert_threshold_reached = True
            self.sample_efficiency = self.total_steps
            self.sample_efficiency_step = episode
        
        # 检查收敛速度
        if not self.convergence_reached and len(self.recent_rewards) >= config.convergence_window:
            reward_std = np.std(list(self.recent_rewards))
            reward_mean = np.mean(list(self.recent_rewards))
            if reward_mean > 0 and reward_std / reward_mean < config.convergence_threshold:
                self.convergence_reached = True
                self.convergence_speed = self.total_steps
                self.convergence_step = episode
        
    def record_update(self, loss, lr):
        """记录更新数据"""
        self.losses.append(loss)
        self.learning_rates.append(lr)
        
    def get_current_metrics(self, episode):
        """获取当前指标字典"""
        metrics = {
            'episode': episode,
            # 农学指标
            'final_yield': self.final_yields[-1] if self.final_yields else 0,
            'irrigation': self.irrigation_amounts[-1] if self.irrigation_amounts else 0,
            'fertilizer': self.fertilizer_amounts[-1] if self.fertilizer_amounts else 0,
            'wue': self.wue_values[-1] if self.wue_values else 0,
            'nue': self.nue_values[-1] if self.nue_values else 0,
            # AI指标
            'episode_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'avg_reward': self.avg_rewards[-1] if self.avg_rewards else 0,
            'sample_efficiency': self.sample_efficiency if self.sample_efficiency else 'N/A',
            'convergence_speed': self.convergence_speed if self.convergence_speed else 'N/A',
            'exploration_coverage': self.exploration_coverages[-1] if self.exploration_coverages else 0,
            # 训练信息
            'total_steps': self.total_steps,
            'loss': self.losses[-1] if self.losses else 0,
            'learning_rate': self.learning_rates[-1] if self.learning_rates else 0
        }
        return metrics
    
    def print_metrics(self, episode, phase="Training"):
        """打印当前指标"""
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
        print(f"    Sample Efficiency:  {metrics['sample_efficiency']} steps")
        print(f"    Convergence Speed:  {metrics['convergence_speed']} steps")
        print(f"    Exploration:        {metrics['exploration_coverage']:.2f}%")
        
        print(f"\n  [Training Info]")
        print(f"    Total Steps:        {metrics['total_steps']}")
        print(f"    Loss:               {metrics['loss']:.4f}")
        print(f"    Learning Rate:      {metrics['learning_rate']:.2e}")
        print(f"{'='*70}\n")
        
    def save_metrics_every_10(self, episode):
        """保存每10轮的指标"""
        metrics = self.get_current_metrics(episode)
        self.metrics_every_10.append(metrics)
        
    def save_all_results(self):
        """保存所有结果"""
        self._save_to_excel()
        self._save_to_pdf()
        self._save_plots()
        print(f"\nAll results saved to: {self.output_dir}")
        
    def _save_to_excel(self):
        """保存到Excel文件"""
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_metrics.xlsx")
        
        # 创建DataFrame
        data = {
            'Episode': list(range(1, len(self.final_yields) + 1)),
            'Final_Yield_kg_ha': self.final_yields,
            'Irrigation_mm': self.irrigation_amounts,
            'Fertilizer_kg_ha': self.fertilizer_amounts,
            'WUE_kg_mm': self.wue_values,
            'NUE_kg_kg': self.nue_values,
            'Episode_Reward': self.episode_rewards,
            'Avg_Reward': self.avg_rewards,
            'Total_Steps': np.cumsum(self.episode_steps),
            'Exploration_Coverage_pct': self.exploration_coverages + [0]*(len(self.final_yields) - len(self.exploration_coverages))
        }
        
        df = pd.DataFrame(data)
        
        # 添加样本效率和收敛速度列
        df['Sample_Efficiency_steps'] = self.sample_efficiency if self.sample_efficiency else 'N/A'
        df['Convergence_Speed_steps'] = self.convergence_speed if self.convergence_speed else 'N/A'
        
        # 保存到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Episodes', index=False)
            
            # 每10轮汇总
            if self.metrics_every_10:
                df_10 = pd.DataFrame(self.metrics_every_10)
                df_10.to_excel(writer, sheet_name='Every_10_Episodes', index=False)
            
            # 整体统计
            summary_data = {
                'Metric': [
                    'Total_Episodes',
                    'Total_Steps',
                    'Avg_Final_Yield_kg_ha',
                    'Max_Final_Yield_kg_ha',
                    'Avg_Irrigation_mm',
                    'Avg_Fertilizer_kg_ha',
                    'Avg_WUE_kg_mm',
                    'Avg_NUE_kg_kg',
                    'Avg_Episode_Reward',
                    'Max_Episode_Reward',
                    'Sample_Efficiency_steps',
                    'Convergence_Speed_steps',
                    'Final_Exploration_Coverage_pct'
                ],
                'Value': [
                    len(self.final_yields),
                    self.total_steps,
                    np.mean(self.final_yields),
                    np.max(self.final_yields),
                    np.mean(self.irrigation_amounts),
                    np.mean(self.fertilizer_amounts),
                    np.mean(self.wue_values),
                    np.mean(self.nue_values),
                    np.mean(self.episode_rewards),
                    np.max(self.episode_rewards),
                    self.sample_efficiency if self.sample_efficiency else 'N/A',
                    self.convergence_speed if self.convergence_speed else 'N/A',
                    self.exploration_coverages[-1] if self.exploration_coverages else 0
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Excel saved: {filepath}")
        
    def _save_to_pdf(self):
        """保存到PDF报告"""
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_report.pdf")
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        
        # 标题
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        elements.append(Paragraph("PPO + LLaMA Training Report", title_style))
        elements.append(Spacer(1, 20))
        
        # 整体统计表
        elements.append(Paragraph("Overall Statistics", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Episodes', str(len(self.final_yields))],
            ['Total Steps', str(self.total_steps)],
            ['Avg Final Yield (kg/ha)', f"{np.mean(self.final_yields):.2f}"],
            ['Max Final Yield (kg/ha)', f"{np.max(self.final_yields):.2f}"],
            ['Avg Irrigation (mm)', f"{np.mean(self.irrigation_amounts):.2f}"],
            ['Avg Fertilizer (kg/ha)', f"{np.mean(self.fertilizer_amounts):.2f}"],
            ['Avg WUE (kg/mm)', f"{np.mean(self.wue_values):.4f}"],
            ['Avg NUE (kg/kg)', f"{np.mean(self.nue_values):.4f}"],
            ['Avg Episode Reward', f"{np.mean(self.episode_rewards):.2f}"],
            ['Max Episode Reward', f"{np.max(self.episode_rewards):.2f}"],
            ['Sample Efficiency (steps)', str(self.sample_efficiency) if self.sample_efficiency else 'N/A'],
            ['Convergence Speed (steps)', str(self.convergence_speed) if self.convergence_speed else 'N/A'],
            ['Final Exploration Coverage (%)', f"{self.exploration_coverages[-1] if self.exploration_coverages else 0:.2f}"]
        ]
        
        table = Table(summary_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # 每10轮指标表
        if self.metrics_every_10:
            elements.append(Paragraph("Metrics Every 10 Episodes", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # 只显示关键指标
            metrics_10_data = [['Episode', 'Yield (kg/ha)', 'WUE', 'NUE', 'Avg Reward', 'Exploration (%)']]
            for m in self.metrics_every_10:
                metrics_10_data.append([
                    str(m['episode']),
                    f"{m['final_yield']:.2f}",
                    f"{m['wue']:.4f}",
                    f"{m['nue']:.4f}",
                    f"{m['avg_reward']:.2f}",
                    f"{m['exploration_coverage']:.2f}"
                ])
            
            table_10 = Table(metrics_10_data, colWidths=[1*inch, 1.2*inch, 1*inch, 1*inch, 1.2*inch, 1.2*inch])
            table_10.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table_10)
        
        doc.build(elements)
        print(f"PDF saved: {filepath}")
        
    def _save_plots(self):
        """保存图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        episodes = list(range(1, len(self.final_yields) + 1))
        
        # 创建多子图
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('PPO + LLaMA Training Metrics', fontsize=16)
        
        # 1. 最终产量
        ax = axes[0, 0]
        ax.plot(episodes, self.final_yields, 'b-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Final Yield (kg/ha)')
        ax.set_title('Final Yield')
        ax.grid(True, alpha=0.3)
        
        # 2. 灌溉量
        ax = axes[0, 1]
        ax.plot(episodes, self.irrigation_amounts, 'g-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Irrigation (mm)')
        ax.set_title('Irrigation Amount')
        ax.grid(True, alpha=0.3)
        
        # 3. 施肥量
        ax = axes[0, 2]
        ax.plot(episodes, self.fertilizer_amounts, 'r-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fertilizer (kg/ha)')
        ax.set_title('Fertilizer Amount')
        ax.grid(True, alpha=0.3)
        
        # 4. WUE
        ax = axes[1, 0]
        ax.plot(episodes, self.wue_values, 'c-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('WUE (kg/mm)')
        ax.set_title('Water Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 5. NUE
        ax = axes[1, 1]
        ax.plot(episodes, self.nue_values, 'm-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('NUE (kg/kg)')
        ax.set_title('Nitrogen Use Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 6. 回报曲线
        ax = axes[1, 2]
        ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.5, label='Episode Reward')
        ax.plot(episodes, self.avg_rewards, 'r-', linewidth=2, label='Avg Reward')
        ax.axhline(y=config.expert_performance_threshold, color='g', linestyle='--', label='Expert Threshold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. 探索覆盖率
        ax = axes[2, 0]
        if self.exploration_coverages:
            ax.plot(episodes[:len(self.exploration_coverages)], self.exploration_coverages, 'orange', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Coverage (%)')
        ax.set_title('Exploration Coverage')
        ax.grid(True, alpha=0.3)
        
        # 8. 损失曲线
        ax = axes[2, 1]
        if self.losses:
            ax.plot(self.losses, 'purple', alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # 9. 学习率曲线
        ax = axes[2, 2]
        if self.learning_rates:
            ax.plot(self.learning_rates, 'brown', alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.output_prefix}_metrics_plot.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filepath}")
        
        # 额外保存单独的回报曲线图
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(episodes, self.episode_rewards, 'b-', alpha=0.5, label='Episode Reward')
        ax2.plot(episodes, self.avg_rewards, 'r-', linewidth=2, label='Avg Reward (10-ep window)')
        
        # 标记样本效率和收敛速度
        if self.sample_efficiency_step:
            ax2.axvline(x=self.sample_efficiency_step, color='g', linestyle='--', 
                       label=f'Sample Efficiency (Ep {self.sample_efficiency_step})')
        if self.convergence_step:
            ax2.axvline(x=self.convergence_step, color='purple', linestyle='--',
                       label=f'Convergence (Ep {self.convergence_step})')
        
        ax2.axhline(y=config.expert_performance_threshold, color='orange', linestyle=':', 
                   label='Expert Threshold')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Training Reward Curve with Key Milestones', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        filepath2 = os.path.join(self.output_dir, f"{self.output_prefix}_reward_curve.png")
        plt.savefig(filepath2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Reward curve saved: {filepath2}")


# ============================================================================
#                              PPO Buffer (修复版v3)
# ============================================================================

@dataclass
class Trajectory:
    """轨迹数据结构 - 支持内在奖励"""
    state: np.ndarray
    action: int
    reward: float              # 外部奖励
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float
    state_str: str
    intrinsic_reward: float = 0.0  # 内在奖励（v3修复：已是完全缩放后的值）


class PPOBuffer:
    """
    PPO Buffer - 支持RND内在奖励
    
    v3修复：不再对内在奖励进行额外缩放
    - intrinsic_reward已在RNDModule中完成所有缩放（归一化 + coef + decay）
    - 在此直接相加：advantages = advantages_ext + advantages_int
    
    v2修复：
    1. 取消外部奖励的batch-level标准化（与基线/论文一致）
    2. 内在奖励独立归一化，不影响外部奖励
    3. Value函数只学习外部奖励
    """
    def __init__(self, gamma, gae_lambda, device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.trajectories: List[Trajectory] = []
        
        # 修改前：没有这两行
        # 修改后：补充奖励归一化统计量（仅用于外部奖励），与新文档3完全对齐
        self.reward_mean = 0.0
        self.reward_std = 1.0

        
    def add(self, state, action, reward, next_state, done, log_prob, value, state_str, intrinsic_reward=0.0):
        """添加轨迹，支持可选的内在奖励"""
        self.trajectories.append(Trajectory(
            state, action, reward, next_state, done, log_prob, value, state_str, intrinsic_reward
        ))
    
    def clear(self): self.trajectories = []
    def __len__(self): return len(self.trajectories)
    
    def compute_gae_and_normalize(self, last_value, use_intrinsic=True):
        n = len(self.trajectories)
        ext_rewards = np.array([t.reward for t in self.trajectories])
        int_rewards = np.array([t.intrinsic_reward for t in self.trajectories])

        # ================= 开始修改 =================
        # 修改前：
        # # v2修复：不再进行batch-level外部奖励归一化，直接使用原始奖励
        # ext_rewards_t = torch.tensor(ext_rewards, dtype=torch.float32, device=self.device)
        # 修改后（与新文档3完全一致）：
        # 1. 外部奖励归一化（使用batch统计量，保持原有稳定性）
        self.reward_mean = ext_rewards.mean()
        self.reward_std = ext_rewards.std() + 1e-8
        norm_ext_rewards = (ext_rewards - self.reward_mean) / self.reward_std
        # 2. 转换为Tensor
        ext_rewards_t = torch.tensor(norm_ext_rewards, dtype=torch.float32, device=self.device)
        # ================= 修改结束 =================
        int_rewards_t = torch.tensor(int_rewards, dtype=torch.float32, device=self.device)
        values_t = torch.tensor([t.value for t in self.trajectories], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor([t.done for t in self.trajectories], dtype=torch.float32, device=self.device)
        
        # GAE计算（只用外部奖励）
        advantages_ext = torch.zeros(n, dtype=torch.float32, device=self.device)
        gae_ext = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones_t[t]
            else:
                next_value = values_t[t + 1]
                next_non_terminal = 1.0 - dones_t[t]
            
            delta_ext = ext_rewards_t[t] + self.gamma * next_value * next_non_terminal - values_t[t]
            gae_ext = delta_ext + self.gamma * self.gae_lambda * next_non_terminal * gae_ext
            advantages_ext[t] = gae_ext
        
        # Value函数的returns（只用外部奖励）
        returns = advantages_ext + values_t
        
        # v3修复：计算用于Policy的Advantages
        if use_intrinsic and int_rewards.sum() != 0:
            # 内在奖励也计算GAE（但不影响Value）
            advantages_int = torch.zeros(n, dtype=torch.float32, device=self.device)
            gae_int = 0.0
            
            for t in reversed(range(n)):
                next_non_terminal = 1.0 - dones_t[t]
                
                # 内在奖励的delta（简化：不使用value bootstrapping）
                delta_int = int_rewards_t[t] + self.gamma * 0.0 * next_non_terminal
                gae_int = delta_int + self.gamma * self.gae_lambda * next_non_terminal * gae_int
                advantages_int[t] = gae_int
            
            # v3修复：直接相加，不再乘intrinsic_coef
            # int_rewards已经在RNDModule中完成了所有缩放
            advantages = advantages_ext + advantages_int
        else:
            advantages = advantages_ext
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def get_mini_batches(self, batch_size):
        indices = np.random.permutation(len(self.trajectories))
        return [indices[start:start + batch_size] for start in range(0, len(self.trajectories), batch_size)]


# ============================================================================
#                              PPO Agent (修复版v3)
# ============================================================================

class PPOAgent:
    def __init__(self, llama_model, tokenizer, config, actual_state_size=None):
        self.config = config
        self.tokenizer = tokenizer
        
        # 1. 网络
        self.embedder = LLaMAEmbedder(llama_model, config.projection_size).to(device)
        self.actor_critic = ActorCriticHead(config.projection_size, config.action_size, config.hidden_size).to(device)
        
        # 2. 优化器
        self.optimizer = optim.AdamW([
            {'params': self.embedder.llama.parameters(), 'lr': config.embedder_lr},
            {'params': self.embedder.projection.parameters(), 'lr': config.projection_lr},
            {'params': self.actor_critic.parameters(), 'lr': config.actor_lr},
        ], betas=config.betas, weight_decay=config.weight_decay)
        
        # 3. 调度器
        total_steps = config.n_episodes * config.max_steps_per_episode // config.update_frequency
        self.lr_scheduler = CosineAnnealingWarmup(self.optimizer, config.warmup_steps, total_steps, config.min_lr_ratio)
        
        # 4. 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=not config.use_bf16)
        
        # 5. Buffer
        self.buffer = PPOBuffer(config.gamma, config.gae_lambda, device)
        
        # 6. 推理缓存 (仅用于act)
        self.inference_cache = {}
        
        # 7. RND模块 - 使用实际状态维度
        state_size_for_rnd = actual_state_size if actual_state_size is not None else config.state_size
        self.rnd = RNDModule(state_size_for_rnd)
        self.actual_state_size = state_size_for_rnd
        
    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding='max_length', 
                              truncation=True, max_length=self.config.token_size).to(device)
    
    @torch.no_grad()
    def get_cached_embedding(self, state_str):
        """推理专用，带缓存"""
        if state_str in self.inference_cache:
            return self.inference_cache[state_str]
        
        inputs = self.tokenize([state_str])
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16 or self.scaler.is_enabled()):
            embed = self.embedder(inputs['input_ids'], inputs['attention_mask'])
        
        if len(self.inference_cache) > self.config.embedding_cache_size:
            self.inference_cache.popitem()
        self.inference_cache[state_str] = embed
        return embed

    def act(self, state):
        """训练时使用，带随机采样"""
        self.embedder.eval()
        self.actor_critic.eval()
        
        state_str = array2str(state)
        embedding = self.get_cached_embedding(state_str)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            dist, value = self.actor_critic.get_action_value(embedding)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def act_greedy(self, state):
        """贪婪策略选择动作（用于验证）"""
        self.embedder.eval()
        self.actor_critic.eval()
        
        state_str = array2str(state)
        embedding = self.get_cached_embedding(state_str)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            dist, value = self.actor_critic.get_action_value(embedding)
            action = torch.argmax(dist.probs)
            
        return action.item()
    
    def update(self, last_value, current_episode=0, total_episodes=None):
        """PPO策略更新"""
        self.embedder.train()
        self.actor_critic.train()
        
        # 确定是否使用内在奖励
        use_intrinsic = self.config.use_rnd_intrinsic
        
        # 计算当前衰减系数
        if use_intrinsic and total_episodes is not None:
            decay_coef = self.rnd.compute_intrinsic_decay(current_episode, total_episodes)
            if decay_coef <= 0:
                use_intrinsic = False
        
        # 1. 计算GAE
        advantages, returns = self.buffer.compute_gae_and_normalize(
            last_value, 
            use_intrinsic=use_intrinsic
        )
        
        # 2. 准备数据
        all_state_strs = [t.state_str for t in self.buffer.trajectories]
        all_actions = torch.tensor([t.action for t in self.buffer.trajectories], dtype=torch.long, device=device)
        all_old_log_probs = torch.tensor([t.log_prob for t in self.buffer.trajectories], dtype=torch.float32, device=device)
        
        total_loss = 0
        n_updates = 0
        
        # 3. PPO Iterations
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
        
        # 4. 清理
        self.buffer.clear()
        self.inference_cache.clear()
        
        return total_loss / max(n_updates, 1)
    
    def save(self, path, episode):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'embedder': self.embedder.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(path, f'model_ep{episode}.pth'))


# ============================================================================
#                              验证函数
# ============================================================================

def validate(agent, env, config, logger, episode):
    """验证函数 - 使用greedy策略"""
    print(f"\n{'='*50}")
    print(f"  Validation Phase - Episode {episode}")
    print(f"  (Using Greedy Policy)")
    print(f"{'='*50}")
    
    n_val_episodes = config.validation_episodes
    val_yields = []
    val_irrigations = []
    val_fertilizers = []
    val_rewards = []
    val_wues = []
    val_nues = []
    
    for val_ep in range(n_val_episodes):
        state = dict2array(env.reset())
        episode_reward, n_amount, w_amount, ep_yield = 0, 0, 0, 0
        done = False
        steps = 0
        
        while not done and steps < config.max_steps_per_episode:
            action = agent.act_greedy(state)
            
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
            
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            steps += 1
            
            if done:
                ep_yield = state[4]
        
        val_yields.append(ep_yield)
        val_irrigations.append(w_amount)
        val_fertilizers.append(n_amount)
        val_rewards.append(episode_reward)
        val_wues.append(ep_yield / w_amount if w_amount > 0 else 0)
        val_nues.append(ep_yield / n_amount if n_amount > 0 else 0)
    
    val_metrics = {
        'episode': episode,
        'final_yield': np.mean(val_yields),
        'irrigation': np.mean(val_irrigations),
        'fertilizer': np.mean(val_fertilizers),
        'wue': np.mean(val_wues),
        'nue': np.mean(val_nues),
        'episode_reward': np.mean(val_rewards),
        'avg_reward': np.mean(val_rewards),
        'exploration_coverage': agent.rnd.get_exploration_coverage(),
        'total_steps': logger.total_steps,
        'loss': logger.losses[-1] if logger.losses else 0,
        'learning_rate': logger.learning_rates[-1] if logger.learning_rates else 0,
        'sample_efficiency': logger.sample_efficiency if logger.sample_efficiency else 'N/A',
        'convergence_speed': logger.convergence_speed if logger.convergence_speed else 'N/A'
    }
    
    print(f"\n  [Validation Results - {n_val_episodes} episodes]")
    print(f"    Avg Yield:         {val_metrics['final_yield']:.2f} kg/ha")
    print(f"    Avg Irrigation:    {val_metrics['irrigation']:.2f} mm")
    print(f"    Avg Fertilizer:    {val_metrics['fertilizer']:.2f} kg/ha")
    print(f"    Avg WUE:           {val_metrics['wue']:.4f} kg/mm")
    print(f"    Avg NUE:           {val_metrics['nue']:.4f} kg/kg")
    print(f"    Avg Reward:        {val_metrics['episode_reward']:.2f}")
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
    
    model = LlamaModel.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype,
        use_cache=False
    ).to(device)
    
    model.gradient_checkpointing_enable()
    print(f"   Precision: {torch_dtype}")
    return model, tokenizer

def train_ppo():
    print("=" * 70)
    print("PPO + LLaMA Training Started (v3 - Fixed RND Double Scaling)")
    print("=" * 70)

    # 1. 初始化
    model_path = '/home/wuyang/models/chinese-llama-2-1.3b'
    llama_model, tokenizer = initialize_llama(model_path, config.use_bf16)
    
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/llama_ppo_rnd_results/logs/dssat-pdi.log',
        'mode': 'all', 'seed': 123456, 'random_weather': False,
        'fileX_template_path': '/home/wuyang/test/val_proj/SIAZ9501.MZX'
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    initial_state = dict2array(env.reset())
    actual_state_size = len(initial_state)
    print(f"   Actual state size: {actual_state_size}")
    
    agent = PPOAgent(llama_model, tokenizer, config, actual_state_size=actual_state_size)
    
    logger = MetricsLogger(config.output_prefix)
    
    scores, yields, best_score = [], [], float('-inf')
    
    print(f"\nStarting training ({config.n_episodes} episodes)")
    print(f"Results will be saved to: {logger.output_dir}")
    print(f"\nv3 Fixes Applied:")
    print(f"  - RND intrinsic reward scaling fixed (no double scaling)")
    print(f"  - scaled_intrinsic stored directly in buffer")
    print(f"  - GAE directly adds intrinsic advantage without extra coef")
    
    pbar = tqdm(range(1, config.n_episodes + 1), 
                desc="Training Progress",
                unit="episode",
                ncols=100)
    
    start_time = time.time()
    
    for episode in pbar:
        ep_start_time = time.time()
        state = dict2array(env.reset())
        episode_reward, n_amount, w_amount, ep_yield = 0, 0, 0, 0
        steps = 0
        done = False
        
        for step in range(config.max_steps_per_episode):
            action, log_prob, value = agent.act(state)
            
            action_dict = {
                'anfer': (action % 5) * 40,
                'amir': int(action / 5) * 6
            }
            if state[0] >= 10000: action_dict['anfer'] = 0
            if state[21] >= 1600: action_dict['amir'] = 0
            
            next_state_raw, _, step_done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not step_done else state
            done = step_done
            
            reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                                next_state, done, config.k1, config.k2, config.k3)
            
            # 计算RND内在奖励
            raw_intrinsic = agent.rnd.get_intrinsic_reward(state)
            agent.rnd.update_predictor(state)
            agent.rnd.update_exploration_stats(state, raw_intrinsic)
            
            # v3修复：获取完全缩放后的内在奖励，直接存入buffer
            if config.use_rnd_intrinsic:
                scaled_intrinsic, raw_intrinsic, decay_coef = agent.rnd.get_scaled_intrinsic_reward(
                    state, episode, config.n_episodes
                )
                # v3修复：存入buffer的是scaled_intrinsic（已包含所有缩放）
                agent.buffer.add(state, action, reward, next_state, done, log_prob, value, 
                               array2str(state), intrinsic_reward=scaled_intrinsic)
            else:
                agent.buffer.add(state, action, reward, next_state, done, log_prob, value, 
                               array2str(state), intrinsic_reward=0.0)
            
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            steps += 1
            
            if done:
                ep_yield = state[4]
                break
        
        exploration_coverage = agent.rnd.get_exploration_coverage()
        
        logger.record_episode(episode, ep_yield, w_amount, n_amount, 
                             episode_reward, steps, exploration_coverage)
        
        scores.append(episode_reward)
        yields.append(ep_yield)
        
        ep_time = time.time() - ep_start_time
        pbar.set_postfix({
            'reward': f'{episode_reward:.0f}',
            'yield': f'{ep_yield:.0f}',
            'wue': f'{logger.wue_values[-1]:.2f}',
            'explore': f'{exploration_coverage:.1f}%'
        })
        
        if episode % config.update_frequency == 0 and len(agent.buffer) >= config.mini_batch_size:
            with torch.no_grad():
                last_val = agent.actor_critic(agent.get_cached_embedding(array2str(state)))[1].item()
            
            loss = agent.update(last_val, current_episode=episode, total_episodes=config.n_episodes)
            lr = agent.lr_scheduler.get_lr()[0]
            logger.record_update(loss, lr)
        
        if episode % 10 == 0:
            logger.print_metrics(episode, phase="Training")
            logger.save_metrics_every_10(episode)
        
        if episode > 650 and episode_reward>1000 and episode_reward > best_score:
            val_metrics = validate(agent, env, config, logger, episode)
        
        if episode_reward > best_score and episode_reward > 1400:
            agent.save(f'/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best', episode)

        if episode_reward > best_score and episode_reward>1000:
            best_score = episode_reward
    
    pbar.close()
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  Training Completed!")
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    logger.print_metrics(config.n_episodes, phase="Final")
    logger.save_all_results()
    
    env.close()
    print("Training finished successfully!")

if __name__ == "__main__":
    os.makedirs(f'/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints', exist_ok=True)
    os.makedirs('/home/wuyang/results/llama_ppo_rnd_results/logs', exist_ok=True)
    train_ppo()