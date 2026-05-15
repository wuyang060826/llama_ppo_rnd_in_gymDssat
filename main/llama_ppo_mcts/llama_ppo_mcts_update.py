#!/usr/bin/env python3
"""
Agri-Reasoner V2: 基于世界模型与MCTS的可解释农业决策系统
优化版 - 支持从LLaMA+PPO+RND checkpoint恢复初始化

核心优化:
1. 支持从RND版本的checkpoint恢复策略网络初始化
2. 网络结构适配: 处理RND版本与MCTS版本的结构差异
3. 参数映射机制: 智能映射兼容的参数，新组件随机初始化
4. 保持MCTS版本的其他功能不变

作者: Agri-Reasoner Team
版本: V2.1 (Checkpoint Adaptation)
"""

import numpy as np
import pandas as pd
import random
import pickle
import json
from collections import deque, OrderedDict, defaultdict
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import gym
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
import requests
import re
import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
warnings.filterwarnings('ignore')

# 显存优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")
if torch.cuda.is_available():
    print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from transformers import LlamaModel, LlamaTokenizerFast, LlamaConfig

# ============================================================================
#                              配置参数 V2.1
# ============================================================================
@dataclass
class ConfigV2:
    """全局配置 V2.1 - 支持Checkpoint恢复"""
    
    # === 路径配置 ===
    llama_path: str = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'
    
    # RND版本的checkpoint路径 (新增)
    rnd_checkpoint_path: str = '/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best_0417/model_ep1338.pth'
    
    baseline_path: str = '/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best_0417/model_ep1338.pth'
    
    world_model_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/world_model'
    policy_sft_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/policy_sft'
    value_sft_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/value_sft'
    final_model_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/final'
    normalizer_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/normalizers'
    lora_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/lora'
    
    # 数据路径
    world_model_data: str = '/home/wuyang/results/llama_ppo_mcts_results/data_0420/world_model_data.pkl'
    policy_sft_data: str = '/home/wuyang/results/llama_ppo_mcts_results/data_0420/policy_sft_data.pkl'
    value_sft_data: str = '/home/wuyang/results/llama_ppo_mcts_results/data_0420/value_sft_data.pkl'
    expert_data: str = '/home/wuyang/results/llama_ppo_mcts_results/data_0420/expert_trajectories.pkl'
    
    # === 环境参数 ===
    state_size: int = 25
    action_size: int = 25
    max_steps: int = 200
    
    # === LoRA配置（新增）===
    lora_r: int = 16  # LoRA秩
    lora_alpha: int = 32  # LoRA缩放因子
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
    
    # === 双路径编码配置（新增）===
    numeric_hidden_size: int = 128
    fusion_hidden_size: int = 256
    use_numeric_path: bool = True  # 是否使用数值路径
    numeric_weight: float = 0.3  # 数值路径权重
    
    # === 世界模型参数（优化） ===
    wm_hidden_size: int = 512
    wm_epochs: int = 150
    wm_batch_size: int = 128
    wm_lr: float = 2e-4
    wm_weight_decay: float = 1e-4
    wm_data_samples: int = 20000
    wm_patience: int = 20
    wm_use_attention: bool = True  # 是否使用注意力机制
    
    # === MCTS参数（优化）===
    mcts_c_puct: float = 2.0
    mcts_gamma: float = 0.99
    mcts_max_depth: int = 15
    mcts_temperature: float = 1.0
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_weight: float = 0.25
    intrinsic_reward_weight: float = 0.01
    
    # === MCTS参与训练配置（关键修复）===
    use_mcts_for_action: bool = True  # 是否使用MCTS选择动作
    use_mcts_for_update: bool = True  # 是否使用MCTS结果更新策略
    mcts_action_frequency: int = 50   # 每隔多少步使用MCTS选择动作（降低频率避免过度依赖MCTS）
    mcts_update_weight: float = 0.01  # MCTS策略目标的权重（大幅降低权重避免破坏已有策略）
    mcts_simulations: int = 10        # MCTS模拟次数（减少以提高速度和稳定性）
    mcts_warmup_episodes: int = 1500  # MCTS预热轮次，延长预热期让世界模型更稳定
    mcts_gradual_start: int = 1500    # MCTS渐进启用起始轮次
    mcts_full_active: int = 2500      # MCTS完全激活轮次
    
    # === SFT参数（优化） ===
    sft_epochs: int = 30
    sft_batch_size: int = 32
    sft_lr: float = 5e-5  # LoRA微调需要稍高学习率
    sft_weight_decay: float = 1e-4
    sft_episodes: int = 200
    sft_patience: int = 8
    sft_warmup_ratio: float = 0.1
    sft_label_smoothing: float = 0.1
    
    # === PPO参数（优化）===
    n_episodes: int = 3000
    update_frequency: int = 8
    ppo_epochs: int = 5
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02  # 增加熵系数
    max_grad_norm: float = 0.5  # 减小梯度裁剪
    
    # === 学习率（优化）===
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    embedder_lr: float = 2e-5  # LoRA学习率
    lora_lr: float = 1e-4  # 专门的LoRA学习率
    projection_lr: float = 1e-4
    world_model_lr: float = 3e-4
    
    # === 网络参数 ===
    token_size: int = 256
    projection_size: int = 256
    hidden_size: int = 256
    
    # === RND版本网络参数 (用于checkpoint适配) ===
    rnd_hidden_size: int = 256  # RND版本的hidden_size
    
    # === 奖励函数 ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === Qwen API ===
    qwen_url: str = "http://10.111.5.16:10096/v1/chat/completions"
    qwen_model: str = "qwen3-llm"
    qwen_timeout: int = 15
    
    # === RND探索参数 ===
    rnd_lr: float = 1e-4
    rnd_update_freq: int = 5
    rnd_hidden_size: int = 256
    intrinsic_scale: float = 0.1
    
    # === 训练策略 ===
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    accumulate_grad_batches: int = 4  # 梯度累积
    
    # === Checkpoint恢复配置 (新增) ===
    load_from_rnd_checkpoint: bool = True  # 是否从RND checkpoint恢复
    rnd_checkpoint_episode: int = -1  # -1表示加载最佳模型
    freeze_llama_initially: bool = True  # 初始时是否冻结LLaMA
    unfreeze_llama_after: int = 100  # 在多少个episode后解冻LLaMA
    
    # === 评估参数 ===
    eval_frequency: int = 50
    eval_episodes: int = 10
    save_frequency: int = 100
    
    # === 专家数据配置 ===
    expert_episodes: int = 50
    expert_reasoning: bool = True

config = ConfigV2()

# 创建目录
for dir_path in [config.world_model_dir, config.policy_sft_dir, 
                 config.value_sft_dir, config.final_model_dir, 
                 config.normalizer_dir, config.lora_dir,
                 '/home/wuyang/results/llama_ppo_mcts_results/data_0420', 
                 '/home/wuyang/results/llama_ppo_mcts_results/data_0420', 
                 '/home/wuyang/results/llama_ppo_mcts_results/data_0420']:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
#                    RND版本网络定义 (用于checkpoint加载)
# ============================================================================

class RNDProjectionLayer(nn.Module):
    """RND版本的投影层定义 - 用于加载checkpoint"""
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
    def forward(self, x): 
        return self.projection(x)


class RNDLLaMAEmbedder(nn.Module):
    """RND版本的LLaMA Embedder - 用于加载checkpoint"""
    def __init__(self, llama_model, projection_size=256):
        super().__init__()
        self.llama = llama_model
        self.hidden_size = llama_model.config.hidden_size
        self.projection_size = projection_size
        self.projection = RNDProjectionLayer(self.hidden_size, projection_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return self.projection(embeddings)


class RNDActorCriticHead(nn.Module):
    """RND版本的Actor-Critic头 - 用于加载checkpoint"""
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


# ============================================================================
#                         状态归一化系统 V2
# ============================================================================
class StateNormalizerV2:
    """
    状态归一化器 V2 - 更鲁棒的归一化方案
    """
    
    def __init__(self, state_size: int, method: str = 'adaptive'):
        self.state_size = state_size
        self.method = method
        
        # 统计量
        self.mean = np.zeros(state_size)
        self.std = np.ones(state_size)
        self.min_val = np.zeros(state_size)
        self.max_val = np.ones(state_size)
        self.median = np.zeros(state_size)
        self.iqr = np.zeros(state_size)
        self.q01 = np.zeros(state_size)
        self.q99 = np.zeros(state_size)
        
        # 维度特定配置
        self.dimension_config = {
            'standardize': [0, 4, 7, 21, 23],
            'minmax': [20, 18],
            'robust': list(range(9, 18)),
            'log': [],
        }
        
        # 运行时统计
        self.running_mean = np.zeros(state_size)
        self.running_var = np.ones(state_size)
        self.count = 0
        
        self.fitted = False
        
    def fit(self, states: np.ndarray, clip_outliers: bool = True):
        """拟合归一化参数"""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        
        if clip_outliers:
            self.q01 = np.percentile(states, 1, axis=0)
            self.q99 = np.percentile(states, 99, axis=0)
            states_clipped = np.clip(states, self.q01, self.q99)
        else:
            states_clipped = states
            
        self.mean = np.mean(states_clipped, axis=0)
        self.std = np.std(states_clipped, axis=0) + 1e-8
        self.min_val = np.min(states_clipped, axis=0)
        self.max_val = np.max(states_clipped, axis=0) + 1e-8
        self.median = np.median(states_clipped, axis=0)
        q75 = np.percentile(states_clipped, 75, axis=0)
        q25 = np.percentile(states_clipped, 25, axis=0)
        self.iqr = (q75 - q25) + 1e-8
        
        self.running_mean = self.mean.copy()
        self.running_var = self.std ** 2
        self.count = len(states_clipped)
        
        self.fitted = True
        return self
    
    def transform(self, states: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """应用归一化变换"""
        if not self.fitted:
            raise ValueError("Normalizer未拟合，请先调用fit方法")
            
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        states_clipped = np.clip(states, self.q01, self.q99)
        normalized = states_clipped.copy()
        
        if self.method == 'hybrid' or self.method == 'adaptive':
            for idx in self.dimension_config['standardize']:
                if idx < self.state_size:
                    normalized[:, idx] = (states_clipped[:, idx] - self.mean[idx]) / self.std[idx]
                    
            for idx in self.dimension_config['minmax']:
                if idx < self.state_size:
                    range_val = self.max_val[idx] - self.min_val[idx]
                    if range_val > 1e-8:
                        normalized[:, idx] = (states_clipped[:, idx] - self.min_val[idx]) / range_val
                    else:
                        normalized[:, idx] = 0.5
                    
            for idx in self.dimension_config['robust']:
                if idx < self.state_size:
                    normalized[:, idx] = (states_clipped[:, idx] - self.median[idx]) / self.iqr[idx]
                    
            all_configured = set()
            for key in self.dimension_config:
                all_configured.update(self.dimension_config[key])
            for idx in range(self.state_size):
                if idx not in all_configured:
                    normalized[:, idx] = (states_clipped[:, idx] - self.mean[idx]) / self.std[idx]
        elif self.method == 'standard':
            normalized = (states_clipped - self.mean) / self.std
        elif self.method == 'minmax':
            normalized = (states_clipped - self.min_val) / (self.max_val - self.min_val)
        elif self.method == 'robust':
            normalized = (states_clipped - self.median) / self.iqr
        
        normalized = np.clip(normalized, -5, 5)
        
        if update_stats and not single_sample:
            self._update_running_stats(states_clipped)
        
        return normalized.squeeze() if single_sample else normalized
    
    def _update_running_stats(self, states: np.ndarray):
        """更新运行统计"""
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_count = len(states)
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean = self.running_mean + delta * batch_count / total_count
        
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.running_var = M2 / total_count
        
        self.count = total_count
    
    def fit_transform(self, states: np.ndarray) -> np.ndarray:
        return self.fit(states).transform(states)
    
    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        """逆变换"""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
            
        denormalized = states.copy()
        
        if self.method == 'hybrid' or self.method == 'adaptive':
            for idx in self.dimension_config['standardize']:
                if idx < self.state_size:
                    denormalized[:, idx] = states[:, idx] * self.std[idx] + self.mean[idx]
                    
            for idx in self.dimension_config['minmax']:
                if idx < self.state_size:
                    denormalized[:, idx] = states[:, idx] * (self.max_val[idx] - self.min_val[idx]) + self.min_val[idx]
                    
            for idx in self.dimension_config['robust']:
                if idx < self.state_size:
                    denormalized[:, idx] = states[:, idx] * self.iqr[idx] + self.median[idx]
                    
            all_configured = set()
            for key in self.dimension_config:
                all_configured.update(self.dimension_config[key])
            for idx in range(self.state_size):
                if idx not in all_configured:
                    denormalized[:, idx] = states[:, idx] * self.std[idx] + self.mean[idx]
        elif self.method == 'standard':
            denormalized = states * self.std + self.mean
        elif self.method == 'minmax':
            denormalized = states * (self.max_val - self.min_val) + self.min_val
        elif self.method == 'robust':
            denormalized = states * self.iqr + self.median
            
        return denormalized.squeeze() if single_sample else denormalized
    
    def save(self, path: str):
        np.savez(path,
                 mean=self.mean, std=self.std,
                 min_val=self.min_val, max_val=self.max_val,
                 median=self.median, iqr=self.iqr,
                 q01=self.q01, q99=self.q99,
                 running_mean=self.running_mean,
                 running_var=self.running_var,
                 count=np.array([self.count]),
                 method=np.array([self.method]),
                 fitted=np.array([self.fitted]))
        
    def load(self, path: str):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.min_val = data['min_val']
        self.max_val = data['max_val']
        self.median = data['median']
        self.iqr = data['iqr']
        self.q01 = data['q01']
        self.q99 = data['q99']
        self.running_mean = data['running_mean']
        self.running_var = data['running_var']
        self.count = int(data['count'][0])
        self.method = str(data['method'][0])
        self.fitted = bool(data['fitted'][0])
        return self


class RewardNormalizerV2:
    """奖励归一化器 V2"""
    
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8, 
                 clip_range: float = 10.0):
        self.momentum = momentum
        self.eps = eps
        self.clip_range = clip_range
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
    def update(self, rewards: np.ndarray):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            
        self.count += batch_count
        
    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        normalized = (rewards - self.mean) / (np.sqrt(self.var) + self.eps)
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def save(self, path: str):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)
        
    def load(self, path: str):
        data = np.load(path)
        self.mean = float(data['mean'])
        self.var = float(data['var'])
        self.count = int(data['count'])
        return self


# ============================================================================
#                         辅助函数
# ============================================================================
def dict2array(state: dict) -> np.ndarray:
    """将字典状态转为数组"""
    if state is None: 
        raise ValueError("状态不能为None")
    new_state = []
    for key in state.keys():
        if key != 'sw': 
            new_state.append(state[key])
        else: 
            new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)

def array2str(state: np.ndarray, precision: int = 1) -> str:
    """状态数组转文本描述"""
    parts = []
    for i, num in enumerate(state):
        if i == 0: 
            parts.append(f"生长天数{int(num/40)}")
        elif i == 4: 
            parts.append(f"预估产量{int(num/100)}百公斤")
        elif i == 7: 
            parts.append(f"地上生物量{int(num/10)}克")
        elif i == 20: 
            parts.append(f"土壤水分{num:.2f}")
        elif i == 21: 
            parts.append(f"累计灌溉{int(num/6)}毫米")
        elif i == 23: 
            parts.append(f"累计降雨{int(num)}毫米")
        elif 9 <= i <= 17: 
            parts.append(f"第{i-8}层氮{int(num*1000)}毫克")
        elif i == 18: 
            parts.append(f"氮素比{num*100:.1f}%")
        else: 
            parts.append(f"参数{i}值{num:.1f}")
    return " ".join(parts)

def array2detailed_str(state: np.ndarray) -> str:
    """生成详细的农学描述"""
    day = state[0]
    growth_stage = ""
    if day < 4000:
        growth_stage = "苗期"
        stage_advice = "适量水肥，促进根系发育"
    elif day < 8000:
        growth_stage = "拔节期"
        stage_advice = "增加水肥，促进茎叶生长"
    elif day < 12000:
        growth_stage = "抽穗开花期（关键期）"
        stage_advice = "充足水肥，确保穗粒发育"
    else:
        growth_stage = "灌浆成熟期"
        stage_advice = "适当控水，促进籽粒饱满"
    
    desc = f"""
【农业状态诊断报告】

一、生长状态
- 生长天数: {int(day)} 天
- 生长阶段: {growth_stage}
- 预估产量: {int(state[4])} kg/ha
- 地上生物量: {int(state[7])} kg/ha

二、土壤水分状况
- 当前土壤水分: {state[20]:.3f} cm³/cm³
- 累计灌溉量: {int(state[21])} mm
- 累计降雨量: {int(state[23])} mm
- 水分状态: {"亏缺" if state[20] < 0.25 else "适宜" if state[20] < 0.4 else "充足"}

三、氮素营养状况
- 土壤氮素百分比: {state[18]*100:.2f}%
- 氮素状态: {"亏缺" if state[18] < 0.015 else "适宜" if state[18] < 0.03 else "充足"}

四、管理建议
- 当前阶段: {stage_advice}
"""
    return desc.strip()

def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3):
    """奖励函数"""
    if done: 
        yield_reward = k1 * state[4]
        nitrogen_cost = k2 * n_action
        water_cost = k3 * w_action
        return yield_reward - nitrogen_cost - water_cost
    return -k2 * n_action - k3 * w_action

def action_to_dict(action: int, state: np.ndarray) -> dict:
    """动作索引转字典"""
    action_dict = {
        'anfer': (action % 5) * 40,
        'amir': int(action / 5) * 6
    }
    if state[0] >= 10000: 
        action_dict['anfer'] = 0
    if state[21] >= 1600: 
        action_dict['amir'] = 0
    return action_dict

def state_hash(state: np.ndarray) -> str:
    """生成状态的唯一哈希值"""
    return hashlib.md5(state.tobytes()).hexdigest()[:16]

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#                         Qwen大模型调用
# ============================================================================
def call_qwen(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """调用本地Qwen模型"""
    payload = {
        "model": config.qwen_model,
        "messages": [{"role": "user", "content": '/no_think' + prompt}],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_think": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(config.qwen_url, headers=headers, 
                                json=payload, timeout=config.qwen_timeout)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                content = re.sub(r'<think.*?</think">', '', content, flags=re.DOTALL)
                return content.strip()
    except Exception as e:
        print(f"⚠️ Qwen调用失败: {e}")
    
    return ""


# ============================================================================
#                         LoRA模块实现
# ============================================================================
class LoRALayer(nn.Module):
    """LoRA层"""
    
    def __init__(self, in_features: int, out_features: int, 
                 r: int = 8, alpha: int = 16, dropout: float = 0.05, dtype=torch.float32):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        dtype = original_output.dtype
        x = x.to(dtype)
        lora_output = self.dropout(x) @ self.lora_A.to(dtype).T @ self.lora_B.to(dtype).T * self.scaling
        return original_output + lora_output


class LoRALinear(nn.Module):
    """带LoRA的线性层"""
    
    def __init__(self, original_linear: nn.Linear, r: int = 8, 
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        
        self.original_linear = original_linear
        self.dtype = original_linear.weight.dtype
        
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            r, alpha, dropout,
            dtype=self.dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        original_output = self.original_linear(x)
        return self.lora(x, original_output)


def apply_lora_to_llama(llama_model, r: int = 16, alpha: int = 32, 
                        dropout: float = 0.05, target_modules: List[str] = None):
    """将LoRA应用到LLaMA模型"""
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    
    lora_params = []
    
    for name, module in llama_model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = llama_model
                    for p in parent_name.split('.'):
                        parent = getattr(parent, p)
                else:
                    parent = llama_model
                    attr_name = name
                
                lora_linear = LoRALinear(module, r, alpha, dropout)
                setattr(parent, attr_name, lora_linear)
                
                lora_params.extend(lora_linear.lora.parameters())
                
                print(f"  ✓ 应用LoRA到: {name} (dtype: {module.weight.dtype})")
    
    return llama_model, lora_params


def get_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    """获取LoRA参数的状态字典"""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name + '.lora.lora_A'] = module.lora.lora_A.data
            lora_state[name + '.lora.lora_B'] = module.lora.lora_B.data
    return lora_state


def load_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]):
    """加载LoRA参数"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if name + '.lora.lora_A' in state_dict:
                module.lora.lora_A.data = state_dict[name + '.lora.lora_A']
            if name + '.lora.lora_B' in state_dict:
                module.lora.lora_B.data = state_dict[name + '.lora.lora_B']


# ============================================================================
#                    Checkpoint加载与参数映射 (核心新增)
# ============================================================================
class CheckpointAdapter:
    """
    Checkpoint适配器 - 处理RND版本到MCTS版本的参数映射
    
    功能:
    1. 加载RND版本的checkpoint
    2. 智能映射兼容的参数
    3. 报告映射状态和统计信息
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.mapping_stats = {
            'loaded_params': 0,
            'mapped_params': 0,
            'skipped_params': 0,
            'new_params': 0
        }
        
    def load_rnd_checkpoint(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """加载RND版本的checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RND checkpoint不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # RND版本可能的key结构
        possible_keys = ['model_state_dict', 'state_dict', 'actor_critic']
        state_dict = None
        
        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is None:
            # 假设checkpoint本身就是state_dict
            state_dict = checkpoint
        
        if self.verbose:
            print(f"\n📦 加载RND Checkpoint: {checkpoint_path}")
            print(f"   参数数量: {len(state_dict)}")
            if 'episode' in checkpoint:
                print(f"   训练轮次: {checkpoint['episode']}")
            if 'score' in checkpoint:
                print(f"   最佳得分: {checkpoint['score']:.2f}")
        
        return state_dict
    
    def map_projection_params(self, rnd_state_dict: Dict, 
                              mcts_encoder: nn.Module) -> Dict[str, Any]:
        """
        映射投影层参数
        
        RND: LLaMAEmbedder.projection (ProjectionLayer)
        MCTS: DualPathStateEncoder.text_projection
        """
        mapping_info = {'success': True, 'details': []}
        
        # 获取MCTS的text_projection
        if hasattr(mcts_encoder, 'text_projection'):
            text_proj = mcts_encoder.text_projection
        else:
            mapping_info['success'] = False
            mapping_info['details'].append("MCTS编码器没有text_projection属性")
            return mapping_info
        
        # RND可能的投影层key前缀
        rnd_proj_prefixes = [
            'embedder.projection.',
            'llama_embedder.projection.',
            'state_encoder.projection.',
            'projection.'
        ]
        
        # 尝试匹配并加载
        for prefix in rnd_proj_prefixes:
            params_found = False
            for key in rnd_state_dict:
                if key.startswith(prefix):
                    params_found = True
                    break
            
            if params_found:
                # 构建参数映射
                proj_state = {}
                for key, value in rnd_state_dict.items():
                    if key.startswith(prefix):
                        # 去掉前缀
                        new_key = key[len(prefix):]
                        
                        # 检查是否是projection.Sequential的参数
                        if new_key.startswith('projection.'):
                            new_key = new_key[len('projection.'):]
                        
                        proj_state[new_key] = value
                
                # 尝试加载到text_projection
                try:
                    if hasattr(text_proj, 'projection'):
                        # text_proj是Sequential
                        text_proj.projection.load_state_dict(proj_state, strict=False)
                    else:
                        text_proj.load_state_dict(proj_state, strict=False)
                    
                    self.mapping_stats['mapped_params'] += len(proj_state)
                    mapping_info['details'].append(f"投影层参数映射成功: {len(proj_state)}个参数")
                except Exception as e:
                    mapping_info['details'].append(f"投影层参数映射失败: {str(e)}")
                
                break
        
        return mapping_info
    
    def map_actor_critic_params(self, rnd_state_dict: Dict,
                                mcts_actor_critic: nn.Module) -> Dict[str, Any]:
        """
        映射Actor-Critic参数
        
        RND: ActorCriticHead.shared, actor_head, critic_head
        MCTS: ActorCriticV2.shared, actor, critic
        
        注意: RND版本shared的输入维度是projection_size(256)
             MCTS版本shared的输入维度也是projection_size(256)
        """
        mapping_info = {'success': True, 'details': []}
        
        # RND可能的Actor-Critic key前缀
        rnd_ac_prefixes = [
            'actor_critic_head.',
            'actor_critic.',
            'policy_head.',
            ''
        ]
        
        # 构建MCTS Actor-Critic的state_dict
        mcts_state = mcts_actor_critic.state_dict()
        
        # 参数映射表: (RND key pattern, MCTS key pattern)
        # ============================================================
        # 修复: 两个版本的shared层结构完全一致，应直接按索引1:1映射
        # 
        # RND版本shared结构:
        #   shared.0: Linear(256, 256)
        #   shared.1: LayerNorm(256)
        #   shared.2: ReLU
        #   shared.3: Dropout
        #   shared.4: Linear(256, 128)
        #   shared.5: LayerNorm(128)
        #   shared.6: ReLU
        #
        # MCTS版本shared结构（完全一致）:
        #   shared.0: Linear(256, 256)
        #   shared.1: LayerNorm(256)
        #   shared.2: ReLU
        #   shared.3: Dropout
        #   shared.4: Linear(256, 128)
        #   shared.5: LayerNorm(128)
        #   shared.6: ReLU
        # ============================================================
        param_mapping = [
            # shared层映射 - 直接按索引1:1映射
            ('shared.0.', 'shared.0.'),  # Linear: [256, 256]
            ('shared.1.', 'shared.1.'),  # LayerNorm: [256]
            ('shared.4.', 'shared.4.'),  # Linear: [256, 128]
            ('shared.5.', 'shared.5.'),  # LayerNorm: [128]
            
            # actor头映射 - 简化结构后直接映射
            ('actor_head.', 'actor.'),  # 单层Linear: [128, 25]
            
            # critic头映射 - 简化结构后直接映射  
            ('critic_head.', 'critic.'),  # 单层Linear: [128, 1]
        ]
        
        mapped_count = 0
        mismatched_count = 0
        matched_details = []
        
        for rnd_key, value in rnd_state_dict.items():
            # 跳过非Actor-Critic参数
            if 'llama' in rnd_key.lower() or 'embedder' in rnd_key.lower():
                continue
            if 'projection' in rnd_key.lower():
                continue
            if 'rnd' in rnd_key.lower():
                continue
            if 'numeric' in rnd_key.lower() or 'fusion' in rnd_key.lower():
                continue
            if 'gate' in rnd_key.lower():
                continue
            
            # 尝试直接匹配
            for rnd_pattern, mcts_pattern in param_mapping:
                if rnd_pattern in rnd_key:
                    # 构建MCTS key
                    mcts_key = rnd_key.replace(rnd_pattern, mcts_pattern)
                    
                    if mcts_key in mcts_state:
                        # 检查维度是否匹配
                        if mcts_state[mcts_key].shape == value.shape:
                            mcts_state[mcts_key] = value.clone()
                            mapped_count += 1
                            matched_details.append(f"✓ {rnd_key} -> {mcts_key} {tuple(value.shape)}")
                        else:
                            mismatched_count += 1
                            if self.verbose:
                                print(f"   ⚠️ 维度不匹配: {rnd_key} {tuple(value.shape)} -> {mcts_key} {tuple(mcts_state[mcts_key].shape)}")
                    break
        
        # 尝试加载修改后的state_dict
        try:
            mcts_actor_critic.load_state_dict(mcts_state, strict=False)
            self.mapping_stats['mapped_params'] = mapped_count
            mapping_info['details'].append(f"✓ Actor-Critic参数映射成功: {mapped_count}个参数")
            if mismatched_count > 0:
                mapping_info['details'].append(f"  维度不匹配跳过: {mismatched_count}个参数")
            
            # 打印详细匹配信息（可选）
            if self.verbose and mapped_count > 0:
                print(f"\n   参数映射详情:")
                for detail in matched_details[:10]:  # 只显示前10个
                    print(f"     {detail}")
                if len(matched_details) > 10:
                    print(f"     ... 还有 {len(matched_details) - 10} 个参数已映射")
                    
        except Exception as e:
            mapping_info['details'].append(f"⚠️ Actor-Critic参数加载警告: {str(e)}")
        
        return mapping_info
    
    def adapt_checkpoint_to_mcts(self, checkpoint_path: str,
                                  mcts_encoder: nn.Module,
                                  mcts_actor_critic: nn.Module) -> Dict[str, Any]:
        """
        完整的checkpoint适配流程
        
        返回:
            包含映射统计和状态的字典
        """
        result = {
            'success': True,
            'checkpoint_path': checkpoint_path,
            'mapping_details': []
        }
        
        # 1. 加载RND checkpoint
        try:
            rnd_state_dict = self.load_rnd_checkpoint(checkpoint_path)
            self.mapping_stats['loaded_params'] = len(rnd_state_dict)
        except Exception as e:
            result['success'] = False
            result['error'] = f"无法加载checkpoint: {str(e)}"
            return result
        
        # 2. 映射投影层参数
        proj_result = self.map_projection_params(rnd_state_dict, mcts_encoder)
        result['mapping_details'].extend(proj_result['details'])
        
        # 3. 映射Actor-Critic参数
        ac_result = self.map_actor_critic_params(rnd_state_dict, mcts_actor_critic)
        result['mapping_details'].extend(ac_result['details'])
        
        # 4. 计算新参数数量
        total_mcts_params = sum(p.numel() for p in mcts_actor_critic.parameters())
        self.mapping_stats['new_params'] = total_mcts_params - self.mapping_stats['mapped_params']
        
        # 5. 打印统计
        if self.verbose:
            print(f"\n📊 Checkpoint适配统计:")
            print(f"   RND参数总数: {self.mapping_stats['loaded_params']}")
            print(f"   成功映射: {self.mapping_stats['mapped_params']}")
            print(f"   新初始化参数: {self.mapping_stats['new_params']}")
        
        return result


# ============================================================================
#                         双路径状态编码器
# ============================================================================
class DualPathStateEncoder(nn.Module):
    """
    双路径状态编码器
    
    架构:
    1. 文本路径: 状态 → 文本 → LLaMA嵌入
    2. 数值路径: 状态 → MLP → 数值特征
    3. 融合层: 文本特征 + 数值特征 → 综合表示
    """
    
    def __init__(self, llama_model, state_size: int, 
                 projection_size: int = 256, numeric_hidden: int = 128,
                 fusion_hidden: int = 256, use_numeric_path: bool = True,
                 numeric_weight: float = 0.3):
        super().__init__()
        
        self.use_numeric_path = use_numeric_path
        self.numeric_weight = numeric_weight
        self.state_size = state_size
        self.projection_size = projection_size
        
        # 文本路径 - LLaMA嵌入
        self.llama = llama_model
        self.hidden_size = llama_model.config.hidden_size
        
        # 文本投影层 (与RND版本结构保持一致以便映射)
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_size * 2),
            nn.LayerNorm(projection_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_size * 2, projection_size),
            nn.LayerNorm(projection_size),
        )
        
        # 数值路径
        if use_numeric_path:
            self.numeric_encoder = nn.Sequential(
                nn.Linear(state_size, numeric_hidden),
                nn.LayerNorm(numeric_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(numeric_hidden, numeric_hidden),
                nn.LayerNorm(numeric_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(numeric_hidden, projection_size),
                nn.LayerNorm(projection_size),
            )
            
            # 融合层
            self.fusion_layer = nn.Sequential(
                nn.Linear(projection_size * 2, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden, projection_size),
                nn.LayerNorm(projection_size),
            )
            
            # 门控机制
            self.gate = nn.Sequential(
                nn.Linear(projection_size * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        # 文本路径
        llama_outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = llama_outputs.last_hidden_state
        
        # 平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embeddings = sum_embeddings / sum_mask
        
        text_features = self.text_projection(text_embeddings)
        
        # 如果不使用数值路径
        if not self.use_numeric_path or numeric_state is None:
            return text_features
        
        # 数值路径
        numeric_features = self.numeric_encoder(numeric_state)
        
        # 门控融合
        concat_features = torch.cat([text_features, numeric_features], dim=-1)
        gate_value = self.gate(concat_features)
        
        # 动态加权融合
        fused = gate_value * numeric_features + (1 - gate_value) * text_features
        
        # 最终融合
        final_features = self.fusion_layer(concat_features)
        
        # 残差连接
        output = final_features + 0.5 * text_features + 0.5 * numeric_features
        
        return output
    
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """仅获取文本特征"""
        llama_outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = llama_outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embeddings = sum_embeddings / sum_mask
        
        return self.text_projection(text_embeddings)
    
    def get_numeric_features(self, numeric_state: torch.Tensor) -> torch.Tensor:
        """仅获取数值特征"""
        return self.numeric_encoder(numeric_state)


# ============================================================================
#                         改进的世界模型 V2
# ============================================================================
class MultiHeadPrediction(nn.Module):
    """多头预测模块"""
    
    def __init__(self, hidden_size: int, output_size: int, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, output_size)
            ) for _ in range(num_heads)
        ])
        
        self.gate = nn.Linear(hidden_size, num_heads)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = self.softmax(self.gate(x))
        head_outputs = torch.stack([head(x) for head in self.heads], dim=1)
        output = torch.einsum('bn,bno->bo', gate_weights, head_outputs)
        return output


class WorldModelV2(nn.Module):
    """
    世界模型 V2 - 更强的预测能力
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 hidden_size: int = 512, use_attention: bool = True):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_attention = use_attention
        
        input_size = state_size + action_size
        
        # 输入嵌入
        self.input_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 自注意力层
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads=8, dropout=0.1, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        
        # 深层残差网络
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_size) for _ in range(6)
        ])
        
        # 状态预测头 - 均值
        self.state_mean_head = MultiHeadPrediction(hidden_size, state_size, num_heads=4)
        
        # 状态预测头 - log标准差
        self.state_std_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, state_size),
            nn.Softplus()
        )
        
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 终止预测头
        self.done_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def _make_residual_block(self, hidden_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor,
                timestep: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        batch_size = state.size(0)
        
        # 输入嵌入
        x = torch.cat([state, action_onehot], dim=-1)
        x = self.input_embed(x)
        
        # 时间嵌入
        if timestep is not None:
            t = timestep.float().unsqueeze(-1) / 1000.0
            t_embed = self.time_embed(t)
            x = x + t_embed
        
        # 自注意力
        if self.use_attention:
            x_seq = x.unsqueeze(1)
            attn_out, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.attn_norm(x + attn_out.squeeze(1))
        
        # 残差块
        for block in self.residual_blocks:
            x = F.gelu(x + block(x))
        
        # 预测头
        next_state_mean = self.state_mean_head(x)
        next_state_std = self.state_std_head(x) + 0.01
        reward = self.reward_head(x)
        done_prob = self.done_head(x)
        value = self.value_head(x)
        
        return next_state_mean, next_state_std, reward, done_prob, value
    
    def get_loss(self, state: torch.Tensor, action_onehot: torch.Tensor,
             next_state: torch.Tensor, reward: torch.Tensor, 
             done: torch.Tensor, value: torch.Tensor = None,
             timestep: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """计算损失"""
        pred_mean, pred_std, pred_reward, pred_done, pred_value = \
            self.forward(state, action_onehot, timestep)
        
        # 状态损失
        state_nll = 0.5 * torch.log(pred_std ** 2) + \
                    0.5 * ((next_state - pred_mean) ** 2) / (pred_std ** 2)
        state_loss = state_nll.mean()
        
        # 奖励损失
        reward_loss = F.smooth_l1_loss(pred_reward, reward)
        
        # 终止损失
        done_loss = F.binary_cross_entropy(pred_done.squeeze(-1), done.squeeze(-1))
        
        # 价值损失
        value_loss = torch.tensor(0.0, device=state.device)
        if value is not None:
            value_loss = F.mse_loss(pred_value.squeeze(-1), value)
        
        # 总损失
        total_loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss + 0.1 * value_loss
        
        return {
            'total': total_loss,
            'state': state_loss,
            'reward': reward_loss,
            'done': done_loss,
            'value': value_loss
        }
    
    def predict(self, state: torch.Tensor, action_onehot: torch.Tensor,
                deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测下一步"""
        mean, std, reward, done, value = self.forward(state, action_onehot)
        
        if deterministic:
            next_state = mean
        else:
            noise = torch.randn_like(mean)
            next_state = mean + std * noise
        
        return next_state, reward, done, value


# ============================================================================
#                         训练系统 V2
# ============================================================================
class TrainingLoggerV2:
    """训练日志记录器 V2"""
    
    def __init__(self, log_dir: str = '/home/wuyang/results/llama_ppo_mcts_results/data_0420'):
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.best_metrics = {}
        
    def log(self, metric_name: str, value: float):
        self.metrics[metric_name].append(value)
        
        if metric_name.endswith('_loss'):
            current_best = self.best_metrics.get(metric_name, float('inf'))
            if value < current_best:
                self.best_metrics[metric_name] = value
        else:
            current_best = self.best_metrics.get(metric_name, float('-inf'))
            if value > current_best:
                self.best_metrics[metric_name] = value
    
    def log_batch(self, metrics: Dict[str, float]):
        for name, value in metrics.items():
            self.log(name, value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1],
                    'best': self.best_metrics.get(name, values[-1])
                }
        return summary
    
    def plot_training_curves(self, save_path: str = None, 
                             figsize: Tuple[int, int] = (16, 12)):
        if not self.metrics:
            return
            
        n_metrics = len(self.metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, (name, values) in enumerate(self.metrics.items()):
            ax = axes[idx]
            ax.plot(values, linewidth=1.5, alpha=0.7)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            if len(values) > 10:
                window = min(20, len(values) // 5)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(values)), moving_avg, 
                       color='red', linewidth=2, label='MA')
                ax.legend(fontsize=8)
                
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
            
        plt.close()
        
    def save_logs(self, path: str):
        data = {
            'metrics': dict(self.metrics),
            'summary': self.get_summary(),
            'duration': time.time() - self.start_time,
            'best_metrics': self.best_metrics
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)


class EarlyStoppingV2:
    """早停机制 V2"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta if self.best_score is not None else True
        else:
            improved = score > self.best_score + self.min_delta if self.best_score is not None else True
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                    
        return self.early_stop
    
    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})


class CosineWarmupScheduler:
    """余弦退火预热调度器"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = max(self.base_lrs[i] * scale, self.min_lr)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ============================================================================
#                         数据收集
# ============================================================================
def collect_world_model_data_v2(env, normalizer: StateNormalizerV2, 
                                n_samples: int = 20000) -> Dict[str, np.ndarray]:
    """收集世界模型训练数据 V2"""
    print(f"\n{'='*70}")
    print("📊 任务1: 收集世界模型训练数据 V2")
    print(f"{'='*70}")
    
    if os.path.exists(config.world_model_data):
        print(f"✓ 发现已有数据文件: {config.world_model_data}")
        with open(config.world_model_data, 'rb') as f:
            data = pickle.load(f)
        print(f"  加载 {len(data['states'])} 条样本")
        
        normalizer.fit(data['states'])
        normalizer.save(os.path.join(config.normalizer_dir, 'state_normalizer.npz'))
        
        return data
    
    print(f"开始收集 {n_samples} 条状态转移样本...")
    
    states, actions, next_states, rewards, dones, timesteps, values = \
        [], [], [], [], [], [], []
    
    pbar = tqdm(total=n_samples, desc="收集数据")
    
    episode_returns = []
    
    while len(states) < n_samples:
        state = dict2array(env.reset())
        done = False
        step = 0
        episode_rewards = []
        
        while not done and step < config.max_steps:
            if random.random() < 0.3:
                action = random.randint(0, config.action_size - 1)
            else:
                action = heuristic_policy_v2(state)
            
            action_dict = action_to_dict(action, state)
            
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            
            reward = get_reward(state, action_dict['anfer'], 
                              action_dict['amir'], next_state, done,
                              config.k1, config.k2, config.k3)
            
            states.append(state.copy())
            actions.append(action)
            next_states.append(next_state.copy())
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            timesteps.append(step)
            episode_rewards.append(reward)
            
            state = next_state
            step += 1
            pbar.update(1)
            
            if len(states) >= n_samples:
                break
        
        if episode_rewards:
            G = 0
            episode_values = []
            for r in reversed(episode_rewards):
                G = r + config.gamma * G
                episode_values.insert(0, G)
            values.extend(episode_values)
            episode_returns.append(G)
    
    pbar.close()
    
    while len(values) < len(states):
        values.append(0.0)
    values = values[:len(states)]
    
    data = {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'timesteps': np.array(timesteps),
        'values': np.array(values)
    }
    
    normalizer.fit(data['states'])
    normalizer.save(os.path.join(config.normalizer_dir, 'state_normalizer.npz'))
    
    with open(config.world_model_data, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n📊 数据统计:")
    print(f"  状态范围: [{data['states'].min():.2f}, {data['states'].max():.2f}]")
    print(f"  奖励范围: [{data['rewards'].min():.2f}, {data['rewards'].max():.2f}]")
    print(f"  平均Episode回报: {np.mean(episode_returns):.2f}")
    print(f"  终止样本比例: {data['dones'].mean():.2%}")
    
    return data


def heuristic_policy_v2(state: np.ndarray) -> int:
    """改进的启发式策略"""
    day = state[0]
    soil_moisture = state[20]
    nitrogen_pct = state[18]
    cumulative_irr = state[21]
    
    if day < 4000:
        if soil_moisture < 0.22:
            return random.choice([5, 10])
        elif nitrogen_pct < 0.012:
            return random.choice([1, 2])
        return 0
    elif day < 8000:
        if soil_moisture < 0.28:
            return random.choice([10, 15])
        elif nitrogen_pct < 0.018:
            return random.choice([2, 3, 7, 8])
        return random.choice([0, 5])
    elif day < 12000:
        if soil_moisture < 0.32 and cumulative_irr < 1400:
            return random.choice([15, 20])
        elif nitrogen_pct < 0.022:
            return random.choice([3, 4, 8, 9])
        return random.choice([5, 10])
    else:
        if soil_moisture < 0.25:
            return 5
        return 0


def train_world_model_v2(data: Dict, normalizer: StateNormalizerV2,
                         force_retrain: bool = False) -> WorldModelV2:
    """训练世界模型 V2"""
    print(f"\n{'='*70}")
    print("🧠 任务1: 训练世界模型 V2")
    print(f"{'='*70}")
    
    model_path = os.path.join(config.world_model_dir, 'world_model_best.pth')
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"✓ 发现已训练模型: {model_path}")
        model = WorldModelV2(config.state_size, config.action_size, 
                            config.wm_hidden_size).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  最佳验证损失: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
        return model
    
    logger = TrainingLoggerV2()
    
    # 数据准备
    states = normalizer.transform(data['states'])
    next_states = normalizer.transform(data['next_states'])
    
    reward_normalizer = RewardNormalizerV2()
    reward_normalizer.update(data['rewards'])
    rewards = reward_normalizer.normalize(data['rewards'])
    reward_normalizer.save(os.path.join(config.normalizer_dir, 'reward_normalizer.npz'))
    
    values_mean = np.mean(data['values'])
    values_std = np.std(data['values']) + 1e-8
    values = (data['values'] - values_mean) / values_std
    
    print(f"\n📊 归一化后数据统计:")
    print(f"  状态: [{states.min():.2f}, {states.max():.2f}]")
    print(f"  奖励: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(data['actions'])
    next_states_t = torch.FloatTensor(next_states)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
    dones_t = torch.FloatTensor(data['dones']).unsqueeze(1)
    timesteps_t = torch.LongTensor(data['timesteps'])
    values_t = torch.FloatTensor(values)
    
    actions_onehot = F.one_hot(actions_t, config.action_size).float()
    
    dataset = TensorDataset(states_t, actions_onehot, next_states_t, 
                           rewards_t, dones_t, timesteps_t, values_t)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.wm_batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.wm_batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    model = WorldModelV2(config.state_size, config.action_size, 
                        config.wm_hidden_size, use_attention=config.wm_use_attention).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.wm_lr, 
                           weight_decay=config.wm_weight_decay)
    
    total_steps = len(train_loader) * config.wm_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStoppingV2(patience=config.wm_patience, mode='min')
    
    best_val_loss = float('inf')
    
    print(f"开始训练 (训练集: {train_size}, 验证集: {val_size})...")
    
    for epoch in range(config.wm_epochs):
        model.train()
        train_losses = defaultdict(float)
        
        for batch in train_loader:
            s = batch[0].to(device)
            a = batch[1].to(device)
            ns = batch[2].to(device)
            r = batch[3].to(device)
            d = batch[4].to(device)
            t = batch[5].to(device)
            v = batch[6].to(device)
            
            # 修复: 传递timestep参数到get_loss
            losses = model.get_loss(s, a, ns, r, d, v, t)
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            for key, value in losses.items():
                train_losses[key] += value.item()
        
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # 验证
        model.eval()
        val_losses = defaultdict(float)
        
        with torch.no_grad():
            for batch in val_loader:
                s = batch[0].to(device)
                a = batch[1].to(device)
                ns = batch[2].to(device)
                r = batch[3].to(device)
                d = batch[4].to(device)
                t = batch[5].to(device)
                v = batch[6].to(device)
                
                # 修复: 验证时也传递timestep参数
                losses = model.get_loss(s, a, ns, r, d, v, t)
                
                for key, value in losses.items():
                    val_losses[key] += value.item()
        
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        logger.log_batch({
            'train_total_loss': train_losses['total'],
            'train_state_loss': train_losses['state'],
            'val_total_loss': val_losses['total'],
            'val_state_loss': val_losses['state'],
            'learning_rate': scheduler.get_lr()
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.wm_epochs} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"LR: {scheduler.get_lr():.2e}")
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_losses': dict(val_losses)
            }, model_path)
        
        if early_stopping(val_losses['total'], model, epoch):
            print(f"⚠️ 早停于 Epoch {epoch+1}")
            break
    
    early_stopping.restore_best_weights(model)
    
    logger.plot_training_curves(
        os.path.join('/home/wuyang/results/llama_ppo_mcts_results/data_0420', 'world_model_training.png')
    )
    
    print(f"\n✓ 世界模型训练完成")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    
    return model


# ============================================================================
#                         策略网络 V2
# ============================================================================
class PolicyNetworkV2(nn.Module):
    """策略网络 V2"""
    
    def __init__(self, state_encoder: DualPathStateEncoder, 
                 projection_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        self.policy_net = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
        )
        
        self.action_head = nn.Linear(hidden_size // 2, action_size)
        self.action_prior = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        features = self.state_encoder(input_ids, attention_mask, numeric_state)
        policy_features = self.policy_net(features)
        logits = self.action_head(policy_features) + self.action_prior
        return logits
    
    def get_action_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         numeric_state: torch.Tensor = None, 
                         temperature: float = 1.0) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask, numeric_state)
        return F.softmax(logits / temperature, dim=-1)
    
    def get_action(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   numeric_state: torch.Tensor = None, 
                   deterministic: bool = False) -> Tuple[int, float, torch.Tensor]:
        logits = self.forward(input_ids, attention_mask, numeric_state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, probs


class ValueNetworkV2(nn.Module):
    """价值网络 V2"""
    
    def __init__(self, state_encoder: DualPathStateEncoder,
                 projection_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        self.value_net = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        features = self.state_encoder(input_ids, attention_mask, numeric_state)
        return self.value_net(features)


# ============================================================================
#                         改进的MCTS V2
# ============================================================================
class MCTSNodeV2:
    """MCTS节点 V2"""
    
    def __init__(self, state: np.ndarray, parent=None, 
                 action: int = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children: Dict[int, 'MCTSNodeV2'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
        self.predicted_reward = 0.0
        self.predicted_done = False
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, total_visits: int, 
                  fpu_reduction: float = 0.0) -> float:
        if self.visit_count == 0:
            if self.parent is not None:
                return self.parent.value() - fpu_reduction + \
                       c_puct * self.prior * math.sqrt(total_visits + 1)
            return float('inf')
        
        q_value = self.value()
        u_value = c_puct * self.prior * math.sqrt(total_visits) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct: float, fpu_reduction: float = 0.0) -> Tuple[int, 'MCTSNodeV2']:
        total_visits = sum(child.visit_count for child in self.children.values())
        
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = child.ucb_score(c_puct, total_visits, fpu_reduction)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_priors: np.ndarray, 
               dirichlet_noise: bool = False, alpha: float = 0.3, weight: float = 0.25):
        if self.is_expanded:
            return
        
        if dirichlet_noise:
            noise = np.random.dirichlet([alpha] * len(action_priors))
            action_priors = (1 - weight) * action_priors + weight * noise
        
        for action in range(len(action_priors)):
            self.children[action] = MCTSNodeV2(
                state=None,
                parent=self,
                action=action,
                prior=action_priors[action]
            )
        
        self.is_expanded = True
    
    def backup(self, value: float, gamma: float = 0.99):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value * gamma
            node = node.parent


class MCTSV2:
    """改进的MCTS V2"""
    
    def __init__(self, world_model: WorldModelV2, 
                 policy_network: PolicyNetworkV2,
                 value_network: ValueNetworkV2,
                 tokenizer, config, normalizer: StateNormalizerV2):
        self.world_model = world_model
        self.policy_network = policy_network
        self.value_network = value_network
        self.tokenizer = tokenizer
        self.config = config
        self.normalizer = normalizer
        
        self.state_cache = {}
        self.cache_max_size = 10000
        
    def _get_cache_key(self, state: np.ndarray, action: int) -> str:
        return f"{state_hash(state)}_{action}"
    
    @torch.no_grad()
    def get_action_probs(self, state: np.ndarray, 
                         add_noise: bool = False) -> np.ndarray:
        state_str = array2str(state)
        state_norm = self.normalizer.transform(state)
        
        inputs = self.tokenizer(
            [state_str], return_tensors='pt',
            padding='max_length', truncation=True,
            max_length=self.config.token_size
        ).to(device)
        
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        
        probs = self.policy_network.get_action_probs(
            inputs['input_ids'], inputs['attention_mask'], state_t
        ).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.dirichlet([self.config.mcts_dirichlet_alpha] * self.config.action_size)
            probs = (1 - self.config.mcts_dirichlet_weight) * probs + \
                   self.config.mcts_dirichlet_weight * noise
        
        return probs
    
    @torch.no_grad()
    def evaluate_state(self, state: np.ndarray) -> float:
        state_str = array2str(state)
        state_norm = self.normalizer.transform(state)
        
        inputs = self.tokenizer(
            [state_str], return_tensors='pt',
            padding='max_length', truncation=True,
            max_length=self.config.token_size
        ).to(device)
        
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        
        value = self.value_network(inputs['input_ids'], inputs['attention_mask'], state_t)
        
        return value.item()
    
    @torch.no_grad()
    def simulate_step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        cache_key = self._get_cache_key(state, action)
        
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]
        
        state_norm = self.normalizer.transform(state)
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
        action_onehot = F.one_hot(
            torch.LongTensor([action]), self.config.action_size
        ).float().to(device)
        
        next_state_mean, _, reward, done_prob, _ = \
            self.world_model(state_t, action_onehot)
        
        next_state_norm = next_state_mean.cpu().numpy()[0]
        next_state = self.normalizer.inverse_transform(next_state_norm)
        reward_val = reward.item()
        done_val = done_prob.item() > 0.5
        
        if len(self.state_cache) < self.cache_max_size:
            self.state_cache[cache_key] = (next_state, reward_val, done_val)
        
        return next_state, reward_val, done_val
    
    def search(self, root_state: np.ndarray, 
               add_noise: bool = True) -> Tuple[np.ndarray, int, float]:
        root = MCTSNodeV2(root_state)
        
        action_probs = self.get_action_probs(root_state, add_noise=add_noise)
        root.expand(action_probs, dirichlet_noise=add_noise,
                   alpha=self.config.mcts_dirichlet_alpha,
                   weight=self.config.mcts_dirichlet_weight)
        
        for sim in range(self.config.mcts_simulations):
            node = root
            state = root_state.copy()
            search_path = [node]
            
            while node.is_expanded and len(node.children) > 0:
                action, node = node.select_child(self.config.mcts_c_puct, fpu_reduction=0.1)
                search_path.append(node)
                
                state, reward, done = self.simulate_step(state, action)
                node.predicted_reward = reward
                node.predicted_done = done
                
                if done:
                    break
            
            if node.predicted_done:
                value = 0
            elif node.visit_count == 0:
                value = self.evaluate_state(state)
            else:
                if not node.is_expanded:
                    action_probs = self.get_action_probs(state, add_noise=False)
                    node.expand(action_probs)
                value = self.evaluate_state(state)
            
            for node in reversed(search_path):
                node.backup(value, gamma=self.config.mcts_gamma)
                value = -value
        
        visits = np.array([root.children[a].visit_count 
                          if a in root.children else 0
                          for a in range(self.config.action_size)])
        
        if self.config.mcts_temperature > 0:
            visits_temp = visits ** (1.0 / self.config.mcts_temperature)
            probs = visits_temp / (visits_temp.sum() + 1e-8)
            best_action = np.random.choice(self.config.action_size, p=probs)
        else:
            best_action = np.argmax(visits)
        
        entropy_before = -np.sum(action_probs * np.log(action_probs + 1e-8))
        visits_norm = visits / (visits.sum() + 1e-8)
        entropy_after = -np.sum(visits_norm * np.log(visits_norm + 1e-8))
        information_gain = entropy_before - entropy_after
        
        return visits, best_action, information_gain


# ============================================================================
#                         PPO训练 V2 (支持从RND恢复)
# ============================================================================
class ActorCriticV2(nn.Module):
    """Actor-Critic网络 V2"""
    
    def __init__(self, state_encoder: DualPathStateEncoder,
                 projection_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        # 共享层 (结构调整以适配RND checkpoint)
        self.shared = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),  # 与RND版本保持一致
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        
        # Actor头 - 简化结构以适配RND checkpoint
        # 注意: 只使用单层Linear，与RND版本保持一致
        self.actor = nn.Linear(hidden_size // 2, action_size)
        
        # Critic头 - 简化结构以适配RND checkpoint
        self.critic = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.state_encoder(input_ids, attention_mask, numeric_state)
        shared_features = self.shared(features)
        
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value
    
    def get_action_value(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         numeric_state: torch.Tensor = None) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(input_ids, attention_mask, numeric_state)
        dist = Categorical(logits=logits)
        return dist, value


class PPOBufferV2:
    """PPO缓冲区 V2 - 支持MCTS策略目标"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = []
        self.states_norm = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.state_strs = []
        # 新增: MCTS相关存储
        self.mcts_probs = []  # MCTS搜索得到的策略分布
        self.advantages = None
        self.returns = None
        
    def add(self, state, state_norm, action, reward, value, log_prob, done, state_str, mcts_prob=None):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.states_norm.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
            self.state_strs.pop(0)
        
        self.states.append(state)
        self.states_norm.append(state_norm)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.state_strs.append(state_str)
        self.mcts_probs.append(mcts_prob)  # 新增: 存储MCTS策略
    
    def clear(self):
        self.states.clear()
        self.states_norm.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.state_strs.clear()
        self.mcts_probs.clear()  # 新增
        self.advantages = None
        self.returns = None
    
    def compute_gae(self, gamma: float, gae_lambda: float):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        self.advantages = torch.FloatTensor(advantages).to(device)
        self.returns = self.advantages + torch.FloatTensor(self.values).to(device)
        
        self.advantages = (self.advantages - self.advantages.mean()) / \
                         (self.advantages.std() + 1e-8)
    
    def __len__(self):
        return len(self.states)


def train_ppo_v2(env, world_model: WorldModelV2, actor_critic: ActorCriticV2,
                 tokenizer, config, normalizer: StateNormalizerV2,
                 start_episode: int = 0) -> ActorCriticV2:
    """PPO训练 V2"""
    print(f"\n{'='*70}")
    print("🚀 PPO + MCTS训练 V2")
    print(f"{'='*70}")
    
    # MCTS
    policy_net = PolicyNetworkV2(actor_critic.state_encoder, config.projection_size,
                                  config.action_size, config.hidden_size).to(device)
    value_net = ValueNetworkV2(actor_critic.state_encoder, config.projection_size,
                               config.hidden_size).to(device)
    
    mcts = MCTSV2(world_model, policy_net, value_net, tokenizer, config, normalizer)
    
    # 优化器
    optimizer = optim.AdamW([
        {'params': actor_critic.state_encoder.text_projection.parameters(), 'lr': config.projection_lr},
        {'params': actor_critic.state_encoder.numeric_encoder.parameters(), 'lr': config.embedder_lr},
        {'params': actor_critic.state_encoder.fusion_layer.parameters(), 'lr': config.embedder_lr},
        {'params': actor_critic.shared.parameters(), 'lr': config.actor_lr},
        {'params': actor_critic.actor.parameters(), 'lr': config.actor_lr},
        {'params': actor_critic.critic.parameters(), 'lr': config.critic_lr},
    ], weight_decay=config.sft_weight_decay)
    
    # LoRA参数单独优化
    lora_params = []
    for name, module in actor_critic.state_encoder.llama.named_modules():
        if isinstance(module, LoRALinear):
            lora_params.extend(module.lora.parameters())
    
    if lora_params:
        lora_optimizer = optim.AdamW(lora_params, lr=config.lora_lr, weight_decay=1e-4)
    else:
        lora_optimizer = None
    
    # 调度器
    total_steps = (config.n_episodes - start_episode) // config.update_frequency
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=int(total_steps * 0.1),
                                      total_steps=total_steps)
    
    buffer = PPOBufferV2()
    scores = []
    best_score = float('-inf')
    
    logger = TrainingLoggerV2()
    
    # 添加训练进度条
    pbar = tqdm(range(start_episode + 1, config.n_episodes + 1), 
                desc="🚀 PPO+MCTS训练", 
                unit="episode",
                initial=start_episode + 1,
                total=config.n_episodes + 1)
    
    for episode in pbar:
        # 解冻LLaMA (如果配置了)
        if config.freeze_llama_initially and episode > config.unfreeze_llama_after:
            for param in actor_critic.state_encoder.llama.parameters():
                param.requires_grad = True
        
        state = dict2array(env.reset())
        state_norm = normalizer.transform(state)
        episode_reward = 0
        done = False
        step = 0
        episode_nitrogen = 0
        episode_water = 0
        
        while not done and step < config.max_steps:
            actor_critic.eval()
            
            state_str = array2str(state)
            inputs = tokenizer([state_str], return_tensors='pt',
                             padding='max_length', truncation=True,
                             max_length=config.token_size).to(device)
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            
            # === 关键修复: MCTS渐进式启用机制 ===
            # 1. 预热期内不使用MCTS，保持恢复的策略稳定
            # 2. 渐进启用期内逐步增加MCTS使用频率
            # 3. 完全激活后才正常使用MCTS
            mcts_prob = None
            
            # 计算当前MCTS启用进度 (0.0 到 1.0)
            if episode <= config.mcts_warmup_episodes:
                mcts_progress = 0.0  # 预热期：完全不使用MCTS
            elif episode < config.mcts_gradual_start:
                mcts_progress = 0.0  # 仍然预热
            elif episode < config.mcts_full_active:
                # 渐进启用期：逐步增加MCTS使用比例
                mcts_progress = (episode - config.mcts_gradual_start) / (config.mcts_full_active - config.mcts_gradual_start)
            else:
                mcts_progress = 1.0  # 完全激活
            
            # 根据进度决定是否使用MCTS
            # 渐进期内，只有部分step会使用MCTS
            use_mcts_this_step = False
            if config.use_mcts_for_action and mcts_progress > 0:
                # 根据进度调整使用频率
                effective_frequency = max(1, int(config.mcts_action_frequency / max(mcts_progress, 0.1)))
                if step % effective_frequency == 0 and random.random() < mcts_progress:
                    use_mcts_this_step = True
            
            if use_mcts_this_step:
                try:
                    # 使用MCTS搜索选择动作
                    visits, mcts_action, info_gain = mcts.search(state, add_noise=True)
                    
                    # 安全检查：确保MCTS选择的动作有效
                    if mcts_action < 0 or mcts_action >= config.action_size:
                        # MCTS返回无效动作，回退到策略网络
                        with torch.no_grad():
                            dist, value = actor_critic.get_action_value(
                                inputs['input_ids'], inputs['attention_mask'], state_t
                            )
                            action = dist.sample()
                            action_int = action.item()
                            log_prob = dist.log_prob(action)
                            mcts_prob = None
                    else:
                        action_int = mcts_action
                        
                        # 计算MCTS策略分布（用于后续策略更新）
                        visits_temp = visits ** (1.0 / config.mcts_temperature)
                        mcts_prob = visits_temp / (visits_temp.sum() + 1e-8)
                        
                        # 获取对应动作的log_prob
                        with torch.no_grad():
                            dist, value = actor_critic.get_action_value(
                                inputs['input_ids'], inputs['attention_mask'], state_t
                            )
                            log_prob = dist.log_prob(torch.tensor(action_int).to(device))
                        
                        logger.log('mcts_info_gain', info_gain)
                except Exception as e:
                    # MCTS搜索出错，回退到策略网络
                    print(f"⚠️ MCTS搜索异常: {e}, 回退到策略网络")
                    with torch.no_grad():
                        dist, value = actor_critic.get_action_value(
                            inputs['input_ids'], inputs['attention_mask'], state_t
                        )
                        action = dist.sample()
                        action_int = action.item()
                        log_prob = dist.log_prob(action)
                        mcts_prob = None
            else:
                # 使用策略网络采样
                with torch.no_grad():
                    dist, value = actor_critic.get_action_value(
                        inputs['input_ids'], inputs['attention_mask'], state_t
                    )
                    action = dist.sample()
                    action_int = action.item()
                    log_prob = dist.log_prob(action)
            
            action_dict = action_to_dict(action_int, state)
            episode_nitrogen += action_dict['anfer']
            episode_water += action_dict['amir']
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            next_state_norm = normalizer.transform(next_state)
            
            env_reward = get_reward(state, action_dict['anfer'],
                                   action_dict['amir'], next_state, done,
                                   config.k1, config.k2, config.k3)
            
            # 存储MCTS策略分布
            buffer.add(state, state_norm, action_int, env_reward, 
                      value.item(), log_prob.item(), done, state_str, mcts_prob)
            
            state = next_state
            state_norm = next_state_norm
            episode_reward += env_reward
            step += 1
        
        scores.append(episode_reward)
        logger.log('episode_reward', episode_reward)
        logger.log('episode_steps', step)
        
        # 更新进度条显示
        avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        pbar.set_postfix({
            'Score': f'{episode_reward:.0f}',
            'Avg': f'{avg_score:.0f}',
            'Best': f'{best_score:.0f}',
            'Steps': step
        })
        
        # 更新
        if episode % config.update_frequency == 0 and len(buffer) >= config.mini_batch_size:
            actor_critic.train()
            
            buffer.compute_gae(config.gamma, config.gae_lambda)
            
            for _ in range(config.ppo_epochs):
                indices = np.random.permutation(len(buffer))
                
                for start in range(0, len(buffer), config.mini_batch_size):
                    end = start + config.mini_batch_size
                    if end > len(buffer):
                        continue
                    
                    mb_indices = indices[start:end]
                    
                    mb_state_strs = [buffer.state_strs[i] for i in mb_indices]
                    mb_states_norm = torch.FloatTensor(
                        np.array([buffer.states_norm[i] for i in mb_indices])
                    ).to(device)
                    mb_actions = torch.LongTensor(
                        [buffer.actions[i] for i in mb_indices]
                    ).to(device)
                    mb_old_log_probs = torch.FloatTensor(
                        [buffer.log_probs[i] for i in mb_indices]
                    ).to(device)
                    mb_advantages = buffer.advantages[mb_indices]
                    mb_returns = buffer.returns[mb_indices]
                    
                    inputs = tokenizer(mb_state_strs, return_tensors='pt',
                                     padding='max_length', truncation=True,
                                     max_length=config.token_size).to(device)
                    
                    dist, values = actor_critic.get_action_value(
                        inputs['input_ids'], inputs['attention_mask'], mb_states_norm
                    )
                    
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 
                                       1 + config.clip_ratio) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(values.squeeze(), mb_returns)
                    
                    # === 关键修复: MCTS策略目标损失 ===
                    # 当有MCTS搜索结果时，将其作为额外监督信号
                    mcts_policy_loss = torch.tensor(0.0, device=device)
                    if config.use_mcts_for_update:
                        # 收集mini_batch中每个样本对应的MCTS概率
                        mb_mcts_probs = []
                        mb_mcts_indices = []
                        for idx, i in enumerate(mb_indices):
                            if buffer.mcts_probs[i] is not None:
                                mb_mcts_probs.append(buffer.mcts_probs[i])
                                mb_mcts_indices.append(idx)
                        
                        if len(mb_mcts_probs) > 0:
                            # 只对有MCTS结果的样本计算KL散度
                            mcts_probs_tensor = torch.FloatTensor(
                                np.array(mb_mcts_probs)
                            ).to(device)
                            
                            # 获取对应样本的logits
                            selected_logits = dist.logits[mb_mcts_indices]
                            
                            # KL散度: KL(MCTS || policy)
                            mcts_policy_loss = F.kl_div(
                                F.log_softmax(selected_logits, dim=-1),
                                mcts_probs_tensor,
                                reduction='batchmean'
                            )
                            logger.log('mcts_policy_loss', mcts_policy_loss.item())
                    
                    loss = (policy_loss + config.value_coef * value_loss 
                           - config.entropy_coef * entropy
                           + config.mcts_update_weight * mcts_policy_loss)  # 新增MCTS损失
                    
                    optimizer.zero_grad()
                    if lora_optimizer:
                        lora_optimizer.zero_grad()
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        actor_critic.parameters(), config.max_grad_norm
                    )
                    
                    optimizer.step()
                    if lora_optimizer:
                        lora_optimizer.step()
            
            buffer.clear()
            scheduler.step()
        
        # 打印
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            current_lr = scheduler.get_lr()
            yield_value = state[4]
            print(f"Episode {episode}/{config.n_episodes} | "
                  f"Score: {episode_reward:.0f} | "
                  f"Avg: {avg_score:.0f} | "
                  f"Steps: {step} | "
                  f"Yield: {yield_value:.1f} kg/ha | "
                  f"N: {episode_nitrogen:.1f} kg/ha | "
                  f"Water: {episode_water:.1f} mm | "
                  f"LR: {current_lr:.2e}")
        
        # 保存最佳
        if episode_reward > best_score:
            best_score = episode_reward
            torch.save({
                'actor_critic': actor_critic.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode': episode,
                'score': best_score
            }, os.path.join(config.final_model_dir, 'best_model.pth'))
        
        # 定期保存
        if episode % config.save_frequency == 0:
            torch.save({
                'actor_critic': actor_critic.state_dict(),
                'episode': episode,
            }, os.path.join(config.final_model_dir, f'checkpoint_ep{episode}.pth'))
        
        # 保存LoRA
        if episode % 200 == 0 and lora_params:
            lora_state = get_lora_state_dict(actor_critic.state_encoder.llama)
            torch.save(lora_state, os.path.join(config.lora_dir, f'lora_ep{episode}.pth'))
    
    logger.plot_training_curves(
        os.path.join('/home/wuyang/results/llama_ppo_mcts_results/data_0420', 'ppo_training.png')
    )
    
    print(f"\n✓ 训练完成！最佳得分: {best_score:.0f}")
    return actor_critic


# ============================================================================
#                         主流程
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print("🌾 Agri-Reasoner V2.1 从RND Checkpoint恢复训练")
    print(f"{'='*70}\n")
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化环境
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/llama_ppo_mcts_results/data_0420/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True
    }
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    # 初始化LLaMA
    print("📚 初始化LLaMA模型...")
    tokenizer = LlamaTokenizerFast.from_pretrained(config.llama_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
    llama_model = LlamaModel.from_pretrained(
        config.llama_path,
        torch_dtype=torch_dtype,
        use_cache=False
    ).to(device)
    
    if config.use_gradient_checkpointing:
        llama_model.gradient_checkpointing_enable()
    
    # 应用LoRA
    print("\n🔧 应用LoRA微调...")
    llama_model, lora_params = apply_lora_to_llama(
        llama_model,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    print(f"  LoRA参数数量: {sum(p.numel() for p in lora_params)}")
    
    # 初始化状态归一化器
    print("\n🔧 初始化状态归一化器...")
    state_normalizer = StateNormalizerV2(config.state_size, method='adaptive')
    
    # === 任务1: 世界模型 ===
    wm_data = collect_world_model_data_v2(env, state_normalizer, config.wm_data_samples)
    world_model = train_world_model_v2(wm_data, state_normalizer)
    
    # === 任务2: 构建双路径编码器 ===
    print("\n🔧 构建双路径状态编码器...")
    state_encoder = DualPathStateEncoder(
        llama_model,
        config.state_size,
        config.projection_size,
        config.numeric_hidden_size,
        config.fusion_hidden_size,
        config.use_numeric_path,
        config.numeric_weight
    ).to(device)
    
    # === 任务3: 构建Actor-Critic ===
    print("\n🔧 构建Actor-Critic网络...")
    actor_critic = ActorCriticV2(
        state_encoder,
        config.projection_size,
        config.action_size,
        config.hidden_size
    ).to(device)
    
    # === 任务3.5: 从RND Checkpoint恢复 (核心新增) ===
    start_episode = 0
    
    if config.load_from_rnd_checkpoint and os.path.exists(config.rnd_checkpoint_path):
        print(f"\n{'='*70}")
        print("📥 从RND Checkpoint恢复策略网络")
        print(f"{'='*70}")
        
        # 创建checkpoint适配器
        adapter = CheckpointAdapter(verbose=True)
        
        # 执行checkpoint适配
        result = adapter.adapt_checkpoint_to_mcts(
            config.rnd_checkpoint_path,
            state_encoder,
            actor_critic
        )
        
        if result['success']:
            print("\n✅ RND Checkpoint恢复成功!")
            for detail in result['mapping_details']:
                print(f"   - {detail}")
            
            # 如果checkpoint包含训练进度，可以继续训练
            try:
                checkpoint = torch.load(config.rnd_checkpoint_path, map_location='cpu')
                if 'episode' in checkpoint:
                    start_episode = checkpoint['episode']
                    print(f"   继续训练起始轮次: {start_episode}")
            except:
                pass
        else:
            print(f"\n⚠️ RND Checkpoint恢复失败: {result.get('error', '未知错误')}")
            print("   将使用随机初始化继续训练...")
    else:
        if config.load_from_rnd_checkpoint:
            print(f"\n⚠️ RND Checkpoint文件不存在: {config.rnd_checkpoint_path}")
            print("   将使用随机初始化继续训练...")
    
    # === 任务4: PPO训练 ===
    actor_critic = train_ppo_v2(
        env, world_model, actor_critic, tokenizer, config, state_normalizer,
        start_episode=start_episode
    )
    
    # === 任务5: 测试 ===
    print(f"\n{'='*70}")
    print("🧪 测试最终模型")
    print(f"{'='*70}")
    
    test_episodes = 20
    test_scores = []
    
    for ep in range(test_episodes):
        state = dict2array(env.reset())
        state_norm = state_normalizer.transform(state)
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config.max_steps:
            state_str = array2str(state)
            inputs = tokenizer([state_str], return_tensors='pt',
                             padding='max_length', truncation=True,
                             max_length=config.token_size).to(device)
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist, _ = actor_critic.get_action_value(
                    inputs['input_ids'], inputs['attention_mask'], state_t
                )
                action = torch.argmax(dist.probs, dim=-1).item()
            
            action_dict = action_to_dict(action, state)
            
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            
            reward = get_reward(state, action_dict['anfer'],
                              action_dict['amir'], next_state, done,
                              config.k1, config.k2, config.k3)
            
            episode_reward += reward
            state = next_state
            state_norm = state_normalizer.transform(state)
            step += 1
        
        test_scores.append(episode_reward)
        print(f"测试 {ep+1}/{test_episodes} | 得分: {episode_reward:.0f}")
    
    print(f"\n✓ 测试完成！")
    print(f"  平均得分: {np.mean(test_scores):.0f} ± {np.std(test_scores):.0f}")
    print(f"  最高得分: {np.max(test_scores):.0f}")
    print(f"  最低得分: {np.min(test_scores):.0f}")
    
    env.close()
    
    print(f"\n{'='*70}")
    print("🎉 Agri-Reasoner V2.1 训练全部完成！")
    print(f"{'='*70}\n")
    
    print("📊 训练总结:")
    print(f"  - RND Checkpoint: {config.rnd_checkpoint_path}")
    print(f"  - 世界模型: {config.world_model_dir}")
    print(f"  - 最终模型: {config.final_model_dir}")
    print(f"  - LoRA权重: {config.lora_dir}")
    print(f"  - 归一化器: {config.normalizer_dir}")
    print(f"  - 训练曲线: /home/wuyang/results/llama_ppo_mcts_results/data_0420/")
    print(f"\n🔧 核心优化:")
    print(f"  - 从RND Checkpoint恢复策略网络初始化")
    print(f"  - LoRA微调: 秩={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  - 双路径编码: 数值权重={config.numeric_weight}")
    print(f"  - 注意力世界模型: {config.wm_use_attention}")


if __name__ == "__main__":
    main()