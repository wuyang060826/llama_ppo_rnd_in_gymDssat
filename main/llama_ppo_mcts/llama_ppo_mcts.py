#!/usr/bin/env python3
"""
Agri-Reasoner V2: 基于世界模型与MCTS的可解释农业决策系统
深度优化版 - 解决架构割裂与信息瓶颈问题

核心优化:
1. LoRA微调LLaMA - 突破冻结限制，参数高效微调
2. 双路径状态编码 - 数值+文本并行，保留精度
3. 统一世界模型接口 - 与策略网络深度耦合
4. 改进的MCTS - 更好的搜索效率
5. 端到端训练流程 - 联合优化

作者: Agri-Reasoner Team
版本: V2.0
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
#                              配置参数 V2
# ============================================================================
@dataclass
class ConfigV2:
    """全局配置 V2 - 深度优化版"""
    
    # === 路径配置 ===
    llama_path: str = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'
    baseline_path: str = '/home/gymusr/llama_ppo_checkpoints/best_baseline_model/best/model_ep1520.pth'
    
    world_model_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/pretrained_v2/world_model'
    policy_sft_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/pretrained_v2/policy_sft'
    value_sft_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/pretrained_v2/value_sft'
    final_model_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/checkpoints_v2/final'
    normalizer_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/pretrained_v2/normalizers'
    lora_dir: str = '/home/gymusr/llama_ppo_mcts_checkpoints/pretrained_v2/lora'
    
    # 数据路径
    world_model_data: str = '/home/gymusr/llama_ppo_mcts_results/data_v2/world_model_data.pkl'
    policy_sft_data: str = '/home/gymusr/llama_ppo_mcts_results/data_v2/policy_sft_data.pkl'
    value_sft_data: str = '/home/gymusr/llama_ppo_mcts_results/data_v2/value_sft_data.pkl'
    expert_data: str = '/home/gymusr/llama_ppo_mcts_results/data_v2/expert_trajectories.pkl'
    
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
    mcts_simulations: int = 100
    mcts_c_puct: float = 2.0
    mcts_gamma: float = 0.99
    mcts_max_depth: int = 15
    mcts_temperature: float = 1.0
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_weight: float = 0.25
    intrinsic_reward_weight: float = 0.01
    
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
                 '/home/gymusr/llama_ppo_mcts_results/data_v2', '/home/gymusr/llama_ppo_mcts_results/logs_v2', '/home/gymusr/llama_ppo_mcts_results/figures_v2']:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
#                         状态归一化系统 V2
# ============================================================================
class StateNormalizerV2:
    """
    状态归一化器 V2 - 更鲁棒的归一化方案
    
    改进:
    1. 基于分位数的裁剪 - 处理极端异常值
    2. 分维度可配置策略
    3. 运行时统计更新
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
        self.q01 = np.zeros(state_size)  # 1%分位数
        self.q99 = np.zeros(state_size)  # 99%分位数
        
        # 维度特定配置（基于农业数据特性）
        self.dimension_config = {
            'standardize': [0, 4, 7, 21, 23],  # 大数值范围用标准化
            'minmax': [20, 18],  # 百分比/比例用Min-Max
            'robust': list(range(9, 18)),  # 氮含量用Robust
            'log': [],  # 对数变换（如有需要）
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
        
        # 处理异常值
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
        
        # 初始化运行统计
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
        
        # 裁剪异常值
        states_clipped = np.clip(states, self.q01, self.q99)
            
        normalized = states_clipped.copy()
        
        if self.method == 'hybrid' or self.method == 'adaptive':
            # 混合策略
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
                    
            # 其余维度用标准化
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
        
        # 裁剪到合理范围
        normalized = np.clip(normalized, -5, 5)
        
        # 可选：更新运行统计
        if update_stats and not single_sample:
            self._update_running_stats(states_clipped)
        
        return normalized.squeeze() if single_sample else normalized
    
    def _update_running_stats(self, states: np.ndarray):
        """更新运行统计（用于在线学习）"""
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
    """奖励归一化器 V2 - 带指数移动平均"""
    
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
            # 指数移动平均
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
    """状态数组转文本描述 - 增强版"""
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
    """生成详细的农学描述 - 增强版"""
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
    """奖励函数 - 增强版"""
    if done: 
        # 收获时的奖励：产量减去成本
        yield_reward = k1 * state[4]
        nitrogen_cost = k2 * n_action
        water_cost = k3 * w_action
        return yield_reward - nitrogen_cost - water_cost
    # 生长过程中的成本
    return -k2 * n_action - k3 * w_action

def action_to_dict(action: int, state: np.ndarray) -> dict:
    """动作索引转字典，带约束"""
    action_dict = {
        'anfer': (action % 5) * 40,
        'amir': int(action / 5) * 6
    }
    # 约束逻辑
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
#                         LoRA模块实现（核心优化）
# ============================================================================
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    
    论文: LoRA: Low-Rank Adaptation of Large Language Models
    核心思想: 冻结预训练权重，在旁边添加低秩分解矩阵
    
    W' = W + BA
    其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 r: int = 8, alpha: int = 16, dropout: float = 0.05, dtype=torch.float32):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA矩阵 - 使用指定的数据类型
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            original_output: 原始线性层的输出
            
        Returns:
            修正后的输出
        """
        # 确保输入和权重使用相同的数据类型
        dtype = original_output.dtype
        x = x.to(dtype)
        
        # LoRA路径: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.to(dtype).T @ self.lora_B.to(dtype).T * self.scaling
        return original_output + lora_output

class LoRALinear(nn.Module):
    """带LoRA的线性层"""
    
    def __init__(self, original_linear: nn.Linear, r: int = 8, 
                 alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        
        self.original_linear = original_linear
        # 获取原始线性层的数据类型
        self.dtype = original_linear.weight.dtype
        
        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            r, alpha, dropout,
            dtype=self.dtype  # 传递原始层的数据类型
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入使用正确的数据类型
        x = x.to(self.dtype)
        original_output = self.original_linear(x)
        return self.lora(x, original_output)

def apply_lora_to_llama(llama_model, r: int = 16, alpha: int = 32, 
                        dropout: float = 0.05, target_modules: List[str] = None):
    """
    将LoRA应用到LLaMA模型
    
    Args:
        llama_model: LLaMA模型实例
        r: LoRA秩
        alpha: LoRA缩放因子
        dropout: Dropout率
        target_modules: 目标模块列表
        
    Returns:
        修改后的模型和LoRA参数列表
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    
    lora_params = []
    
    for name, module in llama_model.named_modules():
        # 检查是否是目标模块
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 获取父模块和属性名
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = llama_model
                    for p in parent_name.split('.'):
                        parent = getattr(parent, p)
                else:
                    parent = llama_model
                    attr_name = name
                
                # 替换为LoRA线性层
                lora_linear = LoRALinear(module, r, alpha, dropout)
                setattr(parent, attr_name, lora_linear)
                
                # 收集LoRA参数
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
#                         双路径状态编码器（核心优化）
# ============================================================================
class DualPathStateEncoder(nn.Module):
    """
    双路径状态编码器
    
    解决问题: 文本描述丢失数值精度
    
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
        
        # 文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_size * 2),
            nn.LayerNorm(projection_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_size * 2, projection_size),
            nn.LayerNorm(projection_size),
        )
        
        # 数值路径 - 直接编码数值状态
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
        
        # 门控机制 - 动态调整两条路径的权重
        if use_numeric_path:
            self.gate = nn.Sequential(
                nn.Linear(projection_size * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: Tokenized文本输入
            attention_mask: 注意力掩码
            numeric_state: 归一化后的数值状态（可选）
            
        Returns:
            融合后的状态表示
        """
        # 文本路径
        llama_outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = llama_outputs.last_hidden_state
        
        # 平均池化（考虑mask）
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embeddings = sum_embeddings / sum_mask
        
        text_features = self.text_projection(text_embeddings)
        
        # 如果不使用数值路径，直接返回文本特征
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
        gate_weights = self.softmax(self.gate(x))  # [batch, num_heads]
        
        head_outputs = torch.stack([head(x) for head in self.heads], dim=1)  # [batch, num_heads, output_size]
        
        # 加权组合
        output = torch.einsum('bn,bno->bo', gate_weights, head_outputs)
        
        return output

class WorldModelV2(nn.Module):
    """
    世界模型 V2 - 更强的预测能力
    
    改进:
    1. 注意力机制 - 建模状态维度间的依赖
    2. 多头预测 - 集成多个预测头
    3. 残差连接 - 更深的网络
    4. 不确定性量化 - 更好的探索
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
        
        # 自注意力层（可选）
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
        
        # 价值预测头（用于MCTS）
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
        """
        前向传播
        
        Returns:
            next_state_mean, next_state_std, reward, done_prob, value
        """
        batch_size = state.size(0)
        
        # 输入嵌入
        x = torch.cat([state, action_onehot], dim=-1)
        x = self.input_embed(x)
        
        # 时间嵌入
        if timestep is not None:
            t = timestep.float().unsqueeze(-1) / 1000.0
            t_embed = self.time_embed(t)
            x = x + t_embed
        
        # 自注意力（将特征作为序列处理）
        if self.use_attention:
            # 添加序列维度
            x_seq = x.unsqueeze(1)  # [batch, 1, hidden]
            attn_out, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.attn_norm(x + attn_out.squeeze(1))
        
        # 残差块
        for block in self.residual_blocks:
            x = F.gelu(x + block(x))
        
        # 预测头
        next_state_mean = self.state_mean_head(x)
        next_state_std = self.state_std_head(x) + 0.01  # 最小标准差
        reward = self.reward_head(x)
        done_prob = self.done_head(x)
        value = self.value_head(x)
        
        return next_state_mean, next_state_std, reward, done_prob, value
    
    def get_loss(self, state: torch.Tensor, action_onehot: torch.Tensor,
             next_state: torch.Tensor, reward: torch.Tensor, 
             done: torch.Tensor, value: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """计算损失"""
        pred_mean, pred_std, pred_reward, pred_done, pred_value = \
            self.forward(state, action_onehot)
        
        # 状态损失 - 使用NLL with variance
        state_nll = 0.5 * torch.log(pred_std ** 2) + \
                    0.5 * ((next_state - pred_mean) ** 2) / (pred_std ** 2)
        state_loss = state_nll.mean()
        
        # 奖励损失
        reward_loss = F.smooth_l1_loss(pred_reward, reward)
        
        # 终止损失 - 修复维度不匹配问题
        done_loss = F.binary_cross_entropy(pred_done.squeeze(-1), done.squeeze(-1))
        
        # 价值损失（如果提供）
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
    
    def __init__(self, log_dir: str = '/home/gymusr/llama_ppo_mcts_results/logs_v2'):
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.best_metrics = {}
        
    def log(self, metric_name: str, value: float):
        self.metrics[metric_name].append(value)
        
        # 更新最佳值
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
            
            # 移动平均
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
            # 线性预热
            scale = self.current_step / self.warmup_steps
        else:
            # 余弦退火
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
            # 更智能的动作选择
            if random.random() < 0.3:
                action = random.randint(0, config.action_size - 1)
            else:
                # 启发式策略
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
        
        # 计算折现回报
        if episode_rewards:
            G = 0
            episode_values = []
            for r in reversed(episode_rewards):
                G = r + config.gamma * G
                episode_values.insert(0, G)
            values.extend(episode_values)
            episode_returns.append(G)
    
    pbar.close()
    
    # 补齐values长度
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
    
    # 根据生长阶段调整
    if day < 4000:  # 苗期
        if soil_moisture < 0.22:
            return random.choice([5, 10])  # 中等灌溉
        elif nitrogen_pct < 0.012:
            return random.choice([1, 2])  # 少量施肥
        return 0
    elif day < 8000:  # 拔节期
        if soil_moisture < 0.28:
            return random.choice([10, 15])  # 中高灌溉
        elif nitrogen_pct < 0.018:
            return random.choice([2, 3, 7, 8])  # 中等施肥+灌溉
        return random.choice([0, 5])
    elif day < 12000:  # 抽穗开花期（关键期）
        if soil_moisture < 0.32 and cumulative_irr < 1400:
            return random.choice([15, 20])  # 高灌溉
        elif nitrogen_pct < 0.022:
            return random.choice([3, 4, 8, 9])  # 高施肥+灌溉
        return random.choice([5, 10])
    else:  # 灌浆成熟期
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
    
    # 奖励归一化
    reward_normalizer = RewardNormalizerV2()
    reward_normalizer.update(data['rewards'])
    rewards = reward_normalizer.normalize(data['rewards'])
    reward_normalizer.save(os.path.join(config.normalizer_dir, 'reward_normalizer.npz'))
    
    # 价值归一化
    values_mean = np.mean(data['values'])
    values_std = np.std(data['values']) + 1e-8
    values = (data['values'] - values_mean) / values_std
    
    # 打印统计
    print(f"\n📊 归一化后数据统计:")
    print(f"  状态: [{states.min():.2f}, {states.max():.2f}]")
    print(f"  奖励: [{rewards.min():.2f}, {rewards.max():.2f}]")
    
    # 转换为张量
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(data['actions'])
    next_states_t = torch.FloatTensor(next_states)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
    dones_t = torch.FloatTensor(data['dones']).unsqueeze(1)
    timesteps_t = torch.LongTensor(data['timesteps'])
    values_t = torch.FloatTensor(values)
    
    actions_onehot = F.one_hot(actions_t, config.action_size).float()
    
    # 数据集划分
    dataset = TensorDataset(states_t, actions_onehot, next_states_t, 
                           rewards_t, dones_t, timesteps_t, values_t)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.wm_batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.wm_batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    # 模型
    model = WorldModelV2(config.state_size, config.action_size, 
                        config.wm_hidden_size, use_attention=config.wm_use_attention).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.wm_lr, 
                           weight_decay=config.wm_weight_decay)
    
    # 调度器
    total_steps = len(train_loader) * config.wm_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    # 早停
    early_stopping = EarlyStoppingV2(patience=config.wm_patience, mode='min')
    
    best_val_loss = float('inf')
    
    print(f"开始训练 (训练集: {train_size}, 验证集: {val_size})...")
    
    for epoch in range(config.wm_epochs):
        # 训练
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
            
            losses = model.get_loss(s, a, ns, r, d, v)
            
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
                
                losses = model.get_loss(s, a, ns, r, d, v)
                
                for key, value in losses.items():
                    val_losses[key] += value.item()
        
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # 日志
        logger.log_batch({
            'train_total_loss': train_losses['total'],
            'train_state_loss': train_losses['state'],
            'val_total_loss': val_losses['total'],
            'val_state_loss': val_losses['state'],
            'learning_rate': scheduler.get_lr()
        })
        
        # 打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.wm_epochs} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"LR: {scheduler.get_lr():.2e}")
        
        # 保存最佳
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_losses': dict(val_losses)
            }, model_path)
        
        # 早停
        if early_stopping(val_losses['total'], model, epoch):
            print(f"⚠️ 早停于 Epoch {epoch+1}")
            break
    
    early_stopping.restore_best_weights(model)
    
    logger.plot_training_curves(
        os.path.join('/home/gymusr/llama_ppo_mcts_results/figures_v2', 'world_model_training.png')
    )
    
    print(f"\n✓ 世界模型训练完成")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    
    return model

# ============================================================================
#                         策略网络 V2
# ============================================================================
class PolicyNetworkV2(nn.Module):
    """策略网络 V2 - 支持双路径输入"""
    
    def __init__(self, state_encoder: DualPathStateEncoder, 
                 projection_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        # 策略头
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
        
        # 动作先验（可学习）
        self.action_prior = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        # 获取融合的状态表示
        features = self.state_encoder(input_ids, attention_mask, numeric_state)
        
        # 策略网络
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
    """价值网络 V2 - 支持双路径输入"""
    
    def __init__(self, state_encoder: DualPathStateEncoder,
                 projection_size: int, hidden_size: int):
        super().__init__()
        
        # 共享编码器
        self.state_encoder = state_encoder
        
        # 价值头
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
        
        # 额外信息
        self.predicted_reward = 0.0
        self.predicted_done = False
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, total_visits: int, 
                  fpu_reduction: float = 0.0) -> float:
        """改进的UCB分数"""
        if self.visit_count == 0:
            # FPU (First Play Urgency) for unvisited nodes
            if self.parent is not None:
                return self.parent.value() - fpu_reduction + \
                       c_puct * self.prior * math.sqrt(total_visits + 1)
            return float('inf')
        
        # PUCT公式
        q_value = self.value()
        u_value = c_puct * self.prior * math.sqrt(total_visits) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct: float, fpu_reduction: float = 0.0) -> Tuple[int, 'MCTSNodeV2']:
        """选择最优子节点"""
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
        """扩展节点"""
        if self.is_expanded:
            return
        
        # 添加Dirichlet噪声（根节点）
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
        """回溯更新"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value * gamma  # 对手视角（这里简化处理）
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
        """获取动作概率"""
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
        
        # 添加噪声（训练时）
        if add_noise:
            noise = np.random.dirichlet([self.config.mcts_dirichlet_alpha] * self.config.action_size)
            probs = (1 - self.config.mcts_dirichlet_weight) * probs + \
                   self.config.mcts_dirichlet_weight * noise
        
        return probs
    
    @torch.no_grad()
    def evaluate_state(self, state: np.ndarray) -> float:
        """评估状态价值"""
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
        """使用世界模型模拟一步"""
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
        """执行MCTS搜索"""
        root = MCTSNodeV2(root_state)
        
        # 获取初始策略
        action_probs = self.get_action_probs(root_state, add_noise=add_noise)
        root.expand(action_probs, dirichlet_noise=add_noise,
                   alpha=self.config.mcts_dirichlet_alpha,
                   weight=self.config.mcts_dirichlet_weight)
        
        for sim in range(self.config.mcts_simulations):
            node = root
            state = root_state.copy()
            search_path = [node]
            
            # 选择
            while node.is_expanded and len(node.children) > 0:
                action, node = node.select_child(self.config.mcts_c_puct, fpu_reduction=0.1)
                search_path.append(node)
                
                state, reward, done = self.simulate_step(state, action)
                node.predicted_reward = reward
                node.predicted_done = done
                
                if done:
                    break
            
            # 评估
            if node.predicted_done:
                value = 0
            elif node.visit_count == 0:
                value = self.evaluate_state(state)
            else:
                # 扩展
                if not node.is_expanded:
                    action_probs = self.get_action_probs(state, add_noise=False)
                    node.expand(action_probs)
                value = self.evaluate_state(state)
            
            # 回溯
            for node in reversed(search_path):
                node.backup(value, gamma=self.config.mcts_gamma)
                value = -value
        
        # 计算访问次数分布
        visits = np.array([root.children[a].visit_count 
                          if a in root.children else 0
                          for a in range(self.config.action_size)])
        
        # 选择动作
        if self.config.mcts_temperature > 0:
            visits_temp = visits ** (1.0 / self.config.mcts_temperature)
            probs = visits_temp / (visits_temp.sum() + 1e-8)
            best_action = np.random.choice(self.config.action_size, p=probs)
        else:
            best_action = np.argmax(visits)
        
        # 计算信息增益
        entropy_before = -np.sum(action_probs * np.log(action_probs + 1e-8))
        visits_norm = visits / (visits.sum() + 1e-8)
        entropy_after = -np.sum(visits_norm * np.log(visits_norm + 1e-8))
        information_gain = entropy_before - entropy_after
        
        return visits, best_action, information_gain

# ============================================================================
#                         PPO训练 V2
# ============================================================================
class ActorCriticV2(nn.Module):
    """Actor-Critic网络 V2"""
    
    def __init__(self, state_encoder: DualPathStateEncoder,
                 projection_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
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
    """PPO缓冲区 V2"""
    
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
        self.advantages = None
        self.returns = None
        
    def add(self, state, state_norm, action, reward, value, log_prob, done, state_str):
        if len(self.states) >= self.max_size:
            # 移除最旧的
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
    
    def clear(self):
        self.states.clear()
        self.states_norm.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.state_strs.clear()
        self.advantages = None
        self.returns = None
    
    def compute_gae(self, gamma: float, gae_lambda: float):
        """计算GAE"""
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
        
        # 标准化优势
        self.advantages = (self.advantages - self.advantages.mean()) / \
                         (self.advantages.std() + 1e-8)
    
    def __len__(self):
        return len(self.states)

def train_ppo_v2(env, world_model: WorldModelV2, actor_critic: ActorCriticV2,
                 tokenizer, config, normalizer: StateNormalizerV2) -> ActorCriticV2:
    """PPO训练 V2"""
    print(f"\n{'='*70}")
    print("🚀 PPO + MCTS训练 V2")
    print(f"{'='*70}")
    
    # MCTS
    # 从actor_critic中提取state_encoder构建独立的策略和价值网络
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
    total_steps = config.n_episodes // config.update_frequency
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=int(total_steps * 0.1),
                                      total_steps=total_steps)
    
    buffer = PPOBufferV2()
    scores = []
    best_score = float('-inf')
    
    logger = TrainingLoggerV2()
    
    for episode in range(1, config.n_episodes + 1):
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
            
            with torch.no_grad():
                dist, value = actor_critic.get_action_value(
                    inputs['input_ids'], inputs['attention_mask'], state_t
                )
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            action_int = action.item()
            action_dict = action_to_dict(action_int, state)
            # 累计施氮量和灌溉量
            episode_nitrogen += action_dict['anfer']
            episode_water += action_dict['amir']
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            next_state_norm = normalizer.transform(next_state)
            
            env_reward = get_reward(state, action_dict['anfer'],
                                   action_dict['amir'], next_state, done,
                                   config.k1, config.k2, config.k3)
            
            buffer.add(state, state_norm, action_int, env_reward, 
                      value.item(), log_prob.item(), done, state_str)
            
            state = next_state
            state_norm = next_state_norm
            episode_reward += env_reward
            step += 1
        
        scores.append(episode_reward)
        logger.log('episode_reward', episode_reward)
        logger.log('episode_steps', step)
        
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
                    
                    # 准备数据
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
                    
                    loss = (policy_loss + config.value_coef * value_loss 
                           - config.entropy_coef * entropy)
                    
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
            # 获取产量数据（状态索引4）
            yield_value = state[4]  # 单位: kg/ha
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
        os.path.join('/home/gymusr/llama_ppo_mcts_results/figures_v2', 'ppo_training.png')
    )
    
    print(f"\n✓ 训练完成！最佳得分: {best_score:.0f}")
    return actor_critic

# ============================================================================
#                         主流程
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print("🌾 Agri-Reasoner V2 深度优化版训练流程")
    print(f"{'='*70}\n")
    
    # 设置随机种子
    set_seed(42)
    
    # 初始化环境
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/gymusr/llama_ppo_mcts_results/logs_v2/dssat-pdi.log',
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
    
    # === 任务4: PPO训练 ===
    actor_critic = train_ppo_v2(
        env, world_model, actor_critic, tokenizer, config, state_normalizer
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
    print("🎉 Agri-Reasoner V2 训练全部完成！")
    print(f"{'='*70}\n")
    
    print("📊 训练总结:")
    print(f"  - 世界模型: {config.world_model_dir}")
    print(f"  - 最终模型: {config.final_model_dir}")
    print(f"  - LoRA权重: {config.lora_dir}")
    print(f"  - 归一化器: {config.normalizer_dir}")
    print(f"  - 训练曲线: /home/gymusr/llama_ppo_mcts_results/figures_v2/")
    print(f"\n🔧 核心优化:")
    print(f"  - LoRA微调: 秩={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  - 双路径编码: 数值权重={config.numeric_weight}")
    print(f"  - 注意力世界模型: {config.wm_use_attention}")

if __name__ == "__main__":
    main()