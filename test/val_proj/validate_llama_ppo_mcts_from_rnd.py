#!/usr/bin/env python3
"""
Agri-Reasoner V2 Checkpoint 验证脚本
=====================================

功能:
1. 加载指定的checkpoint进行验证
2. 支持多checkpoint对比评估
3. 详细的评估指标和可视化
4. 支持渲染模式查看决策过程

使用方法:
    # 验证单个checkpoint
    /opt/gym_dssat_pdi/bin/python  /home/wuyang/test/val_proj/validate_llama_ppo_mcts_from_rnd.py --checkpoint /home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/final/best_model.pth
    
    # 验证最佳模型
    python validate_checkpoint.py --best
    
    # 对比多个checkpoint
    python validate_checkpoint.py --compare /path/to/model1.pth /path/to/model2.pth
    
    # 详细模式（渲染每步决策）
    python validate_checkpoint.py --checkpoint /path/to/checkpoint.pth --verbose

作者: Agri-Reasoner Team
"""

import numpy as np
import pandas as pd
import random
import pickle
import json
import argparse
import os
import sys
import math
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体
# 设置中文字体
import matplotlib.font_manager as fm

# 尝试加载中文字体，如果失败则使用系统默认字体
try:
    # 尝试常见的Noto Sans SC字体路径
    font_paths = [
        '/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.otf',
        '/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf',
        '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
    ]
    
    font_loaded = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
            font_loaded = True
            break
    
    if not font_loaded:
        print("⚠️ 未找到Noto Sans SC字体，将使用系统默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
except Exception as e:
    print(f"⚠️ 字体加载失败: {e}")
    print("⚠️ 将使用系统默认字体")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

from transformers import LlamaModel, LlamaTokenizerFast

# ============================================================================
#                             配置参数
# ============================================================================
@dataclass
class ValidateConfig:
    """验证配置"""
    
    # LLaMA路径
    llama_path: str = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'
    
    # 默认checkpoint路径
    checkpoint_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/final'
    world_model_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/world_model'
    normalizer_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/normalizers'
    lora_dir: str = '/home/wuyang/checkpoints/llama_ppo_mcts_checkpoints/lora'
    
    # 环境参数
    state_size: int = 25
    action_size: int = 25
    max_steps: int = 200
    
    # 网络参数
    token_size: int = 256
    projection_size: int = 256
    hidden_size: int = 256
    numeric_hidden_size: int = 128
    fusion_hidden_size: int = 256
    use_numeric_path: bool = True
    numeric_weight: float = 0.3
    
    # LoRA参数
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
    
    # 验证参数
    eval_episodes: int = 20
    seed: int = 42
    
    # 奖励函数参数
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # DSSAT环境参数
    run_dssat_location: str = '/opt/dssat_pdi/run_dssat'
    log_saving_path: str = '/home/wuyang/results/validation/dssat-pdi.log'
    
    # 输出路径
    output_dir: str = '/home/wuyang/results/validation'

config = ValidateConfig()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
#                             辅助函数
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
    elif day < 8000:
        growth_stage = "拔节期"
    elif day < 12000:
        growth_stage = "抽穗开花期"
    else:
        growth_stage = "灌浆成熟期"
    
    desc = f"""【农业状态诊断】
生长天数: {int(day)} 天 ({growth_stage})
预估产量: {int(state[4])} kg/ha
地上生物量: {int(state[7])} kg/ha
土壤水分: {state[20]:.3f} cm³/cm³
累计灌溉: {int(state[21])} mm
累计降雨: {int(state[23])} mm
氮素百分比: {state[18]*100:.2f}%"""
    return desc


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


def action_to_str(action: int) -> str:
    """动作转文本描述"""
    nitrogen = (action % 5) * 40  # 0, 40, 80, 120, 160 kg/ha
    water = int(action / 5) * 6    # 0, 6, 12, 18, 24 mm
    
    desc_parts = []
    if nitrogen > 0:
        desc_parts.append(f"施氮{nitrogen}kg/ha")
    if water > 0:
        desc_parts.append(f"灌溉{water}mm")
    if not desc_parts:
        desc_parts.append("无操作")
    
    return " + ".join(desc_parts)


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#                             归一化器
# ============================================================================
class StateNormalizerV2:
    """状态归一化器"""
    
    def __init__(self, state_size: int, method: str = 'adaptive'):
        self.state_size = state_size
        self.method = method
        self.mean = np.zeros(state_size)
        self.std = np.ones(state_size)
        self.min_val = np.zeros(state_size)
        self.max_val = np.ones(state_size)
        self.median = np.zeros(state_size)
        self.iqr = np.zeros(state_size)
        self.q01 = np.zeros(state_size)
        self.q99 = np.zeros(state_size)
        self.running_mean = np.zeros(state_size)
        self.running_var = np.ones(state_size)
        self.count = 0
        self.fitted = False
        
        self.dimension_config = {
            'standardize': [0, 4, 7, 21, 23],
            'minmax': [20, 18],
            'robust': list(range(9, 18)),
            'log': [],
        }
        
    def transform(self, states: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return states
            
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        states_clipped = np.clip(states, self.q01, self.q99)
        normalized = states_clipped.copy()
        
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
        
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.squeeze() if single_sample else normalized
    
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


# ============================================================================
#                             LoRA模块
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
    
    return llama_model, lora_params


def load_lora_state_dict(model, state_dict: Dict[str, torch.Tensor]):
    """加载LoRA参数"""
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if name + '.lora.lora_A' in state_dict:
                module.lora.lora_A.data = state_dict[name + '.lora.lora_A']
                loaded_count += 1
            if name + '.lora.lora_B' in state_dict:
                module.lora.lora_B.data = state_dict[name + '.lora.lora_B']
                loaded_count += 1
    return loaded_count


# ============================================================================
#                             双路径状态编码器
# ============================================================================
class DualPathStateEncoder(nn.Module):
    """双路径状态编码器"""
    
    def __init__(self, llama_model, state_size: int,
                 projection_size: int = 256, numeric_hidden: int = 128,
                 fusion_hidden: int = 256, use_numeric_path: bool = True,
                 numeric_weight: float = 0.3):
        super().__init__()
        
        self.use_numeric_path = use_numeric_path
        self.numeric_weight = numeric_weight
        self.state_size = state_size
        self.projection_size = projection_size
        
        self.llama = llama_model
        self.hidden_size = llama_model.config.hidden_size
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_size * 2),
            nn.LayerNorm(projection_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_size * 2, projection_size),
            nn.LayerNorm(projection_size),
        )
        
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
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(projection_size * 2, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden, projection_size),
                nn.LayerNorm(projection_size),
            )
            
            self.gate = nn.Sequential(
                nn.Linear(projection_size * 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                numeric_state: torch.Tensor = None) -> torch.Tensor:
        llama_outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = llama_outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        text_embeddings = sum_embeddings / sum_mask
        
        text_features = self.text_projection(text_embeddings)
        
        if not self.use_numeric_path or numeric_state is None:
            return text_features
        
        numeric_features = self.numeric_encoder(numeric_state)
        
        concat_features = torch.cat([text_features, numeric_features], dim=-1)
        gate_value = self.gate(concat_features)
        
        fused = gate_value * numeric_features + (1 - gate_value) * text_features
        
        final_features = self.fusion_layer(concat_features)
        
        output = final_features + 0.5 * text_features + 0.5 * numeric_features
        
        return output


# ============================================================================
#                             Actor-Critic网络
# ============================================================================
class ActorCriticV2(nn.Module):
    """Actor-Critic网络 V2"""
    
    def __init__(self, state_encoder: DualPathStateEncoder,
                 projection_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.state_encoder = state_encoder
        
        self.shared = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_size // 2, action_size)
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
    
    def get_action_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         numeric_state: torch.Tensor = None) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask, numeric_state)
        return F.softmax(logits, dim=-1)


# ============================================================================
#                             评估器
# ============================================================================
class CheckpointValidator:
    """Checkpoint验证器"""
    
    def __init__(self, config: ValidateConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.device = device
        
        self.tokenizer = None
        self.llama_model = None
        self.state_encoder = None
        self.actor_critic = None
        self.normalizer = None
        self.env = None
        
    def setup(self):
        """初始化所有组件"""
        print(f"\n{'='*70}")
        print("🔧 初始化验证环境")
        print(f"{'='*70}")
        
        # 1. 初始化LLaMA
        print("\n📚 加载LLaMA模型...")
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.config.llama_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.llama_model = LlamaModel.from_pretrained(
            self.config.llama_path,
            torch_dtype=torch_dtype,
            use_cache=False
        ).to(self.device)
        
        # 2. 应用LoRA
        print("🔧 应用LoRA...")
        self.llama_model, lora_params = apply_lora_to_llama(
            self.llama_model,
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules
        )
        
        # 3. 构建状态编码器
        print("🔧 构建状态编码器...")
        self.state_encoder = DualPathStateEncoder(
            self.llama_model,
            self.config.state_size,
            self.config.projection_size,
            self.config.numeric_hidden_size,
            self.config.fusion_hidden_size,
            self.config.use_numeric_path,
            self.config.numeric_weight
        ).to(self.device)
        
        # 4. 构建Actor-Critic
        print("🔧 构建Actor-Critic网络...")
        self.actor_critic = ActorCriticV2(
            self.state_encoder,
            self.config.projection_size,
            self.config.action_size,
            self.config.hidden_size
        ).to(self.device)
        
        # 5. 加载归一化器
        normalizer_path = os.path.join(self.config.normalizer_dir, 'state_normalizer.npz')
        if os.path.exists(normalizer_path):
            print(f"📦 加载归一化器: {normalizer_path}")
            self.normalizer = StateNormalizerV2(self.config.state_size)
            self.normalizer.load(normalizer_path)
        else:
            print("⚠️ 未找到归一化器，将使用原始状态")
            self.normalizer = None
        
        # 6. 初始化环境
        print("🔧 初始化DSSAT环境...")
        os.makedirs(os.path.dirname(self.config.log_saving_path), exist_ok=True)
        
        env_args = {
            'run_dssat_location': self.config.run_dssat_location,
            'log_saving_path': self.config.log_saving_path,
            'mode': 'all',
            'seed': self.config.seed,
            'random_weather': True
        }
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        
        print(f"✓ 初始化完成，使用设备: {self.device}")
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
        
        print(f"\n📦 加载Checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 获取state_dict
        if 'actor_critic' in checkpoint:
            state_dict = checkpoint['actor_critic']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载到模型
        missing_keys, unexpected_keys = self.actor_critic.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  ⚠️ 缺失的键: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠️ 意外的键: {len(unexpected_keys)}")
        
        # 打印checkpoint信息
        if 'episode' in checkpoint:
            print(f"  训练轮次: {checkpoint['episode']}")
        if 'score' in checkpoint:
            print(f"  训练得分: {checkpoint['score']:.2f}")
        if 'best_val_loss' in checkpoint:
            print(f"  验证损失: {checkpoint['best_val_loss']:.4f}")
        
        return checkpoint
    
    def load_lora(self, lora_path: str) -> int:
        """加载LoRA权重"""
        if not os.path.exists(lora_path):
            print(f"⚠️ LoRA文件不存在: {lora_path}")
            return 0
        
        print(f"📦 加载LoRA权重: {lora_path}")
        lora_state = torch.load(lora_path, map_location='cpu', weights_only=False)
        loaded_count = load_lora_state_dict(self.llama_model, lora_state)
        print(f"  加载了 {loaded_count} 个LoRA参数")
        return loaded_count
    
    def evaluate_episode(self, deterministic: bool = True, 
                         render: bool = False) -> Dict[str, Any]:
        """评估单个episode"""
        state = dict2array(self.env.reset())
        state_norm = self.normalizer.transform(state) if self.normalizer else state
        
        episode_data = {
            'states': [state.copy()],
            'actions': [],
            'rewards': [],
            'action_probs': [],
            'values': [],
            'nitrogen_applied': 0,
            'water_applied': 0,
            'done': False,
            'final_yield': 0,
            'steps': 0
        }
        
        done = False
        step = 0
        total_reward = 0
        
        self.actor_critic.eval()
        
        while not done and step < self.config.max_steps:
            # 准备输入
            state_str = array2str(state)
            inputs = self.tokenizer(
                [state_str], return_tensors='pt',
                padding='max_length', truncation=True,
                max_length=self.config.token_size
            ).to(self.device)
            state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            
            # 获取动作
            with torch.no_grad():
                dist, value = self.actor_critic.get_action_value(
                    inputs['input_ids'], inputs['attention_mask'], state_t
                )
                
                if deterministic:
                    action = torch.argmax(dist.probs, dim=-1).item()
                else:
                    action = dist.sample().item()
                
                probs = dist.probs.cpu().numpy()[0]
                value_val = value.item()
            
            # 执行动作
            action_dict = action_to_dict(action, state)
            next_state_raw, _, done, _ = self.env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            
            # 计算奖励
            reward = get_reward(
                state, action_dict['anfer'], action_dict['amir'],
                next_state, done,
                self.config.k1, self.config.k2, self.config.k3
            )
            
            # 记录数据
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['action_probs'].append(probs)
            episode_data['values'].append(value_val)
            episode_data['nitrogen_applied'] += action_dict['anfer']
            episode_data['water_applied'] += action_dict['amir']
            
            if render:
                print(f"\n  Step {step+1}:")
                print(f"    状态: {array2detailed_str(state)}")
                print(f"    动作: {action_to_str(action)}")
                print(f"    奖励: {reward:.2f}")
            
            state = next_state
            state_norm = self.normalizer.transform(state) if self.normalizer else state
            total_reward += reward
            step += 1
        
        episode_data['states'].append(state.copy())
        episode_data['done'] = done
        episode_data['final_yield'] = state[4]
        episode_data['steps'] = step
        episode_data['total_reward'] = total_reward
        episode_data['episode_return'] = total_reward
        
        return episode_data
    
    def evaluate(self, n_episodes: int = None, deterministic: bool = True,
                 render: bool = False) -> Dict[str, Any]:
        """运行评估"""
        if n_episodes is None:
            n_episodes = self.config.eval_episodes
        
        print(f"\n{'='*70}")
        print(f"🧪 开始评估 ({n_episodes} episodes, {'确定性' if deterministic else '随机'}策略)")
        print(f"{'='*70}\n")
        
        all_episodes = []
        scores = []
        yields = []
        nitrogen_used = []
        water_used = []
        steps_list = []
        
        for ep in range(n_episodes):
            episode_data = self.evaluate_episode(deterministic=deterministic, render=render)
            all_episodes.append(episode_data)
            
            scores.append(episode_data['total_reward'])
            yields.append(episode_data['final_yield'])
            nitrogen_used.append(episode_data['nitrogen_applied'])
            water_used.append(episode_data['water_applied'])
            steps_list.append(episode_data['steps'])
            
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"得分: {episode_data['total_reward']:.0f} | "
                  f"产量: {episode_data['final_yield']:.0f} kg/ha | "
                  f"氮: {episode_data['nitrogen_applied']:.0f} kg | "
                  f"水: {episode_data['water_applied']:.0f} mm | "
                  f"步数: {episode_data['steps']}")
        
        # 计算统计
        results = {
            'n_episodes': n_episodes,
            'deterministic': deterministic,
            'episodes': all_episodes,
            'scores': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'all': scores
            },
            'yields': {
                'mean': np.mean(yields),
                'std': np.std(yields),
                'min': np.min(yields),
                'max': np.max(yields),
                'all': yields
            },
            'nitrogen': {
                'mean': np.mean(nitrogen_used),
                'std': np.std(nitrogen_used),
                'all': nitrogen_used
            },
            'water': {
                'mean': np.mean(water_used),
                'std': np.std(water_used),
                'all': water_used
            },
            'steps': {
                'mean': np.mean(steps_list),
                'std': np.std(steps_list),
                'all': steps_list
            }
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any], title: str = "评估结果"):
        """打印评估摘要"""
        print(f"\n{'='*70}")
        print(f"📊 {title}")
        print(f"{'='*70}")
        
        print(f"\n【总得分】")
        print(f"  平均: {results['scores']['mean']:.0f} ± {results['scores']['std']:.0f}")
        print(f"  中位数: {results['scores']['median']:.0f}")
        print(f"  范围: [{results['scores']['min']:.0f}, {results['scores']['max']:.0f}]")
        
        print(f"\n【最终产量 (kg/ha)】")
        print(f"  平均: {results['yields']['mean']:.0f} ± {results['yields']['std']:.0f}")
        print(f"  范围: [{results['yields']['min']:.0f}, {results['yields']['max']:.0f}]")
        
        print(f"\n【资源消耗】")
        print(f"  氮肥: {results['nitrogen']['mean']:.0f} ± {results['nitrogen']['std']:.0f} kg/ha")
        print(f"  灌溉: {results['water']['mean']:.0f} ± {results['water']['std']:.0f} mm")
        
        print(f"\n【效率指标】")
        if results['nitrogen']['mean'] > 0:
            n_efficiency = results['yields']['mean'] / results['nitrogen']['mean']
            print(f"  氮肥效率: {n_efficiency:.2f} kg产量/kg氮肥")
        if results['water']['mean'] > 0:
            w_efficiency = results['yields']['mean'] / results['water']['mean']
            print(f"  水分效率: {w_efficiency:.2f} kg产量/mm灌溉")
        
        print(f"\n【稳定性】")
        cv = results['scores']['std'] / (abs(results['scores']['mean']) + 1e-8) * 100
        print(f"  得分变异系数: {cv:.1f}%")
        
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """绘制评估结果图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 得分分布
        ax = axes[0, 0]
        scores = results['scores']['all']
        ax.hist(scores, bins=15, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'均值: {np.mean(scores):.0f}')
        ax.set_xlabel('得分')
        ax.set_ylabel('频次')
        ax.set_title('得分分布')
        ax.legend()
        
        # 2. 产量分布
        ax = axes[0, 1]
        yields = results['yields']['all']
        ax.hist(yields, bins=15, color='forestgreen', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(yields), color='red', linestyle='--', label=f'均值: {np.mean(yields):.0f}')
        ax.set_xlabel('产量 (kg/ha)')
        ax.set_ylabel('频次')
        ax.set_title('产量分布')
        ax.legend()
        
        # 3. 得分趋势
        ax = axes[0, 2]
        ax.plot(range(1, len(scores)+1), scores, 'o-', color='steelblue', alpha=0.7)
        moving_avg = np.convolve(scores, np.ones(5)/5, mode='valid')
        ax.plot(range(5, len(scores)+1), moving_avg, 'r-', linewidth=2, label='移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('得分')
        ax.set_title('得分趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 资源消耗分布
        ax = axes[1, 0]
        nitrogen = results['nitrogen']['all']
        water = results['water']['all']
        x = np.arange(len(nitrogen))
        width = 0.35
        ax.bar(x - width/2, nitrogen, width, label='氮肥 (kg/ha)', color='coral')
        ax.bar(x + width/2, water, width, label='灌溉 (mm)', color='skyblue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('用量')
        ax.set_title('资源消耗')
        ax.legend()
        
        # 5. 得分vs产量
        ax = axes[1, 1]
        ax.scatter(yields, scores, c=range(len(scores)), cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('产量 (kg/ha)')
        ax.set_ylabel('得分')
        ax.set_title('得分 vs 产量')
        # 拟合线
        z = np.polyfit(yields, scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(yields), max(yields), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8)
        ax.grid(True, alpha=0.3)
        
        # 6. 效率分析
        ax = axes[1, 2]
        n_efficiency = [y/n if n > 0 else 0 for y, n in zip(yields, nitrogen)]
        w_efficiency = [y/w if w > 0 else 0 for y, w in zip(yields, water)]
        ax.scatter(n_efficiency, scores, label='氮肥效率', color='coral', alpha=0.6)
        ax.scatter(w_efficiency, scores, label='水分效率', color='skyblue', alpha=0.6)
        ax.set_xlabel('效率 (kg产量/单位资源)')
        ax.set_ylabel('得分')
        ax.set_title('效率 vs 得分')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 结果图已保存: {save_path}")
        
        plt.close()
        
    def save_results(self, results: Dict[str, Any], checkpoint_path: str, save_dir: str = None):
        """保存评估结果"""
        if save_dir is None:
            save_dir = self.config.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        
        # 保存详细结果
        result_path = os.path.join(save_dir, f'eval_{checkpoint_name}_{timestamp}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        
        # 保存摘要
        summary_path = os.path.join(save_dir, f'summary_{checkpoint_name}_{timestamp}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Checkpoint评估报告\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episode数量: {results['n_episodes']}\n")
            f.write(f"确定性策略: {results['deterministic']}\n\n")
            
            f.write(f"【总得分】\n")
            f.write(f"  平均: {results['scores']['mean']:.0f} ± {results['scores']['std']:.0f}\n")
            f.write(f"  中位数: {results['scores']['median']:.0f}\n")
            f.write(f"  范围: [{results['scores']['min']:.0f}, {results['scores']['max']:.0f}]\n\n")
            
            f.write(f"【最终产量】\n")
            f.write(f"  平均: {results['yields']['mean']:.0f} ± {results['yields']['std']:.0f} kg/ha\n\n")
            
            f.write(f"【资源消耗】\n")
            f.write(f"  氮肥: {results['nitrogen']['mean']:.0f} ± {results['nitrogen']['std']:.0f} kg/ha\n")
            f.write(f"  灌溉: {results['water']['mean']:.0f} ± {results['water']['std']:.0f} mm\n")
        
        print(f"\n📁 结果已保存:")
        print(f"  详细结果: {result_path}")
        print(f"  评估摘要: {summary_path}")
        
        return result_path, summary_path
    
    def cleanup(self):
        """清理资源"""
        if self.env is not None:
            self.env.close()


# ============================================================================
#                             对比评估
# ============================================================================
def compare_checkpoints(checkpoint_paths: List[str], config: ValidateConfig,
                       n_episodes: int = 20):
    """对比多个checkpoint"""
    print(f"\n{'='*70}")
    print(f"📊 多Checkpoint对比评估")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"⚠️ 跳过不存在的checkpoint: {path}")
            continue
        
        validator = CheckpointValidator(config)
        validator.setup()
        validator.load_checkpoint(path)
        
        # 尝试加载对应的LoRA
        checkpoint_dir = os.path.dirname(path)
        lora_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('lora') and f.endswith('.pth')]
        if lora_files:
            latest_lora = sorted(lora_files)[-1]
            validator.load_lora(os.path.join(checkpoint_dir, latest_lora))
        
        results = validator.evaluate(n_episodes=n_episodes)
        all_results[path] = results
        
        validator.cleanup()
    
    # 打印对比结果
    print(f"\n{'='*70}")
    print(f"📊 对比结果汇总")
    print(f"{'='*70}\n")
    
    print(f"{'Checkpoint':<40} {'平均得分':>12} {'平均产量':>15} {'氮肥':>10} {'灌溉':>10}")
    print("-" * 90)
    
    for path, results in all_results.items():
        name = os.path.basename(path)
        print(f"{name:<40} "
              f"{results['scores']['mean']:>12.0f} "
              f"{results['yields']['mean']:>12.0f} kg/ha "
              f"{results['nitrogen']['mean']:>10.0f} "
              f"{results['water']['mean']:>10.0f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 得分对比
    ax = axes[0]
    names = [os.path.basename(p) for p in all_results.keys()]
    means = [r['scores']['mean'] for r in all_results.values()]
    stds = [r['scores']['std'] for r in all_results.values()]
    
    x = range(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], rotation=45, ha='right')
    ax.set_ylabel('平均得分')
    ax.set_title('得分对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 产量对比
    ax = axes[1]
    yields = [r['yields']['mean'] for r in all_results.values()]
    yields_std = [r['yields']['std'] for r in all_results.values()]
    
    bars = ax.bar(x, yields, yerr=yields_std, capsize=5, color='forestgreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:20] + '...' if len(n) > 20 else n for n in names], rotation=45, ha='right')
    ax.set_ylabel('平均产量 (kg/ha)')
    ax.set_title('产量对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(config.output_dir, 'checkpoint_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {save_path}")
    plt.close()
    
    return all_results


# ============================================================================
#                             主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Agri-Reasoner V2 Checkpoint验证工具')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='指定checkpoint路径')
    parser.add_argument('--best', action='store_true',
                       help='使用最佳模型')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                       help='对比多个checkpoint')
    parser.add_argument('--episodes', type=int, default=20,
                       help='评估episode数量')
    parser.add_argument('--stochastic', action='store_true',
                       help='使用随机策略而非确定性策略')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出每步决策')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 更新配置
    if args.output:
        config.output_dir = args.output
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 对比模式
    if args.compare:
        compare_checkpoints(args.compare, config, args.episodes)
        return
    
    # 确定checkpoint路径
    if args.best:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # 查找最新的checkpoint
        checkpoints = [f for f in os.listdir(config.checkpoint_dir) 
                      if f.endswith('.pth')]
        if not checkpoints:
            print(f"❌ 未找到checkpoint文件在: {config.checkpoint_dir}")
            return
        checkpoint_path = os.path.join(config.checkpoint_dir, sorted(checkpoints)[-1])
    
    # 创建验证器
    validator = CheckpointValidator(config)
    
    try:
        # 初始化
        validator.setup()
        
        # 加载checkpoint
        checkpoint = validator.load_checkpoint(checkpoint_path)
        
        # 尝试加载LoRA
        lora_dir = os.path.dirname(checkpoint_path)
        lora_files = [f for f in os.listdir(lora_dir) if f.startswith('lora') and f.endswith('.pth')]
        if lora_files:
            latest_lora = sorted(lora_files)[-1]
            validator.load_lora(os.path.join(lora_dir, latest_lora))
        
        # 评估
        results = validator.evaluate(
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
            render=args.verbose
        )
        
        # 打印摘要
        checkpoint_name = os.path.basename(checkpoint_path)
        validator.print_summary(results, title=f"评估结果 - {checkpoint_name}")
        
        # 绘图
        plot_path = os.path.join(config.output_dir, f'eval_{checkpoint_name.replace(".pth", "")}.png')
        validator.plot_results(results, plot_path)
        
        # 保存结果
        validator.save_results(results, checkpoint_path)
        
    finally:
        validator.cleanup()
    
    print(f"\n{'='*70}")
    print("✅ 验证完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
