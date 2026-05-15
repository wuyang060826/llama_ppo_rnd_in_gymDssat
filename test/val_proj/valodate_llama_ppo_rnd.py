'''
/opt/gym_dssat_pdi/bin/python /home/wuyang/test/val_proj/valodate_llama_ppo_rnd.py --checkpoint /home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best/model_ep1600.pth --n_episodes 5
# 1. 验证所有checkpoint (确定性策略)
python validate_llama_ppo.py \
    --checkpoint_dir /home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best \
    --n_episodes 100

# 2. 验证特定checkpoint
python validate_llama_ppo.py \
    --checkpoint /home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best/model_ep1000.pth \
    --n_episodes 100

# 3. 使用随机策略验证
python validate_llama_ppo.py \
    --stochastic \
    --n_episodes 100

# 4. 记录详细步骤信息
python validate_llama_ppo.py \
    --record_steps \
    --n_episodes 50

# 5. 自定义输出路径
python validate_llama_ppo.py \
    --output_dir /home/wuyang/results/validation_results \
    --n_episodes 100
'''
#!/usr/bin/env python3
"""
PPO + Chinese-LLaMA-2 Checkpoint 验证脚本
用于验证训练好的模型checkpoint效果

功能:
1. 加载指定的checkpoint
2. 在环境中运行多轮验证
3. 计算农学指标: 最终产量、灌溉/施肥量、WUE、NUE
4. 计算AI指标: 平均回报、奖励稳定性、动作分布
5. 生成验证报告: PNG/PDF/Excel格式
6. 支持多个checkpoint对比验证
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
import gym
import os
import glob
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
import argparse
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
    """PPO超参数配置 - 与训练代码保持一致"""
    
    # === 网络参数 ===
    token_size: int = 128
    state_size: int = 25
    action_size: int = 25
    hidden_size: int = 256
    projection_size: int = 256
    
    # === 验证参数 ===
    n_val_episodes: int = 100      # 验证轮数
    max_steps_per_episode: int = 200
    
    # === 奖励函数参数 ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === 优化选项 ===
    use_bf16: bool = True
    
    # === 缓存 ===
    embedding_cache_size: int = 100


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


# ============================================================================
#                              网络定义 (与训练代码一致)
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
#                              验证Agent
# ============================================================================

class ValidationAgent:
    """验证专用Agent - 支持确定性策略和随机策略"""
    
    def __init__(self, llama_model, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        
        # 网络
        self.embedder = LLaMAEmbedder(llama_model, config.projection_size).to(device)
        self.actor_critic = ActorCriticHead(config.projection_size, config.action_size, config.hidden_size).to(device)
        
        # 推理缓存
        self.inference_cache = {}
        
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        
        print(f"   Checkpoint loaded successfully!")
        return checkpoint
        
    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding='max_length', 
                              truncation=True, max_length=self.config.token_size).to(device)
    
    @torch.no_grad()
    def get_cached_embedding(self, state_str):
        """推理专用，带缓存"""
        if state_str in self.inference_cache:
            return self.inference_cache[state_str]
        
        inputs = self.tokenize([state_str])
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            embed = self.embedder(inputs['input_ids'], inputs['attention_mask'])
        
        if len(self.inference_cache) > self.config.embedding_cache_size:
            self.inference_cache.popitem()
        self.inference_cache[state_str] = embed
        return embed
    
    def act(self, state, deterministic=True):
        """
        选择动作
        
        Args:
            state: 状态数组
            deterministic: 是否使用确定性策略（验证时通常为True）
        """
        self.embedder.eval()
        self.actor_critic.eval()
        
        state_str = array2str(state)
        embedding = self.get_cached_embedding(state_str)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            dist, value = self.actor_critic.get_action_value(embedding)
            
            if deterministic:
                action = torch.argmax(dist.probs).item()
            else:
                action = dist.sample().item()
            
            log_prob = dist.log_prob(torch.tensor(action, device=device)).item()
            
        return action, log_prob, value.item()
    
    def get_action_distribution(self, state):
        """获取动作分布（用于分析策略）"""
        self.embedder.eval()
        self.actor_critic.eval()
        
        state_str = array2str(state)
        embedding = self.get_cached_embedding(state_str)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
            dist, value = self.actor_critic.get_action_value(embedding)
            probs = dist.probs.detach().cpu().numpy()
            
        return probs, value.item()
    
    def clear_cache(self):
        """清空推理缓存"""
        self.inference_cache.clear()


# ============================================================================
#                              验证结果记录器
# ============================================================================

class ValidationLogger:
    """验证结果记录器"""
    
    def __init__(self, output_dir, checkpoint_name="checkpoint"):
        self.output_dir = output_dir
        self.checkpoint_name = checkpoint_name
        os.makedirs(output_dir, exist_ok=True)
        
        # 验证指标
        self.episode_rewards = []
        self.final_yields = []
        self.irrigation_amounts = []
        self.fertilizer_amounts = []
        self.wue_values = []
        self.nue_values = []
        self.episode_steps = []
        
        # 详细记录
        self.step_details = []  # 每步详情
        
        # 动作分布统计
        self.action_counts = np.zeros(25)  # 25个离散动作
        self.action_fertilizer_dist = {0: 0, 40: 0, 80: 0, 120: 0, 160: 0}  # 施肥量分布
        self.action_irrigation_dist = {i*6: 0 for i in range(5)}  # 灌溉量分布
        
        # 策略熵统计
        self.policy_entropies = []
        
    def record_episode(self, episode, reward, yield_val, irrigation, fertilizer, steps, entropy=None):
        """记录单轮验证数据"""
        self.episode_rewards.append(reward)
        self.final_yields.append(yield_val)
        self.irrigation_amounts.append(irrigation)
        self.fertilizer_amounts.append(fertilizer)
        self.episode_steps.append(steps)
        
        # 计算WUE和NUE
        wue = yield_val / irrigation if irrigation > 0 else 0
        nue = yield_val / fertilizer if fertilizer > 0 else 0
        self.wue_values.append(wue)
        self.nue_values.append(nue)
        
        if entropy is not None:
            self.policy_entropies.append(entropy)
    
    def record_action(self, action, fertilizer_amount, irrigation_amount):
        """记录动作统计"""
        self.action_counts[action] += 1
        self.action_fertilizer_dist[fertilizer_amount] = self.action_fertilizer_dist.get(fertilizer_amount, 0) + 1
        self.action_irrigation_dist[irrigation_amount] = self.action_irrigation_dist.get(irrigation_amount, 0) + 1
    
    def record_step(self, episode, step, state, action, reward, done):
        """记录单步详情"""
        self.step_details.append({
            'episode': episode,
            'step': step,
            'day': state[0] if state is not None else None,
            'yield': state[4] if state is not None else None,
            'soil_moisture': state[20] if state is not None else None,
            'cumulative_irrigation': state[21] if state is not None else None,
            'action': action,
            'reward': reward,
            'done': done
        })
    
    def get_summary(self):
        """获取汇总统计"""
        summary = {
            'n_episodes': len(self.episode_rewards),
            'total_steps': sum(self.episode_steps),
            
            # 回报指标
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'median_reward': np.median(self.episode_rewards),
            
            # 产量指标
            'mean_yield': np.mean(self.final_yields),
            'std_yield': np.std(self.final_yields),
            'min_yield': np.min(self.final_yields),
            'max_yield': np.max(self.final_yields),
            
            # 灌溉指标
            'mean_irrigation': np.mean(self.irrigation_amounts),
            'std_irrigation': np.std(self.irrigation_amounts),
            
            # 施肥指标
            'mean_fertilizer': np.mean(self.fertilizer_amounts),
            'std_fertilizer': np.std(self.fertilizer_amounts),
            
            # 效率指标
            'mean_wue': np.mean(self.wue_values),
            'std_wue': np.std(self.wue_values),
            'mean_nue': np.mean(self.nue_values),
            'std_nue': np.std(self.nue_values),
            
            # 策略熵
            'mean_entropy': np.mean(self.policy_entropies) if self.policy_entropies else None,
            
            # 成功率（产量>阈值）
            'success_rate': np.mean([1 if y > 5000 else 0 for y in self.final_yields]) * 100,
        }
        return summary
    
    def print_summary(self):
        """打印汇总结果"""
        summary = self.get_summary()
        
        print(f"\n{'='*70}")
        print(f"  Validation Summary - {self.checkpoint_name}")
        print(f"{'='*70}")
        
        print(f"\n  [Overall]")
        print(f"    Episodes:           {summary['n_episodes']}")
        print(f"    Total Steps:        {summary['total_steps']}")
        
        print(f"\n  [Reward Statistics]")
        print(f"    Mean:               {summary['mean_reward']:.2f}")
        print(f"    Std:                {summary['std_reward']:.2f}")
        print(f"    Min:                {summary['min_reward']:.2f}")
        print(f"    Max:                {summary['max_reward']:.2f}")
        print(f"    Median:             {summary['median_reward']:.2f}")
        
        print(f"\n  [Yield Statistics (kg/ha)]")
        print(f"    Mean:               {summary['mean_yield']:.2f}")
        print(f"    Std:                {summary['std_yield']:.2f}")
        print(f"    Min:                {summary['min_yield']:.2f}")
        print(f"    Max:                {summary['max_yield']:.2f}")
        
        print(f"\n  [Resource Usage]")
        print(f"    Mean Irrigation:    {summary['mean_irrigation']:.2f} mm")
        print(f"    Mean Fertilizer:    {summary['mean_fertilizer']:.2f} kg/ha")
        
        print(f"\n  [Efficiency Metrics]")
        print(f"    Mean WUE:           {summary['mean_wue']:.4f} kg/mm")
        print(f"    Mean NUE:           {summary['mean_nue']:.4f} kg/kg")
        
        print(f"\n  [Policy Analysis]")
        print(f"    Mean Entropy:       {summary['mean_entropy']:.4f}" if summary['mean_entropy'] else "    Mean Entropy:       N/A")
        print(f"    Success Rate:       {summary['success_rate']:.1f}%")
        
        print(f"{'='*70}\n")
        
    def save_all_results(self):
        """保存所有结果"""
        self._save_to_excel()
        self._save_to_pdf()
        self._save_plots()
        print(f"\nAll validation results saved to: {self.output_dir}")
        
    def _save_to_excel(self):
        """保存到Excel文件"""
        filepath = os.path.join(self.output_dir, f"{self.checkpoint_name}_validation.xlsx")
        
        # 1. 每轮统计
        df_episodes = pd.DataFrame({
            'Episode': list(range(1, len(self.episode_rewards) + 1)),
            'Reward': self.episode_rewards,
            'Final_Yield_kg_ha': self.final_yields,
            'Irrigation_mm': self.irrigation_amounts,
            'Fertilizer_kg_ha': self.fertilizer_amounts,
            'WUE_kg_mm': self.wue_values,
            'NUE_kg_kg': self.nue_values,
            'Steps': self.episode_steps
        })
        
        # 2. 汇总统计
        summary = self.get_summary()
        df_summary = pd.DataFrame([summary])
        
        # 3. 动作分布
        action_df = pd.DataFrame({
            'Action': list(range(25)),
            'Count': self.action_counts,
            'Frequency': self.action_counts / self.action_counts.sum() * 100
        })
        
        # 4. 施肥量分布
        fert_df = pd.DataFrame({
            'Fertilizer_kg_ha': list(self.action_fertilizer_dist.keys()),
            'Count': list(self.action_fertilizer_dist.values())
        })
        
        # 5. 灌溉量分布
        irr_df = pd.DataFrame({
            'Irrigation_mm': list(self.action_irrigation_dist.keys()),
            'Count': list(self.action_irrigation_dist.values())
        })
        
        # 6. 步骤详情（如果记录了）
        if self.step_details:
            df_steps = pd.DataFrame(self.step_details)
        else:
            df_steps = pd.DataFrame()
        
        # 保存到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_episodes.to_excel(writer, sheet_name='Episodes', index=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            action_df.to_excel(writer, sheet_name='Action_Distribution', index=False)
            fert_df.to_excel(writer, sheet_name='Fertilizer_Distribution', index=False)
            irr_df.to_excel(writer, sheet_name='Irrigation_Distribution', index=False)
            if not df_steps.empty:
                df_steps.to_excel(writer, sheet_name='Step_Details', index=False)
        
        print(f"Excel saved: {filepath}")
        
    def _save_to_pdf(self):
        """保存到PDF报告"""
        filepath = os.path.join(self.output_dir, f"{self.checkpoint_name}_report.pdf")
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1
        )
        
        elements.append(Paragraph(f"Validation Report: {self.checkpoint_name}", title_style))
        elements.append(Spacer(1, 20))
        
        # 汇总统计表
        summary = self.get_summary()
        elements.append(Paragraph("Summary Statistics", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Episodes', str(summary['n_episodes'])],
            ['Total Steps', str(summary['total_steps'])],
            ['Mean Reward', f"{summary['mean_reward']:.2f}"],
            ['Std Reward', f"{summary['std_reward']:.2f}"],
            ['Max Reward', f"{summary['max_reward']:.2f}"],
            ['Mean Yield (kg/ha)', f"{summary['mean_yield']:.2f}"],
            ['Max Yield (kg/ha)', f"{summary['max_yield']:.2f}"],
            ['Mean Irrigation (mm)', f"{summary['mean_irrigation']:.2f}"],
            ['Mean Fertilizer (kg/ha)', f"{summary['mean_fertilizer']:.2f}"],
            ['Mean WUE (kg/mm)', f"{summary['mean_wue']:.4f}"],
            ['Mean NUE (kg/kg)', f"{summary['mean_nue']:.4f}"],
            ['Success Rate (%)', f"{summary['success_rate']:.1f}"],
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
        
        # 动作分布表
        elements.append(Paragraph("Action Distribution", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        action_data = [['Action', 'Count', 'Frequency (%)']]
        for i in range(25):
            freq = self.action_counts[i] / self.action_counts.sum() * 100 if self.action_counts.sum() > 0 else 0
            action_data.append([str(i), str(int(self.action_counts[i])), f"{freq:.2f}"])
        
        table_action = Table(action_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        table_action.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table_action)
        
        doc.build(elements)
        print(f"PDF saved: {filepath}")
        
    def _save_plots(self):
        """保存图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        # 创建多子图
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Validation Results: {self.checkpoint_name}', fontsize=16)
        
        # 1. 回报分布
        ax = axes[0, 0]
        ax.hist(self.episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 产量分布
        ax = axes[0, 1]
        ax.hist(self.final_yields, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(self.final_yields), color='red', linestyle='--', label=f'Mean: {np.mean(self.final_yields):.2f}')
        ax.set_xlabel('Yield (kg/ha)')
        ax.set_ylabel('Frequency')
        ax.set_title('Yield Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 回报曲线
        ax = axes[0, 2]
        ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.7)
        ax.axhline(np.mean(self.episode_rewards), color='red', linestyle='--', label='Mean')
        ax.fill_between(episodes, 
                        np.array(self.episode_rewards) - np.std(self.episode_rewards),
                        np.array(self.episode_rewards) + np.std(self.episode_rewards),
                        alpha=0.2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 产量曲线
        ax = axes[1, 0]
        ax.plot(episodes, self.final_yields, 'g-', alpha=0.7)
        ax.axhline(np.mean(self.final_yields), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Yield (kg/ha)')
        ax.set_title('Yield per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. WUE vs NUE散点图
        ax = axes[1, 1]
        scatter = ax.scatter(self.wue_values, self.nue_values, c=self.episode_rewards, cmap='viridis', alpha=0.7)
        ax.set_xlabel('WUE (kg/mm)')
        ax.set_ylabel('NUE (kg/kg)')
        ax.set_title('WUE vs NUE (colored by reward)')
        plt.colorbar(scatter, ax=ax, label='Reward')
        ax.grid(True, alpha=0.3)
        
        # 6. 动作分布
        ax = axes[1, 2]
        action_freq = self.action_counts / self.action_counts.sum() * 100 if self.action_counts.sum() > 0 else self.action_counts
        ax.bar(range(25), action_freq, color='purple', alpha=0.7)
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency (%)')
        ax.set_title('Action Distribution')
        ax.grid(True, alpha=0.3)
        
        # 7. 灌溉量分布
        ax = axes[2, 0]
        irr_amounts = list(self.action_irrigation_dist.keys())
        irr_counts = list(self.action_irrigation_dist.values())
        ax.bar(irr_amounts, irr_counts, color='cyan', alpha=0.7)
        ax.set_xlabel('Irrigation (mm)')
        ax.set_ylabel('Count')
        ax.set_title('Irrigation Distribution')
        ax.grid(True, alpha=0.3)
        
        # 8. 施肥量分布
        ax = axes[2, 1]
        fert_amounts = list(self.action_fertilizer_dist.keys())
        fert_counts = list(self.action_fertilizer_dist.values())
        ax.bar(fert_amounts, fert_counts, color='orange', alpha=0.7)
        ax.set_xlabel('Fertilizer (kg/ha)')
        ax.set_ylabel('Count')
        ax.set_title('Fertilizer Distribution')
        ax.grid(True, alpha=0.3)
        
        # 9. 策略熵曲线
        ax = axes[2, 2]
        if self.policy_entropies:
            ax.plot(episodes, self.policy_entropies, 'brown', alpha=0.7)
            ax.axhline(np.mean(self.policy_entropies), color='red', linestyle='--', label=f'Mean: {np.mean(self.policy_entropies):.4f}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Entropy')
            ax.set_title('Policy Entropy')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Entropy not recorded', ha='center', va='center')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{self.checkpoint_name}_validation_plots.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plots saved: {filepath}")
        
        # 额外保存回报和产量的对比图
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 回报箱线图
        ax1.boxplot(self.episode_rewards, vert=True)
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Boxplot')
        ax1.grid(True, alpha=0.3)
        
        # 产量箱线图
        ax2.boxplot(self.final_yields, vert=True)
        ax2.set_ylabel('Yield (kg/ha)')
        ax2.set_title('Yield Boxplot')
        ax2.grid(True, alpha=0.3)
        
        fig2.suptitle(f'Validation Boxplots: {self.checkpoint_name}', fontsize=14)
        plt.tight_layout()
        filepath2 = os.path.join(self.output_dir, f"{self.checkpoint_name}_boxplots.png")
        plt.savefig(filepath2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Boxplots saved: {filepath2}")


# ============================================================================
#                              验证函数
# ============================================================================

def initialize_llama(model_path, use_bf16=True):
    """初始化LLaMA模型"""
    print(f"Initializing LLaMA model: {model_path}")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    model = LlamaModel.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype,
        use_cache=True  # 验证时开启KV Cache加速
    ).to(device)
    
    print(f"   Precision: {torch_dtype}")
    return model, tokenizer


def find_checkpoints(checkpoint_dir):
    """查找所有checkpoint文件"""
    pattern = os.path.join(checkpoint_dir, "model_ep*.pth")
    checkpoints = glob.glob(pattern)
    checkpoints.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]))
    return checkpoints


def validate_checkpoint(agent, env, config, logger, deterministic=True, record_steps=False):
    """
    验证单个checkpoint
    
    Args:
        agent: 验证Agent
        env: 环境
        config: 配置
        logger: 验证记录器
        deterministic: 是否使用确定性策略
        record_steps: 是否记录每步详情
    """
    print(f"\n{'='*70}")
    print(f"  Starting Validation - {config.n_val_episodes} episodes")
    print(f"  Policy Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"{'='*70}")
    
    pbar = tqdm(range(1, config.n_val_episodes + 1), 
                desc="Validation Progress",
                unit="episode",
                ncols=100)
    
    for episode in pbar:
        state = dict2array(env.reset())
        episode_reward = 0
        n_amount = 0
        w_amount = 0
        ep_yield = 0
        steps = 0
        episode_entropies = []
        
        for step in range(config.max_steps_per_episode):
            # 选择动作
            action, log_prob, value = agent.act(state, deterministic=deterministic)
            
            # 获取动作分布计算熵
            probs, _ = agent.get_action_distribution(state)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            episode_entropies.append(entropy)
            
            # 动作映射 (与训练代码一致)
            action_dict = {
                'anfer': (action % 5) * 40,  # 施肥量: 0, 40, 80, 120, 160
                'amir': int(action / 5) * 6   # 灌溉量: 0, 6, 12, 18, 24
            }
            
            # 约束
            if state[0] >= 10000:  # 成熟后不施肥
                action_dict['anfer'] = 0
            if state[21] >= 1600:  # 灌溉量上限
                action_dict['amir'] = 0
            
            # 记录动作
            logger.record_action(action, action_dict['anfer'], action_dict['amir'])
            
            # 执行动作
            next_state_raw, _, done, _ = env.step(action_dict)
            next_state = dict2array(next_state_raw) if not done else state
            
            # 计算奖励
            reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                                next_state, done, config.k1, config.k2, config.k3)
            
            # 记录步骤详情
            if record_steps:
                logger.record_step(episode, step, state, action, reward, done)
            
            state = next_state
            episode_reward += reward
            n_amount += action_dict['anfer']
            w_amount += action_dict['amir']
            steps += 1
            
            if done:
                ep_yield = state[4]
                break
        
        # 记录本轮结果
        mean_entropy = np.mean(episode_entropies) if episode_entropies else None
        logger.record_episode(episode, episode_reward, ep_yield, w_amount, n_amount, steps, mean_entropy)
        
        # 更新进度条
        pbar.set_postfix({
            'reward': f'{episode_reward:.0f}',
            'yield': f'{ep_yield:.0f}',
            'irr': f'{w_amount:.0f}',
            'fert': f'{n_amount:.0f}',
            'wue': f'{logger.wue_values[-1]:.2f}',
            'nue': f'{logger.nue_values[-1]:.2f}'
        })

    pbar.close()
    return logger.get_summary()


def compare_checkpoints(results: Dict[str, dict], output_dir):
    """
    比较多个checkpoint的验证结果
    
    Args:
        results: {checkpoint_name: summary_dict}
        output_dir: 输出目录
    """
    if len(results) <= 1:
        return
    
    print(f"\n{'='*70}")
    print(f"  Checkpoint Comparison")
    print(f"{'='*70}")
    
    # 创建对比表格
    df = pd.DataFrame(results).T
    df.index.name = 'Checkpoint'
    
    # 选择关键指标
    key_metrics = ['mean_reward', 'std_reward', 'mean_yield', 'mean_wue', 'mean_nue', 'success_rate']
    df_display = df[[c for c in key_metrics if c in df.columns]]
    
    print(df_display.to_string())
    
    # 保存对比结果
    filepath = os.path.join(output_dir, "checkpoint_comparison.xlsx")
    df.to_excel(filepath)
    print(f"\nComparison saved: {filepath}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Checkpoint Comparison', fontsize=16)
    
    checkpoint_names = list(results.keys())
    
    # 1. 平均回报
    ax = axes[0, 0]
    mean_rewards = [results[c]['mean_reward'] for c in checkpoint_names]
    std_rewards = [results[c]['std_reward'] for c in checkpoint_names]
    ax.bar(range(len(checkpoint_names)), mean_rewards, yerr=std_rewards, capsize=5, color='blue', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('Reward')
    ax.set_title('Mean Reward')
    ax.grid(True, alpha=0.3)
    
    # 2. 平均产量
    ax = axes[0, 1]
    mean_yields = [results[c]['mean_yield'] for c in checkpoint_names]
    ax.bar(range(len(checkpoint_names)), mean_yields, color='green', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title('Mean Yield')
    ax.grid(True, alpha=0.3)
    
    # 3. WUE
    ax = axes[0, 2]
    wues = [results[c]['mean_wue'] for c in checkpoint_names]
    ax.bar(range(len(checkpoint_names)), wues, color='cyan', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('WUE (kg/mm)')
    ax.set_title('Mean WUE')
    ax.grid(True, alpha=0.3)
    
    # 4. NUE
    ax = axes[1, 0]
    nues = [results[c]['mean_nue'] for c in checkpoint_names]
    ax.bar(range(len(checkpoint_names)), nues, color='orange', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('NUE (kg/kg)')
    ax.set_title('Mean NUE')
    ax.grid(True, alpha=0.3)
    
    # 5. 成功率
    ax = axes[1, 1]
    success_rates = [results[c]['success_rate'] for c in checkpoint_names]
    ax.bar(range(len(checkpoint_names)), success_rates, color='purple', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.grid(True, alpha=0.3)
    
    # 6. 综合评分
    ax = axes[1, 2]
    # 计算综合评分 (归一化后加权)
    scores = []
    for c in checkpoint_names:
        r = results[c]
        # 归一化各指标 (简单归一化)
        norm_reward = (r['mean_reward'] - min(mean_rewards)) / (max(mean_rewards) - min(mean_rewards) + 1e-8)
        norm_yield = (r['mean_yield'] - min(mean_yields)) / (max(mean_yields) - min(mean_yields) + 1e-8)
        norm_wue = (r['mean_wue'] - min(wues)) / (max(wues) - min(wues) + 1e-8)
        norm_nue = (r['mean_nue'] - min(nues)) / (max(nues) - min(nues) + 1e-8)
        # 加权平均
        score = 0.3 * norm_reward + 0.3 * norm_yield + 0.2 * norm_wue + 0.2 * norm_nue
        scores.append(score)
    
    ax.bar(range(len(checkpoint_names)), scores, color='red', alpha=0.7)
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in checkpoint_names], fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Composite Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "checkpoint_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {filepath}")


# ============================================================================
#                              主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate PPO+LLaMA checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best',
                        help='Directory containing checkpoint files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint file (if None, validate all)')
    parser.add_argument('--model_path', type=str,
                        default='/home/wuyang/models/chinese-llama-2-1.3b',
                        help='Path to LLaMA model')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of validation episodes')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic policy')
    parser.add_argument('--stochastic', action='store_true',default=True,
                        help='Use stochastic policy instead')
    parser.add_argument('--output_dir', type=str,
                        default='/home/wuyang/test/val_proj/val_result/0417',
                        help='Output directory for results')
    parser.add_argument('--record_steps', action='store_true',
                        help='Record detailed step information')
    
    args = parser.parse_args()
    
    # 确定策略模式
    deterministic = not args.stochastic
    
    print("=" * 70)
    print("PPO + LLaMA Checkpoint Validation")
    print("=" * 70)
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"LLaMA Model: {args.model_path}")
    print(f"Validation Episodes: {args.n_episodes}")
    print(f"Policy Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化配置
    config = PPOConfig()
    config.n_val_episodes = args.n_episodes
    
    # 初始化LLaMA
    llama_model, tokenizer = initialize_llama(args.model_path, config.use_bf16)
    
    # 初始化环境
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/llama_ppo_rnd_results/logs/dssat-pdi-validation.log',
        'mode': 'all', 'seed': 123456, 'random_weather': True
    }
    
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    
    # 创建Agent
    agent = ValidationAgent(llama_model, tokenizer, config)
    
    # 查找checkpoints
    if args.checkpoint:
        # 指定单个checkpoint
        checkpoint_files = [args.checkpoint]
    else:
        # 查找所有checkpoint
        checkpoint_files = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoint_files:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint(s):")
    for f in checkpoint_files:
        print(f"  - {f}")
    
    # 验证所有checkpoint
    all_results = {}
    
    for checkpoint_path in checkpoint_files:
        # 提取checkpoint名称
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        
        # 创建该checkpoint的输出目录
        ckpt_output_dir = os.path.join(args.output_dir, checkpoint_name)
        os.makedirs(ckpt_output_dir, exist_ok=True)
        
        # 加载checkpoint
        agent.load_checkpoint(checkpoint_path)
        
        # 创建记录器
        logger = ValidationLogger(ckpt_output_dir, checkpoint_name)
        
        # 运行验证
        summary = validate_checkpoint(
            agent, env, config, logger,
            deterministic=deterministic,
            record_steps=args.record_steps
        )
        
        all_results[checkpoint_name] = summary
        
        # 打印汇总
        logger.print_summary()
        
        # 保存结果
        logger.save_all_results()
        
        # 清空缓存
        agent.clear_cache()
    
    # 关闭环境
    env.close()
    
    # 如果有多个checkpoint，进行比较
    if len(all_results) > 1:
        compare_checkpoints(all_results, args.output_dir)
    
    print("\n" + "=" * 70)
    print("Validation Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()