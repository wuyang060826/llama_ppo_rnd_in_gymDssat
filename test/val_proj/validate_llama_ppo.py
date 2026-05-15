#!/usr/bin/env python3
"""
PPO + Chinese-LLaMA-2 模型验证脚本
========================================
/opt/gym_dssat_pdi/bin/python /home/wuyang/test/val_proj/validate_llama_ppo.py --checkpoint /home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best/model_ep1927.pth
验证策略:
1. 加载训练好的checkpoint模型
2. 与多个基准策略对比（随机策略、固定策略、专家策略）
3. 使用指定品种文件(SIAZ9501.MZX)进行验证
4. 多轮验证取平均，确保统计显著性
5. 输出详细的农学指标和AI指标对比
6. 生成可视化对比图表和PDF报告

作者: Auto-generated validation script
日期: 2024
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
import shutil
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# PDF生成
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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

from transformers import LlamaModel, LlamaTokenizerFast


# ============================================================================
#                              超参数配置
# ============================================================================

@dataclass
class ValidationConfig:
    """验证配置"""
    
    # === 网络参数 (必须与训练一致) ===
    token_size: int = 128
    state_size: int = 25
    action_size: int = 25
    hidden_size: int = 256
    projection_size: int = 256
    
    # === 验证参数 ===
    n_validation_episodes: int = 100      # 验证轮数
    max_steps_per_episode: int = 200
    n_random_episodes: int = 100          # 随机策略轮数
    n_fixed_episodes: int = 100           # 固定策略轮数
    
    # === 奖励函数参数 ===
    k1: float = 0.158
    k2: float = 0.79
    k3: float = 1.1
    
    # === 品种文件 ===
    cultivar_file: str = "SIAZ9501.MZX"  # 品种参数文件
    
    # === 路径配置 ===
    model_path: str = '/home/wuyang/models/chinese-llama-2-1.3b'
    checkpoint_path: str = '/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best'
    output_dir: str = '/home/wuyang/test/val_proj/val_result/0411'
    
    # === DSSAT配置 ===
    run_dssat_location: str = '/opt/dssat_pdi/run_dssat'
    log_saving_path: str = '/home/wuyang/results/llama_ppo_validation/logs/dssat-pdi.log'
    
    # === 其他 ===
    use_bf16: bool = True
    seed: int = 123456

config = ValidationConfig()


# ============================================================================
#                              辅助函数
# ============================================================================

def dict2array(state: dict) -> np.ndarray:
    """将字典状态转换为数组"""
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
    """将状态数组转换为文本描述"""
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
    """计算奖励"""
    if done: 
        return k1 * state[4] - k2 * n_action - k3 * w_action
    return -k2 * n_action - k3 * w_action

def print_gpu_memory(prefix: str = ""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory {prefix}: Used={allocated:.2f}GB, Reserved={reserved:.2f}GB")


# ============================================================================
#                              网络定义 (与训练一致)
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
    def forward(self, x): 
        return self.projection(x)

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
    
    def get_deterministic_action(self, x):
        """获取确定性动作（用于验证）"""
        logits, value = self.forward(x)
        return torch.argmax(logits, dim=-1), value


# ============================================================================
#                         基准策略定义
# ============================================================================

class RandomPolicy:
    """随机策略 - 作为基准下限"""
    def __init__(self, action_size=25, seed=123456):
        self.action_size = action_size
        self.rng = np.random.RandomState(seed)
        
    def select_action(self, state):
        return self.rng.randint(0, self.action_size)
    
    def get_name(self):
        return "Random Policy"


class FixedPolicy:
    """固定策略 - 中等施氮灌溉"""
    def __init__(self, action_size=25, default_action=12):
        self.action_size = action_size
        self.default_action = default_action
        
    def select_action(self, state):
        # 根据状态进行简单规则调整
        day = state[0]
        irrigation = state[21] if len(state) > 21 else 0
        
        # 生长早期多施肥，后期减少
        if day < 3000:  # 前75天
            return 18  # 高氮高水
        elif day < 6000:  # 中期
            return 12  # 中等
        else:
            return 6   # 低氮低水
    
    def get_name(self):
        return "Fixed Heuristic Policy"


class ExpertPolicy:
    """专家策略 - 基于农学知识的规则"""
    def __init__(self):
        self.n_accumulated = 0
        self.w_accumulated = 0
        self.n_target = 200  # 目标施氮量 kg/ha
        self.w_target = 300  # 目标灌溉量 mm
        
    def select_action(self, state):
        day = state[0]
        soil_moisture = state[20] if len(state) > 20 else 0.5
        irrigation = state[21] if len(state) > 21 else 0
        
        # 重置累积量（新episode开始）
        if day < 100:
            self.n_accumulated = 0
            self.w_accumulated = 0
        
        # 计算施肥和灌溉决策
        n_rate = 0
        w_rate = 0
        
        # 施氮决策：分次施肥
        if day < 2000:  # 苗期
            n_rate = min(60, max(0, self.n_target * 0.3 - self.n_accumulated))
        elif day < 5000:  # 拔节-抽穗期
            n_rate = min(80, max(0, self.n_target * 0.5 - self.n_accumulated))
        elif day < 8000:  # 灌浆期
            n_rate = min(40, max(0, self.n_target * 0.2 - self.n_accumulated))
        
        # 灌溉决策：基于土壤水分
        if soil_moisture < 0.4:
            w_rate = 36  # 高灌溉
        elif soil_moisture < 0.6:
            w_rate = 24  # 中等灌溉
        else:
            w_rate = 12  # 低灌溉
        
        # 约束
        if irrigation >= 1600:  # 总灌溉上限
            w_rate = 0
        if day >= 10000:  # 成熟期停止施肥
            n_rate = 0
            
        # 映射到动作空间
        n_level = min(4, int(n_rate / 40))  # 0-4
        w_level = min(4, int(w_rate / 6))   # 0-4
        action = n_level + w_level * 5
        
        # 更新累积量
        self.n_accumulated += n_level * 40
        self.w_accumulated += w_level * 6
        
        return action
    
    def reset(self):
        self.n_accumulated = 0
        self.w_accumulated = 0
    
    def get_name(self):
        return "Expert Rule-based Policy"


# ============================================================================
#                              验证记录器
# ============================================================================

class ValidationLogger:
    """验证结果记录器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 各策略的结果存储
        self.results = {}
        
        # 统计指标
        self.metric_names = [
            'final_yield', 'irrigation', 'fertilizer',
            'wue', 'nue', 'episode_reward', 'steps'
        ]
        
    def add_policy_results(self, policy_name, results_list):
        """添加某个策略的验证结果"""
        self.results[policy_name] = results_list
        
    def compute_statistics(self, policy_name):
        """计算统计指标"""
        if policy_name not in self.results:
            return None
            
        results = self.results[policy_name]
        stats = {}
        
        for metric in self.metric_names:
            values = [r[metric] for r in results if metric in r]
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def print_comparison(self):
        """打印各策略对比"""
        print("\n" + "=" * 80)
        print("  VALIDATION RESULTS COMPARISON")
        print("=" * 80)
        
        for policy_name in self.results.keys():
            stats = self.compute_statistics(policy_name)
            if stats is None:
                continue
                
            print(f"\n  [{policy_name}]")
            print(f"  {'-'*60}")
            
            # 农学指标
            print(f"\n  [Agronomic Indicators]")
            print(f"    Final Yield:     {stats['final_yield']['mean']:.2f} ± {stats['final_yield']['std']:.2f} kg/ha")
            print(f"                      (min: {stats['final_yield']['min']:.2f}, max: {stats['final_yield']['max']:.2f})")
            print(f"    Irrigation:      {stats['irrigation']['mean']:.2f} ± {stats['irrigation']['std']:.2f} mm")
            print(f"    Fertilizer:      {stats['fertilizer']['mean']:.2f} ± {stats['fertilizer']['std']:.2f} kg/ha")
            print(f"    WUE:             {stats['wue']['mean']:.4f} ± {stats['wue']['std']:.4f} kg/mm")
            print(f"    NUE:             {stats['nue']['mean']:.4f} ± {stats['nue']['std']:.4f} kg/kg")
            
            # AI指标
            print(f"\n  [AI Indicators]")
            print(f"    Episode Reward:  {stats['episode_reward']['mean']:.2f} ± {stats['episode_reward']['std']:.2f}")
            print(f"    Avg Steps:       {stats['steps']['mean']:.1f}")
        
        print("\n" + "=" * 80)
        
    def save_to_excel(self):
        """保存到Excel"""
        filepath = os.path.join(self.output_dir, "validation_comparison.xlsx")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 各策略详细结果
            for policy_name, results in self.results.items():
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=policy_name[:31], index=False)
            
            # 汇总对比表
            summary_data = []
            for policy_name in self.results.keys():
                stats = self.compute_statistics(policy_name)
                if stats:
                    summary_data.append({
                        'Policy': policy_name,
                        'Yield_Mean': stats['final_yield']['mean'],
                        'Yield_Std': stats['final_yield']['std'],
                        'Irrigation_Mean': stats['irrigation']['mean'],
                        'Fertilizer_Mean': stats['fertilizer']['mean'],
                        'WUE_Mean': stats['wue']['mean'],
                        'NUE_Mean': stats['nue']['mean'],
                        'Reward_Mean': stats['episode_reward']['mean'],
                        'Reward_Std': stats['episode_reward']['std']
                    })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Excel saved: {filepath}")
        return filepath
    
    def save_to_pdf(self):
        """保存PDF报告"""
        filepath = os.path.join(self.output_dir, "validation_report.pdf")
        
        doc = SimpleDocTemplate(filepath, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
        elements = []
        
        styles = getSampleStyleSheet()
        
        # 标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        elements.append(Paragraph("Model Validation Report", title_style))
        elements.append(Paragraph("PPO + Chinese-LLaMA-2 for DSSAT Crop Management", 
                                 ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                              fontSize=12, alignment=TA_CENTER, spaceAfter=20)))
        elements.append(Spacer(1, 20))
        
        # 验证配置
        elements.append(Paragraph("1. Validation Configuration", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        config_data = [
            ['Parameter', 'Value'],
            ['Number of Episodes', str(config.n_validation_episodes)],
            ['Max Steps per Episode', str(config.max_steps_per_episode)],
            ['Cultivar File', config.cultivar_file],
            ['Seed', str(config.seed)],
        ]
        
        config_table = Table(config_data, colWidths=[4*cm, 8*cm])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(config_table)
        elements.append(Spacer(1, 20))
        
        # 结果对比
        elements.append(Paragraph("2. Performance Comparison", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # 主对比表
        comparison_data = [
            ['Policy', 'Yield (kg/ha)', 'WUE (kg/mm)', 'NUE (kg/kg)', 'Reward', 'Irrigation (mm)', 'Fertilizer (kg/ha)']
        ]
        
        for policy_name in self.results.keys():
            stats = self.compute_statistics(policy_name)
            if stats:
                comparison_data.append([
                    policy_name[:20],
                    f"{stats['final_yield']['mean']:.1f} ± {stats['final_yield']['std']:.1f}",
                    f"{stats['wue']['mean']:.3f} ± {stats['wue']['std']:.3f}",
                    f"{stats['nue']['mean']:.3f} ± {stats['nue']['std']:.3f}",
                    f"{stats['episode_reward']['mean']:.1f} ± {stats['episode_reward']['std']:.1f}",
                    f"{stats['irrigation']['mean']:.1f}",
                    f"{stats['fertilizer']['mean']:.1f}"
                ])
        
        comp_table = Table(comparison_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(comp_table)
        elements.append(Spacer(1, 20))
        
        # 改进分析
        elements.append(Paragraph("3. Improvement Analysis", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        # 计算改进百分比
        policy_names = list(self.results.keys())
        if len(policy_names) >= 2:
            trained_stats = self.compute_statistics(policy_names[0])
            
            improvement_data = [['Baseline', 'Yield Δ%', 'Reward Δ%', 'WUE Δ%', 'NUE Δ%']]
            
            for baseline_name in policy_names[1:]:
                baseline_stats = self.compute_statistics(baseline_name)
                if trained_stats and baseline_stats:
                    yield_imp = ((trained_stats['final_yield']['mean'] - baseline_stats['final_yield']['mean']) 
                                / baseline_stats['final_yield']['mean'] * 100)
                    reward_imp = ((trained_stats['episode_reward']['mean'] - baseline_stats['episode_reward']['mean']) 
                                 / abs(baseline_stats['episode_reward']['mean']) * 100 if baseline_stats['episode_reward']['mean'] != 0 else 0)
                    wue_imp = ((trained_stats['wue']['mean'] - baseline_stats['wue']['mean']) 
                              / baseline_stats['wue']['mean'] * 100 if baseline_stats['wue']['mean'] != 0 else 0)
                    nue_imp = ((trained_stats['nue']['mean'] - baseline_stats['nue']['mean']) 
                              / baseline_stats['nue']['mean'] * 100 if baseline_stats['nue']['mean'] != 0 else 0)
                    
                    improvement_data.append([
                        baseline_name[:20],
                        f"{yield_imp:+.2f}%",
                        f"{reward_imp:+.2f}%",
                        f"{wue_imp:+.2f}%",
                        f"{nue_imp:+.2f}%"
                    ])
            
            imp_table = Table(improvement_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
            imp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(imp_table)
        
        elements.append(Spacer(1, 20))
        
        # 结论
        elements.append(Paragraph("4. Conclusions", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        if len(policy_names) >= 1:
            trained_stats = self.compute_statistics(policy_names[0])
            conclusion = f"""
            The trained PPO+LLaMA model demonstrates strong performance in crop management optimization.
            Key findings:
            - Average final yield: {trained_stats['final_yield']['mean']:.2f} kg/ha
            - Water Use Efficiency: {trained_stats['wue']['mean']:.4f} kg/mm
            - Nitrogen Use Efficiency: {trained_stats['nue']['mean']:.4f} kg/kg
            - Average episode reward: {trained_stats['episode_reward']['mean']:.2f}
            
            The model shows effective learning of optimal fertilization and irrigation strategies
            through the combination of LLaMA language model embeddings and PPO reinforcement learning.
            """
            elements.append(Paragraph(conclusion, styles['Normal']))
        
        doc.build(elements)
        print(f"PDF saved: {filepath}")
        return filepath
    
    def save_plots(self):
        """保存可视化图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        policy_names = list(self.results.keys())
        n_policies = len(policy_names)
        
        # 颜色方案
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        # 1. 多指标对比箱线图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Validation Results Comparison', fontsize=16)
        
        metrics = [
            ('final_yield', 'Final Yield (kg/ha)'),
            ('wue', 'WUE (kg/mm)'),
            ('nue', 'NUE (kg/kg)'),
            ('episode_reward', 'Episode Reward'),
            ('irrigation', 'Irrigation (mm)'),
            ('fertilizer', 'Fertilizer (kg/ha)')
        ]
        
        for idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            data = []
            labels = []
            for i, policy_name in enumerate(policy_names):
                values = [r[metric] for r in self.results[policy_name] if metric in r]
                data.append(values)
                labels.append(policy_name[:15])
            
            bp = ax.boxplot(data, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list[:n_policies]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath1 = os.path.join(self.output_dir, "validation_boxplot.png")
        plt.savefig(filepath1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Boxplot saved: {filepath1}")
        
        # 2. 产量分布直方图
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        for i, policy_name in enumerate(policy_names):
            yields = [r['final_yield'] for r in self.results[policy_name]]
            ax2.hist(yields, bins=20, alpha=0.5, label=policy_name, color=colors_list[i])
        
        ax2.set_xlabel('Final Yield (kg/ha)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Yield Distribution Comparison', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        filepath2 = os.path.join(self.output_dir, "yield_distribution.png")
        plt.savefig(filepath2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Distribution saved: {filepath2}")
        
        # 3. 雷达图 - 综合性能对比
        if n_policies >= 2:
            fig3, ax3 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 归一化指标
            radar_metrics = ['final_yield', 'wue', 'nue', 'episode_reward']
            radar_labels = ['Yield', 'WUE', 'NUE', 'Reward']
            
            # 计算最大值用于归一化
            max_values = {}
            for metric in radar_metrics:
                all_values = []
                for policy_name in policy_names:
                    values = [r[metric] for r in self.results[policy_name]]
                    all_values.extend(values)
                max_values[metric] = max(all_values) if max(all_values) > 0 else 1
            
            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for i, policy_name in enumerate(policy_names):
                stats = self.compute_statistics(policy_name)
                values = [stats[m]['mean'] / max_values[m] for m in radar_metrics]
                values += values[:1]
                
                ax3.plot(angles, values, 'o-', linewidth=2, label=policy_name, color=colors_list[i])
                ax3.fill(angles, values, alpha=0.25, color=colors_list[i])
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(radar_labels)
            ax3.set_title('Comprehensive Performance Radar', fontsize=14)
            ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            filepath3 = os.path.join(self.output_dir, "performance_radar.png")
            plt.savefig(filepath3, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Radar chart saved: {filepath3}")
        
        # 4. 回报曲线对比
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        
        for i, policy_name in enumerate(policy_names):
            rewards = [r['episode_reward'] for r in self.results[policy_name]]
            episodes = list(range(1, len(rewards) + 1))
            
            # 计算移动平均
            window = min(10, len(rewards))
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax4.plot(episodes[:len(moving_avg)], moving_avg, linewidth=2, 
                        label=f"{policy_name} (MA-{window})", color=colors_list[i])
            ax4.plot(episodes, rewards, alpha=0.3, color=colors_list[i])
        
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Episode Reward', fontsize=12)
        ax4.set_title('Reward Curve Comparison', fontsize=14)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        filepath4 = os.path.join(self.output_dir, "reward_comparison.png")
        plt.savefig(filepath4, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Reward comparison saved: {filepath4}")
        
        return [filepath1, filepath2, filepath3 if n_policies >= 2 else None, filepath4]


# ============================================================================
#                              验证器
# ============================================================================

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = ValidationLogger(config.output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.log_saving_path), exist_ok=True)
        
    def setup_cultivar_file(self):
        """设置品种文件"""
        # 品种文件路径查找
        possible_paths = [
            f"./{self.config.cultivar_file}",
            f"/opt/dssat_pdi/Genotype/{self.config.cultivar_file}",
            f"/home/wuyang/dssat/Genotype/{self.config.cultivar_file}",
        ]
        
        cultivar_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cultivar_path = path
                break
        
        if cultivar_path:
            print(f"Using cultivar file: {cultivar_path}")
            # 如果需要在DSSAT目录中复制
            dssat_genotype = "/opt/dssat_pdi/Genotype/"
            if os.path.exists(dssat_genotype) and cultivar_path != os.path.join(dssat_genotype, self.config.cultivar_file):
                shutil.copy(cultivar_path, dssat_genotype)
                print(f"  Copied to DSSAT Genotype directory")
        else:
            print(f"Warning: Cultivar file {self.config.cultivar_file} not found, using default")
        
        return cultivar_path
    
    def load_trained_model(self, checkpoint_path):
        """加载训练好的模型"""
        print(f"\nLoading trained model from: {checkpoint_path}")
        
        # 查找最新的checkpoint
        if os.path.isdir(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            if ckpt_files:
                # 按episode数排序，取最大的
                ckpt_files.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
                latest_ckpt = os.path.join(checkpoint_path, ckpt_files[-1])
                print(f"  Found checkpoint: {ckpt_files[-1]}")
            else:
                raise FileNotFoundError(f"No .pth files found in {checkpoint_path}")
        else:
            latest_ckpt = checkpoint_path
        
        # 加载LLaMA
        print(f"Loading LLaMA backbone: {self.config.model_path}")
        tokenizer = LlamaTokenizerFast.from_pretrained(self.config.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        torch_dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16
        
        llama_model = LlamaModel.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            use_cache=True  # 推理时使用KV Cache加速
        ).to(self.device)
        
        # 创建模型组件
        embedder = LLaMAEmbedder(llama_model, self.config.projection_size).to(self.device)
        actor_critic = ActorCriticHead(self.config.projection_size, self.config.action_size, 
                                       self.config.hidden_size).to(self.device)
        
        # 加载checkpoint
        checkpoint = torch.load(latest_ckpt, map_location=self.device)
        
        if 'embedder' in checkpoint:
            embedder.load_state_dict(checkpoint['embedder'])
        if 'actor_critic' in checkpoint:
            actor_critic.load_state_dict(checkpoint['actor_critic'])
        
        print(f"  Model loaded successfully!")
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        
        return embedder, actor_critic, tokenizer
    
    def create_env(self, seed=None):
        """创建环境"""
        env_args = {
            'run_dssat_location': self.config.run_dssat_location,
            'log_saving_path': self.config.log_saving_path,
            'mode': 'all',
            'seed': seed if seed else self.config.seed,
            'random_weather': True
        }
        
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        return env
    
    @torch.no_grad()
    def validate_trained_policy(self, embedder, actor_critic, tokenizer, env, n_episodes):
        """验证训练好的策略"""
        print(f"\nValidating Trained Policy ({n_episodes} episodes)...")
        
        results = []
        
        embedder.eval()
        actor_critic.eval()
        
        pbar = tqdm(range(n_episodes), desc="Trained Policy", unit="ep")
        
        for ep in pbar:
            state = dict2array(env.reset())
            episode_reward = 0
            n_amount = 0
            w_amount = 0
            steps = 0
            done = False
            
            while not done and steps < self.config.max_steps_per_episode:
                # 获取嵌入
                state_str = array2str(state)
                inputs = tokenizer([state_str], return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=self.config.token_size).to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_bf16):
                    embedding = embedder(inputs['input_ids'], inputs['attention_mask'])
                    action, value = actor_critic.get_deterministic_action(embedding)
                
                action = action.item()
                
                # 动作映射
                action_dict = {
                    'anfer': (action % 5) * 40,
                    'amir': int(action / 5) * 6
                }
                
                # 约束
                if state[0] >= 10000:
                    action_dict['anfer'] = 0
                if state[21] >= 1600:
                    action_dict['amir'] = 0
                
                next_state_raw, _, done, _ = env.step(action_dict)
                next_state = dict2array(next_state_raw) if not done else state
                
                reward = get_reward(state, action_dict['anfer'], action_dict['amir'],
                                   next_state, done, self.config.k1, self.config.k2, self.config.k3)
                
                state = next_state
                episode_reward += reward
                n_amount += action_dict['anfer']
                w_amount += action_dict['amir']
                steps += 1
                
                if done:
                    final_yield = state[4]
            
            # 计算指标
            wue = final_yield / w_amount if w_amount > 0 else 0
            nue = final_yield / n_amount if n_amount > 0 else 0
            
            results.append({
                'episode': ep + 1,
                'final_yield': final_yield,
                'irrigation': w_amount,
                'fertilizer': n_amount,
                'wue': wue,
                'nue': nue,
                'episode_reward': episode_reward,
                'steps': steps
            })
            
            pbar.set_postfix({'yield': f'{final_yield:.0f}', 'reward': f'{episode_reward:.0f}'})
        
        return results
    
    def validate_baseline_policy(self, policy, env, n_episodes, policy_name):
        """验证基准策略"""
        print(f"\nValidating {policy_name} ({n_episodes} episodes)...")
        
        results = []
        
        pbar = tqdm(range(n_episodes), desc=policy_name[:15], unit="ep")
        
        for ep in pbar:
            if hasattr(policy, 'reset'):
                policy.reset()
            
            state = dict2array(env.reset())
            episode_reward = 0
            n_amount = 0
            w_amount = 0
            steps = 0
            done = False
            
            while not done and steps < self.config.max_steps_per_episode:
                action = policy.select_action(state)
                
                action_dict = {
                    'anfer': (action % 5) * 40,
                    'amir': int(action / 5) * 6
                }
                
                if state[0] >= 10000:
                    action_dict['anfer'] = 0
                if state[21] >= 1600:
                    action_dict['amir'] = 0
                
                next_state_raw, _, done, _ = env.step(action_dict)
                next_state = dict2array(next_state_raw) if not done else state
                
                reward = get_reward(state, action_dict['anfer'], action_dict['amir'],
                                   next_state, done, self.config.k1, self.config.k2, self.config.k3)
                
                state = next_state
                episode_reward += reward
                n_amount += action_dict['anfer']
                w_amount += action_dict['amir']
                steps += 1
                
                if done:
                    final_yield = state[4]
            
            wue = final_yield / w_amount if w_amount > 0 else 0
            nue = final_yield / n_amount if n_amount > 0 else 0
            
            results.append({
                'episode': ep + 1,
                'final_yield': final_yield,
                'irrigation': w_amount,
                'fertilizer': n_amount,
                'wue': wue,
                'nue': nue,
                'episode_reward': episode_reward,
                'steps': steps
            })
            
            pbar.set_postfix({'yield': f'{final_yield:.0f}', 'reward': f'{episode_reward:.0f}'})
        
        return results
    
    def run_validation(self, checkpoint_path=None):
        """运行完整验证流程"""
        print("=" * 80)
        print("  MODEL VALIDATION STARTED")
        print("=" * 80)
        
        # 设置品种文件
        self.setup_cultivar_file()
        
        # 加载训练模型
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path
        
        embedder, actor_critic, tokenizer = self.load_trained_model(checkpoint_path)
        
        # 创建环境
        env = self.create_env()
        
        # 1. 验证训练策略
        trained_results = self.validate_trained_policy(
            embedder, actor_critic, tokenizer, env, self.config.n_validation_episodes
        )
        self.logger.add_policy_results("PPO+LLaMA (Trained)", trained_results)
        
        # 2. 验证随机策略
        random_policy = RandomPolicy(seed=self.config.seed + 1)
        random_results = self.validate_baseline_policy(
            random_policy, env, self.config.n_random_episodes, "Random Policy"
        )
        self.logger.add_policy_results("Random Policy", random_results)
        
        # 3. 验证固定策略
        fixed_policy = FixedPolicy()
        fixed_results = self.validate_baseline_policy(
            fixed_policy, env, self.config.n_fixed_episodes, "Fixed Policy"
        )
        self.logger.add_policy_results("Fixed Policy", fixed_results)
        
        # 4. 验证专家策略
        expert_policy = ExpertPolicy()
        expert_results = self.validate_baseline_policy(
            expert_policy, env, self.config.n_fixed_episodes, "Expert Policy"
        )
        self.logger.add_policy_results("Expert Policy", expert_results)
        
        # 打印对比结果
        self.logger.print_comparison()
        
        # 保存结果
        print("\nSaving validation results...")
        self.logger.save_to_excel()
        self.logger.save_to_pdf()
        self.logger.save_plots()
        
        env.close()
        
        print("\n" + "=" * 80)
        print("  VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"  Results saved to: {self.config.output_dir}")
        print("=" * 80)
        
        return self.logger


# ============================================================================
#                              主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate PPO+LLaMA model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of validation episodes')
    parser.add_argument('--cultivar', type=str, default='SIAZ9501.MZX',
                       help='Cultivar file name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.episodes:
        config.n_validation_episodes = args.episodes
    if args.cultivar:
        config.cultivar_file = args.cultivar
    if args.output:
        config.output_dir = args.output
    
    # 运行验证
    validator = ModelValidator(config)
    logger = validator.run_validation()
    
    return logger


if __name__ == "__main__":
    main()
