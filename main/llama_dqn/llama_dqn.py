#!/usr/bin/env python3
"""
llama-dqn-gym-dssat-gpu-baseline (Enhanced Version):
1. DQN+Chinese-LLaMA-2+gym-dssat GPU深度优化版本 (A100 40G适配)
2. llama-1.3b参数全参可微调
3. 新增农学指标和AI指标追踪与输出
"""

import numpy as np
import pandas as pd
import random
from collections import namedtuple, deque
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import os
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 显存碎片优化
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 导入transformers
from transformers import LlamaModel, LlamaTokenizerFast

# -------------------------- 参数配置 --------------------------
# 针对A100 40G优化：减小Batch Size以容纳LLaMA梯度
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256            # 保守设置，保证LLaMA前向传播不OOM
GAMMA = 0.99
TAU = 8
LR = 5e-5                 # 配合更小的Batch Size调整学习率
UPDATE_EVERY = 16          # 提高更新频率
N_EPISODES = 2000
TOKEN_SIZE = 100          # 状态描述不需要512那么长，128足够
STATE_SIZE = 25           # 原始状态维度
ACTION_SIZE = 25

# 优化器参数
betas = (0.9, 0.999)
weight_decay = 0.01

# 输出路径前缀
OUTPUT_PREFIX = "llama_dqn"
OUTPUT_DIR = '/home/wuyang/results/llama_dqn_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标性能阈值（用于样本效率计算）
TARGET_PERFORMANCE_SCORE = 1000  # 专家水平阈值
CONVERGENCE_THRESHOLD = 50       # 收敛判断阈值
CONVERGENCE_WINDOW = 50          # 收敛判断窗口大小

# -------------------------- 辅助函数 --------------------------
def dict2array(state):
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state)

def array2str(state):
    state_str = ""
    for i, num in enumerate(state):
        # 简化描述，节省Token
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
    return state_str.strip()

def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3, k4):
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action
    else:
        return -k2 * n_action - k3 * w_action

# -------------------------- 农学指标计算函数 --------------------------
def calculate_wue(final_yield, total_irrigation):
    """
    水分利用率(WUE): 单位灌溉水量产生的产量
    单位: kg/mm
    计算方式: 最终产量除以累计灌溉量
    """
    if total_irrigation <= 0:
        return 0.0
    return final_yield / total_irrigation

def calculate_nue(final_yield, total_fertilizer):
    """
    氮肥利用率(NUE): 单位施氮量产生的产量
    单位: kg/kg
    计算方式: 最终产量除以累计施氮量
    """
    if total_fertilizer <= 0:
        return 0.0
    return final_yield / total_fertilizer

# -------------------------- RND网络定义（探索覆盖率） --------------------------
class RNDNetwork(nn.Module):
    """随机网络蒸馏(RND)网络，用于计算探索覆盖率"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(RNDNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RNDModule:
    """RND模块，用于追踪探索覆盖率"""
    def __init__(self, state_size, hidden_dim=256, output_dim=128, lr=1e-4):
        self.target_network = RNDNetwork(state_size, hidden_dim, output_dim).to(device)
        self.predictor_network = RNDNetwork(state_size, hidden_dim, output_dim).to(device)
        
        # 冻结目标网络
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)
        
        # 探索追踪
        self.visited_states = set()
        self.novel_state_threshold = 0.5  # 新颖状态阈值
        self.total_state_space_estimate = 10000  # 状态空间估计大小
        
    def get_prediction_error(self, state):
        """计算预测误差"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predictor_output = self.predictor_network(state_tensor)
        error = F.mse_loss(predictor_output, target_output, reduction='none').mean().item()
        
        return error
    
    def update(self, state):
        """更新预测网络并返回预测误差"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predictor_output = self.predictor_network(state_tensor)
        error = F.mse_loss(predictor_output, target_output)
        
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()
        
        return error.item()
    
    def record_exploration(self, state, prediction_error):
        """记录探索状态"""
        state_key = tuple(np.round(state, 2))
        
        if prediction_error > self.novel_state_threshold:
            self.visited_states.add(state_key)
            return True
        return False
    
    def get_exploration_coverage(self):
        """计算探索覆盖率"""
        if self.total_state_space_estimate == 0:
            return 0.0
        coverage = len(self.visited_states) / self.total_state_space_estimate * 100
        return min(coverage, 100.0)  # 限制最大100%

# -------------------------- AI指标追踪类 --------------------------
class AIMetricsTracker:
    """AI指标追踪器"""
    def __init__(self, target_performance=TARGET_PERFORMANCE_SCORE, 
                 convergence_threshold=CONVERGENCE_THRESHOLD,
                 convergence_window=CONVERGENCE_WINDOW):
        self.target_performance = target_performance
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        
        # 追踪变量
        self.episode_scores = []
        self.total_steps = 0
        self.sample_efficiency_achieved = False
        self.sample_efficiency_step = None
        self.convergence_achieved = False
        self.convergence_step = None
        
    def update(self, episode_score, steps_in_episode):
        """更新指标追踪"""
        self.episode_scores.append(episode_score)
        self.total_steps += steps_in_episode
        
        # 检查样本效率（首次达到目标性能）
        if not self.sample_efficiency_achieved and episode_score >= self.target_performance:
            self.sample_efficiency_achieved = True
            self.sample_efficiency_step = self.total_steps
        
        # 检查收敛速度
        if not self.convergence_achieved and len(self.episode_scores) >= self.convergence_window:
            recent_scores = self.episode_scores[-self.convergence_window:]
            score_std = np.std(recent_scores)
            score_mean = np.mean(recent_scores)
            
            # 当最近窗口内的分数标准差小于阈值且均值较高时认为收敛
            if score_std < self.convergence_threshold and score_mean > self.target_performance * 0.8:
                self.convergence_achieved = True
                self.convergence_step = self.total_steps
    
    def get_sample_efficiency(self):
        """获取样本效率"""
        return self.sample_efficiency_step if self.sample_efficiency_achieved else -1
    
    def get_convergence_speed(self):
        """获取收敛速度"""
        return self.convergence_step if self.convergence_achieved else -1
    
    def get_average_return(self, last_n=None):
        """获取平均回报"""
        if len(self.episode_scores) == 0:
            return 0.0
        if last_n is None:
            return np.mean(self.episode_scores)
        return np.mean(self.episode_scores[-last_n:])

# -------------------------- 指标输出函数 --------------------------
def print_metrics(epoch, scores, n_amount_list, w_amount_list, yield_list, 
                  ai_tracker, rnd_module, prefix="", is_final=False):
    """打印农学指标和AI指标"""
    if len(scores) == 0:
        return
    
    # 计算农学指标
    final_yield = yield_list[-1] if yield_list else 0
    total_irrigation = w_amount_list[-1] if w_amount_list else 0
    total_fertilizer = n_amount_list[-1] if n_amount_list else 0
    
    wue = calculate_wue(final_yield, total_irrigation)
    nue = calculate_nue(final_yield, total_fertilizer)
    
    # 计算AI指标
    avg_return = ai_tracker.get_average_return()
    sample_efficiency = ai_tracker.get_sample_efficiency()
    convergence_speed = ai_tracker.get_convergence_speed()
    exploration_coverage = rnd_module.get_exploration_coverage()
    
    if is_final:
        # 最终输出 - 使用全部数据计算平均值
        final_avg_yield = np.mean(yield_list) if yield_list else 0
        final_avg_irrigation = np.mean(w_amount_list) if w_amount_list else 0
        final_avg_fertilizer = np.mean(n_amount_list) if n_amount_list else 0
        final_avg_score = np.mean(scores) if scores else 0
        
        final_wue = calculate_wue(final_avg_yield, final_avg_irrigation)
        final_nue = calculate_nue(final_avg_yield, final_avg_fertilizer)
        
        print("\n" + "="*70)
        print("                    FINAL TRAINING METRICS SUMMARY")
        print("="*70)
        print("\n[ Agronomic Metrics ]")
        print("-"*50)
        print(f"  Final Yield (avg):       {final_avg_yield:>12.2f} kg/ha")
        print(f"  Irrigation Amount (avg): {final_avg_irrigation:>12.2f} mm")
        print(f"  Fertilizer Amount (avg): {final_avg_fertilizer:>12.2f} kg/ha")
        print(f"  WUE (Water Use Eff.):    {final_wue:>12.2f} kg/mm")
        print(f"  NUE (Nitrogen Use Eff.): {final_nue:>12.2f} kg/kg")
        
        print("\n[ AI Metrics ]")
        print("-"*50)
        print(f"  Average Return:          {final_avg_score:>12.2f}")
        print(f"  Sample Efficiency:       {sample_efficiency:>12d} steps")
        print(f"  Convergence Speed:       {convergence_speed:>12d} steps")
        print(f"  Exploration Coverage:    {exploration_coverage:>11.2f} %")
        print("="*70 + "\n")
    else:
        # 每10轮输出
        print(f"\n{prefix} Epoch {epoch} Metrics:")
        print("-"*50)
        print("  [Agronomic Metrics]")
        print(f"    Final Yield:        {final_yield:>10.2f} kg/ha")
        print(f"    Irrigation Amount:  {total_irrigation:>10.2f} mm")
        print(f"    Fertilizer Amount:  {total_fertilizer:>10.2f} kg/ha")
        print(f"    WUE:                {wue:>10.2f} kg/mm")
        print(f"    NUE:                {nue:>10.2f} kg/kg")
        print("  [AI Metrics]")
        print(f"    Average Return:     {avg_return:>10.2f}")
        print(f"    Sample Efficiency:  {sample_efficiency:>10d} steps" if sample_efficiency > 0 else "    Sample Efficiency:  Not achieved")
        print(f"    Convergence Speed:  {convergence_speed:>10d} steps" if convergence_speed > 0 else "    Convergence Speed:  Not converged")
        print(f"    Exploration:        {exploration_coverage:>9.2f} %")
        print("-"*50)

# -------------------------- 结果保存函数 --------------------------
def save_results(scores, n_amount_list, w_amount_list, yield_list, 
                 ai_tracker, rnd_module, output_dir, prefix):
    """保存结果到多种格式"""
    
    # 1. 计算所有指标
    n_episodes = len(scores)
    
    # 农学指标计算
    wue_list = [calculate_wue(y, w) if w > 0 else 0 for y, w in zip(yield_list, w_amount_list)]
    nue_list = [calculate_nue(y, n) if n > 0 else 0 for y, n in zip(yield_list, n_amount_list)]
    
    # AI指标
    sample_efficiency = ai_tracker.get_sample_efficiency()
    convergence_speed = ai_tracker.get_convergence_speed()
    exploration_coverage = rnd_module.get_exploration_coverage()
    
    # 滑动平均回报
    window_size = 10
    moving_avg_scores = [np.mean(scores[max(0, i-window_size+1):i+1]) for i in range(len(scores))]
    
    # 2. 保存Excel文件
    df_episodes = pd.DataFrame({
        'Episode': list(range(1, n_episodes + 1)),
        'Score': scores,
        'Moving_Avg_Score': moving_avg_scores,
        'Final_Yield_kg_ha': yield_list,
        'Irrigation_mm': w_amount_list,
        'Fertilizer_kg_ha': n_amount_list,
        'WUE_kg_mm': wue_list,
        'NUE_kg_kg': nue_list
    })
    
    # 汇总指标
    df_summary = pd.DataFrame({
        'Metric': [
            'Total_Episodes', 'Average_Score', 'Max_Score', 'Min_Score',
            'Average_Yield_kg_ha', 'Average_Irrigation_mm', 'Average_Fertilizer_kg_ha',
            'Average_WUE_kg_mm', 'Average_NUE_kg_kg',
            'Sample_Efficiency_steps', 'Convergence_Speed_steps', 'Exploration_Coverage_percent'
        ],
        'Value': [
            n_episodes, np.mean(scores), np.max(scores), np.min(scores),
            np.mean(yield_list), np.mean(w_amount_list), np.mean(n_amount_list),
            np.mean(wue_list), np.mean(nue_list),
            sample_efficiency, convergence_speed, exploration_coverage
        ]
    })
    
    excel_path = os.path.join(output_dir, f'{prefix}_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_episodes.to_excel(writer, sheet_name='Episode_Details', index=False)
        df_summary.to_excel(writer, sheet_name='Summary_Metrics', index=False)
    print(f"Excel results saved to: {excel_path}")
    
    # 3. 保存PDF图表
    pdf_path = os.path.join(output_dir, f'{prefix}_plots.pdf')
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 训练得分
    axes[0, 0].plot(scores, 'b-', alpha=0.6, label='Score')
    axes[0, 0].plot(moving_avg_scores, 'r-', linewidth=2, label='Moving Avg (10)')
    axes[0, 0].set_title('Training Score', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 最终产量
    axes[0, 1].plot(yield_list, 'g-')
    axes[0, 1].set_title('Final Yield', fontsize=12)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Yield (kg/ha)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 氮肥使用量
    axes[0, 2].plot(n_amount_list, 'm-')
    axes[0, 2].set_title('Fertilizer Amount', fontsize=12)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Fertilizer (kg/ha)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 灌溉量
    axes[1, 0].plot(w_amount_list, 'c-')
    axes[1, 0].set_title('Irrigation Amount', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Irrigation (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # WUE
    axes[1, 1].plot(wue_list, 'orange')
    axes[1, 1].set_title('Water Use Efficiency (WUE)', fontsize=12)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('WUE (kg/mm)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # NUE
    axes[1, 2].plot(nue_list, 'purple')
    axes[1, 2].set_title('Nitrogen Use Efficiency (NUE)', fontsize=12)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('NUE (kg/kg)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 累计平均回报
    cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores) + 1)
    axes[2, 0].plot(cumulative_avg, 'b-')
    axes[2, 0].axhline(y=TARGET_PERFORMANCE_SCORE, color='r', linestyle='--', label='Target')
    axes[2, 0].set_title('Cumulative Average Return', fontsize=12)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Average Return')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 探索覆盖率（模拟累积）
    exploration_progress = []
    visited_set = set()
    for i, s in enumerate(zip(n_amount_list, w_amount_list)):
        state_key = (round(s[0], 0), round(s[1], 0))
        visited_set.add(state_key)
        exploration_progress.append(len(visited_set) / 100 * 100)  # 估计状态空间
    axes[2, 1].plot(exploration_progress, 'g-')
    axes[2, 1].set_title('Exploration Coverage', fontsize=12)
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Coverage (%)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 性能指标对比
    metrics_names = ['Avg Score', 'Avg Yield', 'Avg WUE', 'Avg NUE']
    metrics_values = [np.mean(scores)/100, np.mean(yield_list)/1000, 
                      np.mean(wue_list), np.mean(nue_list)]
    axes[2, 2].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'purple'])
    axes[2, 2].set_title('Key Metrics Comparison', fontsize=12)
    axes[2, 2].set_ylabel('Normalized Value')
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf', dpi=150)
    plt.close()
    print(f"PDF plots saved to: {pdf_path}")
    
    # 4. 保存PNG图表
    png_path = os.path.join(output_dir, f'{prefix}_plots.png')
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # 训练得分
    axes[0, 0].plot(scores, 'b-', alpha=0.6, label='Score')
    axes[0, 0].plot(moving_avg_scores, 'r-', linewidth=2, label='Moving Avg (10)')
    axes[0, 0].set_title('Training Score', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 最终产量
    axes[0, 1].plot(yield_list, 'g-')
    axes[0, 1].set_title('Final Yield', fontsize=12)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Yield (kg/ha)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 氮肥使用量
    axes[0, 2].plot(n_amount_list, 'm-')
    axes[0, 2].set_title('Fertilizer Amount', fontsize=12)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Fertilizer (kg/ha)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 灌溉量
    axes[1, 0].plot(w_amount_list, 'c-')
    axes[1, 0].set_title('Irrigation Amount', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Irrigation (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # WUE
    axes[1, 1].plot(wue_list, 'orange')
    axes[1, 1].set_title('Water Use Efficiency (WUE)', fontsize=12)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('WUE (kg/mm)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # NUE
    axes[1, 2].plot(nue_list, 'purple')
    axes[1, 2].set_title('Nitrogen Use Efficiency (NUE)', fontsize=12)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('NUE (kg/kg)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 累计平均回报
    axes[2, 0].plot(cumulative_avg, 'b-')
    axes[2, 0].axhline(y=TARGET_PERFORMANCE_SCORE, color='r', linestyle='--', label='Target')
    axes[2, 0].set_title('Cumulative Average Return', fontsize=12)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Average Return')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 探索覆盖率
    axes[2, 1].plot(exploration_progress, 'g-')
    axes[2, 1].set_title('Exploration Coverage', fontsize=12)
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Coverage (%)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 性能指标对比
    axes[2, 2].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'purple'])
    axes[2, 2].set_title('Key Metrics Comparison', fontsize=12)
    axes[2, 2].set_ylabel('Normalized Value')
    axes[2, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(png_path, format='png', dpi=150)
    plt.close()
    print(f"PNG plots saved to: {png_path}")
    
    return excel_path, pdf_path, png_path

# -------------------------- 模型初始化 --------------------------
print("初始化 Chinese-LLaMA-2-1.3B 模型 (FP16 + Gradient Checkpointing)...")
model_path = '/home/gymusr/gym-dssat-rl-project-baseline/chinese-llama-2-1.3b'

# 1. Tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. LLaMA Model (FP16)
llama_model = LlamaModel.from_pretrained(
    model_path, 
    torch_dtype=torch.float16
).to(device)

# 3. 开启梯度检查点以大幅降低显存占用
llama_model.gradient_checkpointing_enable()
print("✓ LLaMA 梯度检查点已开启")

# -------------------------- 网络定义 --------------------------

class LLaMAEmbedder(nn.Module):
    """封装LLaMA模型，只负责生成状态嵌入"""
    def __init__(self, model):
        super(LLaMAEmbedder, self).__init__()
        self.llama = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        # 使用平均池化 而不是展平
        # 原始代码使用 last_hidden_state.view(batch, -1)，维度为 [batch, seq_len*hidden_size]
        # 这会导致全连接层参数量爆炸。改为平均池化后，维度为 [batch, hidden_size]
        last_hidden_state = outputs.last_hidden_state # [batch, seq_len, hidden_size]
        
        # 扩展 attention_mask 以便进行加权平均
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask # [batch, hidden_size]

class QNetwork(nn.Module):
    """Q网络，输入为LLaMA的Embedding向量"""
    def __init__(self, input_size, action_size, fc1_units=256, fc2_units=128):
        super(QNetwork, self).__init__()
        # input_size 现在是 llama_model.config.hidden_size (2048)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------- Replay Buffer --------------------------

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# -------------------------- Agent --------------------------

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # 1. Embedder
        self.embedder = LLaMAEmbedder(llama_model)
        
        # 2. Q Networks
        hidden_size = llama_model.config.hidden_size
        self.qnetwork_local = QNetwork(hidden_size, action_size).to(device)
        self.qnetwork_target = QNetwork(hidden_size, action_size).to(device)
        
        # 3. Optimizer: 只优化需要梯度的部分
        # 注意：如果显存依然紧张，可以冻结 embedder 参数：lr=0 或不传入优化器
        params = list(self.embedder.parameters()) + list(self.qnetwork_local.parameters())
        self.optimizer = optim.AdamW(params, lr=LR, betas=betas, weight_decay=weight_decay)
        
        # 4. Memory & Cache
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        
        # 嵌入缓存：state_str -> embedding_tensor
        self.emb_cache = {}
        self.cache_max_size = 5000 # 限制缓存大小防止内存溢出

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if done:
            for _ in range(7): # 经验复用
                self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def get_state_embedding(self, state_np):
        """将原始numpy状态转换为embedding，带缓存"""
        state_str = array2str(state_np)
        
        if state_str in self.emb_cache:
            return self.emb_cache[state_str]
            
        with torch.no_grad():
            token = tokenizer(state_str, add_special_tokens=True, max_length=TOKEN_SIZE, 
                             truncation=True, padding='max_length', return_tensors='pt')
            input_ids = token["input_ids"].to(device)
            attention_mask = token["attention_mask"].to(device)
            
            # LLaMA Embedder 前向传播
            embedding = self.embedder(input_ids, attention_mask) # [1, hidden_size]
            
            # 存入缓存
            emb_cpu = embedding.cpu()
            if len(self.emb_cache) < self.cache_max_size:
                self.emb_cache[state_str] = emb_cpu
            
            return emb_cpu

    def act(self, state, eps):
        # 1. 获取Embedding
        emb_tensor = self.get_state_embedding(state).to(device) # [1, hidden_size]
        
        # 2. Q Network 推理
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(emb_tensor)
        self.qnetwork_local.train()
        
        # 3. Epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # 1. 批量处理 Embedding
        # 为了加速，这里批量 Tokenize 和 Forward
        state_str_list = [array2str(s) for s in states]
        next_state_str_list = [array2str(s) for s in next_states]
        
        # Tokenize
        tokens = tokenizer(state_str_list, add_special_tokens=True, max_length=TOKEN_SIZE,
                         truncation=True, padding='max_length', return_tensors='pt')
        next_tokens = tokenizer(next_state_str_list, add_special_tokens=True, max_length=TOKEN_SIZE,
                              truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        next_input_ids = next_tokens['input_ids'].to(device)
        next_attention_mask = next_tokens['attention_mask'].to(device)

        # 2. 前向传播 (使用混合精度)
        with torch.cuda.amp.autocast():
            # 计算 State Embeddings
            state_embeddings = self.embedder(input_ids, attention_mask)
            
            # 计算 Next State Embeddings (Target网络不需要梯度，但embedder共享)
            # 注意：这里为了简化，即使target network也共享embedder
            # 这意味着我们实际上在训练一个Encoder
            with torch.no_grad():
                next_state_embeddings = self.embedder(next_input_ids, next_attention_mask)
            
            # Q-values
            Q_targets_next = self.qnetwork_target(next_state_embeddings).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.qnetwork_local(state_embeddings).gather(1, actions)
            
            loss = F.mse_loss(Q_expected, Q_targets)

        # 3. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        # 对LLaMA部分进行更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1.0)
        self.optimizer.step()

        # 4. 更新目标网络
        if self.t_step % (TAU * UPDATE_EVERY) == 0:
            self.update_target_net(self.qnetwork_local, self.qnetwork_target)

    def update_target_net(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())
    
    def save(self, name, output_dir):
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 保存时同时保存 embedder 以防万一
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'embedder': self.embedder.state_dict()
        }, os.path.join(checkpoint_dir, f'model{name}.pth'))
        print(f'模型已保存: {checkpoint_dir}/model{name}.pth')

# -------------------------- 主训练函数 --------------------------

def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix} GPU显存: 已分配={allocated:.2f} GB, 已预留={reserved:.2f} GB")

def dqn_train():
    print("="*60)
    print("     LLaMA-DQN Training with Enhanced Metrics Tracking")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Total Episodes: {N_EPISODES}")
    print("="*60 + "\n")
    
    print("初始化环境...")
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/llama_dqn_results/training.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True,
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("✓ 环境初始化成功")
    except Exception as e:
        print(f"✗ 环境初始化失败: {e}")
        return

    agent = Agent(STATE_SIZE, ACTION_SIZE)
    print("✓ Agent 初始化成功")
    
    # 初始化RND模块和AI指标追踪器
    rnd_module = RNDModule(STATE_SIZE)
    ai_tracker = AIMetricsTracker()
    print("✓ RND Module and AI Metrics Tracker initialized")
    
    print_gpu_memory("初始化后")

    scores = []
    n_amount_list = []
    w_amount_list = []
    yield_list = []
    eps = 1.0
    eps_decay = 0.994
    eps_end = 0.01
    
    k1, k2, k3, k4 = 0.158, 0.79, 1.1, 0
    
    # 记录训练开始时间
    train_start_time = time.time()
    
    print(f"\n开始训练 (共 {N_EPISODES} 轮)...")
    print("-"*60)
    
    # 使用tqdm显示进度条
    pbar = tqdm(range(1, N_EPISODES + 1), desc="Training Progress", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i_episode in pbar:
        episode_start_time = time.time()
        
        state = env.reset()
        state = dict2array(state)
        score = 0
        n_amount = 0
        w_amount = 0
        y = 0
        steps_in_episode = 0
        
        for t in range(200):
            action1 = agent.act(state, eps)
            action = {
                'anfer': (action1 % 5) * 40,
                'amir': int(action1 / 5) * 6,
            }
                    
            if state[0] >= 10000:
                action['anfer'] = 0
            if state[21] >= 1600:
                action['amir'] = 0
                    
            next_state, _, done, _ = env.step(action)
            steps_in_episode += 1
            
            # 更新RND模块并记录探索
            prediction_error = rnd_module.update(state)
            rnd_module.record_exploration(state, prediction_error)
            
            if done:
                y = state[4]
                next_state = state
                reward = get_reward(state, action['anfer'], action['amir'], next_state, done, k1, k2, k3, k4)
                agent.step(state, action1, reward, next_state, done)
                score += reward
                break
            
            n_amount += action['anfer']
            w_amount += action['amir']
            next_state = dict2array(next_state)
            reward = get_reward(state, action['anfer'], action['amir'], next_state, done, k1, k2, k3, k4)
            agent.step(state, action1, reward, next_state, done)
            state = next_state
            score += reward
        
        # 更新AI指标追踪器
        ai_tracker.update(score, steps_in_episode)
        
        n_amount_list.append(n_amount)
        w_amount_list.append(w_amount)
        yield_list.append(y)
        scores.append(score)
        
        if score > 1400 and y > 11000:
            agent.save(str(i_episode), OUTPUT_DIR)
        
        eps = max(eps_end, eps_decay * eps)
        
        # 更新进度条信息
        pbar.set_postfix({
            'Score': f'{score:.1f}',
            'Yield': f'{y:.0f}',
            'Eps': f'{eps:.3f}'
        })
        
        # 每10轮输出详细指标
        if i_episode % 10 == 0:
            print_metrics(i_episode, scores, n_amount_list, w_amount_list, 
                         yield_list, ai_tracker, rnd_module, prefix="[Check]")
            print_gpu_memory("训练中")

    # 训练结束，输出最终指标
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    
    print("\n" + "="*60)
    print(f"Training Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {total_train_time/3600:.2f} hours ({total_train_time:.1f} seconds)")
    print("="*60)
    
    # 输出最终指标汇总
    print_metrics(N_EPISODES, scores, n_amount_list, w_amount_list, 
                 yield_list, ai_tracker, rnd_module, is_final=True)
    
    # 保存所有结果
    print("\nSaving results...")
    save_results(scores, n_amount_list, w_amount_list, yield_list, 
                 ai_tracker, rnd_module, OUTPUT_DIR, OUTPUT_PREFIX)
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print("="*60)
    
    # 最后清理显存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    dqn_train()