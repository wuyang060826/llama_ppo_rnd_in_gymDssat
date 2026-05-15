#!/usr/bin/env python3
"""DQN+BERT+gym-dssat 模型验证代码 - 完整版本
/opt/gym_dssat_pdi/bin/python /home/wuyang/test/val_proj/val_bert_dqn.py --checkpoint /home/wuyang/test/val_proj/val_model/final_model/model1576.pth
验证策略说明：
1. 加载训练好的checkpoint模型
2. 与多种基准策略对比验证（随机、固定、启发式）
3. 支持使用特定文件进行确定性验证
4. 多轮统计验证确保结果可靠性
5. 全面的农学指标和AI指标评估
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
import gym
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import warnings
import glob
import json
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 导入transformers
from transformers import DistilBertModel, BertTokenizerFast

# ==================== 参数配置 ====================
TOKEN_SIZE = 27
STATE_SIZE = 25
ACTION_SIZE = 25

# 验证参数
N_VALIDATION_EPISODES = 5  # 每个策略的验证轮数
VALIDATION_SEEDS = [123456]  # 用于多轮验证的随机种子

# 奖励参数（与训练保持一致）
K1, K2, K3, K4 = 0.158, 0.79, 1.1, 0

# 文件路径配置
CHECKPOINT_DIR = '/home/wuyang/test/val_proj/val_model/final_model'
VALIDATION_FILE = 'SIAZ9501.MZX'  # 用户的验证文件
RESULTS_DIR = '/home/wuyang/test/val_proj/result/0410'


# ==================== 辅助函数 ====================
def dict2array(state):
    """将字典状态转换为数组"""
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state)


def array2str(state):
    """将状态数组转换为字符串用于BERT编码"""
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


def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3, k4):
    """计算奖励"""
    if done:
        return k1 * state[4] - k2 * n_action - k3 * w_action
    else:
        return -k2 * n_action - k3 * w_action


# ==================== 初始化模型和tokenizer ====================
print("Initializing DistilBERT model...")
model_name = 'distilbert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name, use_fast=True)
distilbert_model = DistilBertModel.from_pretrained(model_name).to(device)


# ==================== Q网络定义（与训练代码一致）====================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256, fc3_units=256):
        super(QNetwork, self).__init__()
        self.distilbert = distilbert_model
        self.fc1 = nn.Linear(distilbert_model.config.hidden_size * state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = distilbert_out.last_hidden_state
        reshaped_hidden_states = last_hidden_state.view(last_hidden_state.shape[0], -1)
        
        fc1_out = F.relu(self.fc1(reshaped_hidden_states))
        fc2_out = F.relu(self.fc2(fc1_out))
        return self.fc3(fc2_out)


# ==================== 验证Agent ====================
class ValidationAgent:
    """用于验证的Agent，只包含推理功能"""
    def __init__(self, state_size, action_size, checkpoint_path=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # 初始化网络
        self.qnetwork = QNetwork(TOKEN_SIZE, action_size).to(device)
        
        # 加载checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练好的模型权重"""
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.qnetwork.load_state_dict(state_dict)
            print(f"Checkpoint loaded successfully from: {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
    
    def act(self, state, deterministic=True):
        """选择动作（验证时使用确定性策略）"""
        state_str = array2str(state)
        token = tokenizer(state_str, add_special_tokens=True, max_length=TOKEN_SIZE, 
                         truncation=True, padding='max_length', return_tensors='pt')
        input_ids = token["input_ids"].to(device)
        attention_mask = token["attention_mask"].to(device)
        
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(input_ids, attention_mask)
        
        if deterministic:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # 使用softmax进行概率采样（可用于分析策略多样性）
            probs = F.softmax(action_values, dim=1).cpu().data.numpy()[0]
            return np.random.choice(np.arange(self.action_size), p=probs)


# ==================== 基准策略 ====================
class RandomPolicy:
    """随机策略基线"""
    def __init__(self, action_size=25, seed=None):
        self.action_size = action_size
        self.rng = np.random.RandomState(seed)
        self.name = "Random Policy"
    
    def act(self, state):
        return self.rng.randint(0, self.action_size)


class FixedPolicy:
    """固定策略（固定的灌溉和施肥量）"""
    def __init__(self, nitrogen_amount=80, irrigation_amount=12):
        """
        nitrogen_amount: 固定施肥量 (kg/ha)
        irrigation_amount: 固定灌溉量 (mm)
        """
        self.nitrogen_amount = nitrogen_amount
        self.irrigation_amount = irrigation_amount
        self.name = f"Fixed Policy (N={nitrogen_amount}kg/ha, W={irrigation_amount}mm)"
        
        # 将连续动作映射到离散动作空间
        # action = n_level + 5 * w_level
        n_level = nitrogen_amount // 40  # 0-4
        w_level = irrigation_amount // 6  # 0-4
        self.fixed_action = min(n_level, 4) + 5 * min(w_level, 4)
    
    def act(self, state):
        return self.fixed_action


class HeuristicPolicy:
    """启发式策略（基于领域知识的简单规则）"""
    def __init__(self):
        self.name = "Heuristic Policy"
    
    def act(self, state):
        """
        基于简单规则的决策：
        - 如果土壤水分低，增加灌溉
        - 如果氮含量低，增加施肥
        - 如果作物生长后期，减少投入
        """
        day = state[0]  # 生长天数
        soil_water = state[21]  # 土壤水分
        nitrogen = state[7] if len(state) > 7 else 50  # 氮含量
        
        # 基础动作
        n_level = 2  # 中等施肥
        w_level = 2  # 中等灌溉
        
        # 根据土壤水分调整灌溉
        if soil_water < 800:
            w_level = 3  # 增加灌溉
        elif soil_water < 500:
            w_level = 4  # 大量灌溉
        elif soil_water > 1200:
            w_level = 1  # 减少灌溉
        elif soil_water > 1500:
            w_level = 0  # 不灌溉
        
        # 根据氮含量调整施肥
        if nitrogen < 20:
            n_level = 3  # 增加施肥
        elif nitrogen > 80:
            n_level = 1  # 减少施肥
        
        # 生长后期（接近收获）减少投入
        if day > 150:
            n_level = max(0, n_level - 1)
            w_level = max(0, w_level - 1)
        if day > 180:
            n_level = 0
            w_level = max(0, w_level - 1)
        
        # 安全边界
        n_level = min(max(n_level, 0), 4)
        w_level = min(max(w_level, 0), 4)
        
        return n_level + 5 * w_level


class ConservativePolicy:
    """保守策略（低投入策略）"""
    def __init__(self):
        self.name = "Conservative Policy (Low Input)"
    
    def act(self, state):
        # 低施肥、低灌溉
        n_level = 1
        w_level = 1
        return n_level + 5 * w_level


class AggressivePolicy:
    """激进策略（高投入策略）"""
    def __init__(self):
        self.name = "Aggressive Policy (High Input)"
    
    def act(self, state):
        # 高施肥、高灌溉
        n_level = 3
        w_level = 3
        return n_level + 5 * w_level


# ==================== 动作转换函数 ====================
def action_to_dict(action):
    """将离散动作转换为实际操作参数"""
    return {
        'anfer': (action % 5) * 40,  # 施肥量: 0, 40, 80, 120, 160 kg/ha
        'amir': int(action / 5) * 6,  # 灌溉量: 0, 6, 12, 18, 24 mm
    }


# ==================== 验证指标追踪器 ====================
class ValidationMetricsTracker:
    """验证指标追踪器"""
    def __init__(self):
        self.episodes_data = []
        
    def add_episode(self, policy_name, episode_num, yield_val, irrigation, fertilizer, 
                    score, steps, seed, final_state=None):
        """添加一轮验证数据"""
        # 计算WUE和NUE
        wue = yield_val / irrigation if irrigation > 0 else 0
        nue = yield_val / fertilizer if fertilizer > 0 else 0
        
        episode_data = {
            'policy': policy_name,
            'episode': episode_num,
            'yield_kg_ha': yield_val,
            'irrigation_mm': irrigation,
            'fertilizer_kg_ha': fertilizer,
            'wue_kg_mm': wue,
            'nue_kg_kg': nue,
            'score': score,
            'steps': steps,
            'seed': seed,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.episodes_data.append(episode_data)
        
        return episode_data
    
    def get_policy_summary(self, policy_name):
        """获取特定策略的汇总统计"""
        policy_data = [d for d in self.episodes_data if d['policy'] == policy_name]
        if not policy_data:
            return None
        
        summary = {
            'policy': policy_name,
            'n_episodes': len(policy_data),
            'yield_mean': np.mean([d['yield_kg_ha'] for d in policy_data]),
            'yield_std': np.std([d['yield_kg_ha'] for d in policy_data]),
            'yield_max': np.max([d['yield_kg_ha'] for d in policy_data]),
            'yield_min': np.min([d['yield_kg_ha'] for d in policy_data]),
            'irrigation_mean': np.mean([d['irrigation_mm'] for d in policy_data]),
            'fertilizer_mean': np.mean([d['fertilizer_kg_ha'] for d in policy_data]),
            'wue_mean': np.mean([d['wue_kg_mm'] for d in policy_data]),
            'nue_mean': np.mean([d['nue_kg_kg'] for d in policy_data]),
            'score_mean': np.mean([d['score'] for d in policy_data]),
            'score_std': np.std([d['score'] for d in policy_data]),
        }
        return summary
    
    def get_all_summaries(self):
        """获取所有策略的汇总"""
        policies = set(d['policy'] for d in self.episodes_data)
        return {p: self.get_policy_summary(p) for p in policies}


# ==================== 验证环境创建 ====================
def create_validation_env(use_validation_file=False, seed=123456):
    """创建验证环境"""
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/gymusr/bert_dqn_results/logs/dssat-pdi-validation.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': False,'fileX_template_path':'/home/wuyang/test/val_proj/SIAZ9501.MZX'
    }
    
    # 如果有验证文件，配置使用
    if use_validation_file and os.path.exists(f'/opt/dssat_pdi/run_dssat/{VALIDATION_FILE}'):
        env_args['experiment_file'] = VALIDATION_FILE
        print(f"Using validation file: {VALIDATION_FILE}")
    
    return gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)


# ==================== 单轮验证函数 ====================
def run_single_episode(env, policy, policy_name, seed):
    """运行单轮验证"""
    np.random.seed(seed)
    random.seed(seed)
    
    state = env.reset()
    state = dict2array(state)
    
    score = 0
    n_amount = 0
    w_amount = 0
    y = 0
    steps = 0
    
    trajectory = []  # 记录轨迹用于分析
    
    for t in range(200):  # 最大步数限制
        # 获取动作
        if hasattr(policy, 'act'):
            if isinstance(policy, ValidationAgent):
                action = policy.act(state, deterministic=True)
            else:
                action = policy.act(state)
        else:
            action = policy(state)
        
        action_dict = action_to_dict(action)
        
        # 应用安全约束
        if state[0] >= 10000:  # 生长后期
            action_dict['anfer'] = 0
        if state[21] >= 1600:  # 土壤水分充足
            action_dict['amir'] = 0
        
        # 执行动作
        next_state, _, done, _ = env.step(action_dict)
        steps += 1
        
        # 记录轨迹
        trajectory.append({
            'step': t,
            'state': state.copy(),
            'action': action,
            'action_dict': action_dict.copy()
        })
        
        if done:
            y = state[4]  # 最终产量
            next_state = state
            reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                              next_state, done, K1, K2, K3, K4)
            score += reward
            break
        
        n_amount += action_dict['anfer']
        w_amount += action_dict['amir']
        next_state = dict2array(next_state)
        reward = get_reward(state, action_dict['anfer'], action_dict['amir'], 
                          next_state, done, K1, K2, K3, K4)
        state = next_state
        score += reward
    
    return {
        'yield': y,
        'irrigation': w_amount,
        'fertilizer': n_amount,
        'score': score,
        'steps': steps,
        'trajectory': trajectory
    }


# ==================== 策略验证函数 ====================
def validate_policy(env, policy, policy_name, n_episodes, seeds):
    """验证单个策略"""
    results = []
    
    print(f"\nValidating {policy_name}...")
    
    for i in tqdm(range(n_episodes), desc=f"  {policy_name}", leave=False):
        seed = seeds[i % len(seeds)]
        
        try:
            result = run_single_episode(env, policy, policy_name, seed)
            results.append(result)
        except Exception as e:
            print(f"    Error in episode {i+1}: {e}")
            continue
    
    return results


# ==================== 找到最佳checkpoint ====================
def find_best_checkpoint(checkpoint_dir):
    """找到最佳checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # 查找所有checkpoint文件
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model*.pth'))
    
    if not checkpoint_files:
        print("No checkpoint files found")
        return None
    
    # 按修改时间排序，返回最新的
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files[:5]:  # 只显示前5个
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        print(f"  - {os.path.basename(f)} (modified: {mtime})")
    
    return checkpoint_files[0]


# ==================== 统计检验函数 ====================
def perform_statistical_test(dqn_scores, baseline_scores, baseline_name):
    """执行统计显著性检验"""
    from scipy import stats
    
    # t检验
    t_stat, p_value_t = stats.ttest_ind(dqn_scores, baseline_scores)
    
    # Mann-Whitney U检验（非参数）
    u_stat, p_value_u = stats.mannwhitneyu(dqn_scores, baseline_scores, alternative='two-sided')
    
    # 效应量 (Cohen's d)
    mean_diff = np.mean(dqn_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt((np.std(dqn_scores)**2 + np.std(baseline_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'baseline': baseline_name,
        't_statistic': t_stat,
        'p_value_ttest': p_value_t,
        'u_statistic': u_stat,
        'p_value_mannwhitney': p_value_u,
        'cohens_d': cohens_d,
        'significant_005': p_value_t < 0.05,
        'significant_001': p_value_t < 0.01
    }


# ==================== 结果可视化 ====================
def plot_validation_results(metrics_tracker, output_dir, timestamp):
    """生成验证结果可视化图表"""
    
    # 获取所有策略的汇总
    summaries = metrics_tracker.get_all_summaries()
    policy_names = list(summaries.keys())
    
    # 创建综合图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 产量对比（箱线图）
    ax1 = fig.add_subplot(3, 3, 1)
    yield_data = []
    for policy in policy_names:
        policy_episodes = [d for d in metrics_tracker.episodes_data if d['policy'] == policy]
        yield_data.append([d['yield_kg_ha'] for d in policy_episodes])
    
    bp1 = ax1.boxplot(yield_data, labels=[p[:15] + '...' if len(p) > 15 else p for p in policy_names], 
                      patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(policy_names)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_title('Yield Comparison (kg/ha)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Yield (kg/ha)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 得分对比（箱线图）
    ax2 = fig.add_subplot(3, 3, 2)
    score_data = []
    for policy in policy_names:
        policy_episodes = [d for d in metrics_tracker.episodes_data if d['policy'] == policy]
        score_data.append([d['score'] for d in policy_episodes])
    
    bp2 = ax2.boxplot(score_data, labels=[p[:15] + '...' if len(p) > 15 else p for p in policy_names],
                      patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Score Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. 资源使用对比（双轴柱状图）
    ax3 = fig.add_subplot(3, 3, 3)
    x = np.arange(len(policy_names))
    width = 0.35
    
    irrigation_means = [summaries[p]['irrigation_mean'] for p in policy_names]
    fertilizer_means = [summaries[p]['fertilizer_mean'] for p in policy_names]
    
    bars1 = ax3.bar(x - width/2, irrigation_means, width, label='Irrigation (mm)', color='steelblue')
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, fertilizer_means, width, label='Fertilizer (kg/ha)', color='coral')
    
    ax3.set_xlabel('Policy')
    ax3.set_ylabel('Irrigation (mm)', color='steelblue')
    ax3_twin.set_ylabel('Fertilizer (kg/ha)', color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p[:10] + '...' if len(p) > 10 else p for p in policy_names], rotation=45, ha='right')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.set_title('Resource Usage Comparison', fontsize=12, fontweight='bold')
    
    # 4. WUE对比
    ax4 = fig.add_subplot(3, 3, 4)
    wue_means = [summaries[p]['wue_mean'] for p in policy_names]
    wue_stds = [np.std([d['wue_kg_mm'] for d in metrics_tracker.episodes_data 
                        if d['policy'] == p]) for p in policy_names]
    
    bars4 = ax4.bar(policy_names, wue_means, yerr=wue_stds, capsize=5, color='forestgreen', alpha=0.7)
    ax4.set_title('Water Use Efficiency (WUE)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('WUE (kg/mm)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. NUE对比
    ax5 = fig.add_subplot(3, 3, 5)
    nue_means = [summaries[p]['nue_mean'] for p in policy_names]
    nue_stds = [np.std([d['nue_kg_kg'] for d in metrics_tracker.episodes_data 
                        if d['policy'] == p]) for p in policy_names]
    
    bars5 = ax5.bar(policy_names, nue_means, yerr=nue_stds, capsize=5, color='darkorange', alpha=0.7)
    ax5.set_title('Nitrogen Use Efficiency (NUE)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('NUE (kg/kg)')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 综合性能雷达图
    ax6 = fig.add_subplot(3, 3, 6, projection='polar')
    categories = ['Yield', 'Score', 'WUE', 'NUE', 'Resource Efficiency']
    
    # 标准化各项指标
    def normalize(values, reverse=False):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1] * len(values)
        norm = [(v - min_val) / (max_val - min_val) for v in values]
        return [1 - n for n in norm] if reverse else norm
    
    yield_norm = normalize([summaries[p]['yield_mean'] for p in policy_names])
    score_norm = normalize([summaries[p]['score_mean'] for p in policy_names])
    wue_norm = normalize([summaries[p]['wue_mean'] for p in policy_names])
    nue_norm = normalize([summaries[p]['nue_mean'] for p in policy_names])
    # 资源效率（低资源使用为好）
    resource_norm = normalize([summaries[p]['irrigation_mean'] + summaries[p]['fertilizer_mean'] 
                               for p in policy_names], reverse=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, policy in enumerate(policy_names):
        values = [yield_norm[i], score_norm[i], wue_norm[i], nue_norm[i], resource_norm[i]]
        values += values[:1]
        ax6.plot(angles, values, 'o-', linewidth=2, label=policy[:10])
        ax6.fill(angles, values, alpha=0.1)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_title('Comprehensive Performance Radar', fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # 7. 产量vs资源使用散点图
    ax7 = fig.add_subplot(3, 3, 7)
    for i, policy in enumerate(policy_names):
        policy_data = [d for d in metrics_tracker.episodes_data if d['policy'] == policy]
        yields = [d['yield_kg_ha'] for d in policy_data]
        irrigation = [d['irrigation_mm'] for d in policy_data]
        ax7.scatter(irrigation, yields, alpha=0.6, s=50, label=policy[:10], color=colors[i])
    
    ax7.set_xlabel('Total Irrigation (mm)')
    ax7.set_ylabel('Final Yield (kg/ha)')
    ax7.set_title('Yield vs Irrigation', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. 得分稳定性（标准差对比）
    ax8 = fig.add_subplot(3, 3, 8)
    score_stds = [summaries[p]['score_std'] for p in policy_names]
    
    bars8 = ax8.bar(policy_names, score_stds, color='mediumpurple', alpha=0.7)
    ax8.set_title('Score Stability (Std Dev)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Score Standard Deviation')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. 各策略排名热力图
    ax9 = fig.add_subplot(3, 3, 9)
    rank_metrics = ['yield_mean', 'score_mean', 'wue_mean', 'nue_mean']
    rank_data = []
    
    for metric in rank_metrics:
        values = [summaries[p][metric] for p in policy_names]
        ranks = [sorted(values, reverse=True).index(v) + 1 for v in values]
        rank_data.append(ranks)
    
    im = ax9.imshow(rank_data, cmap='RdYlGn_r', aspect='auto')
    ax9.set_xticks(np.arange(len(policy_names)))
    ax9.set_yticks(np.arange(len(rank_metrics)))
    ax9.set_xticklabels([p[:8] for p in policy_names], rotation=45, ha='right')
    ax9.set_yticklabels(['Yield', 'Score', 'WUE', 'NUE'])
    ax9.set_title('Policy Rankings (1=Best)', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(rank_metrics)):
        for j in range(len(policy_names)):
            text = ax9.text(j, i, rank_data[i][j], ha="center", va="center", 
                           color="black" if rank_data[i][j] <= 2 else "white")
    
    plt.colorbar(im, ax=ax9, label='Rank')
    
    plt.tight_layout()
    
    # 保存图表
    png_path = os.path.join(output_dir, f'validation_results_{timestamp}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Validation plots saved to: {png_path}")
    
    pdf_path = os.path.join(output_dir, f'validation_results_{timestamp}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Validation plots saved to: {pdf_path}")
    
    plt.close()


# ==================== 保存验证结果 ====================
def save_validation_results(metrics_tracker, statistical_tests, output_dir):
    """保存验证结果到文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 保存详细结果到Excel
    df = pd.DataFrame(metrics_tracker.episodes_data)
    excel_path = os.path.join(output_dir, f'validation_details_{timestamp}.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Detailed results saved to: {excel_path}")
    
    # 2. 保存汇总结果
    summaries = metrics_tracker.get_all_summaries()
    summary_df = pd.DataFrame(summaries).T
    summary_path = os.path.join(output_dir, f'validation_summary_{timestamp}.xlsx')
    summary_df.to_excel(summary_path)
    print(f"Summary results saved to: {summary_path}")
    
    # 3. 保存统计检验结果
    if statistical_tests:
        test_df = pd.DataFrame(statistical_tests)
        test_path = os.path.join(output_dir, f'statistical_tests_{timestamp}.xlsx')
        test_df.to_excel(test_path, index=False)
        print(f"Statistical test results saved to: {test_path}")
    
    # 4. 生成JSON格式的汇总报告
    report = {
        'timestamp': timestamp,
        'validation_config': {
            'n_episodes_per_policy': N_VALIDATION_EPISODES,
            'seeds': VALIDATION_SEEDS,
            'device': str(device)
        },
        'summaries': summaries,
        'statistical_tests': statistical_tests
    }
    
    json_path = os.path.join(output_dir, f'validation_report_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON report saved to: {json_path}")
    
    # 5. 生成可视化图表
    plot_validation_results(metrics_tracker, output_dir, timestamp)
    
    return excel_path, summary_path


# ==================== 打印验证报告 ====================
def print_validation_report(metrics_tracker, statistical_tests):
    """打印详细的验证报告"""
    summaries = metrics_tracker.get_all_summaries()
    policy_names = list(summaries.keys())
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS REPORT")
    print("="*80)
    
    # 排序策略（按产量均值降序）
    sorted_policies = sorted(policy_names, key=lambda p: summaries[p]['yield_mean'], reverse=True)
    
    print("\n" + "-"*80)
    print("POLICY PERFORMANCE SUMMARY (Sorted by Yield)")
    print("-"*80)
    
    print(f"\n{'Policy':<35} {'Yield':>12} {'Score':>12} {'WUE':>10} {'NUE':>10} {'Irrig':>8} {'Fert':>8}")
    print("-"*95)
    
    for policy in sorted_policies:
        s = summaries[policy]
        name = policy[:33] + '..' if len(policy) > 35 else policy
        print(f"{name:<35} {s['yield_mean']:>10.1f}±{s['yield_std']:<5.1f} "
              f"{s['score_mean']:>10.1f}±{s['score_std']:<5.1f} "
              f"{s['wue_mean']:>10.3f} {s['nue_mean']:>10.3f} "
              f"{s['irrigation_mean']:>8.1f} {s['fertilizer_mean']:>8.1f}")
    
    # 找出DQN策略
    dqn_policy = None
    for p in policy_names:
        if 'DQN' in p or 'Checkpoint' in p:
            dqn_policy = p
            break
    
    if dqn_policy and statistical_tests:
        print("\n" + "-"*80)
        print("STATISTICAL SIGNIFICANCE TESTS (DQN vs Baselines)")
        print("-"*80)
        
        for test in statistical_tests:
            baseline = test['baseline']
            print(f"\n  DQN vs {baseline}:")
            print(f"    t-statistic:     {test['t_statistic']:.4f}")
            print(f"    p-value (t-test): {test['p_value_ttest']:.6f} {'***' if test['p_value_ttest'] < 0.001 else '**' if test['p_value_ttest'] < 0.01 else '*' if test['p_value_ttest'] < 0.05 else ''}")
            print(f"    p-value (U-test): {test['p_value_mannwhitney']:.6f}")
            print(f"    Cohen's d:       {test['cohens_d']:.4f} ({'large' if abs(test['cohens_d']) > 0.8 else 'medium' if abs(test['cohens_d']) > 0.5 else 'small'})")
            print(f"    Significant (α=0.05): {'Yes' if test['significant_005'] else 'No'}")
            print(f"    Significant (α=0.01): {'Yes' if test['significant_001'] else 'No'}")
    
    # 性能提升计算
    if dqn_policy:
        dqn_summary = summaries[dqn_policy]
        print("\n" + "-"*80)
        print("PERFORMANCE IMPROVEMENT (DQN vs Best Baseline)")
        print("-"*80)
        
        baseline_policies = [p for p in policy_names if p != dqn_policy]
        if baseline_policies:
            best_baseline = max(baseline_policies, key=lambda p: summaries[p]['yield_mean'])
            best_baseline_summary = summaries[best_baseline]
            
            yield_improvement = (dqn_summary['yield_mean'] - best_baseline_summary['yield_mean']) / best_baseline_summary['yield_mean'] * 100
            score_improvement = (dqn_summary['score_mean'] - best_baseline_summary['score_mean']) / abs(best_baseline_summary['score_mean']) * 100 if best_baseline_summary['score_mean'] != 0 else 0
            wue_improvement = (dqn_summary['wue_mean'] - best_baseline_summary['wue_mean']) / best_baseline_summary['wue_mean'] * 100 if best_baseline_summary['wue_mean'] != 0 else 0
            nue_improvement = (dqn_summary['nue_mean'] - best_baseline_summary['nue_mean']) / best_baseline_summary['nue_mean'] * 100 if best_baseline_summary['nue_mean'] != 0 else 0
            
            print(f"\n  Best Baseline: {best_baseline}")
            print(f"  Yield Improvement:    {yield_improvement:+.2f}%")
            print(f"  Score Improvement:    {score_improvement:+.2f}%")
            print(f"  WUE Improvement:      {wue_improvement:+.2f}%")
            print(f"  NUE Improvement:      {nue_improvement:+.2f}%")
    
    print("\n" + "="*80)


# ==================== 主验证函数 ====================
def validate_model(checkpoint_path=None, use_validation_file=True):
    """主验证函数"""
    print("\n" + "="*60)
    print("DQN+BERT+gym-dssat Model Validation")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 查找或使用指定的checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
    
    if checkpoint_path is None:
        print("No checkpoint found. Will validate baseline policies only.")
        dqn_agent = None
        dqn_policy_name = None
    else:
        print(f"\nUsing checkpoint: {checkpoint_path}")
        dqn_agent = ValidationAgent(STATE_SIZE, ACTION_SIZE, checkpoint_path)
        dqn_policy_name = f"DQN-BERT ({os.path.basename(checkpoint_path)})"
    
    # 初始化指标追踪器
    metrics_tracker = ValidationMetricsTracker()
    
    # 创建验证环境
    print("\nInitializing validation environment...")
    try:
        env = create_validation_env(use_validation_file=use_validation_file, seed=42)
        print("Environment initialized successfully")
    except Exception as e:
        print(f"Environment initialization failed: {e}")
        return None
    
    # 定义要验证的策略列表
    policies = [
        ('Random', RandomPolicy(action_size=ACTION_SIZE, seed=42)),
        ('Fixed-Low', FixedPolicy(nitrogen_amount=40, irrigation_amount=6)),
        ('Fixed-Mid', FixedPolicy(nitrogen_amount=80, irrigation_amount=12)),
        ('Fixed-High', FixedPolicy(nitrogen_amount=120, irrigation_amount=18)),
        ('Conservative', ConservativePolicy()),
        ('Aggressive', AggressivePolicy()),
        ('Heuristic', HeuristicPolicy()),
    ]
    
    # 添加DQN策略
    if dqn_agent is not None:
        policies.append((dqn_policy_name, dqn_agent))
    
    # 验证每个策略
    print(f"\nValidating {len(policies)} policies, {N_VALIDATION_EPISODES} episodes each...")
    
    for policy_name, policy in policies:
        # 为每个策略重置环境种子
        env.close()
        
        try:
            env = create_validation_env(use_validation_file=use_validation_file, 
                                       seed=VALIDATION_SEEDS[0])
        except:
            continue
        
        # 运行验证
        results = validate_policy(env, policy, policy_name, N_VALIDATION_EPISODES, VALIDATION_SEEDS)
        
        # 记录结果
        for i, result in enumerate(results):
            metrics_tracker.add_episode(
                policy_name=policy_name,
                episode_num=i + 1,
                yield_val=result['yield'],
                irrigation=result['irrigation'],
                fertilizer=result['fertilizer'],
                score=result['score'],
                steps=result['steps'],
                seed=VALIDATION_SEEDS[i % len(VALIDATION_SEEDS)]
            )
    
    # 执行统计检验
    statistical_tests = []
    if dqn_agent is not None:
        dqn_scores = [d['score'] for d in metrics_tracker.episodes_data 
                      if d['policy'] == dqn_policy_name]
        
        baseline_names = [p[0] for p in policies if p[0] != dqn_policy_name]
        
        for baseline_name in baseline_names:
            baseline_scores = [d['score'] for d in metrics_tracker.episodes_data 
                              if d['policy'] == baseline_name]
            
            if len(dqn_scores) > 1 and len(baseline_scores) > 1:
                test_result = perform_statistical_test(dqn_scores, baseline_scores, baseline_name)
                statistical_tests.append(test_result)
    
    # 关闭环境
    env.close()
    
    # 打印报告
    print_validation_report(metrics_tracker, statistical_tests)
    
    # 保存结果
    print("\nSaving validation results...")
    save_validation_results(metrics_tracker, statistical_tests, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("Validation completed successfully!")
    print("="*60)
    
    return metrics_tracker, statistical_tests


# ==================== 命令行接口 ====================
def main():
    # 更新全局参数
    global N_VALIDATION_EPISODES, RESULTS_DIR
    
    parser = argparse.ArgumentParser(description='Validate DQN+BERT model for crop management')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: auto-detect latest)')
    parser.add_argument('--n_episodes', type=int, default=N_VALIDATION_EPISODES,
                       help=f'Number of validation episodes per policy (default: {N_VALIDATION_EPISODES})')
    parser.add_argument('--use_validation_file', action='store_true', default=True,
                       help='Use SIAZ9501.MZX file for deterministic validation')
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                       help='Output directory for validation results')
    
    args = parser.parse_args()
    
    N_VALIDATION_EPISODES = args.n_episodes
    RESULTS_DIR = args.output_dir

    
    # 运行验证
    validate_model(
        checkpoint_path=args.checkpoint,
        use_validation_file=args.use_validation_file
    )


if __name__ == "__main__":
    main()