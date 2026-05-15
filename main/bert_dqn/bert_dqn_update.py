#!/usr/bin/env python3
"""
DQN+BERT+gym-dssat GPU优化版本 - 完整指标追踪版本
严格参考作者开源项目(jingwu6/LM_AG)和论文(arXiv:2403.19839)优化

主要优化点：
1. 与作者原始代码保持一致的训练参数
2. 梯度裁剪 (-1, 1)
3. 终止状态经验重复添加 7 次
4. 硬更新目标网络策略
5. epsilon衰减策略 (eps_end=0, eps_decay=0.994)
6. 学习率 1e-5, 批大小 512, 更新频率 16
7. 【新增】测量噪声增强 (Measurement Noise Augmentation) - 提升泛化能力
8. 保留用户的RND探索覆盖率和指标追踪改进
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

# 设备配置 - 自动检测并使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 导入transformers
from transformers import DistilBertModel, BertTokenizerFast

# ==================== 参数配置 ====================
# 严格参考作者原始代码: llm512_256_batch512_tau8_update16_lr5_irr6_clip_aug.ipynb
BUFFER_SIZE = int(1e5)  # replay buffer size - 与作者一致
BATCH_SIZE = 512        # minibatch size - 与作者一致
GAMMA = 0.99            # discount factor - 与作者一致
TAU = 8                 # for update of target network parameters - 与作者一致
LR = 1e-5               # learning rate - 与作者一致
UPDATE_EVERY = 16       # how often to update the network - 与作者一致

# 优化器参数 - 与作者一致（不使用weight_decay）
betas = (0.9, 0.999)

# 网络参数
TOKEN_SIZE = 27
STATE_SIZE = 25
ACTION_SIZE = 25

# 探索参数 - 与作者一致
EPS_START = 1.0
EPS_END = 0              # 作者使用0而非0.01
EPS_DECAY = 0.994        # 与作者一致

# 训练参数
N_EPISODES = 3000        # 与作者一致
MAX_T = 500              # 与作者一致

# ==================== 【新增】测量噪声增强参数 ====================
# 参考论文5.2节 "Policy Evaluation with Measurement Noises"
# 在训练时添加噪声提高模型鲁棒性和泛化能力
USE_NOISE_AUGMENTATION = True      # 是否启用噪声增强
NOISE_TYPE = 'gaussian'            # 噪声类型: 'gaussian' 或 'uniform'
NOISE_RATIO = 0.05                 # 噪声比例 (相对于状态值的百分比)
NOISE_CLIP = 0.1                   # 噪声裁剪上限 (防止过大扰动)
STATE_NOISE_MASK = None            # 状态噪声掩码 (None表示对所有状态添加噪声)
                                   # 可设置为列表指定哪些状态索引需要添加噪声

# 目标性能阈值（用于计算样本效率和收敛速度）
TARGET_PERFORMANCE_SCORE = 1000
CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.1

# RND探索覆盖率参数
RND_HIDDEN_SIZE = 256
RND_LR = 1e-4
STATE_NORMALIZATION_DECAY = 0.99
EXPLORATION_BONUS_SCALE = 0.01


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
    """将状态数组转换为字符串用于BERT编码 - 与作者一致"""
    state_str = ""
    for i, num in enumerate(state):
        if i == 0:
            state_str += str(round(num / 40)) + " "
        elif i == 4:
            state_str += str(round(num / 100)) + " "
        elif i == 7:
            state_str += str(round(num / 10)) + " "
        elif i == 20:
            state_str += str(round(num / 100)) + " "  # 250
        elif i == 21:
            state_str += str(round(num / 6)) + " "    # 12
        elif i == 23:
            state_str += str(round(num)) + " "        # 10
        elif i >= 9 and i <= 17:
            state_str += str(round(num * 1000)) + " "
        elif i == 18:
            state_str += str(round(num * 100)) + " "
        else:
            state_str += str(round(num)) + " "
    return state_str


def get_reward(state, n_action, w_action, next_state, done, k1, k2, k3, k4):
    """计算奖励 - 与作者一致"""
    if done:
        reward = k1 * state[4] - k2 * n_action - k3 * w_action
        return reward
    else:
        reward = -k2 * n_action - k3 * w_action
        return reward


# ==================== 【新增】噪声增强函数 ====================
def add_measurement_noise(state, noise_type='gaussian', noise_ratio=0.05, 
                          noise_clip=0.1, noise_mask=None):
    """
    为状态添加测量噪声 - 提高模型鲁棒性
    
    参考论文5.2节 "Policy Evaluation with Measurement Noises"
    在训练时对状态添加噪声，模拟真实世界中传感器测量的不确定性
    
    参数:
        state: 原始状态数组
        noise_type: 噪声类型 ('gaussian' 或 'uniform')
        noise_ratio: 噪声相对于状态值的比例
        noise_clip: 噪声裁剪上限，防止过大扰动
        noise_mask: 指定哪些状态索引需要添加噪声 (None表示全部)
    
    返回:
        添加噪声后的状态数组
    """
    if not USE_NOISE_AUGMENTATION:
        return state
    
    noisy_state = state.copy()
    
    # 确定需要添加噪声的状态索引
    if noise_mask is None:
        indices = range(len(state))
    else:
        indices = noise_mask
    
    for i in indices:
        if i >= len(state):
            continue
            
        state_val = state[i]
        
        # 根据状态值计算噪声幅度
        if abs(state_val) < 1e-6:
            # 状态值接近0时，使用固定小噪声
            noise_scale = noise_ratio
        else:
            # 噪声幅度与状态值成比例
            noise_scale = abs(state_val) * noise_ratio
        
        # 生成噪声
        if noise_type == 'gaussian':
            # 高斯噪声 (模拟传感器测量误差)
            noise = np.random.normal(0, noise_scale)
        else:  # uniform
            # 均匀噪声 (模拟量化误差)
            noise = np.random.uniform(-noise_scale, noise_scale)
        
        # 裁剪噪声幅度
        noise = np.clip(noise, -noise_clip * abs(state_val) if abs(state_val) > 1e-6 else -noise_clip,
                        noise_clip * abs(state_val) if abs(state_val) > 1e-6 else noise_clip)
        
        noisy_state[i] = state_val + noise
    
    return noisy_state


def add_batch_noise(states, noise_type='gaussian', noise_ratio=0.05, 
                    noise_clip=0.1, noise_mask=None):
    """
    批量为状态添加测量噪声 - 用于批量训练
    
    参数:
        states: 状态数组 (batch_size, state_dim)
        其他参数同 add_measurement_noise
    
    返回:
        添加噪声后的状态数组
    """
    if not USE_NOISE_AUGMENTATION:
        return states
    
    noisy_states = states.copy()
    batch_size = states.shape[0]
    
    for b in range(batch_size):
        noisy_states[b] = add_measurement_noise(
            states[b], noise_type, noise_ratio, noise_clip, noise_mask
        )
    
    return noisy_states


# ==================== 初始化模型和tokenizer ====================
print("Initializing DistilBERT model...")
model_name = 'distilbert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name, use_fast=True)
distilbert_model = DistilBertModel.from_pretrained(model_name).to(device)

total_params = sum(p.numel() for p in distilbert_model.parameters())
print(f"DistilBERT Total parameters: {total_params}")


# ==================== RND网络（用于探索覆盖率计算）====================
class RNDNetwork(nn.Module):
    """随机网络蒸馏(RND)的目标网络和预测网络"""
    def __init__(self, input_size, hidden_size):
        super(RNDNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x):
        return self.network(x)


class RNDModule:
    """RND模块用于计算探索覆盖率"""
    def __init__(self, state_size, hidden_size=RND_HIDDEN_SIZE):
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # 目标网络（固定参数）
        self.target_network = RNDNetwork(state_size, hidden_size).to(device)
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # 预测网络（可训练）
        self.predictor_network = RNDNetwork(state_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=RND_LR)
        
        # 状态归一化参数
        self.running_mean = np.zeros(state_size)
        self.running_var = np.ones(state_size)
        self.count = 0
        
        # 探索追踪
        self.visited_states = set()
        self.novel_states_count = 0
        self.total_states_count = 0
        self.exploration_bonus_threshold = 0.5
        
    def normalize_state(self, state):
        """归一化状态"""
        self.count += 1
        delta = state - self.running_mean
        self.running_mean += delta / self.count
        delta2 = state - self.running_mean
        self.running_var += delta * delta2
        
        normalized_state = (state - self.running_mean) / (np.sqrt(self.running_var / self.count) + 1e-8)
        return np.clip(normalized_state, -5, 5)
    
    def get_prediction_error(self, state):
        """计算预测误差"""
        normalized_state = self.normalize_state(state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predict_output = self.predictor_network(state_tensor)
        error = F.mse_loss(predict_output, target_output).item()
        
        return error
    
    def update_predictor(self, state):
        """更新预测网络"""
        normalized_state = self.normalize_state(state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            target_output = self.target_network(state_tensor)
        
        predict_output = self.predictor_network(state_tensor)
        loss = F.mse_loss(predict_output, target_output)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_exploration_coverage(self, state):
        """更新探索覆盖率"""
        self.total_states_count += 1
        
        # 计算预测误差作为新颖性度量
        prediction_error = self.get_prediction_error(state)
        
        # 状态离散化用于追踪已访问状态
        discretized_state = tuple(np.round(state / 10).astype(int))
        
        # 如果预测误差较高，说明是新状态
        if prediction_error > self.exploration_bonus_threshold:
            if discretized_state not in self.visited_states:
                self.visited_states.add(discretized_state)
                self.novel_states_count += 1
        
        # 更新预测网络
        self.update_predictor(state)
        
        return prediction_error
    
    def get_exploration_coverage(self):
        """获取探索覆盖率"""
        if self.total_states_count == 0:
            return 0.0
        coverage = len(self.visited_states) / max(self.total_states_count * 0.1, 1) * 100
        return min(coverage, 100.0)


# ==================== Q网络定义 - 与作者一致 ====================
class QNetwork(nn.Module):
    """Q网络 - 使用DistilBERT编码状态，与作者架构一致"""
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256, fc3_units=256):
        super(QNetwork, self).__init__()
        self.distilbert = distilbert_model
        # FC层配置与作者一致: 512 -> 256 -> action_size
        self.fc1 = nn.Linear(distilbert_model.config.hidden_size * state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # 注意：作者没有冻结DistilBERT参数，允许端到端训练

    def forward(self, input_ids, attention_mask):
        # DistilBERT编码 - 允许梯度回传
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = distilbert_out.last_hidden_state
        reshaped_hidden_states = last_hidden_state.view(last_hidden_state.shape[0], -1)
        
        # FC层前向传播
        fc1_out = F.relu(self.fc1(reshaped_hidden_states))
        fc2_out = F.relu(self.fc2(fc1_out))
        return self.fc3(fc2_out)


# ==================== Replay Buffer - 与作者一致 ====================
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# ==================== 指标追踪器 ====================
class MetricsTracker:
    """追踪和计算所有训练指标"""
    def __init__(self):
        # 农学指标
        self.yield_list = []
        self.irrigation_list = []
        self.fertilizer_list = []
        self.wue_list = []
        self.nue_list = []
        
        # AI指标
        self.scores = []
        self.episode_steps = []
        self.total_steps = 0
        self.sample_efficiency_step = None
        self.convergence_step = None
        
        # 探索相关
        self.exploration_coverage_list = []
        
        # 目标性能追踪
        self.target_achieved = False
        self.convergence_achieved = False
        
    def update_agronomic_metrics(self, yield_val, irrigation, fertilizer):
        """更新农学指标"""
        self.yield_list.append(yield_val)
        self.irrigation_list.append(irrigation)
        self.fertilizer_list.append(fertilizer)
        
        # 计算WUE和NUE
        wue = yield_val / irrigation if irrigation > 0 else 0.0
        nue = yield_val / fertilizer if fertilizer > 0 else 0.0
        self.wue_list.append(wue)
        self.nue_list.append(nue)
        
        return wue, nue
    
    def update_ai_metrics(self, score, steps, exploration_coverage):
        """更新AI指标"""
        self.scores.append(score)
        self.episode_steps.append(steps)
        self.total_steps += steps
        self.exploration_coverage_list.append(exploration_coverage)
        
        # 检查样本效率
        if not self.target_achieved and score >= TARGET_PERFORMANCE_SCORE:
            self.target_achieved = True
            self.sample_efficiency_step = self.total_steps
        
        # 检查收敛速度
        if not self.convergence_achieved and len(self.scores) >= CONVERGENCE_WINDOW:
            recent_scores = self.scores[-CONVERGENCE_WINDOW:]
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            if mean_score > 0 and std_score / mean_score < CONVERGENCE_THRESHOLD:
                self.convergence_achieved = True
                self.convergence_step = self.total_steps - sum(self.episode_steps[-CONVERGENCE_WINDOW:])
    
    def get_summary_metrics(self):
        """获取汇总指标"""
        summary = {
            'avg_yield': np.mean(self.yield_list) if self.yield_list else 0,
            'max_yield': np.max(self.yield_list) if self.yield_list else 0,
            'avg_irrigation': np.mean(self.irrigation_list) if self.irrigation_list else 0,
            'avg_fertilizer': np.mean(self.fertilizer_list) if self.fertilizer_list else 0,
            'avg_wue': np.mean(self.wue_list) if self.wue_list else 0,
            'avg_nue': np.mean(self.nue_list) if self.nue_list else 0,
            'avg_score': np.mean(self.scores) if self.scores else 0,
            'max_score': np.max(self.scores) if self.scores else 0,
            'sample_efficiency': self.sample_efficiency_step if self.sample_efficiency_step else 'Not achieved',
            'convergence_speed': self.convergence_step if self.convergence_step else 'Not converged',
            'avg_exploration_coverage': np.mean(self.exploration_coverage_list) if self.exploration_coverage_list else 0,
            'final_exploration_coverage': self.exploration_coverage_list[-1] if self.exploration_coverage_list else 0,
            'total_episodes': len(self.scores),
            'total_steps': self.total_steps
        }
        return summary


# ==================== Agent - 与作者实现保持一致 + 噪声增强 ====================
class Agent:
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network - 与作者配置一致
        self.qnetwork_local = QNetwork(TOKEN_SIZE, action_size).to(device)
        self.qnetwork_target = QNetwork(TOKEN_SIZE, action_size).to(device)
        # 优化器配置与作者一致 - 只使用betas，不使用weight_decay
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, betas=betas)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        
        # 初始化RND模块
        self.rnd = RNDModule(state_size)
    
    def step(self, state, action, reward, next_state, done):
        """处理每一步的经验并学习 - 与作者实现一致"""
        # 保存经验到replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # 更新探索覆盖率
        self.rnd.update_exploration_coverage(state)
        
        # 关键技巧：终止状态经验重复添加7次 - 与作者一致
        if done:
            for _ in range(7):
                self.memory.add(state, action, reward, next_state, done)
        
        # 每UPDATE_EVERY步学习一次 - 与作者一致
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps):
        """Returns actions for given state as per current policy.
        
        【新增】在训练时对状态添加噪声以提高鲁棒性
        """
        # 【新增】训练模式下添加测量噪声
        if USE_NOISE_AUGMENTATION and self.qnetwork_local.training:
            noisy_state = add_measurement_noise(
                state, 
                noise_type=NOISE_TYPE,
                noise_ratio=NOISE_RATIO,
                noise_clip=NOISE_CLIP,
                noise_mask=STATE_NOISE_MASK
            )
        else:
            noisy_state = state
        
        # 状态转换为字符串并tokenize - 与作者一致
        state_str = array2str(noisy_state)
        token = tokenizer(state_str, add_special_tokens=True, max_length=TOKEN_SIZE, 
                         truncation=True, padding='max_length', return_tensors='pt')
        input_ids = token["input_ids"].to(device)
        attention_mask = token["attention_mask"].to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(input_ids, attention_mask)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection - 与作者一致
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        【新增】在训练时对批量状态添加噪声
        """
        states, actions, rewards, next_states, dones = experiences

        # 【新增】对批量状态添加噪声增强 - 提高鲁棒性
        if USE_NOISE_AUGMENTATION:
            noisy_states = add_batch_noise(
                states,
                noise_type=NOISE_TYPE,
                noise_ratio=NOISE_RATIO,
                noise_clip=NOISE_CLIP,
                noise_mask=STATE_NOISE_MASK
            )
            noisy_next_states = add_batch_noise(
                next_states,
                noise_type=NOISE_TYPE,
                noise_ratio=NOISE_RATIO,
                noise_clip=NOISE_CLIP,
                noise_mask=STATE_NOISE_MASK
            )
        else:
            noisy_states = states
            noisy_next_states = next_states

        # 状态字符串转换和tokenization - 与作者一致
        state_str_list = [array2str(state) for state in noisy_states]
        token = tokenizer(state_str_list, add_special_tokens=True, max_length=TOKEN_SIZE,
                         truncation=True, padding='max_length', return_tensors='pt')
        input_ids_batch = token['input_ids'].to(device)
        attention_mask_batch = token['attention_mask'].to(device)

        next_state_str_list = [array2str(next_state) for next_state in noisy_next_states]
        next_token = tokenizer(next_state_str_list, add_special_tokens=True, max_length=TOKEN_SIZE,
                              truncation=True, padding='max_length', return_tensors='pt')
        next_input_ids_batch = next_token["input_ids"].to(device)
        next_attention_mask_batch = next_token["attention_mask"].to(device)

        # 计算目标Q值 - 与作者一致
        Q_targets_next = self.qnetwork_target(next_input_ids_batch, next_attention_mask_batch).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 计算当前Q值 - 与作者一致
        Q_expected = self.qnetwork_local(input_ids_batch, attention_mask_batch).gather(1, actions)

        # 计算损失并反向传播 - 与作者一致
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        
        # 关键技巧：梯度裁剪 (-1, 1) - 与作者一致
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 关键技巧：硬更新目标网络 - 每TAU*UPDATE_EVERY步更新 - 与作者一致
        if self.t_step % (TAU * UPDATE_EVERY) == 0:
            self.update_target_net(self.qnetwork_local, self.qnetwork_target)

    def update_target_net(self, local_model, target_model):
        """硬更新目标网络 - 与作者一致"""
        target_model.load_state_dict(local_model.state_dict())
    
    def get_exploration_coverage(self):
        """获取当前探索覆盖率"""
        return self.rnd.get_exploration_coverage()
    
    def save(self, name, output_dir='/home/wuyang/checkpoints/bert_dqn_checkpoints/0417'):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), f'{output_dir}/model{name}.pth')
        print(f'Model saved as {output_dir}/model{name}.pth')
        return


# ==================== 结果保存函数 ====================
def save_results(metrics_tracker, output_dir='/home/wuyang/results/bert_dqn_results/0417'):
    """保存所有结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 保存Excel文件
    df = pd.DataFrame({
        'Episode': list(range(1, len(metrics_tracker.scores) + 1)),
        'Score': metrics_tracker.scores,
        'Yield_kg_ha': metrics_tracker.yield_list,
        'Irrigation_mm': metrics_tracker.irrigation_list,
        'Fertilizer_kg_ha': metrics_tracker.fertilizer_list,
        'WUE_kg_mm': metrics_tracker.wue_list,
        'NUE_kg_kg': metrics_tracker.nue_list,
        'Episode_Steps': metrics_tracker.episode_steps,
        'Exploration_Coverage_percent': metrics_tracker.exploration_coverage_list
    })
    excel_path = os.path.join(output_dir, f'training_results_{timestamp}.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Excel results saved to: {excel_path}")
    
    # 2. 保存汇总指标
    summary = metrics_tracker.get_summary_metrics()
    summary['noise_augmentation'] = USE_NOISE_AUGMENTATION
    summary['noise_type'] = NOISE_TYPE
    summary['noise_ratio'] = NOISE_RATIO
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f'summary_metrics_{timestamp}.xlsx')
    summary_df.to_excel(summary_path, index=False)
    print(f"Summary metrics saved to: {summary_path}")
    
    # 3. 绘制并保存图表
    plot_and_save_figures(metrics_tracker, output_dir, timestamp)
    
    return excel_path, summary_path


def plot_and_save_figures(metrics_tracker, output_dir, timestamp):
    """绘制并保存所有图表"""
    episodes = list(range(1, len(metrics_tracker.scores) + 1))
    
    # 创建大型综合图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 训练得分
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(episodes, metrics_tracker.scores, 'b-', linewidth=0.8)
    ax1.set_title('Training Score', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. 最终产量
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(episodes, metrics_tracker.yield_list, 'g-', linewidth=0.8)
    ax2.set_title('Final Yield (kg/ha)', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Yield (kg/ha)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 灌溉量
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(episodes, metrics_tracker.irrigation_list, 'c-', linewidth=0.8)
    ax3.set_title('Irrigation Amount (mm)', fontsize=12)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Irrigation (mm)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 施肥量
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(episodes, metrics_tracker.fertilizer_list, 'm-', linewidth=0.8)
    ax4.set_title('Fertilizer Amount (kg/ha)', fontsize=12)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Fertilizer (kg/ha)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 水分利用率 (WUE)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(episodes, metrics_tracker.wue_list, 'r-', linewidth=0.8)
    ax5.set_title('Water Use Efficiency (WUE) (kg/mm)', fontsize=12)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('WUE (kg/mm)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 氮肥利用率 (NUE)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(episodes, metrics_tracker.nue_list, 'orange', linewidth=0.8)
    ax6.set_title('Nitrogen Use Efficiency (NUE) (kg/kg)', fontsize=12)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('NUE (kg/kg)')
    ax6.grid(True, alpha=0.3)
    
    # 7. 探索覆盖率
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(episodes, metrics_tracker.exploration_coverage_list, 'purple', linewidth=0.8)
    ax7.set_title('Exploration Coverage (%)', fontsize=12)
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Coverage (%)')
    ax7.grid(True, alpha=0.3)
    
    # 8. 每轮步数
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(episodes, metrics_tracker.episode_steps, 'brown', linewidth=0.8)
    ax8.set_title('Steps per Episode', fontsize=12)
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Steps')
    ax8.grid(True, alpha=0.3)
    
    # 9. 得分移动平均
    ax9 = fig.add_subplot(3, 3, 9)
    window = min(50, len(metrics_tracker.scores))
    if window > 1:
        moving_avg = pd.Series(metrics_tracker.scores).rolling(window=window).mean()
        ax9.plot(episodes, moving_avg, 'b-', linewidth=1.5, label=f'{window}-Episode Moving Avg')
    ax9.plot(episodes, metrics_tracker.scores, 'b-', linewidth=0.3, alpha=0.5, label='Raw Score')
    ax9.axhline(y=TARGET_PERFORMANCE_SCORE, color='r', linestyle='--', label=f'Target: {TARGET_PERFORMANCE_SCORE}')
    ax9.set_title('Score with Moving Average', fontsize=12)
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Score')
    ax9.legend(loc='upper left', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存PNG
    png_path = os.path.join(output_dir, f'training_plots_{timestamp}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"PNG plots saved to: {png_path}")
    
    # 保存PDF
    pdf_path = os.path.join(output_dir, f'training_plots_{timestamp}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF plots saved to: {pdf_path}")
    
    plt.close()
    
    # 创建农学指标图表
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # 产量分布
    axes2[0, 0].hist(metrics_tracker.yield_list, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes2[0, 0].set_title('Yield Distribution')
    axes2[0, 0].set_xlabel('Yield (kg/ha)')
    axes2[0, 0].set_ylabel('Frequency')
    
    # 灌溉量分布
    axes2[0, 1].hist(metrics_tracker.irrigation_list, bins=30, color='cyan', edgecolor='black', alpha=0.7)
    axes2[0, 1].set_title('Irrigation Distribution')
    axes2[0, 1].set_xlabel('Irrigation (mm)')
    axes2[0, 1].set_ylabel('Frequency')
    
    # 施肥量分布
    axes2[0, 2].hist(metrics_tracker.fertilizer_list, bins=30, color='magenta', edgecolor='black', alpha=0.7)
    axes2[0, 2].set_title('Fertilizer Distribution')
    axes2[0, 2].set_xlabel('Fertilizer (kg/ha)')
    axes2[0, 2].set_ylabel('Frequency')
    
    # WUE趋势
    if len(metrics_tracker.wue_list) > 50:
        wue_ma = pd.Series(metrics_tracker.wue_list).rolling(window=50).mean()
        axes2[1, 0].plot(episodes, wue_ma, 'r-', linewidth=1.5)
    axes2[1, 0].set_title('WUE Trend (50-Episode MA)')
    axes2[1, 0].set_xlabel('Episode')
    axes2[1, 0].set_ylabel('WUE (kg/mm)')
    
    # NUE趋势
    if len(metrics_tracker.nue_list) > 50:
        nue_ma = pd.Series(metrics_tracker.nue_list).rolling(window=50).mean()
        axes2[1, 1].plot(episodes, nue_ma, 'orange', linewidth=1.5)
    axes2[1, 1].set_title('NUE Trend (50-Episode MA)')
    axes2[1, 1].set_xlabel('Episode')
    axes2[1, 1].set_ylabel('NUE (kg/kg)')
    
    # 产量vs资源使用散点图
    scatter = axes2[1, 2].scatter(metrics_tracker.irrigation_list, metrics_tracker.yield_list, 
                                   c=metrics_tracker.fertilizer_list, cmap='viridis', alpha=0.6, s=10)
    axes2[1, 2].set_title('Yield vs Irrigation (color: Fertilizer)')
    axes2[1, 2].set_xlabel('Irrigation (mm)')
    axes2[1, 2].set_ylabel('Yield (kg/ha)')
    plt.colorbar(scatter, ax=axes2[1, 2], label='Fertilizer (kg/ha)')
    
    plt.tight_layout()
    
    agronomy_png = os.path.join(output_dir, f'agronomy_metrics_{timestamp}.png')
    plt.savefig(agronomy_png, dpi=150, bbox_inches='tight')
    print(f"Agronomy PNG saved to: {agronomy_png}")
    
    agronomy_pdf = os.path.join(output_dir, f'agronomy_metrics_{timestamp}.pdf')
    plt.savefig(agronomy_pdf, format='pdf', bbox_inches='tight')
    print(f"Agronomy PDF saved to: {agronomy_pdf}")
    
    plt.close()


def print_metrics_report(metrics_tracker, episode=None, is_final=False):
    """打印指标报告"""
    if is_final:
        print("\n" + "="*80)
        print("FINAL TRAINING METRICS REPORT")
        print("="*80)
    else:
        print(f"\n--- Metrics Report (Episode {episode}) ---")
    
    # 农学指标
    print("\n[ Agronomic Metrics ]")
    if is_final:
        print(f"  Final Yield (avg):           {np.mean(metrics_tracker.yield_list):.2f} kg/ha")
        print(f"  Final Yield (max):           {np.max(metrics_tracker.yield_list):.2f} kg/ha")
        print(f"  Irrigation (avg):            {np.mean(metrics_tracker.irrigation_list):.2f} mm")
        print(f"  Fertilizer (avg):            {np.mean(metrics_tracker.fertilizer_list):.2f} kg/ha")
        print(f"  WUE (avg):                   {np.mean(metrics_tracker.wue_list):.4f} kg/mm")
        print(f"  NUE (avg):                   {np.mean(metrics_tracker.nue_list):.4f} kg/kg")
    else:
        idx = episode - 1
        print(f"  Final Yield:                 {metrics_tracker.yield_list[idx]:.2f} kg/ha")
        print(f"  Irrigation:                  {metrics_tracker.irrigation_list[idx]:.2f} mm")
        print(f"  Fertilizer:                  {metrics_tracker.fertilizer_list[idx]:.2f} kg/ha")
        print(f"  WUE:                         {metrics_tracker.wue_list[idx]:.4f} kg/mm")
        print(f"  NUE:                         {metrics_tracker.nue_list[idx]:.4f} kg/kg")
    
    # AI指标
    print("\n[ AI Metrics ]")
    if is_final:
        print(f"  Average Return:              {np.mean(metrics_tracker.scores):.2f}")
        print(f"  Max Return:                  {np.max(metrics_tracker.scores):.2f}")
        print(f"  Sample Efficiency:           {metrics_tracker.sample_efficiency_step if metrics_tracker.sample_efficiency_step else 'Target not achieved'}")
        print(f"  Convergence Speed:           {metrics_tracker.convergence_step if metrics_tracker.convergence_step else 'Not converged'}")
        print(f"  Exploration Coverage (avg):  {np.mean(metrics_tracker.exploration_coverage_list):.2f}%")
        print(f"  Exploration Coverage (final):{metrics_tracker.exploration_coverage_list[-1]:.2f}%")
        print(f"  Total Episodes:              {len(metrics_tracker.scores)}")
        print(f"  Total Steps:                 {metrics_tracker.total_steps}")
        
        # 打印噪声增强配置
        print("\n[ Noise Augmentation Config ]")
        print(f"  Enabled:                     {USE_NOISE_AUGMENTATION}")
        print(f"  Noise Type:                  {NOISE_TYPE}")
        print(f"  Noise Ratio:                 {NOISE_RATIO}")
        print(f"  Noise Clip:                  {NOISE_CLIP}")
    else:
        idx = episode - 1
        print(f"  Score:                       {metrics_tracker.scores[idx]:.2f}")
        print(f"  Exploration Coverage:        {metrics_tracker.exploration_coverage_list[idx]:.2f}%")
        print(f"  Total Steps:                 {metrics_tracker.total_steps}")
        
        if metrics_tracker.sample_efficiency_step:
            print(f"  Sample Efficiency:           {metrics_tracker.sample_efficiency_step} steps (achieved)")
        else:
            print(f"  Sample Efficiency:           Target not yet achieved")
        
        if metrics_tracker.convergence_step:
            print(f"  Convergence:                 Converged at step {metrics_tracker.convergence_step}")
    
    if is_final:
        print("\n" + "="*80)


# ==================== 主训练函数 ====================
def dqn_train():
    """DQN训练主函数 - 与作者实现一致"""
    print("\n" + "="*60)
    print("DQN+BERT+gym-dssat Training - GPU Optimized Version")
    print("Reference: jingwu6/LM_AG, arXiv:2403.19839")
    print("="*60)
    
    print(f"\n[ Noise Augmentation: {USE_NOISE_AUGMENTATION} ]")
    if USE_NOISE_AUGMENTATION:
        print(f"  - Noise Type: {NOISE_TYPE}")
        print(f"  - Noise Ratio: {NOISE_RATIO}")
        print(f"  - Noise Clip: {NOISE_CLIP}")
    
    print("\nInitializing environment...")
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/results/bert_dqn_results/logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True,
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("Environment initialized successfully")
    except Exception as e:
        print(f"Environment initialization failed: {e}")
        return

    agent = Agent(STATE_SIZE, ACTION_SIZE)
    print("Agent initialized successfully")
    
    # 初始化指标追踪器
    metrics_tracker = MetricsTracker()

    # 探索参数 - 与作者一致
    eps = EPS_START
    eps_decay = EPS_DECAY
    eps_end = EPS_END
    
    # 奖励函数系数 - 与作者一致
    k1, k2, k3, k4 = 0.158, 0.79, 1.1, 0

    print(f"\nStarting training ({N_EPISODES} episodes)...")
    print(f"Target performance score: {TARGET_PERFORMANCE_SCORE}")
    print("-"*60)
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 使用tqdm进度条
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
        steps = 0
        
        for t in range(MAX_T):
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
            steps += 1
            
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
        
        # 获取探索覆盖率
        exploration_coverage = agent.get_exploration_coverage()
        
        # 更新指标
        wue, nue = metrics_tracker.update_agronomic_metrics(y, w_amount, n_amount)
        metrics_tracker.update_ai_metrics(score, steps, exploration_coverage)
        
        # 保存表现良好的模型 - 与作者一致
        if score > 1400 and y > 11000:
            agent.save(str(i_episode))
        
        # Epsilon衰减 - 与作者一致
        eps = max(eps_end, eps_decay * eps)
        
        # 更新进度条描述
        episode_time = time.time() - episode_start_time
        pbar.set_postfix({
            'Score': f'{score:.1f}',
            'Yield': f'{y:.0f}',
            'Eps': f'{eps:.3f}',
            'Time': f'{episode_time:.1f}s'
        })
        
        # 每10个episode打印详细指标
        if i_episode % 10 == 0:
            print_metrics_report(metrics_tracker, episode=i_episode, is_final=False)
    
    pbar.close()
    
    # 计算总训练时间
    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time/3600:.2f} hours ({total_training_time:.2f} seconds)")
    
    # 打印最终报告
    print_metrics_report(metrics_tracker, is_final=True)
    
    # 保存结果
    print("\nSaving results...")
    excel_path, summary_path = save_results(metrics_tracker, output_dir='/home/wuyang/results/bert_dqn_results/0417')
    
    # 关闭环境
    env.close()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    return metrics_tracker


# ==================== 主入口 ====================
if __name__ == "__main__":
    metrics = dqn_train()