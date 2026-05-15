#!/usr/bin/env python3
"""
纯数值输入DQN基准模型 - 论文经典基准
=====================================
功能特点：
1. 纯数值状态输入（无BERT/LLM）
2. 标准DQN架构作为经典强化学习基准
3. 离散动作空间处理
4. 完整的农学指标与AI指标输出

农学指标：
- 最终产量 (Yield, kg/ha)
- 灌溉量 (Irrigation, L/ha)
- 施肥量 (N fertilizer, kg/ha)
- 水分利用率 WUE (Water Use Efficiency, kg/m³)
- 氮肥利用率 NUE (Nitrogen Use Efficiency, kg/kg)

AI指标：
- 平均回报 (Average Return)
- 样本效率 (Sample Efficiency)
- 收敛速度 (Convergence Speed)
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
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局配置 ====================
# 设备配置 - 自动检测并使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ==================== 超参数配置 ====================
# 经验回放参数
BUFFER_SIZE = int(1e5)      # 经验回放缓冲区大小
BATCH_SIZE = 512             # 批量大小（经典DQN推荐值）

# 学习参数
GAMMA = 0.99                # 折扣因子
TAU = 1e-3                  # 软更新参数 (用于目标网络软更新)
LR = 1e-4                   # 学习率
UPDATE_EVERY = 4            # 每隔多少步更新一次网络

# 训练参数
N_EPISODES = 2000           # 训练轮次
MAX_STEPS = 200             # 每轮最大步数

# 状态与动作空间维度
STATE_SIZE = 25             # 状态维度
ACTION_SIZE = 25            # 离散动作数量 (5个施肥级别 × 5个灌溉级别)

# 探索参数
EPS_START = 1.0             # 初始探索率
EPS_END = 0.01              # 最终探索率
EPS_DECAY = 0.995           # 探索率衰减

# 优化器参数
BETAS = (0.9, 0.999)        # Adam优化器beta参数
WEIGHT_DECAY = 1e-5         # 权重衰减（L2正则化）

# 奖励函数系数
K1 = 0.158                  # 产量奖励系数
K2 = 0.79                   # 氮肥成本系数
K3 = 1.1                    # 灌溉成本系数
K4 = 0                      # 额外惩罚项

# 结果保存路径
RESULTS_DIR = '/home/wuyang/results/only_dqn_results/0410'
CHECKPOINT_DIR = '/home/wuyang/checkpoints/only_dqn_checkpoint'

# ==================== 辅助函数 ====================
def dict2array(state):
    """
    将字典状态转换为numpy数组
    处理gym-dssat返回的状态字典
    """
    new_state = []
    for key in state.keys():
        if key != 'sw':
            new_state.append(state[key])
        else:
            new_state += list(state['sw'])
    return np.asarray(new_state, dtype=np.float32)


def get_reward(state, n_action, w_action, next_state, done, k1=K1, k2=K2, k3=K3, k4=K4):
    """
    计算奖励值
    
    奖励函数设计：
    - 终止状态：产量奖励 - 氮肥成本 - 灌溉成本
    - 非终止状态：- 氮肥成本 - 灌溉成本
    
    参数:
        state: 当前状态
        n_action: 氮肥施用量 (kg/ha)
        w_action: 灌溉量 (L/ha)
        next_state: 下一状态
        done: 是否终止
        k1-k4: 奖励系数
    
    返回:
        reward: 奖励值
    """
    if done:
        # 终止状态：考虑最终产量
        return k1 * state[4] - k2 * n_action - k3 * w_action
    else:
        # 非终止状态：仅考虑成本
        return -k2 * n_action - k3 * w_action


def action_to_management(action_id):
    """
    将离散动作ID转换为管理决策
    
    动作空间设计：
    - 动作ID: 0-24 (共25个离散动作)
    - 氮肥级别: 5个 (0, 40, 80, 120, 160 kg/ha)
    - 灌溉级别: 5个 (0, 6, 12, 18, 24 L/ha)
    
    映射关系:
        action_id = n_level * 5 + w_level
        n_level = action_id // 5
        w_level = action_id % 5
    
    返回:
        dict: {'anfer': 氮肥量, 'amir': 灌溉量}
    """
    n_level = action_id % 5
    w_level = action_id // 5
    
    return {
        'anfer': n_level * 40,      # 氮肥: 0, 40, 80, 120, 160 kg/ha
        'amir': w_level * 6,        # 灌溉: 0, 6, 12, 18, 24 L/ha
    }


def calculate_wue(yield_kg, water_L):
    """
    计算水分利用率 (Water Use Efficiency)
    WUE = 产量 / 总用水量 (kg/m³)
    注意：需要将灌溉量从L转换为m³
    """
    if water_L <= 0:
        return 0.0
    water_m3 = water_L / 1000  # L转m³
    return yield_kg / water_m3


def calculate_nue(yield_kg, n_amount):
    """
    计算氮肥利用率 (Nitrogen Use Efficiency)
    NUE = 产量 / 氮肥施用量 (kg/kg)
    """
    if n_amount <= 0:
        return 0.0
    return yield_kg / n_amount


# ==================== Q网络定义 ====================
class QNetwork(nn.Module):
    """
    标准DQN Q网络
    
    架构说明：
    - 输入层：状态向量 (STATE_SIZE维)
    - 隐藏层1：128个神经元 + ReLU
    - 隐藏层2：128个神经元 + ReLU
    - 输出层：各动作的Q值 (ACTION_SIZE维)
    
    这是最经典的DQN网络结构，作为论文基准。
    """
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        """
        初始化Q网络
        
        参数:
            state_size: 状态空间维度
            action_size: 动作空间维度
            hidden_sizes: 隐藏层神经元数量列表
        """
        super(QNetwork, self).__init__()
        
        # 构建网络层
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 状态张量 [batch_size, state_size]
        
        返回:
            Q值张量 [batch_size, action_size]
        """
        return self.network(state)


# ==================== 经验回放缓冲区 ====================
class ReplayBuffer:
    """
    经验回放缓冲区 (Experience Replay Buffer)
    
    功能：
    - 存储智能体与环境交互的经验
    - 提供随机采样功能，打破数据相关性
    - 支持优先经验回放扩展
    """
    
    def __init__(self, action_size, buffer_size, batch_size):
        """
        初始化经验回放缓冲区
        
        参数:
            action_size: 动作空间维度
            buffer_size: 缓冲区最大容量
            batch_size: 采样批量大小
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", 
            field_names=["state", "action", "reward", "next_state", "done"]
        )
    
    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        随机采样一批经验
        
        返回:
            tuple: (states, actions, rewards, next_states, dones)
                   所有元素都是PyTorch张量，已移至GPU
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # 转换为张量
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)
        
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.memory)


# ==================== DQN智能体 ====================
class DQNAgent:
    """
    DQN智能体
    
    实现功能：
    1. ε-贪婪策略探索
    2. 经验回放学习
    3. 目标网络软更新
    4. 模型保存与加载
    
    作为论文的经典基准算法。
    """
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], 
                 double_dqn=False, dueling=False):
        """
        初始化DQN智能体
        
        参数:
            state_size: 状态空间维度
            action_size: 动作空间维度
            hidden_sizes: 隐藏层大小
            double_dqn: 是否使用Double DQN
            dueling: 是否使用Dueling网络架构
        """
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn
        self.dueling = dueling
        
        # Q网络（局部网络和目标网络）
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_sizes).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_sizes).to(device)
        
        # 复制权重到目标网络
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), 
            lr=LR, 
            betas=BETAS, 
            weight_decay=WEIGHT_DECAY
        )
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        
        # 计步器
        self.t_step = 0
        
        # 学习步数计数器（用于AI指标）
        self.learning_steps = 0
        
        print(f"DQN智能体初始化完成:")
        print(f"  - 状态维度: {state_size}")
        print(f"  - 动作数量: {action_size}")
        print(f"  - 网络结构: {hidden_sizes}")
        print(f"  - 参数总量: {sum(p.numel() for p in self.qnetwork_local.parameters()):,}")
    
    def step(self, state, action, reward, next_state, done):
        """
        执行一步交互并学习
        
        参数:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 保存经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 终止状态额外保存（增加稀疏奖励的重要性）
        if done:
            for _ in range(3):
                self.memory.add(state, action, reward, next_state, done)
        
        # 更新计数器
        self.t_step += 1
        
        # 定期学习
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.0):
        """
        根据当前策略选择动作
        
        参数:
            state: 当前状态
            eps: 探索率 (ε-贪婪策略)
        
        返回:
            int: 选择的动作ID
        """
        # 转换为张量
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # 评估模式
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        # ε-贪婪策略
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """
        从经验中学习
        
        参数:
            experiences: 采样的经验批次
            gamma: 折扣因子
        """
        states, actions, rewards, next_states, dones = experiences
        
        # 计算目标Q值
        if self.double_dqn:
            # Double DQN: 使用local网络选择动作，target网络评估
            next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        else:
            # 标准DQN: 直接使用target网络的最大Q值
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 计算TD目标
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 计算当前Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # 计算损失
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # 更新学习步数
        self.learning_steps += 1
        
        # 软更新目标网络
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """
        软更新目标网络参数
        θ_target = τ * θ_local + (1 - τ) * θ_target
        
        参数:
            local_model: 源网络
            target_model: 目标网络
            tau: 软更新系数
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
    
    def save(self, filename):
        """保存模型"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        filepath = os.path.join(CHECKPOINT_DIR, f'{filename}.pth')
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_steps': self.learning_steps,
        }, filepath)
        print(f'✓ 模型已保存: {filepath}')
    
    def load(self, filename):
        """加载模型"""
        filepath = os.path.join(CHECKPOINT_DIR, f'{filename}.pth')
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.learning_steps = checkpoint.get('learning_steps', 0)
            print(f'✓ 模型已加载: {filepath}')
            return True
        return False


# ==================== 指标记录器 ====================
class MetricsLogger:
    """
    指标记录器
    
    记录并计算农学指标和AI指标
    """
    
    def __init__(self):
        """初始化记录器"""
        # 农学指标
        self.yields = []            # 产量列表
        self.n_amounts = []         # 氮肥使用量列表
        self.w_amounts = []         # 灌溉量列表
        self.wues = []              # 水分利用率列表
        self.nues = []              # 氮肥利用率列表
        
        # AI指标
        self.returns = []           # 累计回报列表
        self.episode_lengths = []   # 每轮步数
        self.losses = []            # 损失值列表
        self.epsilons = []          # 探索率列表
        
        # 样本效率追踪
        self.total_steps = 0        # 总步数
        self.total_learning_steps = 0  # 总学习步数
        
        # 收敛追踪
        self.best_avg_return = -float('inf')
        self.convergence_episode = None
        self.convergence_threshold = 0.95  # 收敛判定阈值
    
    def log_episode(self, yield_val, n_amount, w_amount, total_reward, 
                    episode_length, epsilon):
        """
        记录一个episode的指标
        """
        self.yields.append(yield_val)
        self.n_amounts.append(n_amount)
        self.w_amounts.append(w_amount)
        self.returns.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.epsilons.append(epsilon)
        
        # 计算WUE和NUE
        wue = calculate_wue(yield_val, w_amount)
        nue = calculate_nue(yield_val, n_amount)
        self.wues.append(wue)
        self.nues.append(nue)
    
    def log_loss(self, loss):
        """记录损失值"""
        self.losses.append(loss)
    
    def add_steps(self, steps, learning_steps):
        """增加步数计数"""
        self.total_steps += steps
        self.total_learning_steps += learning_steps
    
    def get_agronomic_metrics(self, last_n=10):
        """
        获取农学指标（最近N轮）
        """
        if len(self.yields) < last_n:
            last_n = len(self.yields)
        
        if last_n == 0:
            return {}
        
        return {
            'avg_yield': np.mean(self.yields[-last_n:]),
            'std_yield': np.std(self.yields[-last_n:]),
            'max_yield': np.max(self.yields[-last_n:]),
            'avg_n_amount': np.mean(self.n_amounts[-last_n:]),
            'avg_w_amount': np.mean(self.w_amounts[-last_n:]),
            'avg_wue': np.mean(self.wues[-last_n:]),
            'avg_nue': np.mean(self.nues[-last_n:]),
        }
    
    def get_ai_metrics(self, last_n=10):
        """
        获取AI指标（最近N轮）
        """
        if len(self.returns) < last_n:
            last_n = len(self.returns)
        
        if last_n == 0:
            return {}
        
        avg_return = np.mean(self.returns[-last_n:])
        
        # 检查收敛
        if self.convergence_episode is None and len(self.returns) >= 100:
            recent_avg = np.mean(self.returns[-100:])
            if recent_avg >= self.best_avg_return * self.convergence_threshold:
                self.convergence_episode = len(self.returns)
            if recent_avg > self.best_avg_return:
                self.best_avg_return = recent_avg
        
        # 样本效率：达到某性能水平所需样本数
        # 这里简化为每1000步的平均回报
        sample_efficiency = np.mean(self.returns[:min(1000, len(self.returns))]) if len(self.returns) > 0 else 0
        
        return {
            'avg_return': avg_return,
            'std_return': np.std(self.returns[-last_n:]),
            'max_return': np.max(self.returns[-last_n:]),
            'sample_efficiency': sample_efficiency,
            'convergence_episode': self.convergence_episode,
            'avg_episode_length': np.mean(self.episode_lengths[-last_n:]),
            'total_steps': self.total_steps,
        }
    
    def get_overall_metrics(self):
        """获取整体指标"""
        return {
            'agronomic': {
                'avg_yield': np.mean(self.yields) if self.yields else 0,
                'max_yield': np.max(self.yields) if self.yields else 0,
                'total_n': np.sum(self.n_amounts),
                'total_w': np.sum(self.w_amounts),
                'avg_wue': np.mean(self.wues) if self.wues else 0,
                'avg_nue': np.mean(self.nues) if self.nues else 0,
            },
            'ai': {
                'avg_return': np.mean(self.returns) if self.returns else 0,
                'max_return': np.max(self.returns) if self.returns else 0,
                'final_avg_return': np.mean(self.returns[-100:]) if len(self.returns) >= 100 else np.mean(self.returns),
                'total_episodes': len(self.returns),
                'total_steps': self.total_steps,
                'convergence_episode': self.convergence_episode,
            }
        }


# ==================== 训练函数 ====================
def train_dqn():
    """
    主训练函数
    """
    print("=" * 60)
    print("纯数值输入DQN基准模型 - 开始训练")
    print("=" * 60)
    
    # 创建目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 初始化环境
    print("\n[1] 初始化环境...")
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/data/logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': True,
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("✓ 环境初始化成功")
    except Exception as e:
        print(f"✗ 环境初始化失败: {e}")
        print("  请确保gym-dssat已正确安装")
        return None
    
    # 初始化智能体
    print("\n[2] 初始化DQN智能体...")
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    
    # 初始化指标记录器
    logger = MetricsLogger()
    
    # 探索参数
    eps = EPS_START
    
    print(f"\n[3] 开始训练 (共 {N_EPISODES} 轮)...")
    print("-" * 60)
    
    # 训练开始时间
    train_start_time = time.time()
    
    # 训练循环
    for i_episode in tqdm(range(1, N_EPISODES + 1), desc="训练进度"):
        episode_start_time = time.time()
        
        # 重置环境
        state = env.reset()
        state = dict2array(state)
        
        # Episode累计值
        score = 0
        n_amount = 0
        w_amount = 0
        yield_val = 0
        episode_steps = 0
        
        # Episode循环
        for t in range(MAX_STEPS):
            # 选择动作
            action_id = agent.act(state, eps)
            action = action_to_management(action_id)
            
            # 应用约束（防止过量施肥/灌溉）
            if state[0] >= 10000:  # 累积氮肥过高
                action['anfer'] = 0
            if state[21] >= 1600:  # 土壤水分过高
                action['amir'] = 0
            
            # 执行动作
            next_state, _, done, _ = env.step(action)
            
            episode_steps += 1
            
            # 处理终止状态
            if done:
                yield_val = state[4]
                next_state = state
                reward = get_reward(state, action['anfer'], action['amir'], 
                                   next_state, done)
                agent.step(state, action_id, reward, next_state, done)
                score += reward
                break
            
            # 累计资源使用
            n_amount += action['anfer']
            w_amount += action['amir']
            
            # 转换下一状态
            next_state = dict2array(next_state)
            
            # 计算奖励
            reward = get_reward(state, action['anfer'], action['amir'], 
                               next_state, done)
            
            # 学习
            agent.step(state, action_id, reward, next_state, done)
            
            # 更新状态
            state = next_state
            score += reward
        
        # 记录指标
        logger.log_episode(yield_val, n_amount, w_amount, score, 
                          episode_steps, eps)
        logger.add_steps(episode_steps, agent.learning_steps)
        
        # 更新探索率
        eps = max(EPS_END, EPS_DECAY * eps)
        
        # 保存优秀模型
        if score > 1400 and yield_val > 11000:
            agent.save(f'best_model_ep{i_episode}')
        
        # 每10轮打印详细指标
        if i_episode % 10 == 0:
            episode_time = time.time() - episode_start_time
            elapsed_time = time.time() - train_start_time
            
            # 获取指标
            agronomic = logger.get_agronomic_metrics(10)
            ai_metrics = logger.get_ai_metrics(10)
            
            print(f"\n{'='*60}")
            print(f"第 {i_episode}/{N_EPISODES} 轮 [已用时: {elapsed_time/60:.1f}分钟]")
            print(f"{'='*60}")
            
            print("\n【农学指标】(最近10轮平均)")
            print(f"  • 产量: {agronomic['avg_yield']:.2f} ± {agronomic['std_yield']:.2f} kg/ha")
            print(f"  • 最高产量: {agronomic['max_yield']:.2f} kg/ha")
            print(f"  • 氮肥用量: {agronomic['avg_n_amount']:.2f} kg/ha")
            print(f"  • 灌溉量: {agronomic['avg_w_amount']:.2f} L/ha")
            print(f"  • 水分利用率(WUE): {agronomic['avg_wue']:.4f} kg/m³")
            print(f"  • 氮肥利用率(NUE): {agronomic['avg_nue']:.2f} kg/kg")
            
            print("\n【AI指标】(最近10轮)")
            print(f"  • 平均回报: {ai_metrics['avg_return']:.2f} ± {ai_metrics['std_return']:.2f}")
            print(f"  • 最高回报: {ai_metrics['max_return']:.2f}")
            print(f"  • 平均步数: {ai_metrics['avg_episode_length']:.1f}")
            print(f"  • 探索率: {eps:.4f}")
            print(f"  • 本轮用时: {episode_time:.2f}秒")
            
            if ai_metrics['convergence_episode']:
                print(f"  • 收敛轮次: {ai_metrics['convergence_episode']}")
        
        # 每100轮保存检查点
        if i_episode % 1000 == 0:
            agent.save(f'checkpoint_ep{i_episode}')
    
    # 训练结束
    total_time = time.time() - train_start_time
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    
    # 输出整体指标
    overall = logger.get_overall_metrics()
    
    print("\n【整体农学指标】")
    print(f"  • 平均产量: {overall['agronomic']['avg_yield']:.2f} kg/ha")
    print(f"  • 最高产量: {overall['agronomic']['max_yield']:.2f} kg/ha")
    print(f"  • 累计氮肥: {overall['agronomic']['total_n']:.2f} kg")
    print(f"  • 累计灌溉: {overall['agronomic']['total_w']:.2f} L")
    print(f"  • 平均WUE: {overall['agronomic']['avg_wue']:.4f} kg/m³")
    print(f"  • 平均NUE: {overall['agronomic']['avg_nue']:.2f} kg/kg")
    
    print("\n【整体AI指标】")
    print(f"  • 总轮次: {overall['ai']['total_episodes']}")
    print(f"  • 总步数: {overall['ai']['total_steps']}")
    print(f"  • 平均回报: {overall['ai']['avg_return']:.2f}")
    print(f"  • 最终平均回报(后100轮): {overall['ai']['final_avg_return']:.2f}")
    print(f"  • 最高回报: {overall['ai']['max_return']:.2f}")
    if overall['ai']['convergence_episode']:
        print(f"  • 收敛轮次: {overall['ai']['convergence_episode']}")
    print(f"  • 总训练时间: {total_time/60:.1f} 分钟")
    
    # 保存结果
    save_results(logger, agent)
    
    # 关闭环境
    env.close()
    
    return logger, agent


def save_results(logger, agent):
    """
    保存训练结果
    """
    print("\n[4] 保存结果...")
    
    # 保存详细数据到Excel
    df = pd.DataFrame({
        'Episode': range(1, len(logger.yields) + 1),
        'Yield': logger.yields,
        'N_Amount': logger.n_amounts,
        'W_Amount': logger.w_amounts,
        'WUE': logger.wues,
        'NUE': logger.nues,
        'Return': logger.returns,
        'Episode_Length': logger.episode_lengths,
        'Epsilon': logger.epsilons,
    })
    
    excel_path = os.path.join(RESULTS_DIR, 'dqn_baseline_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"✓ 数据已保存: {excel_path}")
    
    # 保存最终模型
    agent.save('final_model')
    
    # 绘制训练曲线
    plot_training_curves(logger)


def plot_training_curves(logger):
    """
    绘制训练曲线
    """
    print("\n[5] 绘制训练曲线...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 回报曲线
    ax = axes[0, 0]
    ax.plot(logger.returns, alpha=0.6, label='回报')
    # 添加移动平均
    window = min(50, len(logger.returns) // 5)
    if window > 1:
        ma = np.convolve(logger.returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.returns)), ma, 'r-', linewidth=2, label=f'移动平均({window})')
    ax.set_xlabel('轮次')
    ax.set_ylabel('回报')
    ax.set_title('训练回报曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 产量曲线
    ax = axes[0, 1]
    ax.plot(logger.yields, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.yields, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.yields)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('产量 (kg/ha)')
    ax.set_title('产量变化曲线')
    ax.grid(True, alpha=0.3)
    
    # 3. 探索率曲线
    ax = axes[0, 2]
    ax.plot(logger.epsilons)
    ax.set_xlabel('轮次')
    ax.set_ylabel('探索率 (ε)')
    ax.set_title('探索率衰减曲线')
    ax.grid(True, alpha=0.3)
    
    # 4. 氮肥使用量
    ax = axes[1, 0]
    ax.plot(logger.n_amounts, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.n_amounts, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.n_amounts)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('氮肥 (kg/ha)')
    ax.set_title('氮肥使用量')
    ax.grid(True, alpha=0.3)
    
    # 5. 灌溉量
    ax = axes[1, 1]
    ax.plot(logger.w_amounts, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.w_amounts, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.w_amounts)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('灌溉 (L/ha)')
    ax.set_title('灌溉量')
    ax.grid(True, alpha=0.3)
    
    # 6. 水分利用率WUE
    ax = axes[1, 2]
    ax.plot(logger.wues, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.wues, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.wues)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('WUE (kg/m³)')
    ax.set_title('水分利用率(WUE)')
    ax.grid(True, alpha=0.3)
    
    # 7. 氮肥利用率NUE
    ax = axes[2, 0]
    ax.plot(logger.nues, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.nues, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.nues)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('NUE (kg/kg)')
    ax.set_title('氮肥利用率(NUE)')
    ax.grid(True, alpha=0.3)
    
    # 8. Episode长度
    ax = axes[2, 1]
    ax.plot(logger.episode_lengths, alpha=0.6)
    if window > 1:
        ma = np.convolve(logger.episode_lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(logger.episode_lengths)), ma, 'r-', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('步数')
    ax.set_title('每轮步数')
    ax.grid(True, alpha=0.3)
    
    # 9. 指标汇总表
    ax = axes[2, 2]
    ax.axis('off')
    
    overall = logger.get_overall_metrics()
    
    summary_text = """
    【最终指标汇总】
    
    农学指标:
    ─────────────────
    平均产量: {:.2f} kg/ha
    最高产量: {:.2f} kg/ha
    平均WUE: {:.4f} kg/m³
    平均NUE: {:.2f} kg/kg
    
    AI指标:
    ─────────────────
    平均回报: {:.2f}
    最终回报: {:.2f}
    总轮次: {}
    总步数: {}
    """.format(
        overall['agronomic']['avg_yield'],
        overall['agronomic']['max_yield'],
        overall['agronomic']['avg_wue'],
        overall['agronomic']['avg_nue'],
        overall['ai']['avg_return'],
        overall['ai']['final_avg_return'],
        overall['ai']['total_episodes'],
        overall['ai']['total_steps'],
    )
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(RESULTS_DIR, 'dqn_baseline_training_curves.pdf')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {plot_path}")
    
    # 同时保存PNG格式
    png_path = os.path.join(RESULTS_DIR, 'dqn_baseline_training_curves.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {png_path}")
    
    plt.close()


# ==================== 验证函数 ====================
def validate_agent(agent, n_episodes=100):
    """
    验证训练好的智能体
    
    参数:
        agent: 训练好的DQN智能体
        n_episodes: 验证轮数
    """
    print("\n" + "=" * 60)
    print("验证阶段")
    print("=" * 60)
    
    # 初始化环境
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': '/home/wuyang/data/logs/dssat-pdi-validation.log',
        'mode': 'all',
        'seed': 654321,  # 不同的种子用于验证
        'random_weather': True,
    }
    
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return
    
    # 验证指标记录
    val_logger = MetricsLogger()
    
    print(f"\n开始验证 ({n_episodes} 轮)...")
    
    for i_episode in tqdm(range(1, n_episodes + 1), desc="验证进度"):
        state = env.reset()
        state = dict2array(state)
        
        score = 0
        n_amount = 0
        w_amount = 0
        yield_val = 0
        episode_steps = 0
        
        for t in range(MAX_STEPS):
            # 使用贪婪策略（无探索）
            action_id = agent.act(state, eps=0.0)
            action = action_to_management(action_id)
            
            # 应用约束
            if state[0] >= 10000:
                action['anfer'] = 0
            if state[21] >= 1600:
                action['amir'] = 0
            
            next_state, _, done, _ = env.step(action)
            episode_steps += 1
            
            if done:
                yield_val = state[4]
                score += get_reward(state, action['anfer'], action['amir'], 
                                   state, done)
                break
            
            n_amount += action['anfer']
            w_amount += action['amir']
            next_state = dict2array(next_state)
            score += get_reward(state, action['anfer'], action['amir'], 
                               next_state, done)
            state = next_state
        
        val_logger.log_episode(yield_val, n_amount, w_amount, score, 
                               episode_steps, 0.0)
        
        # 每10轮输出
        if i_episode % 10 == 0:
            agronomic = val_logger.get_agronomic_metrics(10)
            ai_metrics = val_logger.get_ai_metrics(10)
            
            print(f"\n验证轮次 {i_episode}/{n_episodes}")
            print(f"  【农学】产量: {agronomic['avg_yield']:.2f} kg/ha, "
                  f"氮肥: {agronomic['avg_n_amount']:.2f} kg/ha, "
                  f"灌溉: {agronomic['avg_w_amount']:.2f} L/ha")
            print(f"  【农学】WUE: {agronomic['avg_wue']:.4f} kg/m³, "
                  f"NUE: {agronomic['avg_nue']:.2f} kg/kg")
            print(f"  【AI】平均回报: {ai_metrics['avg_return']:.2f}")
    
    # 输出验证结果汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    overall = val_logger.get_overall_metrics()
    
    print("\n【验证农学指标】")
    print(f"  • 平均产量: {overall['agronomic']['avg_yield']:.2f} kg/ha")
    print(f"  • 最高产量: {overall['agronomic']['max_yield']:.2f} kg/ha")
    print(f"  • 平均氮肥: {overall['agronomic']['avg_wue']:.4f} kg/m³")
    print(f"  • 平均NUE: {overall['agronomic']['avg_nue']:.2f} kg/kg")
    
    print("\n【验证AI指标】")
    print(f"  • 平均回报: {overall['ai']['avg_return']:.2f}")
    print(f"  • 最高回报: {overall['ai']['max_return']:.2f}")
    
    # 保存验证结果
    df = pd.DataFrame({
        'Episode': range(1, len(val_logger.yields) + 1),
        'Yield': val_logger.yields,
        'N_Amount': val_logger.n_amounts,
        'W_Amount': val_logger.w_amounts,
        'WUE': val_logger.wues,
        'NUE': val_logger.nues,
        'Return': val_logger.returns,
    })
    
    val_path = os.path.join(RESULTS_DIR, 'dqn_baseline_validation.xlsx')
    df.to_excel(val_path, index=False)
    print(f"\n✓ 验证结果已保存: {val_path}")
    
    env.close()
    
    return val_logger


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("纯数值输入DQN基准模型")
    print("论文经典基准 - 无预训练模型版本")
    print("=" * 60)
    
    # 打印配置信息
    print("\n【配置参数】")
    print(f"  • 设备: {device}")
    print(f"  • 训练轮次: {N_EPISODES}")
    print(f"  • 批量大小: {BATCH_SIZE}")
    print(f"  • 学习率: {LR}")
    print(f"  • 折扣因子: {GAMMA}")
    print(f"  • 状态维度: {STATE_SIZE}")
    print(f"  • 动作数量: {ACTION_SIZE}")
    print(f"  • 探索率范围: {EPS_START} -> {EPS_END}")
    
    # 执行训练
    logger, agent = train_dqn()
    
    # 验证
    if logger is not None:
        val_logger = validate_agent(agent, n_episodes=100)
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)