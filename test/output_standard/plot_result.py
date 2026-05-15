#!/usr/bin/env python3
"""
从训练日志生成可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import re
from typing import List, Dict

# ============================================================================
#                              配置
# ============================================================================

# 输出路径
OUTPUT_DIR = "/home/gymusr/only_ppo_results"
OUTPUT_PREFIX = "only_ppo"

# 奖励函数参数 (与原代码一致)
K1 = 0.158
K2 = 0.79
K3 = 1.1

# 专家水平阈值
EXPERT_PERFORMANCE_THRESHOLD = 500.0
CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.05

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
#                              数据解析
# ============================================================================

def parse_log_data(log_text: str) -> List[Dict]:
    """解析日志文本，提取训练指标"""
    pattern = r'Episode\s+(\d+)/\d+\s*\|\s*Score:\s*([-+]?\d*\.?\d+)\s*\|\s*Avg:\s*([-+]?\d*\.?\d+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*Yield:\s*([-+]?\d*\.?\d+)\s*kg/ha\s*\|\s*N:\s*([-+]?\d*\.?\d+)\s*kg/ha\s*\|\s*Water:\s*([-+]?\d*\.?\d+)\s*mm\s*\|\s*LR:\s*([-+]?\d*\.?\d+e[-+]?\d+)'
    
    records = []
    for match in re.finditer(pattern, log_text):
        record = {
            'episode': int(match.group(1)),
            'score': float(match.group(2)),
            'avg_reward': float(match.group(3)),
            'steps': int(match.group(4)),
            'yield_val': float(match.group(5)),
            'n_amount': float(match.group(6)),
            'water_amount': float(match.group(7)),
            'lr': float(match.group(8))
        }
        records.append(record)
    
    return records

def calculate_metrics(records: List[Dict]) -> pd.DataFrame:
    """计算衍生指标"""
    df = pd.DataFrame(records)
    
    # 计算WUE (水分利用效率) = 产量 / 灌溉量
    df['wue'] = df.apply(lambda x: x['yield_val'] / x['water_amount'] if x['water_amount'] > 0 else 0, axis=1)
    
    # 计算NUE (氮肥利用效率) = 产量 / 施肥量
    df['nue'] = df.apply(lambda x: x['yield_val'] / x['n_amount'] if x['n_amount'] > 0 else 0, axis=1)
    
    # 计算累积步数
    df['total_steps'] = df['steps'].cumsum()
    
    # 计算移动平均回报 (窗口=10)
    df['rolling_avg_reward'] = df['score'].rolling(window=10, min_periods=1).mean()
    
    # 计算样本效率和收敛速度
    sample_efficiency_episode = None
    convergence_episode = None
    
    # 样本效率：首次达到专家阈值
    for i, row in df.iterrows():
        if row['avg_reward'] >= EXPERT_PERFORMANCE_THRESHOLD:
            sample_efficiency_episode = row['episode']
            break
    
    # 收敛速度：最近50轮回报方差/均值 < 5%
    if len(df) >= CONVERGENCE_WINDOW:
        for i in range(CONVERGENCE_WINDOW, len(df)):
            window_rewards = df['score'].iloc[i-CONVERGENCE_WINDOW:i].values
            mean_r = np.mean(window_rewards)
            std_r = np.std(window_rewards)
            if mean_r > 0 and std_r / mean_r < CONVERGENCE_THRESHOLD:
                convergence_episode = df['episode'].iloc[i]
                break
    
    # 添加到DataFrame
    df['sample_efficiency_ep'] = sample_efficiency_episode
    df['convergence_ep'] = convergence_episode
    
    return df

# ============================================================================
#                              日志数据
# ============================================================================

LOG_DATA = """
Episode 1/2000 | Score: -8106 | Avg: -920 | Steps: 165 | Yield: 10293.5 kg/ha | N: 10080.0 kg/ha | Water: 1608.0 mm | LR: 1.00e+00
Episode 2/2000 | Score: -8261 | Avg: -920 | Steps: 162 | Yield: 9112.1 kg/ha | N: 10040.0 kg/ha | Water: 1608.0 mm | LR: 9.95e-01
Episode 3/2000 | Score: -7937 | Avg: -920 | Steps: 163 | Yield: 10962.5 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 9.90e-01
Episode 4/2000 | Score: -7994 | Avg: -920 | Steps: 157 | Yield: 11200.0 kg/ha | N: 10120.0 kg/ha | Water: 1608.0 mm | LR: 9.85e-01
Episode 5/2000 | Score: -7899 | Avg: -920 | Steps: 159 | Yield: 11400.6 kg/ha | N: 10040.0 kg/ha | Water: 1608.0 mm | LR: 9.80e-01
Episode 6/2000 | Score: -8118 | Avg: -920 | Steps: 164 | Yield: 10172.7 kg/ha | N: 10080.0 kg/ha | Water: 1602.0 mm | LR: 9.75e-01
Episode 7/2000 | Score: -7657 | Avg: -920 | Steps: 155 | Yield: 12893.7 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 9.70e-01
Episode 8/2000 | Score: -7877 | Avg: -920 | Steps: 160 | Yield: 11341.3 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 9.66e-01
Episode 9/2000 | Score: -8129 | Avg: -920 | Steps: 163 | Yield: 9830.2 kg/ha | N: 10000.0 kg/ha | Water: 1620.0 mm | LR: 9.61e-01
Episode 10/2000 | Score: -7964 | Avg: -920 | Steps: 164 | Yield: 10745.8 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 9.56e-01
Episode 11/2000 | Score: -7838 | Avg: -920 | Steps: 164 | Yield: 11985.9 kg/ha | N: 10080.0 kg/ha | Water: 1608.0 mm | LR: 9.51e-01
Episode 12/2000 | Score: -7829 | Avg: -920 | Steps: 158 | Yield: 11730.9 kg/ha | N: 10000.0 kg/ha | Water: 1620.0 mm | LR: 9.46e-01
Episode 13/2000 | Score: -7976 | Avg: -920 | Steps: 163 | Yield: 10713.8 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 9.42e-01
Episode 14/2000 | Score: -7953 | Avg: -920 | Steps: 163 | Yield: 11456.5 kg/ha | N: 10120.0 kg/ha | Water: 1608.0 mm | LR: 9.37e-01
Episode 15/2000 | Score: -8154 | Avg: -920 | Steps: 160 | Yield: 9744.3 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 9.32e-01
Episode 16/2000 | Score: -7832 | Avg: -920 | Steps: 166 | Yield: 11585.8 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 9.28e-01
Episode 17/2000 | Score: -7835 | Avg: -920 | Steps: 156 | Yield: 11604.2 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 9.23e-01
Episode 18/2000 | Score: -7818 | Avg: -920 | Steps: 159 | Yield: 11714.1 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 9.18e-01
Episode 19/2000 | Score: -8255 | Avg: -920 | Steps: 164 | Yield: 9108.9 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 9.14e-01
Episode 20/2000 | Score: -7870 | Avg: -920 | Steps: 160 | Yield: 11344.4 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 9.09e-01
Episode 21/2000 | Score: -7789 | Avg: -920 | Steps: 156 | Yield: 12141.5 kg/ha | N: 10040.0 kg/ha | Water: 1614.0 mm | LR: 9.05e-01
Episode 22/2000 | Score: -7967 | Avg: -920 | Steps: 153 | Yield: 11413.2 kg/ha | N: 10120.0 kg/ha | Water: 1614.0 mm | LR: 9.00e-01
Episode 23/2000 | Score: -7903 | Avg: -920 | Steps: 160 | Yield: 11173.5 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 8.96e-01
Episode 24/2000 | Score: -8082 | Avg: -920 | Steps: 157 | Yield: 10200.1 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 8.91e-01
Episode 25/2000 | Score: -7731 | Avg: -920 | Steps: 161 | Yield: 12421.7 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 8.87e-01
Episode 26/2000 | Score: -7980 | Avg: -920 | Steps: 157 | Yield: 10974.5 kg/ha | N: 10040.0 kg/ha | Water: 1620.0 mm | LR: 8.82e-01
Episode 27/2000 | Score: -7818 | Avg: -920 | Steps: 153 | Yield: 11915.1 kg/ha | N: 10040.0 kg/ha | Water: 1608.0 mm | LR: 8.78e-01
Episode 28/2000 | Score: -7835 | Avg: -920 | Steps: 165 | Yield: 11651.2 kg/ha | N: 10000.0 kg/ha | Water: 1614.0 mm | LR: 8.73e-01
Episode 29/2000 | Score: -7983 | Avg: -920 | Steps: 160 | Yield: 10753.3 kg/ha | N: 10000.0 kg/ha | Water: 1620.0 mm | LR: 8.69e-01
Episode 30/2000 | Score: -8020 | Avg: -920 | Steps: 155 | Yield: 10476.3 kg/ha | N: 10000.0 kg/ha | Water: 1614.0 mm | LR: 8.65e-01
Episode 31/2000 | Score: -7722 | Avg: -920 | Steps: 160 | Yield: 12476.9 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 8.60e-01
Episode 32/2000 | Score: -8102 | Avg: -920 | Steps: 162 | Yield: 10201.4 kg/ha | N: 10040.0 kg/ha | Water: 1620.0 mm | LR: 8.56e-01
Episode 33/2000 | Score: -7928 | Avg: -920 | Steps: 163 | Yield: 11573.0 kg/ha | N: 10120.0 kg/ha | Water: 1602.0 mm | LR: 8.52e-01
Episode 34/2000 | Score: -7937 | Avg: -920 | Steps: 153 | Yield: 11115.9 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 8.48e-01
Episode 35/2000 | Score: -8049 | Avg: -920 | Steps: 164 | Yield: 10253.1 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 8.43e-01
Episode 36/2000 | Score: -8021 | Avg: -920 | Steps: 165 | Yield: 10273.9 kg/ha | N: 9840.0 kg/ha | Water: 1614.0 mm | LR: 8.39e-01
Episode 37/2000 | Score: -7892 | Avg: -920 | Steps: 163 | Yield: 11202.9 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 8.35e-01
Episode 38/2000 | Score: -7975 | Avg: -920 | Steps: 163 | Yield: 10879.7 kg/ha | N: 10040.0 kg/ha | Water: 1602.0 mm | LR: 8.31e-01
Episode 39/2000 | Score: -7892 | Avg: -920 | Steps: 160 | Yield: 11442.8 kg/ha | N: 9920.0 kg/ha | Water: 1608.0 mm | LR: 8.27e-01
Episode 40/2000 | Score: -7522 | Avg: -920 | Steps: 156 | Yield: 12084.2 kg/ha | N: 9800.0 kg/ha | Water: 1536.0 mm | LR: 8.22e-01
Episode 41/2000 | Score: -7854 | Avg: -920 | Steps: 169 | Yield: 11967.1 kg/ha | N: 10080.0 kg/ha | Water: 1620.0 mm | LR: 8.18e-01
Episode 42/2000 | Score: -7782 | Avg: -920 | Steps: 167 | Yield: 12299.0 kg/ha | N: 10080.0 kg/ha | Water: 1602.0 mm | LR: 8.14e-01
Episode 43/2000 | Score: -7937 | Avg: -920 | Steps: 159 | Yield: 10918.3 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 8.10e-01
Episode 44/2000 | Score: -7969 | Avg: -920 | Steps: 155 | Yield: 11161.1 kg/ha | N: 10080.0 kg/ha | Water: 1608.0 mm | LR: 8.06e-01
Episode 45/2000 | Score: -8026 | Avg: -920 | Steps: 164 | Yield: 11038.0 kg/ha | N: 10120.0 kg/ha | Water: 1614.0 mm | LR: 8.02e-01
Episode 46/2000 | Score: -7297 | Avg: -920 | Steps: 154 | Yield: 10652.1 kg/ha | N: 9280.0 kg/ha | Water: 1458.0 mm | LR: 7.98e-01
Episode 47/2000 | Score: -8038 | Avg: -920 | Steps: 165 | Yield: 10522.4 kg/ha | N: 10040.0 kg/ha | Water: 1608.0 mm | LR: 7.94e-01
Episode 48/2000 | Score: -7974 | Avg: -920 | Steps: 165 | Yield: 10683.6 kg/ha | N: 10000.0 kg/ha | Water: 1602.0 mm | LR: 7.90e-01
Episode 49/2000 | Score: -7999 | Avg: -920 | Steps: 155 | Yield: 10769.8 kg/ha | N: 10040.0 kg/ha | Water: 1608.0 mm | LR: 7.86e-01
Episode 50/2000 | Score: -8002 | Avg: -920 | Steps: 164 | Yield: 10549.8 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 7.82e-01
Episode 51/2000 | Score: -7768 | Avg: -920 | Steps: 156 | Yield: 10031.0 kg/ha | N: 9560.0 kg/ha | Water: 1608.0 mm | LR: 7.78e-01
Episode 52/2000 | Score: -7726 | Avg: -920 | Steps: 161 | Yield: 11935.3 kg/ha | N: 9880.0 kg/ha | Water: 1614.0 mm | LR: 7.74e-01
Episode 53/2000 | Score: -7916 | Avg: -920 | Steps: 157 | Yield: 11537.0 kg/ha | N: 10080.0 kg/ha | Water: 1614.0 mm | LR: 7.71e-01
Episode 54/2000 | Score: -7926 | Avg: -920 | Steps: 167 | Yield: 11715.8 kg/ha | N: 10120.0 kg/ha | Water: 1620.0 mm | LR: 7.67e-01
Episode 55/2000 | Score: -7634 | Avg: -920 | Steps: 154 | Yield: 11081.6 kg/ha | N: 9520.0 kg/ha | Water: 1608.0 mm | LR: 7.63e-01
Episode 56/2000 | Score: -7472 | Avg: -920 | Steps: 162 | Yield: 11862.6 kg/ha | N: 9480.0 kg/ha | Water: 1602.0 mm | LR: 7.59e-01
Episode 57/2000 | Score: -7791 | Avg: -920 | Steps: 158 | Yield: 11386.2 kg/ha | N: 10000.0 kg/ha | Water: 1518.0 mm | LR: 7.55e-01
Episode 58/2000 | Score: -7813 | Avg: -920 | Steps: 157 | Yield: 11747.2 kg/ha | N: 10000.0 kg/ha | Water: 1608.0 mm | LR: 7.51e-01
Episode 59/2000 | Score: -7066 | Avg: -920 | Steps: 157 | Yield: 11358.0 kg/ha | N: 8800.0 kg/ha | Water: 1620.0 mm | LR: 7.48e-01
Episode 60/2000 | Score: -6752 | Avg: -920 | Steps: 156 | Yield: 12813.1 kg/ha | N: 8920.0 kg/ha | Water: 1566.0 mm | LR: 7.44e-01
Episode 61/2000 | Score: -6177 | Avg: -920 | Steps: 154 | Yield: 10784.4 kg/ha | N: 7960.0 kg/ha | Water: 1350.0 mm | LR: 7.40e-01
Episode 62/2000 | Score: -6794 | Avg: -920 | Steps: 161 | Yield: 10955.2 kg/ha | N: 8480.0 kg/ha | Water: 1602.0 mm | LR: 7.37e-01
Episode 63/2000 | Score: -7828 | Avg: -920 | Steps: 158 | Yield: 9811.7 kg/ha | N: 9600.0 kg/ha | Water: 1602.0 mm | LR: 7.33e-01
Episode 64/2000 | Score: -7780 | Avg: -920 | Steps: 171 | Yield: 11753.0 kg/ha | N: 9840.0 kg/ha | Water: 1608.0 mm | LR: 7.29e-01
Episode 65/2000 | Score: -6605 | Avg: -920 | Steps: 164 | Yield: 11431.8 kg/ha | N: 8400.0 kg/ha | Water: 1614.0 mm | LR: 7.26e-01
Episode 66/2000 | Score: -7437 | Avg: -920 | Steps: 161 | Yield: 10527.5 kg/ha | N: 9120.0 kg/ha | Water: 1608.0 mm | LR: 7.22e-01
Episode 67/2000 | Score: -7236 | Avg: -920 | Steps: 157 | Yield: 12107.1 kg/ha | N: 9400.0 kg/ha | Water: 1566.0 mm | LR: 7.18e-01
Episode 68/2000 | Score: -7761 | Avg: -920 | Steps: 157 | Yield: 11522.5 kg/ha | N: 10040.0 kg/ha | Water: 1482.0 mm | LR: 7.15e-01
Episode 69/2000 | Score: -6474 | Avg: -920 | Steps: 158 | Yield: 11336.1 kg/ha | N: 8400.0 kg/ha | Water: 1452.0 mm | LR: 7.11e-01
Episode 70/2000 | Score: -7845 | Avg: -920 | Steps: 157 | Yield: 10588.1 kg/ha | N: 9960.0 kg/ha | Water: 1482.0 mm | LR: 7.08e-01
Episode 71/2000 | Score: -7553 | Avg: -920 | Steps: 162 | Yield: 12413.7 kg/ha | N: 9800.0 kg/ha | Water: 1548.0 mm | LR: 7.04e-01
Episode 72/2000 | Score: -7311 | Avg: -920 | Steps: 149 | Yield: 10411.2 kg/ha | N: 9280.0 kg/ha | Water: 1350.0 mm | LR: 7.01e-01
Episode 73/2000 | Score: -7602 | Avg: -920 | Steps: 160 | Yield: 12589.3 kg/ha | N: 9880.0 kg/ha | Water: 1554.0 mm | LR: 6.97e-01
Episode 74/2000 | Score: -7648 | Avg: -920 | Steps: 160 | Yield: 11148.4 kg/ha | N: 9520.0 kg/ha | Water: 1602.0 mm | LR: 6.94e-01
Episode 75/2000 | Score: -6772 | Avg: -920 | Steps: 156 | Yield: 11830.2 kg/ha | N: 8960.0 kg/ha | Water: 1374.0 mm | LR: 6.90e-01
Episode 76/2000 | Score: -6269 | Avg: -920 | Steps: 159 | Yield: 10054.2 kg/ha | N: 7840.0 kg/ha | Water: 1398.0 mm | LR: 6.87e-01
Episode 77/2000 | Score: -7220 | Avg: -920 | Steps: 159 | Yield: 10941.8 kg/ha | N: 8920.0 kg/ha | Water: 1614.0 mm | LR: 6.83e-01
Episode 78/2000 | Score: -7094 | Avg: -920 | Steps: 161 | Yield: 11303.6 kg/ha | N: 9160.0 kg/ha | Water: 1482.0 mm | LR: 6.80e-01
Episode 79/2000 | Score: -5777 | Avg: -920 | Steps: 152 | Yield: 12326.2 kg/ha | N: 7800.0 kg/ha | Water: 1380.0 mm | LR: 6.76e-01
Episode 80/2000 | Score: -6985 | Avg: -920 | Steps: 159 | Yield: 11945.4 kg/ha | N: 8960.0 kg/ha | Water: 1596.0 mm | LR: 6.73e-01
Episode 81/2000 | Score: -7211 | Avg: -920 | Steps: 161 | Yield: 10553.4 kg/ha | N: 9000.0 kg/ha | Water: 1608.0 mm | LR: 6.70e-01
Episode 82/2000 | Score: -8071 | Avg: -920 | Steps: 161 | Yield: 10755.7 kg/ha | N: 10120.0 kg/ha | Water: 1614.0 mm | LR: 6.66e-01
Episode 83/2000 | Score: -6120 | Avg: -920 | Steps: 159 | Yield: 10968.2 kg/ha | N: 7920.0 kg/ha | Water: 1410.0 mm | LR: 6.63e-01
Episode 84/2000 | Score: -7256 | Avg: -920 | Steps: 156 | Yield: 11459.4 kg/ha | N: 9280.0 kg/ha | Water: 1572.0 mm | LR: 6.60e-01
Episode 85/2000 | Score: -6003 | Avg: -920 | Steps: 162 | Yield: 11800.1 kg/ha | N: 7720.0 kg/ha | Water: 1608.0 mm | LR: 6.56e-01
Episode 86/2000 | Score: -6254 | Avg: -920 | Steps: 155 | Yield: 11641.3 kg/ha | N: 8080.0 kg/ha | Water: 1416.0 mm | LR: 6.53e-01
Episode 87/2000 | Score: -6879 | Avg: -920 | Steps: 158 | Yield: 10300.8 kg/ha | N: 8720.0 kg/ha | Water: 1338.0 mm | LR: 6.50e-01
Episode 88/2000 | Score: -7158 | Avg: -920 | Steps: 165 | Yield: 10027.0 kg/ha | N: 9000.0 kg/ha | Water: 1374.0 mm | LR: 6.47e-01
Episode 89/2000 | Score: -6537 | Avg: -920 | Steps: 163 | Yield: 12106.0 kg/ha | N: 8800.0 kg/ha | Water: 1362.0 mm | LR: 6.43e-01
Episode 90/2000 | Score: -5236 | Avg: -920 | Steps: 157 | Yield: 12426.4 kg/ha | N: 7400.0 kg/ha | Water: 1230.0 mm | LR: 6.40e-01
Episode 91/2000 | Score: -7683 | Avg: -920 | Steps: 167 | Yield: 11277.6 kg/ha | N: 9800.0 kg/ha | Water: 1560.0 mm | LR: 6.37e-01
Episode 92/2000 | Score: -6867 | Avg: -920 | Steps: 164 | Yield: 9766.2 kg/ha | N: 8400.0 kg/ha | Water: 1566.0 mm | LR: 6.34e-01
Episode 93/2000 | Score: -5447 | Avg: -920 | Steps: 153 | Yield: 10762.4 kg/ha | N: 7280.0 kg/ha | Water: 1206.0 mm | LR: 6.31e-01
Episode 94/2000 | Score: -6288 | Avg: -920 | Steps: 163 | Yield: 10666.3 kg/ha | N: 8280.0 kg/ha | Water: 1302.0 mm | LR: 6.27e-01
Episode 95/2000 | Score: -5739 | Avg: -920 | Steps: 163 | Yield: 11463.1 kg/ha | N: 7360.0 kg/ha | Water: 1554.0 mm | LR: 6.24e-01
Episode 96/2000 | Score: -5579 | Avg: -920 | Steps: 155 | Yield: 11353.1 kg/ha | N: 7720.0 kg/ha | Water: 1158.0 mm | LR: 6.21e-01
Episode 97/2000 | Score: -6677 | Avg: -920 | Steps: 159 | Yield: 9787.9 kg/ha | N: 8320.0 kg/ha | Water: 1362.0 mm | LR: 6.18e-01
Episode 98/2000 | Score: -6266 | Avg: -920 | Steps: 156 | Yield: 11074.0 kg/ha | N: 8080.0 kg/ha | Water: 1386.0 mm | LR: 6.15e-01
Episode 99/2000 | Score: -6287 | Avg: -920 | Steps: 164 | Yield: 10341.6 kg/ha | N: 7880.0 kg/ha | Water: 1530.0 mm | LR: 6.12e-01
Episode 100/2000 | Score: -6645 | Avg: -920 | Steps: 153 | Yield: 10007.6 kg/ha | N: 8280.0 kg/ha | Water: 1428.0 mm | LR: 6.09e-01
Episode 101/2000 | Score: -6305 | Avg: -920 | Steps: 162 | Yield: 11787.3 kg/ha | N: 8160.0 kg/ha | Water: 1512.0 mm | LR: 6.06e-01
Episode 102/2000 | Score: -5324 | Avg: -920 | Steps: 160 | Yield: 11929.7 kg/ha | N: 7320.0 kg/ha | Water: 1296.0 mm | LR: 6.03e-01
Episode 103/2000 | Score: -5837 | Avg: -920 | Steps: 162 | Yield: 12164.7 kg/ha | N: 8000.0 kg/ha | Water: 1308.0 mm | LR: 6.00e-01
Episode 104/2000 | Score: -5859 | Avg: -920 | Steps: 155 | Yield: 9249.4 kg/ha | N: 7480.0 kg/ha | Water: 1248.0 mm | LR: 5.97e-01
Episode 105/2000 | Score: -5393 | Avg: -920 | Steps: 156 | Yield: 10704.2 kg/ha | N: 6720.0 kg/ha | Water: 1614.0 mm | LR: 5.94e-01
Episode 106/2000 | Score: -5840 | Avg: -920 | Steps: 163 | Yield: 12033.2 kg/ha | N: 8120.0 kg/ha | Water: 1206.0 mm | LR: 5.91e-01
Episode 107/2000 | Score: -5860 | Avg: -920 | Steps: 158 | Yield: 11570.9 kg/ha | N: 7760.0 kg/ha | Water: 1416.0 mm | LR: 5.88e-01
Episode 108/2000 | Score: -5844 | Avg: -920 | Steps: 161 | Yield: 12920.8 kg/ha | N: 8040.0 kg/ha | Water: 1308.0 mm | LR: 5.85e-01
Episode 109/2000 | Score: -4725 | Avg: -920 | Steps: 163 | Yield: 12628.4 kg/ha | N: 6760.0 kg/ha | Water: 1236.0 mm | LR: 5.82e-01
Episode 110/2000 | Score: -5224 | Avg: -920 | Steps: 160 | Yield: 10327.3 kg/ha | N: 6840.0 kg/ha | Water: 1302.0 mm | LR: 5.79e-01
Episode 111/2000 | Score: -5524 | Avg: -920 | Steps: 158 | Yield: 10084.6 kg/ha | N: 7000.0 kg/ha | Water: 1368.0 mm | LR: 5.76e-01
Episode 112/2000 | Score: -4602 | Avg: -920 | Steps: 155 | Yield: 9937.7 kg/ha | N: 6160.0 kg/ha | Water: 1140.0 mm | LR: 5.73e-01
Episode 113/2000 | Score: -6110 | Avg: -920 | Steps: 166 | Yield: 9387.7 kg/ha | N: 7520.0 kg/ha | Water: 1404.0 mm | LR: 5.70e-01
Episode 114/2000 | Score: -4902 | Avg: -920 | Steps: 165 | Yield: 11014.4 kg/ha | N: 6680.0 kg/ha | Water: 1200.0 mm | LR: 5.68e-01
Episode 115/2000 | Score: -5996 | Avg: -920 | Steps: 162 | Yield: 10724.3 kg/ha | N: 7640.0 kg/ha | Water: 1464.0 mm | LR: 5.65e-01
Episode 116/2000 | Score: -6187 | Avg: -920 | Steps: 162 | Yield: 11167.6 kg/ha | N: 8160.0 kg/ha | Water: 1368.0 mm | LR: 5.62e-01
Episode 117/2000 | Score: -6128 | Avg: -920 | Steps: 157 | Yield: 11418.9 kg/ha | N: 8320.0 kg/ha | Water: 1236.0 mm | LR: 5.59e-01
Episode 118/2000 | Score: -4344 | Avg: -920 | Steps: 150 | Yield: 11661.1 kg/ha | N: 6160.0 kg/ha | Water: 1188.0 mm | LR: 5.56e-01
Episode 119/2000 | Score: -4319 | Avg: -920 | Steps: 160 | Yield: 12974.0 kg/ha | N: 6200.0 kg/ha | Water: 1308.0 mm | LR: 5.54e-01
Episode 120/2000 | Score: -4482 | Avg: -920 | Steps: 163 | Yield: 12069.5 kg/ha | N: 6400.0 kg/ha | Water: 1200.0 mm | LR: 5.51e-01
Episode 121/2000 | Score: -4995 | Avg: -920 | Steps: 161 | Yield: 11295.6 kg/ha | N: 6920.0 kg/ha | Water: 1188.0 mm | LR: 5.48e-01
Episode 122/2000 | Score: -5316 | Avg: -920 | Steps: 163 | Yield: 11423.5 kg/ha | N: 7360.0 kg/ha | Water: 1182.0 mm | LR: 5.45e-01
Episode 123/2000 | Score: -5725 | Avg: -920 | Steps: 158 | Yield: 9869.1 kg/ha | N: 7520.0 kg/ha | Water: 1140.0 mm | LR: 5.43e-01
Episode 124/2000 | Score: -5096 | Avg: -920 | Steps: 164 | Yield: 11072.1 kg/ha | N: 6760.0 kg/ha | Water: 1368.0 mm | LR: 5.40e-01
Episode 125/2000 | Score: -4610 | Avg: -920 | Steps: 157 | Yield: 10124.9 kg/ha | N: 6200.0 kg/ha | Water: 1140.0 mm | LR: 5.37e-01
Episode 126/2000 | Score: -5771 | Avg: -920 | Steps: 161 | Yield: 10559.1 kg/ha | N: 7520.0 kg/ha | Water: 1362.0 mm | LR: 5.34e-01
Episode 127/2000 | Score: -6020 | Avg: -920 | Steps: 164 | Yield: 10930.7 kg/ha | N: 7960.0 kg/ha | Water: 1326.0 mm | LR: 5.32e-01
Episode 128/2000 | Score: -4947 | Avg: -920 | Steps: 160 | Yield: 11577.3 kg/ha | N: 6840.0 kg/ha | Water: 1248.0 mm | LR: 5.29e-01
Episode 129/2000 | Score: -5411 | Avg: -920 | Steps: 156 | Yield: 11813.0 kg/ha | N: 7560.0 kg/ha | Water: 1146.0 mm | LR: 5.26e-01
Episode 130/2000 | Score: -4459 | Avg: -920 | Steps: 160 | Yield: 12339.8 kg/ha | N: 6560.0 kg/ha | Water: 1080.0 mm | LR: 5.24e-01
Episode 131/2000 | Score: -6124 | Avg: -920 | Steps: 159 | Yield: 11454.2 kg/ha | N: 8000.0 kg/ha | Water: 1392.0 mm | LR: 5.21e-01
Episode 132/2000 | Score: -4895 | Avg: -920 | Steps: 161 | Yield: 11967.4 kg/ha | N: 6920.0 kg/ha | Water: 1164.0 mm | LR: 5.19e-01
Episode 133/2000 | Score: -4906 | Avg: -920 | Steps: 167 | Yield: 9901.0 kg/ha | N: 6400.0 kg/ha | Water: 1194.0 mm | LR: 5.16e-01
Episode 134/2000 | Score: -5096 | Avg: -920 | Steps: 156 | Yield: 11512.0 kg/ha | N: 7040.0 kg/ha | Water: 1230.0 mm | LR: 5.13e-01
Episode 135/2000 | Score: -4773 | Avg: -920 | Steps: 168 | Yield: 11836.6 kg/ha | N: 6600.0 kg/ha | Water: 1224.0 mm | LR: 5.11e-01
Episode 136/2000 | Score: -5400 | Avg: -920 | Steps: 160 | Yield: 10780.7 kg/ha | N: 7120.0 kg/ha | Water: 1344.0 mm | LR: 5.08e-01
Episode 137/2000 | Score: -5775 | Avg: -920 | Steps: 124 | Yield: 0.0 kg/ha | N: 5840.0 kg/ha | Water: 1050.0 mm | LR: 5.06e-01
Episode 138/2000 | Score: -4202 | Avg: -920 | Steps: 156 | Yield: 11137.4 kg/ha | N: 5560.0 kg/ha | Water: 1392.0 mm | LR: 5.03e-01
Episode 139/2000 | Score: -4482 | Avg: -920 | Steps: 155 | Yield: 9087.6 kg/ha | N: 5880.0 kg/ha | Water: 1110.0 mm | LR: 5.01e-01
Episode 140/2000 | Score: -4933 | Avg: -920 | Steps: 163 | Yield: 10254.6 kg/ha | N: 6320.0 kg/ha | Water: 1320.0 mm | LR: 4.98e-01
Episode 141/2000 | Score: -4997 | Avg: -920 | Steps: 158 | Yield: 12853.4 kg/ha | N: 7200.0 kg/ha | Water: 1218.0 mm | LR: 4.96e-01
Episode 142/2000 | Score: -4630 | Avg: -920 | Steps: 155 | Yield: 11503.1 kg/ha | N: 6800.0 kg/ha | Water: 978.0 mm | LR: 4.93e-01
Episode 143/2000 | Score: -5933 | Avg: -920 | Steps: 158 | Yield: 11010.6 kg/ha | N: 8000.0 kg/ha | Water: 1230.0 mm | LR: 4.91e-01
Episode 144/2000 | Score: -4992 | Avg: -920 | Steps: 162 | Yield: 12318.2 kg/ha | N: 7120.0 kg/ha | Water: 1194.0 mm | LR: 4.88e-01
Episode 145/2000 | Score: -4442 | Avg: -920 | Steps: 155 | Yield: 10478.4 kg/ha | N: 6240.0 kg/ha | Water: 1062.0 mm | LR: 4.86e-01
Episode 146/2000 | Score: -3844 | Avg: -920 | Steps: 160 | Yield: 11557.7 kg/ha | N: 5400.0 kg/ha | Water: 1236.0 mm | LR: 4.83e-01
Episode 147/2000 | Score: -3895 | Avg: -920 | Steps: 162 | Yield: 11673.8 kg/ha | N: 5560.0 kg/ha | Water: 1110.0 mm | LR: 4.81e-01
Episode 148/2000 | Score: -4491 | Avg: -920 | Steps: 162 | Yield: 10532.6 kg/ha | N: 6480.0 kg/ha | Water: 942.0 mm | LR: 4.79e-01
Episode 149/2000 | Score: -4916 | Avg: -920 | Steps: 161 | Yield: 11480.0 kg/ha | N: 6720.0 kg/ha | Water: 1194.0 mm | LR: 4.76e-01
Episode 150/2000 | Score: -5032 | Avg: -920 | Steps: 158 | Yield: 10896.9 kg/ha | N: 6680.0 kg/ha | Water: 1308.0 mm | LR: 4.74e-01
Episode 151/2000 | Score: -4461 | Avg: -920 | Steps: 160 | Yield: 11647.2 kg/ha | N: 6200.0 kg/ha | Water: 1212.0 mm | LR: 4.71e-01
Episode 152/2000 | Score: -5029 | Avg: -920 | Steps: 161 | Yield: 11408.7 kg/ha | N: 7160.0 kg/ha | Water: 1068.0 mm | LR: 4.69e-01
Episode 153/2000 | Score: -5251 | Avg: -920 | Steps: 158 | Yield: 12392.8 kg/ha | N: 7000.0 kg/ha | Water: 1422.0 mm | LR: 4.67e-01
Episode 154/2000 | Score: -4383 | Avg: -920 | Steps: 160 | Yield: 11751.3 kg/ha | N: 5960.0 kg/ha | Water: 1374.0 mm | LR: 4.64e-01
Episode 155/2000 | Score: -4827 | Avg: -920 | Steps: 168 | Yield: 9773.6 kg/ha | N: 6360.0 kg/ha | Water: 1224.0 mm | LR: 4.62e-01
Episode 156/2000 | Score: -5264 | Avg: -920 | Steps: 163 | Yield: 9612.9 kg/ha | N: 6840.0 kg/ha | Water: 1254.0 mm | LR: 4.60e-01
Episode 157/2000 | Score: -4552 | Avg: -920 | Steps: 160 | Yield: 10011.5 kg/ha | N: 6120.0 kg/ha | Water: 1152.0 mm | LR: 4.58e-01
Episode 158/2000 | Score: -4196 | Avg: -920 | Steps: 167 | Yield: 8730.3 kg/ha | N: 5240.0 kg/ha | Water: 1236.0 mm | LR: 4.55e-01
Episode 159/2000 | Score: -4303 | Avg: -920 | Steps: 157 | Yield: 11649.1 kg/ha | N: 6080.0 kg/ha | Water: 1104.0 mm | LR: 4.53e-01
Episode 160/2000 | Score: -4788 | Avg: -920 | Steps: 157 | Yield: 11244.5 kg/ha | N: 6240.0 kg/ha | Water: 1452.0 mm | LR: 4.51e-01
Episode 161/2000 | Score: -3356 | Avg: -920 | Steps: 155 | Yield: 12304.2 kg/ha | N: 5040.0 kg/ha | Water: 1146.0 mm | LR: 4.48e-01
Episode 162/2000 | Score: -5139 | Avg: -920 | Steps: 160 | Yield: 10958.6 kg/ha | N: 6600.0 kg/ha | Water: 1482.0 mm | LR: 4.46e-01
Episode 163/2000 | Score: -4969 | Avg: -920 | Steps: 164 | Yield: 11498.1 kg/ha | N: 6960.0 kg/ha | Water: 1170.0 mm | LR: 4.44e-01
Episode 164/2000 | Score: -4117 | Avg: -920 | Steps: 159 | Yield: 10691.1 kg/ha | N: 5560.0 kg/ha | Water: 1146.0 mm | LR: 4.42e-01
Episode 165/2000 | Score: -4094 | Avg: -920 | Steps: 152 | Yield: 10697.1 kg/ha | N: 5960.0 kg/ha | Water: 978.0 mm | LR: 4.40e-01
Episode 166/2000 | Score: -4339 | Avg: -920 | Steps: 157 | Yield: 11802.1 kg/ha | N: 5920.0 kg/ha | Water: 1290.0 mm | LR: 4.37e-01
Episode 167/2000 | Score: -4513 | Avg: -920 | Steps: 159 | Yield: 12041.2 kg/ha | N: 6080.0 kg/ha | Water: 1380.0 mm | LR: 4.35e-01
Episode 168/2000 | Score: -5108 | Avg: -920 | Steps: 167 | Yield: 11928.1 kg/ha | N: 7040.0 kg/ha | Water: 1272.0 mm | LR: 4.33e-01
Episode 169/2000 | Score: -4083 | Avg: -920 | Steps: 158 | Yield: 10885.3 kg/ha | N: 5680.0 kg/ha | Water: 1086.0 mm | LR: 4.31e-01
Episode 170/2000 | Score: -4501 | Avg: -920 | Steps: 161 | Yield: 11221.2 kg/ha | N: 6120.0 kg/ha | Water: 1308.0 mm | LR: 4.29e-01
Episode 171/2000 | Score: -4061 | Avg: -920 | Steps: 155 | Yield: 9622.4 kg/ha | N: 5480.0 kg/ha | Water: 1098.0 mm | LR: 4.27e-01
Episode 172/2000 | Score: -4110 | Avg: -920 | Steps: 158 | Yield: 10253.5 kg/ha | N: 5800.0 kg/ha | Water: 1044.0 mm | LR: 4.24e-01
Episode 173/2000 | Score: -4290 | Avg: -920 | Steps: 163 | Yield: 11360.6 kg/ha | N: 6200.0 kg/ha | Water: 1050.0 mm | LR: 4.22e-01
Episode 174/2000 | Score: -5278 | Avg: -920 | Steps: 157 | Yield: 12025.1 kg/ha | N: 7080.0 kg/ha | Water: 1320.0 mm | LR: 4.20e-01
Episode 175/2000 | Score: -3941 | Avg: -920 | Steps: 159 | Yield: 11060.6 kg/ha | N: 5600.0 kg/ha | Water: 1092.0 mm | LR: 4.18e-01
Episode 176/2000 | Score: -4282 | Avg: -920 | Steps: 157 | Yield: 10979.8 kg/ha | N: 5680.0 kg/ha | Water: 1344.0 mm | LR: 4.16e-01
Episode 177/2000 | Score: -4227 | Avg: -920 | Steps: 168 | Yield: 11197.0 kg/ha | N: 6120.0 kg/ha | Water: 1032.0 mm | LR: 4.14e-01
Episode 178/2000 | Score: -3176 | Avg: -920 | Steps: 163 | Yield: 10090.3 kg/ha | N: 4200.0 kg/ha | Water: 1302.0 mm | LR: 4.12e-01
Episode 179/2000 | Score: -4643 | Avg: -920 | Steps: 157 | Yield: 11423.8 kg/ha | N: 6560.0 kg/ha | Water: 1122.0 mm | LR: 4.10e-01
Episode 180/2000 | Score: -3299 | Avg: -920 | Steps: 162 | Yield: 11460.2 kg/ha | N: 4680.0 kg/ha | Water: 1284.0 mm | LR: 4.08e-01
Episode 181/2000 | Score: -3218 | Avg: -920 | Steps: 158 | Yield: 10470.1 kg/ha | N: 4640.0 kg/ha | Water: 1068.0 mm | LR: 4.06e-01
Episode 182/2000 | Score: -4501 | Avg: -920 | Steps: 164 | Yield: 11832.5 kg/ha | N: 6360.0 kg/ha | Water: 1212.0 mm | LR: 4.04e-01
Episode 183/2000 | Score: -4538 | Avg: -920 | Steps: 159 | Yield: 11938.0 kg/ha | N: 6520.0 kg/ha | Water: 1146.0 mm | LR: 4.02e-01
Episode 184/2000 | Score: -3360 | Avg: -920 | Steps: 158 | Yield: 9911.7 kg/ha | N: 4640.0 kg/ha | Water: 1134.0 mm | LR: 4.00e-01
Episode 185/2000 | Score: -4603 | Avg: -920 | Steps: 165 | Yield: 9502.2 kg/ha | N: 6200.0 kg/ha | Water: 1050.0 mm | LR: 3.98e-01
Episode 186/2000 | Score: -3753 | Avg: -920 | Steps: 163 | Yield: 11908.7 kg/ha | N: 5720.0 kg/ha | Water: 1002.0 mm | LR: 3.96e-01
Episode 187/2000 | Score: -3367 | Avg: -920 | Steps: 162 | Yield: 10777.7 kg/ha | N: 4880.0 kg/ha | Water: 1092.0 mm | LR: 3.94e-01
Episode 188/2000 | Score: -4208 | Avg: -920 | Steps: 163 | Yield: 11585.4 kg/ha | N: 6040.0 kg/ha | Water: 1140.0 mm | LR: 3.92e-01
Episode 189/2000 | Score: -3245 | Avg: -920 | Steps: 168 | Yield: 8526.4 kg/ha | N: 4000.0 kg/ha | Water: 1290.0 mm | LR: 3.90e-01
Episode 190/2000 | Score: -3208 | Avg: -920 | Steps: 164 | Yield: 11152.2 kg/ha | N: 4520.0 kg/ha | Water: 1254.0 mm | LR: 3.88e-01
Episode 191/2000 | Score: -3690 | Avg: -920 | Steps: 157 | Yield: 11412.5 kg/ha | N: 5320.0 kg/ha | Water: 1092.0 mm | LR: 3.86e-01
Episode 192/2000 | Score: -3694 | Avg: -920 | Steps: 160 | Yield: 11048.3 kg/ha | N: 5160.0 kg/ha | Water: 1164.0 mm | LR: 3.84e-01
Episode 193/2000 | Score: -4746 | Avg: -920 | Steps: 158 | Yield: 11715.6 kg/ha | N: 6840.0 kg/ha | Water: 1044.0 mm | LR: 3.82e-01
Episode 194/2000 | Score: -2873 | Avg: -920 | Steps: 153 | Yield: 10085.4 kg/ha | N: 4400.0 kg/ha | Water: 882.0 mm | LR: 3.80e-01
Episode 195/2000 | Score: -4011 | Avg: -920 | Steps: 164 | Yield: 10822.9 kg/ha | N: 5320.0 kg/ha | Water: 1380.0 mm | LR: 3.78e-01
Episode 196/2000 | Score: -3527 | Avg: -920 | Steps: 158 | Yield: 11315.0 kg/ha | N: 4960.0 kg/ha | Water: 1200.0 mm | LR: 3.76e-01
Episode 197/2000 | Score: -3710 | Avg: -920 | Steps: 167 | Yield: 10402.8 kg/ha | N: 5240.0 kg/ha | Water: 1104.0 mm | LR: 3.74e-01
Episode 198/2000 | Score: -3119 | Avg: -920 | Steps: 159 | Yield: 12228.3 kg/ha | N: 4360.0 kg/ha | Water: 1374.0 mm | LR: 3.73e-01
Episode 199/2000 | Score: -3427 | Avg: -920 | Steps: 159 | Yield: 11840.0 kg/ha | N: 4840.0 kg/ha | Water: 1242.0 mm | LR: 3.71e-01
Episode 200/2000 | Score: -3377 | Avg: -920 | Steps: 162 | Yield: 12080.2 kg/ha | N: 5320.0 kg/ha | Water: 984.0 mm | LR: 3.69e-01
Episode 201/2000 | Score: -3357 | Avg: -920 | Steps: 153 | Yield: 12472.8 kg/ha | N: 5240.0 kg/ha | Water: 1080.0 mm | LR: 3.67e-01
Episode 202/2000 | Score: -3659 | Avg: -920 | Steps: 154 | Yield: 11709.7 kg/ha | N: 5240.0 kg/ha | Water: 1182.0 mm | LR: 3.65e-01
Episode 203/2000 | Score: -3034 | Avg: -920 | Steps: 158 | Yield: 10162.8 kg/ha | N: 4520.0 kg/ha | Water: 972.0 mm | LR: 3.63e-01
Episode 204/2000 | Score: -2271 | Avg: -920 | Steps: 156 | Yield: 12471.4 kg/ha | N: 4040.0 kg/ha | Water: 936.0 mm | LR: 3.61e-01
Episode 205/2000 | Score: -3475 | Avg: -920 | Steps: 155 | Yield: 7495.5 kg/ha | N: 4360.0 kg/ha | Water: 1104.0 mm | LR: 3.60e-01
Episode 206/2000 | Score: -2535 | Avg: -920 | Steps: 162 | Yield: 10853.8 kg/ha | N: 3600.0 kg/ha | Water: 1266.0 mm | LR: 3.58e-01
Episode 207/2000 | Score: -2931 | Avg: -920 | Steps: 161 | Yield: 12154.6 kg/ha | N: 4520.0 kg/ha | Water: 1146.0 mm | LR: 3.56e-01
Episode 208/2000 | Score: -4060 | Avg: -920 | Steps: 165 | Yield: 10151.6 kg/ha | N: 5440.0 kg/ha | Water: 1218.0 mm | LR: 3.54e-01
Episode 209/2000 | Score: -3231 | Avg: -920 | Steps: 162 | Yield: 11729.4 kg/ha | N: 5040.0 kg/ha | Water: 984.0 mm | LR: 3.53e-01
Episode 210/2000 | Score: -3510 | Avg: -920 | Steps: 160 | Yield: 10761.1 kg/ha | N: 5040.0 kg/ha | Water: 984.0 mm | LR: 3.51e-01
Episode 211/2000 | Score: -3211 | Avg: -920 | Steps: 165 | Yield: 11030.9 kg/ha | N: 4800.0 kg/ha | Water: 1056.0 mm | LR: 3.49e-01
Episode 212/2000 | Score: -3884 | Avg: -920 | Steps: 152 | Yield: 10985.0 kg/ha | N: 5640.0 kg/ha | Water: 954.0 mm | LR: 3.47e-01
Episode 213/2000 | Score: -3393 | Avg: -920 | Steps: 163 | Yield: 12377.8 kg/ha | N: 4800.0 kg/ha | Water: 1368.0 mm | LR: 3.46e-01
Episode 214/2000 | Score: -2779 | Avg: -920 | Steps: 159 | Yield: 12558.7 kg/ha | N: 4400.0 kg/ha | Water: 1170.0 mm | LR: 3.44e-01
Episode 215/2000 | Score: -2983 | Avg: -920 | Steps: 165 | Yield: 9105.2 kg/ha | N: 4360.0 kg/ha | Water: 882.0 mm | LR: 3.42e-01
Episode 216/2000 | Score: -3344 | Avg: -920 | Steps: 167 | Yield: 8853.6 kg/ha | N: 4600.0 kg/ha | Water: 1002.0 mm | LR: 3.40e-01
Episode 217/2000 | Score: -3016 | Avg: -920 | Steps: 156 | Yield: 11333.0 kg/ha | N: 4240.0 kg/ha | Water: 1284.0 mm | LR: 3.39e-01
Episode 218/2000 | Score: -2225 | Avg: -920 | Steps: 160 | Yield: 9097.3 kg/ha | N: 3040.0 kg/ha | Water: 1146.0 mm | LR: 3.37e-01
Episode 219/2000 | Score: -3114 | Avg: -920 | Steps: 165 | Yield: 9627.4 kg/ha | N: 4240.0 kg/ha | Water: 1116.0 mm | LR: 3.35e-01
Episode 220/2000 | Score: -3087 | Avg: -920 | Steps: 156 | Yield: 11279.0 kg/ha | N: 4680.0 kg/ha | Water: 1002.0 mm | LR: 3.34e-01
Episode 221/2000 | Score: -3555 | Avg: -920 | Steps: 157 | Yield: 10359.6 kg/ha | N: 4960.0 kg/ha | Water: 1134.0 mm | LR: 3.32e-01
Episode 222/2000 | Score: -3211 | Avg: -920 | Steps: 170 | Yield: 11794.1 kg/ha | N: 4560.0 kg/ha | Water: 1326.0 mm | LR: 3.30e-01
Episode 223/2000 | Score: -3521 | Avg: -920 | Steps: 164 | Yield: 10876.1 kg/ha | N: 4800.0 kg/ha | Water: 1212.0 mm | LR: 3.29e-01
Episode 224/2000 | Score: -3276 | Avg: -920 | Steps: 163 | Yield: 12476.1 kg/ha | N: 4920.0 kg/ha | Water: 1098.0 mm | LR: 3.27e-01
Episode 225/2000 | Score: -2497 | Avg: -920 | Steps: 160 | Yield: 10522.5 kg/ha | N: 3880.0 kg/ha | Water: 966.0 mm | LR: 3.25e-01
Episode 226/2000 | Score: -3025 | Avg: -920 | Steps: 162 | Yield: 11920.7 kg/ha | N: 4760.0 kg/ha | Water: 1044.0 mm | LR: 3.24e-01
Episode 227/2000 | Score: -2875 | Avg: -920 | Steps: 151 | Yield: 12431.7 kg/ha | N: 4440.0 kg/ha | Water: 1182.0 mm | LR: 3.22e-01
Episode 228/2000 | Score: -2865 | Avg: -920 | Steps: 159 | Yield: 12231.6 kg/ha | N: 4720.0 kg/ha | Water: 960.0 mm | LR: 3.21e-01
Episode 229/2000 | Score: -2461 | Avg: -920 | Steps: 163 | Yield: 11654.9 kg/ha | N: 4160.0 kg/ha | Water: 924.0 mm | LR: 3.19e-01
Episode 230/2000 | Score: -2575 | Avg: -920 | Steps: 161 | Yield: 9853.3 kg/ha | N: 3760.0 kg/ha | Water: 1056.0 mm | LR: 3.17e-01
Episode 231/2000 | Score: -2039 | Avg: -920 | Steps: 161 | Yield: 8640.8 kg/ha | N: 3240.0 kg/ha | Water: 768.0 mm | LR: 3.16e-01
Episode 232/2000 | Score: -2609 | Avg: -920 | Steps: 164 | Yield: 10696.8 kg/ha | N: 3800.0 kg/ha | Water: 1110.0 mm | LR: 3.14e-01
Episode 233/2000 | Score: -3083 | Avg: -920 | Steps: 164 | Yield: 11219.3 kg/ha | N: 4360.0 kg/ha | Water: 1236.0 mm | LR: 3.13e-01
Episode 234/2000 | Score: -3301 | Avg: -920 | Steps: 159 | Yield: 10885.9 kg/ha | N: 4440.0 kg/ha | Water: 1290.0 mm | LR: 3.11e-01
Episode 235/2000 | Score: -3126 | Avg: -920 | Steps: 157 | Yield: 10053.5 kg/ha | N: 4480.0 kg/ha | Water: 1068.0 mm | LR: 3.09e-01
Episode 236/2000 | Score: -2299 | Avg: -920 | Steps: 151 | Yield: 10486.4 kg/ha | N: 3160.0 kg/ha | Water: 1194.0 mm | LR: 3.08e-01
Episode 237/2000 | Score: -3335 | Avg: -920 | Steps: 162 | Yield: 8147.6 kg/ha | N: 3760.0 kg/ha | Water: 1392.0 mm | LR: 3.06e-01
Episode 238/2000 | Score: -2956 | Avg: -920 | Steps: 164 | Yield: 11120.8 kg/ha | N: 4320.0 kg/ha | Water: 1176.0 mm | LR: 3.05e-01
Episode 239/2000 | Score: -3511 | Avg: -920 | Steps: 157 | Yield: 9724.6 kg/ha | N: 4960.0 kg/ha | Water: 1026.0 mm | LR: 3.03e-01
Episode 240/2000 | Score: -2660 | Avg: -920 | Steps: 154 | Yield: 11719.0 kg/ha | N: 3840.0 kg/ha | Water: 1326.0 mm | LR: 3.02e-01
Episode 241/2000 | Score: -2218 | Avg: -920 | Steps: 164 | Yield: 11629.2 kg/ha | N: 3680.0 kg/ha | Water: 1032.0 mm | LR: 3.00e-01
Episode 242/2000 | Score: -2846 | Avg: -920 | Steps: 158 | Yield: 10107.7 kg/ha | N: 4000.0 kg/ha | Water: 1074.0 mm | LR: 2.99e-01
Episode 243/2000 | Score: -2447 | Avg: -920 | Steps: 154 | Yield: 10533.0 kg/ha | N: 3640.0 kg/ha | Water: 1008.0 mm | LR: 2.97e-01
Episode 244/2000 | Score: -2733 | Avg: -920 | Steps: 157 | Yield: 12052.1 kg/ha | N: 4000.0 kg/ha | Water: 1302.0 mm | LR: 2.96e-01
Episode 245/2000 | Score: -2960 | Avg: -920 | Steps: 157 | Yield: 11451.3 kg/ha | N: 4360.0 kg/ha | Water: 1176.0 mm | LR: 2.94e-01
Episode 246/2000 | Score: -2998 | Avg: -920 | Steps: 159 | Yield: 9374.2 kg/ha | N: 3840.0 kg/ha | Water: 1308.0 mm | LR: 2.93e-01
Episode 247/2000 | Score: -2415 | Avg: -920 | Steps: 163 | Yield: 12149.4 kg/ha | N: 4200.0 kg/ha | Water: 912.0 mm | LR: 2.91e-01
Episode 248/2000 | Score: -1793 | Avg: -920 | Steps: 165 | Yield: 11403.2 kg/ha | N: 3080.0 kg/ha | Water: 1056.0 mm | LR: 2.90e-01
Episode 249/2000 | Score: -2255 | Avg: -920 | Steps: 158 | Yield: 10998.9 kg/ha | N: 3320.0 kg/ha | Water: 1176.0 mm | LR: 2.88e-01
Episode 250/2000 | Score: -3162 | Avg: -920 | Steps: 153 | Yield: 10588.8 kg/ha | N: 4600.0 kg/ha | Water: 1086.0 mm | LR: 2.87e-01
Episode 251/2000 | Score: -2458 | Avg: -920 | Steps: 158 | Yield: 11732.4 kg/ha | N: 4000.0 kg/ha | Water: 984.0 mm | LR: 2.86e-01
Episode 252/2000 | Score: -2580 | Avg: -920 | Steps: 160 | Yield: 10424.0 kg/ha | N: 3680.0 kg/ha | Water: 1188.0 mm | LR: 2.84e-01
Episode 253/2000 | Score: -3259 | Avg: -920 | Steps: 163 | Yield: 11103.0 kg/ha | N: 4640.0 kg/ha | Water: 1104.0 mm | LR: 2.83e-01
Episode 254/2000 | Score: -2815 | Avg: -920 | Steps: 155 | Yield: 9817.6 kg/ha | N: 4240.0 kg/ha | Water: 912.0 mm | LR: 2.81e-01
Episode 255/2000 | Score: -2957 | Avg: -920 | Steps: 165 | Yield: 10395.8 kg/ha | N: 4360.0 kg/ha | Water: 1038.0 mm | LR: 2.80e-01
Episode 256/2000 | Score: -2721 | Avg: -920 | Steps: 166 | Yield: 5286.1 kg/ha | N: 3440.0 kg/ha | Water: 744.0 mm | LR: 2.79e-01
Episode 257/2000 | Score: -2378 | Avg: -920 | Steps: 160 | Yield: 7015.1 kg/ha | N: 3120.0 kg/ha | Water: 894.0 mm | LR: 2.77e-01
Episode 258/2000 | Score: -2527 | Avg: -920 | Steps: 156 | Yield: 8389.0 kg/ha | N: 3440.0 kg/ha | Water: 1032.0 mm | LR: 2.76e-01
Episode 259/2000 | Score: -2484 | Avg: -920 | Steps: 159 | Yield: 10646.3 kg/ha | N: 3720.0 kg/ha | Water: 1092.0 mm | LR: 2.74e-01
Episode 260/2000 | Score: -2232 | Avg: -920 | Steps: 158 | Yield: 10408.2 kg/ha | N: 3320.0 kg/ha | Water: 1128.0 mm | LR: 2.73e-01
Episode 261/2000 | Score: -1961 | Avg: -920 | Steps: 161 | Yield: 12192.1 kg/ha | N: 3200.0 kg/ha | Water: 1224.0 mm | LR: 2.72e-01
Episode 262/2000 | Score: -2039 | Avg: -920 | Steps: 157 | Yield: 10763.8 kg/ha | N: 2960.0 kg/ha | Water: 1164.0 mm | LR: 2.70e-01
Episode 263/2000 | Score: -3010 | Avg: -920 | Steps: 162 | Yield: 9581.8 kg/ha | N: 4360.0 kg/ha | Water: 900.0 mm | LR: 2.69e-01
Episode 264/2000 | Score: -2418 | Avg: -920 | Steps: 161 | Yield: 13094.8 kg/ha | N: 3760.0 kg/ha | Water: 1350.0 mm | LR: 2.68e-01
Episode 265/2000 | Score: -2863 | Avg: -920 | Steps: 159 | Yield: 6275.7 kg/ha | N: 3280.0 kg/ha | Water: 1038.0 mm | LR: 2.66e-01
Episode 266/2000 | Score: -1928 | Avg: -920 | Steps: 155 | Yield: 11607.1 kg/ha | N: 3200.0 kg/ha | Water: 1110.0 mm | LR: 2.65e-01
Episode 267/2000 | Score: -2465 | Avg: -920 | Steps: 160 | Yield: 11209.0 kg/ha | N: 3800.0 kg/ha | Water: 1122.0 mm | LR: 2.64e-01
Episode 268/2000 | Score: -1905 | Avg: -920 | Steps: 157 | Yield: 9672.6 kg/ha | N: 3120.0 kg/ha | Water: 852.0 mm | LR: 2.62e-01
Episode 269/2000 | Score: -1999 | Avg: -920 | Steps: 163 | Yield: 11263.3 kg/ha | N: 3360.0 kg/ha | Water: 930.0 mm | LR: 2.61e-01
Episode 270/2000 | Score: -2377 | Avg: -920 | Steps: 160 | Yield: 11489.4 kg/ha | N: 3720.0 kg/ha | Water: 1140.0 mm | LR: 2.60e-01
Episode 271/2000 | Score: -2790 | Avg: -920 | Steps: 166 | Yield: 9713.3 kg/ha | N: 3920.0 kg/ha | Water: 1092.0 mm | LR: 2.58e-01
Episode 272/2000 | Score: -2096 | Avg: -920 | Steps: 165 | Yield: 9853.0 kg/ha | N: 3080.0 kg/ha | Water: 1080.0 mm | LR: 2.57e-01
Episode 273/2000 | Score: -1977 | Avg: -920 | Steps: 162 | Yield: 11551.3 kg/ha | N: 3160.0 kg/ha | Water: 1134.0 mm | LR: 2.56e-01
Episode 274/2000 | Score: -2512 | Avg: -920 | Steps: 153 | Yield: 10471.3 kg/ha | N: 3640.0 kg/ha | Water: 1116.0 mm | LR: 2.55e-01
Episode 275/2000 | Score: -1778 | Avg: -920 | Steps: 157 | Yield: 11510.4 kg/ha | N: 2840.0 kg/ha | Water: 1218.0 mm | LR: 2.53e-01
Episode 276/2000 | Score: -2682 | Avg: -920 | Steps: 155 | Yield: 11805.5 kg/ha | N: 4080.0 kg/ha | Water: 1122.0 mm | LR: 2.52e-01
Episode 277/2000 | Score: -2198 | Avg: -920 | Steps: 165 | Yield: 8489.6 kg/ha | N: 2920.0 kg/ha | Water: 1068.0 mm | LR: 2.51e-01
Episode 278/2000 | Score: -2463 | Avg: -920 | Steps: 162 | Yield: 11450.5 kg/ha | N: 3680.0 kg/ha | Water: 1200.0 mm | LR: 2.49e-01
Episode 279/2000 | Score: -1503 | Avg: -920 | Steps: 158 | Yield: 11940.1 kg/ha | N: 2720.0 kg/ha | Water: 1110.0 mm | LR: 2.48e-01
Episode 280/2000 | Score: -1405 | Avg: -920 | Steps: 160 | Yield: 11212.1 kg/ha | N: 2360.0 kg/ha | Water: 1152.0 mm | LR: 2.47e-01
Episode 281/2000 | Score: -1996 | Avg: -920 | Steps: 163 | Yield: 11716.3 kg/ha | N: 3320.0 kg/ha | Water: 1032.0 mm | LR: 2.46e-01
Episode 282/2000 | Score: -2691 | Avg: -920 | Steps: 159 | Yield: 10159.0 kg/ha | N: 3400.0 kg/ha | Water: 1440.0 mm | LR: 2.45e-01
Episode 283/2000 | Score: -2302 | Avg: -920 | Steps: 165 | Yield: 10661.0 kg/ha | N: 3760.0 kg/ha | Water: 912.0 mm | LR: 2.43e-01
Episode 284/2000 | Score: -2007 | Avg: -920 | Steps: 158 | Yield: 11697.9 kg/ha | N: 3560.0 kg/ha | Water: 936.0 mm | LR: 2.42e-01
Episode 285/2000 | Score: -1968 | Avg: -920 | Steps: 162 | Yield: 10906.1 kg/ha | N: 3160.0 kg/ha | Water: 1074.0 mm | LR: 2.41e-01
Episode 286/2000 | Score: -2005 | Avg: -920 | Steps: 163 | Yield: 11664.9 kg/ha | N: 2960.0 kg/ha | Water: 1332.0 mm | LR: 2.40e-01
Episode 287/2000 | Score: -1280 | Avg: -920 | Steps: 163 | Yield: 12909.7 kg/ha | N: 2640.0 kg/ha | Water: 1110.0 mm | LR: 2.38e-01
Episode 288/2000 | Score: -1811 | Avg: -920 | Steps: 164 | Yield: 10876.1 kg/ha | N: 2880.0 kg/ha | Water: 1128.0 mm | LR: 2.37e-01
Episode 289/2000 | Score: -3161 | Avg: -920 | Steps: 162 | Yield: 3607.6 kg/ha | N: 3640.0 kg/ha | Water: 702.0 mm | LR: 2.36e-01
Episode 290/2000 | Score: -1804 | Avg: -920 | Steps: 163 | Yield: 11170.3 kg/ha | N: 2680.0 kg/ha | Water: 1296.0 mm | LR: 2.35e-01
Episode 291/2000 | Score: -1830 | Avg: -920 | Steps: 157 | Yield: 11331.9 kg/ha | N: 3280.0 kg/ha | Water: 924.0 mm | LR: 2.34e-01
Episode 292/2000 | Score: -1971 | Avg: -920 | Steps: 161 | Yield: 8536.2 kg/ha | N: 2840.0 kg/ha | Water: 954.0 mm | LR: 2.33e-01
Episode 293/2000 | Score: -1690 | Avg: -920 | Steps: 157 | Yield: 11388.3 kg/ha | N: 2840.0 kg/ha | Water: 1098.0 mm | LR: 2.31e-01
Episode 294/2000 | Score: -2307 | Avg: -920 | Steps: 161 | Yield: 10812.4 kg/ha | N: 3520.0 kg/ha | Water: 1122.0 mm | LR: 2.30e-01
Episode 295/2000 | Score: -2355 | Avg: -920 | Steps: 162 | Yield: 12021.5 kg/ha | N: 4040.0 kg/ha | Water: 954.0 mm | LR: 2.29e-01
Episode 296/2000 | Score: -2287 | Avg: -920 | Steps: 160 | Yield: 12364.1 kg/ha | N: 4200.0 kg/ha | Water: 810.0 mm | LR: 2.28e-01
Episode 297/2000 | Score: -2365 | Avg: -920 | Steps: 164 | Yield: 11129.6 kg/ha | N: 3240.0 kg/ha | Water: 1422.0 mm | LR: 2.27e-01
Episode 298/2000 | Score: -2844 | Avg: -920 | Steps: 161 | Yield: 12149.1 kg/ha | N: 4760.0 kg/ha | Water: 900.0 mm | LR: 2.26e-01
Episode 299/2000 | Score: -2007 | Avg: -920 | Steps: 159 | Yield: 8656.3 kg/ha | N: 2760.0 kg/ha | Water: 1074.0 mm | LR: 2.25e-01
Episode 300/2000 | Score: -2057 | Avg: -920 | Steps: 160 | Yield: 9821.0 kg/ha | N: 3240.0 kg/ha | Water: 954.0 mm | LR: 2.23e-01
Episode 301/2000 | Score: -1877 | Avg: -920 | Steps: 160 | Yield: 11116.8 kg/ha | N: 2720.0 kg/ha | Water: 1338.0 mm | LR: 2.22e-01
Episode 302/2000 | Score: -2329 | Avg: -920 | Steps: 163 | Yield: 10286.5 kg/ha | N: 3640.0 kg/ha | Water: 888.0 mm | LR: 2.21e-01
Episode 303/2000 | Score: -1704 | Avg: -920 | Steps: 157 | Yield: 12042.7 kg/ha | N: 2920.0 kg/ha | Water: 1170.0 mm | LR: 2.20e-01
Episode 304/2000 | Score: -2177 | Avg: -920 | Steps: 154 | Yield: 8777.3 kg/ha | N: 3400.0 kg/ha | Water: 798.0 mm | LR: 2.19e-01
Episode 305/2000 | Score: -1923 | Avg: -920 | Steps: 160 | Yield: 9974.8 kg/ha | N: 2600.0 kg/ha | Water: 1302.0 mm | LR: 2.18e-01
Episode 306/2000 | Score: -1804 | Avg: -920 | Steps: 164 | Yield: 10449.5 kg/ha | N: 2920.0 kg/ha | Water: 1026.0 mm | LR: 2.17e-01
Episode 307/2000 | Score: -1139 | Avg: -920 | Steps: 149 | Yield: 9260.9 kg/ha | N: 1840.0 kg/ha | Water: 1020.0 mm | LR: 2.16e-01
Episode 308/2000 | Score: -1880 | Avg: -920 | Steps: 162 | Yield: 12598.6 kg/ha | N: 3280.0 kg/ha | Water: 1116.0 mm | LR: 2.15e-01
Episode 309/2000 | Score: -1722 | Avg: -920 | Steps: 161 | Yield: 11515.6 kg/ha | N: 2880.0 kg/ha | Water: 1122.0 mm | LR: 2.14e-01
Episode 310/2000 | Score: -1130 | Avg: -920 | Steps: 156 | Yield: 11749.8 kg/ha | N: 2160.0 kg/ha | Water: 1152.0 mm | LR: 2.12e-01
Episode 311/2000 | Score: -2530 | Avg: -920 | Steps: 162 | Yield: 11984.9 kg/ha | N: 4240.0 kg/ha | Water: 942.0 mm | LR: 2.11e-01
Episode 312/2000 | Score: -1262 | Avg: -920 | Steps: 163 | Yield: 11881.3 kg/ha | N: 2320.0 kg/ha | Water: 1170.0 mm | LR: 2.10e-01
Episode 313/2000 | Score: -1855 | Avg: -920 | Steps: 152 | Yield: 8835.1 kg/ha | N: 2720.0 kg/ha | Water: 990.0 mm | LR: 2.09e-01
Episode 314/2000 | Score: -1954 | Avg: -920 | Steps: 158 | Yield: 12700.1 kg/ha | N: 3520.0 kg/ha | Water: 1038.0 mm | LR: 2.08e-01
Episode 315/2000 | Score: -1994 | Avg: -920 | Steps: 161 | Yield: 10218.4 kg/ha | N: 3360.0 kg/ha | Water: 810.0 mm | LR: 2.07e-01
Episode 316/2000 | Score: -1795 | Avg: -920 | Steps: 166 | Yield: 12318.6 kg/ha | N: 3000.0 kg/ha | Water: 1218.0 mm | LR: 2.06e-01
Episode 317/2000 | Score: -1668 | Avg: -920 | Steps: 150 | Yield: 11840.8 kg/ha | N: 3320.0 kg/ha | Water: 798.0 mm | LR: 2.05e-01
Episode 318/2000 | Score: -1096 | Avg: -920 | Steps: 156 | Yield: 11419.6 kg/ha | N: 2000.0 kg/ha | Water: 1188.0 mm | LR: 2.04e-01
Episode 319/2000 | Score: -2051 | Avg: -920 | Steps: 161 | Yield: 10753.2 kg/ha | N: 3160.0 kg/ha | Water: 1128.0 mm | LR: 2.03e-01
Episode 320/2000 | Score: -1320 | Avg: -920 | Steps: 159 | Yield: 9921.8 kg/ha | N: 2160.0 kg/ha | Water: 1062.0 mm | LR: 2.02e-01
Episode 321/2000 | Score: -1056 | Avg: -920 | Steps: 160 | Yield: 10372.8 kg/ha | N: 2000.0 kg/ha | Water: 1002.0 mm | LR: 2.01e-01
Episode 322/2000 | Score: -1825 | Avg: -920 | Steps: 158 | Yield: 10166.2 kg/ha | N: 3040.0 kg/ha | Water: 936.0 mm | LR: 2.00e-01
Episode 323/2000 | Score: -1648 | Avg: -920 | Steps: 160 | Yield: 10889.1 kg/ha | N: 2720.0 kg/ha | Water: 1062.0 mm | LR: 1.99e-01
Episode 324/2000 | Score: -1786 | Avg: -920 | Steps: 158 | Yield: 11266.3 kg/ha | N: 2760.0 kg/ha | Water: 1248.0 mm | LR: 1.98e-01
Episode 325/2000 | Score: -1344 | Avg: -920 | Steps: 161 | Yield: 11620.9 kg/ha | N: 2440.0 kg/ha | Water: 1110.0 mm | LR: 1.97e-01
Episode 326/2000 | Score: -1714 | Avg: -920 | Steps: 164 | Yield: 11642.6 kg/ha | N: 2760.0 kg/ha | Water: 1248.0 mm | LR: 1.96e-01
Episode 327/2000 | Score: -1808 | Avg: -920 | Steps: 153 | Yield: 10468.9 kg/ha | N: 2920.0 kg/ha | Water: 1038.0 mm | LR: 1.95e-01
Episode 328/2000 | Score: -1333 | Avg: -920 | Steps: 155 | Yield: 9892.2 kg/ha | N: 1560.0 kg/ha | Water: 1488.0 mm | LR: 1.94e-01
Episode 329/2000 | Score: -1835 | Avg: -920 | Steps: 168 | Yield: 10394.8 kg/ha | N: 3000.0 kg/ha | Water: 972.0 mm | LR: 1.93e-01
Episode 330/2000 | Score: -1519 | Avg: -920 | Steps: 160 | Yield: 12241.4 kg/ha | N: 2760.0 kg/ha | Water: 1122.0 mm | LR: 1.92e-01
Episode 331/2000 | Score: -2178 | Avg: -920 | Steps: 157 | Yield: 11435.7 kg/ha | N: 2840.0 kg/ha | Water: 1548.0 mm | LR: 1.91e-01
Episode 332/2000 | Score: -1738 | Avg: -920 | Steps: 162 | Yield: 10496.9 kg/ha | N: 2720.0 kg/ha | Water: 1134.0 mm | LR: 1.90e-01
Episode 333/2000 | Score: -1466 | Avg: -920 | Steps: 161 | Yield: 9769.5 kg/ha | N: 2440.0 kg/ha | Water: 972.0 mm | LR: 1.89e-01
Episode 334/2000 | Score: -1372 | Avg: -920 | Steps: 158 | Yield: 12339.6 kg/ha | N: 2920.0 kg/ha | Water: 888.0 mm | LR: 1.88e-01
Episode 335/2000 | Score: -2130 | Avg: -920 | Steps: 160 | Yield: 9673.9 kg/ha | N: 2960.0 kg/ha | Water: 1200.0 mm | LR: 1.87e-01
Episode 336/2000 | Score: -1662 | Avg: -920 | Steps: 155 | Yield: 10717.5 kg/ha | N: 2760.0 kg/ha | Water: 1056.0 mm | LR: 1.87e-01
Episode 337/2000 | Score: -1256 | Avg: -920 | Steps: 153 | Yield: 11339.3 kg/ha | N: 2320.0 kg/ha | Water: 1092.0 mm | LR: 1.86e-01
Episode 338/2000 | Score: -939 | Avg: -920 | Steps: 156 | Yield: 12352.5 kg/ha | N: 2040.0 kg/ha | Water: 1116.0 mm | LR: 1.85e-01
Episode 339/2000 | Score: -1314 | Avg: -920 | Steps: 159 | Yield: 12576.7 kg/ha | N: 2560.0 kg/ha | Water: 1134.0 mm | LR: 1.84e-01
Episode 340/2000 | Score: -1843 | Avg: -920 | Steps: 161 | Yield: 11135.7 kg/ha | N: 3040.0 kg/ha | Water: 1092.0 mm | LR: 1.83e-01
Episode 341/2000 | Score: -1314 | Avg: -920 | Steps: 159 | Yield: 11658.3 kg/ha | N: 2560.0 kg/ha | Water: 1002.0 mm | LR: 1.82e-01
Episode 342/2000 | Score: -1257 | Avg: -920 | Steps: 163 | Yield: 10248.4 kg/ha | N: 2480.0 kg/ha | Water: 822.0 mm | LR: 1.81e-01
Episode 343/2000 | Score: -1468 | Avg: -920 | Steps: 158 | Yield: 11881.6 kg/ha | N: 2280.0 kg/ha | Water: 1380.0 mm | LR: 1.80e-01
Episode 344/2000 | Score: -2015 | Avg: -920 | Steps: 159 | Yield: 11347.9 kg/ha | N: 2840.0 kg/ha | Water: 1398.0 mm | LR: 1.79e-01
Episode 345/2000 | Score: -1566 | Avg: -920 | Steps: 153 | Yield: 10695.5 kg/ha | N: 2400.0 kg/ha | Water: 1212.0 mm | LR: 1.78e-01
Episode 346/2000 | Score: -1911 | Avg: -920 | Steps: 163 | Yield: 9760.7 kg/ha | N: 2400.0 kg/ha | Water: 1392.0 mm | LR: 1.77e-01
Episode 347/2000 | Score: -1458 | Avg: -920 | Steps: 156 | Yield: 11992.2 kg/ha | N: 2600.0 kg/ha | Water: 1146.0 mm | LR: 1.77e-01
Episode 348/2000 | Score: -1685 | Avg: -920 | Steps: 161 | Yield: 10095.0 kg/ha | N: 2840.0 kg/ha | Water: 930.0 mm | LR: 1.76e-01
Episode 349/2000 | Score: -1703 | Avg: -920 | Steps: 158 | Yield: 11768.1 kg/ha | N: 2880.0 kg/ha | Water: 1158.0 mm | LR: 1.75e-01
Episode 350/2000 | Score: -1228 | Avg: -920 | Steps: 162 | Yield: 11780.0 kg/ha | N: 2600.0 kg/ha | Water: 906.0 mm | LR: 1.74e-01
Episode 351/2000 | Score: -1054 | Avg: -920 | Steps: 161 | Yield: 11485.7 kg/ha | N: 2320.0 kg/ha | Water: 930.0 mm | LR: 1.73e-01
Episode 352/2000 | Score: -1196 | Avg: -920 | Steps: 165 | Yield: 11900.0 kg/ha | N: 2440.0 kg/ha | Water: 1032.0 mm | LR: 1.72e-01
Episode 353/2000 | Score: -1297 | Avg: -920 | Steps: 154 | Yield: 9428.0 kg/ha | N: 2040.0 kg/ha | Water: 1056.0 mm | LR: 1.71e-01
Episode 354/2000 | Score: -1428 | Avg: -920 | Steps: 162 | Yield: 11420.7 kg/ha | N: 2680.0 kg/ha | Water: 1002.0 mm | LR: 1.70e-01
Episode 355/2000 | Score: -1163 | Avg: -920 | Steps: 156 | Yield: 10769.6 kg/ha | N: 2440.0 kg/ha | Water: 840.0 mm | LR: 1.70e-01
Episode 356/2000 | Score: -1229 | Avg: -920 | Steps: 152 | Yield: 11906.5 kg/ha | N: 2280.0 kg/ha | Water: 1080.0 mm | LR: 1.69e-01
Episode 357/2000 | Score: -1120 | Avg: -920 | Steps: 158 | Yield: 11127.6 kg/ha | N: 2240.0 kg/ha | Water: 996.0 mm | LR: 1.68e-01
Episode 358/2000 | Score: -1243 | Avg: -920 | Steps: 160 | Yield: 10654.4 kg/ha | N: 2000.0 kg/ha | Water: 1212.0 mm | LR: 1.67e-01
Episode 359/2000 | Score: -887 | Avg: -920 | Steps: 156 | Yield: 10667.1 kg/ha | N: 1520.0 kg/ha | Water: 1218.0 mm | LR: 1.66e-01
Episode 360/2000 | Score: -1418 | Avg: -920 | Steps: 152 | Yield: 11597.9 kg/ha | N: 2280.0 kg/ha | Water: 1260.0 mm | LR: 1.65e-01
Episode 361/2000 | Score: -1821 | Avg: -920 | Steps: 164 | Yield: 11033.2 kg/ha | N: 3200.0 kg/ha | Water: 930.0 mm | LR: 1.65e-01
Episode 362/2000 | Score: -1102 | Avg: -920 | Steps: 158 | Yield: 9099.7 kg/ha | N: 1720.0 kg/ha | Water: 1062.0 mm | LR: 1.64e-01
Episode 363/2000 | Score: -1040 | Avg: -920 | Steps: 160 | Yield: 12052.7 kg/ha | N: 2400.0 kg/ha | Water: 918.0 mm | LR: 1.63e-01
Episode 364/2000 | Score: -943 | Avg: -920 | Steps: 160 | Yield: 10518.2 kg/ha | N: 1720.0 kg/ha | Water: 1098.0 mm | LR: 1.62e-01
Episode 365/2000 | Score: -827 | Avg: -920 | Steps: 157 | Yield: 12832.4 kg/ha | N: 2320.0 kg/ha | Water: 900.0 mm | LR: 1.61e-01
Episode 366/2000 | Score: -1421 | Avg: -920 | Steps: 162 | Yield: 9948.6 kg/ha | N: 2280.0 kg/ha | Water: 1002.0 mm | LR: 1.60e-01
Episode 367/2000 | Score: -1670 | Avg: -920 | Steps: 156 | Yield: 7387.8 kg/ha | N: 2480.0 kg/ha | Water: 786.0 mm | LR: 1.60e-01
Episode 368/2000 | Score: -1089 | Avg: -920 | Steps: 155 | Yield: 12131.6 kg/ha | N: 2320.0 kg/ha | Water: 1032.0 mm | LR: 1.59e-01
Episode 369/2000 | Score: -1195 | Avg: -920 | Steps: 155 | Yield: 11208.6 kg/ha | N: 2200.0 kg/ha | Water: 1104.0 mm | LR: 1.58e-01
Episode 370/2000 | Score: -866 | Avg: -920 | Steps: 158 | Yield: 12375.0 kg/ha | N: 2320.0 kg/ha | Water: 864.0 mm | LR: 1.57e-01
Episode 371/2000 | Score: -1150 | Avg: -920 | Steps: 165 | Yield: 11944.5 kg/ha | N: 2760.0 kg/ha | Water: 750.0 mm | LR: 1.57e-01
Episode 372/2000 | Score: -882 | Avg: -920 | Steps: 167 | Yield: 11544.9 kg/ha | N: 2040.0 kg/ha | Water: 966.0 mm | LR: 1.56e-01
Episode 373/2000 | Score: -1028 | Avg: -920 | Steps: 159 | Yield: 11157.2 kg/ha | N: 2160.0 kg/ha | Water: 894.0 mm | LR: 1.55e-01
Episode 374/2000 | Score: -1260 | Avg: -920 | Steps: 163 | Yield: 10510.1 kg/ha | N: 2360.0 kg/ha | Water: 960.0 mm | LR: 1.54e-01
Episode 375/2000 | Score: -540 | Avg: -920 | Steps: 156 | Yield: 11312.8 kg/ha | N: 1400.0 kg/ha | Water: 1098.0 mm | LR: 1.53e-01
Episode 376/2000 | Score: -1454 | Avg: -920 | Steps: 162 | Yield: 7034.8 kg/ha | N: 2320.0 kg/ha | Water: 654.0 mm | LR: 1.53e-01
Episode 377/2000 | Score: -1121 | Avg: -920 | Steps: 160 | Yield: 10179.6 kg/ha | N: 1960.0 kg/ha | Water: 1062.0 mm | LR: 1.52e-01
Episode 378/2000 | Score: -1106 | Avg: -920 | Steps: 157 | Yield: 11146.4 kg/ha | N: 2320.0 kg/ha | Water: 912.0 mm | LR: 1.51e-01
Episode 379/2000 | Score: -1184 | Avg: -920 | Steps: 154 | Yield: 11960.2 kg/ha | N: 2640.0 kg/ha | Water: 858.0 mm | LR: 1.50e-01
Episode 380/2000 | Score: -784 | Avg: -920 | Steps: 162 | Yield: 10851.7 kg/ha | N: 1560.0 kg/ha | Water: 1116.0 mm | LR: 1.50e-01
Episode 381/2000 | Score: -948 | Avg: -920 | Steps: 160 | Yield: 11708.8 kg/ha | N: 2240.0 kg/ha | Water: 894.0 mm | LR: 1.49e-01
Episode 382/2000 | Score: -636 | Avg: -920 | Steps: 158 | Yield: 11591.3 kg/ha | N: 1720.0 kg/ha | Water: 996.0 mm | LR: 1.48e-01
Episode 383/2000 | Score: -456 | Avg: -920 | Steps: 158 | Yield: 12404.1 kg/ha | N: 1640.0 kg/ha | Water: 990.0 mm | LR: 1.47e-01
Episode 384/2000 | Score: -711 | Avg: -920 | Steps: 161 | Yield: 10293.9 kg/ha | N: 1680.0 kg/ha | Water: 906.0 mm | LR: 1.47e-01
Episode 385/2000 | Score: -711 | Avg: -920 | Steps: 158 | Yield: 11239.5 kg/ha | N: 1880.0 kg/ha | Water: 876.0 mm | LR: 1.46e-01
Episode 386/2000 | Score: -1127 | Avg: -920 | Steps: 161 | Yield: 10591.4 kg/ha | N: 2040.0 kg/ha | Water: 948.0 mm | LR: 1.45e-01
Episode 387/2000 | Score: -758 | Avg: -920 | Steps: 161 | Yield: 11913.7 kg/ha | N: 1880.0 kg/ha | Water: 1038.0 mm | LR: 1.44e-01
Episode 388/2000 | Score: -521 | Avg: -920 | Steps: 156 | Yield: 10817.7 kg/ha | N: 1480.0 kg/ha | Water: 930.0 mm | LR: 1.44e-01
Episode 389/2000 | Score: -927 | Avg: -920 | Steps: 164 | Yield: 10467.5 kg/ha | N: 2080.0 kg/ha | Water: 840.0 mm | LR: 1.43e-01
Episode 390/2000 | Score: -1004 | Avg: -920 | Steps: 159 | Yield: 11293.9 kg/ha | N: 2360.0 kg/ha | Water: 828.0 mm | LR: 1.42e-01
Episode 391/2000 | Score: -958 | Avg: -920 | Steps: 157 | Yield: 10152.2 kg/ha | N: 2040.0 kg/ha | Water: 864.0 mm | LR: 1.42e-01
Episode 392/2000 | Score: -1085 | Avg: -920 | Steps: 161 | Yield: 12199.0 kg/ha | N: 2240.0 kg/ha | Water: 1044.0 mm | LR: 1.41e-01
Episode 393/2000 | Score: -560 | Avg: -920 | Steps: 156 | Yield: 10463.4 kg/ha | N: 1600.0 kg/ha | Water: 834.0 mm | LR: 1.40e-01
Episode 394/2000 | Score: -986 | Avg: -920 | Steps: 158 | Yield: 8882.2 kg/ha | N: 1880.0 kg/ha | Water: 822.0 mm | LR: 1.39e-01
Episode 395/2000 | Score: -608 | Avg: -920 | Steps: 156 | Yield: 9667.2 kg/ha | N: 1600.0 kg/ha | Water: 792.0 mm | LR: 1.39e-01
Episode 396/2000 | Score: -572 | Avg: -920 | Steps: 164 | Yield: 10166.1 kg/ha | N: 1680.0 kg/ha | Water: 774.0 mm | LR: 1.38e-01
Episode 397/2000 | Score: -867 | Avg: -920 | Steps: 160 | Yield: 10002.1 kg/ha | N: 1920.0 kg/ha | Water: 846.0 mm | LR: 1.37e-01
Episode 398/2000 | Score: -1048 | Avg: -920 | Steps: 158 | Yield: 9954.5 kg/ha | N: 1880.0 kg/ha | Water: 1032.0 mm | LR: 1.37e-01
Episode 399/2000 | Score: -562 | Avg: -920 | Steps: 165 | Yield: 11701.8 kg/ha | N: 1840.0 kg/ha | Water: 870.0 mm | LR: 1.36e-01
Episode 400/2000 | Score: -1021 | Avg: -920 | Steps: 161 | Yield: 11549.6 kg/ha | N: 2200.0 kg/ha | Water: 978.0 mm | LR: 1.35e-01
Episode 401/2000 | Score: -890 | Avg: -920 | Steps: 154 | Yield: 12196.4 kg/ha | N: 2120.0 kg/ha | Water: 1026.0 mm | LR: 1.35e-01
Episode 402/2000 | Score: -758 | Avg: -920 | Steps: 157 | Yield: 11656.7 kg/ha | N: 2040.0 kg/ha | Water: 870.0 mm | LR: 1.34e-01
Episode 403/2000 | Score: -690 | Avg: -920 | Steps: 163 | Yield: 10169.7 kg/ha | N: 1520.0 kg/ha | Water: 972.0 mm | LR: 1.33e-01
Episode 404/2000 | Score: -180 | Avg: -920 | Steps: 159 | Yield: 11547.8 kg/ha | N: 1360.0 kg/ha | Water: 822.0 mm | LR: 1.33e-01
Episode 405/2000 | Score: -883 | Avg: -920 | Steps: 162 | Yield: 11521.1 kg/ha | N: 2160.0 kg/ha | Water: 882.0 mm | LR: 1.32e-01
Episode 406/2000 | Score: -578 | Avg: -920 | Steps: 158 | Yield: 9526.4 kg/ha | N: 1360.0 kg/ha | Water: 882.0 mm | LR: 1.31e-01
Episode 407/2000 | Score: -862 | Avg: -920 | Steps: 164 | Yield: 9667.4 kg/ha | N: 1880.0 kg/ha | Water: 822.0 mm | LR: 1.31e-01
Episode 408/2000 | Score: -370 | Avg: -920 | Steps: 157 | Yield: 10055.0 kg/ha | N: 1360.0 kg/ha | Water: 804.0 mm | LR: 1.30e-01
Episode 409/2000 | Score: -523 | Avg: -920 | Steps: 160 | Yield: 9946.4 kg/ha | N: 1440.0 kg/ha | Water: 870.0 mm | LR: 1.29e-01
Episode 410/2000 | Score: -1242 | Avg: -920 | Steps: 158 | Yield: 9554.9 kg/ha | N: 2080.0 kg/ha | Water: 1008.0 mm | LR: 1.29e-01
Episode 411/2000 | Score: -979 | Avg: -920 | Steps: 161 | Yield: 9840.0 kg/ha | N: 2080.0 kg/ha | Water: 810.0 mm | LR: 1.28e-01
Episode 412/2000 | Score: -566 | Avg: -920 | Steps: 159 | Yield: 11933.5 kg/ha | N: 1960.0 kg/ha | Water: 792.0 mm | LR: 1.27e-01
Episode 413/2000 | Score: -221 | Avg: -920 | Steps: 161 | Yield: 11315.1 kg/ha | N: 1240.0 kg/ha | Water: 936.0 mm | LR: 1.27e-01
Episode 414/2000 | Score: -258 | Avg: -920 | Steps: 163 | Yield: 12364.5 kg/ha | N: 1680.0 kg/ha | Water: 792.0 mm | LR: 1.26e-01
Episode 415/2000 | Score: -1130 | Avg: -920 | Steps: 165 | Yield: 12487.6 kg/ha | N: 2560.0 kg/ha | Water: 954.0 mm | LR: 1.26e-01
Episode 416/2000 | Score: -512 | Avg: -920 | Steps: 164 | Yield: 12126.9 kg/ha | N: 1880.0 kg/ha | Water: 828.0 mm | LR: 1.25e-01
Episode 417/2000 | Score: -1281 | Avg: -920 | Steps: 157 | Yield: 8748.4 kg/ha | N: 2160.0 kg/ha | Water: 870.0 mm | LR: 1.24e-01
Episode 418/2000 | Score: -692 | Avg: -920 | Steps: 165 | Yield: 12134.0 kg/ha | N: 2160.0 kg/ha | Water: 792.0 mm | LR: 1.24e-01
Episode 419/2000 | Score: -469 | Avg: -920 | Steps: 163 | Yield: 10593.5 kg/ha | N: 1520.0 kg/ha | Water: 822.0 mm | LR: 1.23e-01
Episode 420/2000 | Score: -854 | Avg: -920 | Steps: 159 | Yield: 11281.7 kg/ha | N: 2160.0 kg/ha | Water: 822.0 mm | LR: 1.22e-01
Episode 421/2000 | Score: -694 | Avg: -920 | Steps: 165 | Yield: 11615.9 kg/ha | N: 1840.0 kg/ha | Water: 954.0 mm | LR: 1.22e-01
Episode 422/2000 | Score: -777 | Avg: -920 | Steps: 154 | Yield: 8932.1 kg/ha | N: 1600.0 kg/ha | Water: 840.0 mm | LR: 1.21e-01
Episode 423/2000 | Score: -780 | Avg: -920 | Steps: 159 | Yield: 10152.8 kg/ha | N: 1840.0 kg/ha | Water: 834.0 mm | LR: 1.21e-01
Episode 424/2000 | Score: -678 | Avg: -920 | Steps: 160 | Yield: 9720.5 kg/ha | N: 1400.0 kg/ha | Water: 972.0 mm | LR: 1.20e-01
Episode 425/2000 | Score: -783 | Avg: -920 | Steps: 160 | Yield: 12377.8 kg/ha | N: 2240.0 kg/ha | Water: 852.0 mm | LR: 1.19e-01
Episode 426/2000 | Score: -1140 | Avg: -920 | Steps: 166 | Yield: 12134.4 kg/ha | N: 2400.0 kg/ha | Water: 1038.0 mm | LR: 1.19e-01
Episode 427/2000 | Score: -584 | Avg: -920 | Steps: 161 | Yield: 9617.1 kg/ha | N: 1560.0 kg/ha | Water: 792.0 mm | LR: 1.18e-01
Episode 428/2000 | Score: -637 | Avg: -920 | Steps: 160 | Yield: 10975.1 kg/ha | N: 1800.0 kg/ha | Water: 834.0 mm | LR: 1.18e-01
Episode 429/2000 | Score: -405 | Avg: -920 | Steps: 161 | Yield: 10225.0 kg/ha | N: 1480.0 kg/ha | Water: 762.0 mm | LR: 1.17e-01
Episode 430/2000 | Score: -804 | Avg: -920 | Steps: 156 | Yield: 8640.7 kg/ha | N: 1560.0 kg/ha | Water: 852.0 mm | LR: 1.16e-01
Episode 431/2000 | Score: -418 | Avg: -920 | Steps: 159 | Yield: 11445.4 kg/ha | N: 1640.0 kg/ha | Water: 834.0 mm | LR: 1.16e-01
Episode 432/2000 | Score: -445 | Avg: -920 | Steps: 165 | Yield: 10750.6 kg/ha | N: 1400.0 kg/ha | Water: 810.0 mm | LR: 1.15e-01
Episode 433/2000 | Score: -513 | Avg: -920 | Steps: 164 | Yield: 10530.1 kg/ha | N: 1600.0 kg/ha | Water: 732.0 mm | LR: 1.15e-01
Episode 434/2000 | Score: -820 | Avg: -920 | Steps: 165 | Yield: 11543.4 kg/ha | N: 1840.0 kg/ha | Water: 984.0 mm | LR: 1.14e-01
Episode 435/2000 | Score: -471 | Avg: -920 | Steps: 158 | Yield: 11932.7 kg/ha | N: 1640.0 kg/ha | Water: 936.0 mm | LR: 1.14e-01
Episode 436/2000 | Score: -647 | Avg: -920 | Steps: 163 | Yield: 11389.8 kg/ha | N: 1720.0 kg/ha | Water: 960.0 mm | LR: 1.13e-01
Episode 437/2000 | Score: -1735 | Avg: -920 | Steps: 165 | Yield: 6712.7 kg/ha | N: 2120.0 kg/ha | Water: 978.0 mm | LR: 1.12e-01
Episode 438/2000 | Score: 97 | Avg: -920 | Steps: 159 | Yield: 11380.9 kg/ha | N: 1000.0 kg/ha | Water: 816.0 mm | LR: 1.12e-01
Episode 439/2000 | Score: -1280 | Avg: -920 | Steps: 162 | Yield: 9411.7 kg/ha | N: 2040.0 kg/ha | Water: 1038.0 mm | LR: 1.11e-01
Episode 440/2000 | Score: -362 | Avg: -920 | Steps: 165 | Yield: 10933.0 kg/ha | N: 1200.0 kg/ha | Water: 1026.0 mm | LR: 1.11e-01
Episode 441/2000 | Score: -755 | Avg: -920 | Steps: 155 | Yield: 11500.5 kg/ha | N: 1760.0 kg/ha | Water: 1062.0 mm | LR: 1.10e-01
Episode 442/2000 | Score: -1113 | Avg: -920 | Steps: 160 | Yield: 10577.4 kg/ha | N: 2440.0 kg/ha | Water: 750.0 mm | LR: 1.10e-01
Episode 443/2000 | Score: -435 | Avg: -920 | Steps: 162 | Yield: 12254.9 kg/ha | N: 1600.0 kg/ha | Water: 978.0 mm | LR: 1.09e-01
Episode 444/2000 | Score: -13 | Avg: -920 | Steps: 163 | Yield: 11772.5 kg/ha | N: 1120.0 kg/ha | Water: 870.0 mm | LR: 1.09e-01
Episode 445/2000 | Score: -811 | Avg: -920 | Steps: 158 | Yield: 11073.4 kg/ha | N: 1880.0 kg/ha | Water: 966.0 mm | LR: 1.08e-01
Episode 446/2000 | Score: -670 | Avg: -920 | Steps: 159 | Yield: 10965.5 kg/ha | N: 1680.0 kg/ha | Water: 966.0 mm | LR: 1.07e-01
Episode 447/2000 | Score: -386 | Avg: -920 | Steps: 162 | Yield: 11632.0 kg/ha | N: 1680.0 kg/ha | Water: 786.0 mm | LR: 1.07e-01
Episode 448/2000 | Score: -1679 | Avg: -920 | Steps: 159 | Yield: 8167.8 kg/ha | N: 2240.0 kg/ha | Water: 1062.0 mm | LR: 1.06e-01
Episode 449/2000 | Score: -474 | Avg: -920 | Steps: 164 | Yield: 11527.1 kg/ha | N: 1360.0 kg/ha | Water: 1086.0 mm | LR: 1.06e-01
Episode 450/2000 | Score: -77 | Avg: -920 | Steps: 163 | Yield: 11484.6 kg/ha | N: 1160.0 kg/ha | Water: 858.0 mm | LR: 1.05e-01
Episode 451/2000 | Score: 167 | Avg: -920 | Steps: 159 | Yield: 12006.6 kg/ha | N: 1080.0 kg/ha | Water: 756.0 mm | LR: 1.05e-01
Episode 452/2000 | Score: -373 | Avg: -920 | Steps: 159 | Yield: 9987.8 kg/ha | N: 1400.0 kg/ha | Water: 744.0 mm | LR: 1.04e-01
Episode 453/2000 | Score: -820 | Avg: -920 | Steps: 161 | Yield: 2489.4 kg/ha | N: 920.0 kg/ha | Water: 414.0 mm | LR: 1.04e-01
Episode 454/2000 | Score: -90 | Avg: -920 | Steps: 163 | Yield: 11008.5 kg/ha | N: 1280.0 kg/ha | Water: 744.0 mm | LR: 1.03e-01
Episode 455/2000 | Score: -233 | Avg: -920 | Steps: 162 | Yield: 11767.9 kg/ha | N: 1640.0 kg/ha | Water: 696.0 mm | LR: 1.03e-01
Episode 456/2000 | Score: -1244 | Avg: -920 | Steps: 161 | Yield: 9203.3 kg/ha | N: 1920.0 kg/ha | Water: 1068.0 mm | LR: 1.02e-01
Episode 457/2000 | Score: -300 | Avg: -920 | Steps: 155 | Yield: 11015.3 kg/ha | N: 1480.0 kg/ha | Water: 780.0 mm | LR: 1.02e-01
Episode 458/2000 | Score: -528 | Avg: -920 | Steps: 162 | Yield: 10221.8 kg/ha | N: 1560.0 kg/ha | Water: 816.0 mm | LR: 1.01e-01
Episode 459/2000 | Score: -383 | Avg: -920 | Steps: 158 | Yield: 10900.0 kg/ha | N: 1520.0 kg/ha | Water: 822.0 mm | LR: 1.01e-01
Episode 460/2000 | Score: -380 | Avg: -920 | Steps: 165 | Yield: 11535.0 kg/ha | N: 1720.0 kg/ha | Water: 738.0 mm | LR: 1.00e-01
Episode 461/2000 | Score: 136 | Avg: -920 | Steps: 163 | Yield: 11568.7 kg/ha | N: 1080.0 kg/ha | Water: 744.0 mm | LR: 9.97e-02
Episode 462/2000 | Score: -188 | Avg: -920 | Steps: 166 | Yield: 12651.2 kg/ha | N: 1600.0 kg/ha | Water: 792.0 mm | LR: 9.92e-02
Episode 463/2000 | Score: -18 | Avg: -920 | Steps: 161 | Yield: 9357.9 kg/ha | N: 800.0 kg/ha | Water: 786.0 mm | LR: 9.87e-02
Episode 464/2000 | Score: -95 | Avg: -920 | Steps: 165 | Yield: 11795.5 kg/ha | N: 1560.0 kg/ha | Water: 648.0 mm | LR: 9.82e-02
Episode 465/2000 | Score: 114 | Avg: -920 | Steps: 156 | Yield: 10656.9 kg/ha | N: 960.0 kg/ha | Water: 726.0 mm | LR: 9.77e-02
Episode 466/2000 | Score: -450 | Avg: -920 | Steps: 160 | Yield: 11537.3 kg/ha | N: 2000.0 kg/ha | Water: 618.0 mm | LR: 9.72e-02
Episode 467/2000 | Score: 148 | Avg: -920 | Steps: 156 | Yield: 11183.5 kg/ha | N: 1400.0 kg/ha | Water: 438.0 mm | LR: 9.67e-02
Episode 468/2000 | Score: -209 | Avg: -920 | Steps: 160 | Yield: 9521.3 kg/ha | N: 1160.0 kg/ha | Water: 696.0 mm | LR: 9.62e-02
Episode 469/2000 | Score: -476 | Avg: -920 | Steps: 154 | Yield: 11621.1 kg/ha | N: 1600.0 kg/ha | Water: 906.0 mm | LR: 9.58e-02
Episode 470/2000 | Score: -348 | Avg: -920 | Steps: 159 | Yield: 10720.1 kg/ha | N: 1240.0 kg/ha | Water: 966.0 mm | LR: 9.53e-02
Episode 471/2000 | Score: -472 | Avg: -920 | Steps: 157 | Yield: 11076.3 kg/ha | N: 1560.0 kg/ha | Water: 888.0 mm | LR: 9.48e-02
Episode 472/2000 | Score: -83 | Avg: -920 | Steps: 158 | Yield: 9954.0 kg/ha | N: 1160.0 kg/ha | Water: 660.0 mm | LR: 9.43e-02
Episode 473/2000 | Score: -54 | Avg: -920 | Steps: 163 | Yield: 9511.3 kg/ha | N: 960.0 kg/ha | Water: 714.0 mm | LR: 9.39e-02
Episode 474/2000 | Score: -21 | Avg: -920 | Steps: 161 | Yield: 10728.3 kg/ha | N: 1120.0 kg/ha | Water: 756.0 mm | LR: 9.34e-02
Episode 475/2000 | Score: -1095 | Avg: -920 | Steps: 156 | Yield: 12346.4 kg/ha | N: 2320.0 kg/ha | Water: 1074.0 mm | LR: 9.29e-02
Episode 476/2000 | Score: -216 | Avg: -920 | Steps: 156 | Yield: 11153.3 kg/ha | N: 1360.0 kg/ha | Water: 822.0 mm | LR: 9.25e-02
Episode 477/2000 | Score: -603 | Avg: -920 | Steps: 159 | Yield: 10851.0 kg/ha | N: 1680.0 kg/ha | Water: 900.0 mm | LR: 9.20e-02
Episode 478/2000 | Score: -281 | Avg: -920 | Steps: 155 | Yield: 10685.6 kg/ha | N: 1400.0 kg/ha | Water: 756.0 mm | LR: 9.15e-02
Episode 479/2000 | Score: 54 | Avg: -920 | Steps: 165 | Yield: 10969.3 kg/ha | N: 1200.0 kg/ha | Water: 630.0 mm | LR: 9.11e-02
Episode 480/2000 | Score: -168 | Avg: -920 | Steps: 158 | Yield: 11623.5 kg/ha | N: 1560.0 kg/ha | Water: 690.0 mm | LR: 9.06e-02
Episode 481/2000 | Score: -445 | Avg: -920 | Steps: 169 | Yield: 8388.7 kg/ha | N: 1280.0 kg/ha | Water: 690.0 mm | LR: 9.02e-02
Episode 482/2000 | Score: 202 | Avg: -920 | Steps: 151 | Yield: 11247.0 kg/ha | N: 1200.0 kg/ha | Water: 558.0 mm | LR: 8.97e-02
Episode 483/2000 | Score: -453 | Avg: -920 | Steps: 161 | Yield: 7835.3 kg/ha | N: 1280.0 kg/ha | Water: 618.0 mm | LR: 8.93e-02
Episode 484/2000 | Score: 166 | Avg: -920 | Steps: 160 | Yield: 11476.6 kg/ha | N: 1160.0 kg/ha | Water: 612.0 mm | LR: 8.88e-02
Episode 485/2000 | Score: -210 | Avg: -920 | Steps: 159 | Yield: 12776.0 kg/ha | N: 1560.0 kg/ha | Water: 894.0 mm | LR: 8.84e-02
Episode 486/2000 | Score: -673 | Avg: -920 | Steps: 159 | Yield: 9057.1 kg/ha | N: 1360.0 kg/ha | Water: 936.0 mm | LR: 8.79e-02
Episode 487/2000 | Score: 225 | Avg: -920 | Steps: 157 | Yield: 9617.3 kg/ha | N: 680.0 kg/ha | Water: 660.0 mm | LR: 8.75e-02
Episode 488/2000 | Score: -726 | Avg: -920 | Steps: 165 | Yield: 10090.2 kg/ha | N: 1960.0 kg/ha | Water: 702.0 mm | LR: 8.71e-02
Episode 489/2000 | Score: -86 | Avg: -920 | Steps: 154 | Yield: 12655.5 kg/ha | N: 1480.0 kg/ha | Water: 804.0 mm | LR: 8.66e-02
Episode 490/2000 | Score: -683 | Avg: -920 | Steps: 163 | Yield: 12111.9 kg/ha | N: 1800.0 kg/ha | Water: 1056.0 mm | LR: 8.62e-02
Episode 491/2000 | Score: -336 | Avg: -920 | Steps: 155 | Yield: 10840.0 kg/ha | N: 1320.0 kg/ha | Water: 828.0 mm | LR: 8.58e-02
Episode 492/2000 | Score: 218 | Avg: -920 | Steps: 158 | Yield: 11853.1 kg/ha | N: 960.0 kg/ha | Water: 786.0 mm | LR: 8.53e-02
Episode 493/2000 | Score: -404 | Avg: -920 | Steps: 162 | Yield: 10672.5 kg/ha | N: 1560.0 kg/ha | Water: 780.0 mm | LR: 8.49e-02
Episode 494/2000 | Score: -242 | Avg: -920 | Steps: 160 | Yield: 9054.3 kg/ha | N: 1240.0 kg/ha | Water: 630.0 mm | LR: 8.45e-02
Episode 495/2000 | Score: -128 | Avg: -920 | Steps: 155 | Yield: 9738.6 kg/ha | N: 1040.0 kg/ha | Water: 756.0 mm | LR: 8.41e-02
Episode 496/2000 | Score: -347 | Avg: -920 | Steps: 159 | Yield: 11272.1 kg/ha | N: 1240.0 kg/ha | Water: 1032.0 mm | LR: 8.36e-02
Episode 497/2000 | Score: -959 | Avg: -920 | Steps: 156 | Yield: 11483.4 kg/ha | N: 2000.0 kg/ha | Water: 1038.0 mm | LR: 8.32e-02
Episode 498/2000 | Score: 205 | Avg: -920 | Steps: 162 | Yield: 9690.2 kg/ha | N: 1120.0 kg/ha | Water: 372.0 mm | LR: 8.28e-02
Episode 499/2000 | Score: -361 | Avg: -920 | Steps: 163 | Yield: 7028.2 kg/ha | N: 760.0 kg/ha | Water: 792.0 mm | LR: 8.24e-02
Episode 500/2000 | Score: -10 | Avg: -920 | Steps: 161 | Yield: 9786.5 kg/ha | N: 800.0 kg/ha | Water: 840.0 mm | LR: 8.20e-02
Episode 501/2000 | Score: -65 | Avg: -920 | Steps: 160 | Yield: 11528.2 kg/ha | N: 1160.0 kg/ha | Water: 882.0 mm | LR: 8.16e-02
Episode 502/2000 | Score: 275 | Avg: -920 | Steps: 162 | Yield: 11622.3 kg/ha | N: 800.0 kg/ha | Water: 816.0 mm | LR: 8.12e-02
Episode 503/2000 | Score: -492 | Avg: -920 | Steps: 156 | Yield: 11177.4 kg/ha | N: 1440.0 kg/ha | Water: 990.0 mm | LR: 8.08e-02
Episode 504/2000 | Score: 346 | Avg: -920 | Steps: 158 | Yield: 12259.9 kg/ha | N: 920.0 kg/ha | Water: 786.0 mm | LR: 8.04e-02
Episode 505/2000 | Score: -583 | Avg: -920 | Steps: 160 | Yield: 8467.0 kg/ha | N: 1320.0 kg/ha | Water: 798.0 mm | LR: 8.00e-02
Episode 506/2000 | Score: 387 | Avg: -920 | Steps: 159 | Yield: 11423.0 kg/ha | N: 600.0 kg/ha | Water: 846.0 mm | LR: 7.96e-02
Episode 507/2000 | Score: 415 | Avg: -920 | Steps: 157 | Yield: 12147.5 kg/ha | N: 920.0 kg/ha | Water: 678.0 mm | LR: 7.92e-02
Episode 508/2000 | Score: -1152 | Avg: -920 | Steps: 162 | Yield: 9642.5 kg/ha | N: 2000.0 kg/ha | Water: 990.0 mm | LR: 7.88e-02
Episode 509/2000 | Score: -174 | Avg: -920 | Steps: 158 | Yield: 10000.4 kg/ha | N: 1280.0 kg/ha | Water: 600.0 mm | LR: 7.84e-02
Episode 510/2000 | Score: 93 | Avg: -920 | Steps: 168 | Yield: 11254.2 kg/ha | N: 920.0 kg/ha | Water: 756.0 mm | LR: 7.80e-02
Episode 511/2000 | Score: 191 | Avg: -920 | Steps: 159 | Yield: 11672.9 kg/ha | N: 840.0 kg/ha | Water: 900.0 mm | LR: 7.76e-02
Episode 512/2000 | Score: -219 | Avg: -920 | Steps: 162 | Yield: 11911.0 kg/ha | N: 1640.0 kg/ha | Water: 726.0 mm | LR: 7.72e-02
Episode 513/2000 | Score: 181 | Avg: -920 | Steps: 164 | Yield: 12569.8 kg/ha | N: 1200.0 kg/ha | Water: 750.0 mm | LR: 7.68e-02
Episode 514/2000 | Score: -280 | Avg: -920 | Steps: 158 | Yield: 9234.1 kg/ha | N: 1240.0 kg/ha | Water: 684.0 mm | LR: 7.64e-02
Episode 515/2000 | Score: -878 | Avg: -920 | Steps: 160 | Yield: 11641.7 kg/ha | N: 1760.0 kg/ha | Water: 1206.0 mm | LR: 7.60e-02
Episode 516/2000 | Score: 194 | Avg: -920 | Steps: 156 | Yield: 11094.7 kg/ha | N: 1080.0 kg/ha | Water: 642.0 mm | LR: 7.57e-02
Episode 517/2000 | Score: -656 | Avg: -920 | Steps: 161 | Yield: 9866.1 kg/ha | N: 1600.0 kg/ha | Water: 864.0 mm | LR: 7.53e-02
Episode 518/2000 | Score: -1073 | Avg: -920 | Steps: 163 | Yield: 11994.0 kg/ha | N: 1760.0 kg/ha | Water: 1428.0 mm | LR: 7.49e-02
Episode 519/2000 | Score: 26 | Avg: -920 | Steps: 160 | Yield: 11593.3 kg/ha | N: 1360.0 kg/ha | Water: 636.0 mm | LR: 7.45e-02
Episode 520/2000 | Score: 24 | Avg: -920 | Steps: 162 | Yield: 11407.8 kg/ha | N: 1240.0 kg/ha | Water: 726.0 mm | LR: 7.42e-02
Episode 521/2000 | Score: -154 | Avg: -920 | Steps: 165 | Yield: 10919.5 kg/ha | N: 1280.0 kg/ha | Water: 732.0 mm | LR: 7.38e-02
Episode 522/2000 | Score: 427 | Avg: -920 | Steps: 167 | Yield: 10522.4 kg/ha | N: 520.0 kg/ha | Water: 750.0 mm | LR: 7.34e-02
Episode 523/2000 | Score: -994 | Avg: -920 | Steps: 163 | Yield: 10760.9 kg/ha | N: 2040.0 kg/ha | Water: 960.0 mm | LR: 7.31e-02
Episode 524/2000 | Score: 398 | Avg: -920 | Steps: 161 | Yield: 10329.0 kg/ha | N: 760.0 kg/ha | Water: 564.0 mm | LR: 7.27e-02
Episode 525/2000 | Score: 263 | Avg: -920 | Steps: 155 | Yield: 12678.6 kg/ha | N: 1160.0 kg/ha | Water: 720.0 mm | LR: 7.23e-02
Episode 526/2000 | Score: 285 | Avg: -920 | Steps: 162 | Yield: 9028.6 kg/ha | N: 520.0 kg/ha | Water: 624.0 mm | LR: 7.20e-02
Episode 527/2000 | Score: 270 | Avg: -920 | Steps: 156 | Yield: 9435.1 kg/ha | N: 560.0 kg/ha | Water: 708.0 mm | LR: 7.16e-02
Episode 528/2000 | Score: 277 | Avg: -920 | Steps: 156 | Yield: 11330.7 kg/ha | N: 880.0 kg/ha | Water: 744.0 mm | LR: 7.12e-02
Episode 529/2000 | Score: 301 | Avg: -920 | Steps: 159 | Yield: 11824.0 kg/ha | N: 640.0 kg/ha | Water: 924.0 mm | LR: 7.09e-02
Episode 530/2000 | Score: -371 | Avg: -920 | Steps: 163 | Yield: 10403.6 kg/ha | N: 880.0 kg/ha | Water: 1200.0 mm | LR: 7.05e-02
Episode 531/2000 | Score: 392 | Avg: -920 | Steps: 162 | Yield: 11483.2 kg/ha | N: 840.0 kg/ha | Water: 690.0 mm | LR: 7.02e-02
Episode 532/2000 | Score: 131 | Avg: -920 | Steps: 161 | Yield: 10082.9 kg/ha | N: 840.0 kg/ha | Water: 714.0 mm | LR: 6.98e-02
Episode 533/2000 | Score: 256 | Avg: -920 | Steps: 160 | Yield: 11007.6 kg/ha | N: 840.0 kg/ha | Water: 624.0 mm | LR: 6.95e-02
Episode 534/2000 | Score: -141 | Avg: -920 | Steps: 161 | Yield: 9199.6 kg/ha | N: 840.0 kg/ha | Water: 846.0 mm | LR: 6.91e-02
Episode 535/2000 | Score: -198 | Avg: -920 | Steps: 155 | Yield: 11008.2 kg/ha | N: 1240.0 kg/ha | Water: 756.0 mm | LR: 6.88e-02
Episode 536/2000 | Score: 76 | Avg: -920 | Steps: 164 | Yield: 9778.1 kg/ha | N: 840.0 kg/ha | Water: 732.0 mm | LR: 6.84e-02
Episode 537/2000 | Score: 105 | Avg: -920 | Steps: 157 | Yield: 11768.1 kg/ha | N: 1160.0 kg/ha | Water: 762.0 mm | LR: 6.81e-02
Episode 538/2000 | Score: 60 | Avg: -920 | Steps: 153 | Yield: 9099.3 kg/ha | N: 800.0 kg/ha | Water: 678.0 mm | LR: 6.78e-02
Episode 539/2000 | Score: 73 | Avg: -920 | Steps: 162 | Yield: 10726.5 kg/ha | N: 800.0 kg/ha | Water: 888.0 mm | LR: 6.74e-02
Episode 540/2000 | Score: 202 | Avg: -920 | Steps: 157 | Yield: 11566.0 kg/ha | N: 1080.0 kg/ha | Water: 702.0 mm | LR: 6.71e-02
Episode 541/2000 | Score: -188 | Avg: -920 | Steps: 169 | Yield: 10317.7 kg/ha | N: 1240.0 kg/ha | Water: 762.0 mm | LR: 6.68e-02
Episode 542/2000 | Score: -285 | Avg: -920 | Steps: 160 | Yield: 9858.8 kg/ha | N: 1280.0 kg/ha | Water: 756.0 mm | LR: 6.64e-02
Episode 543/2000 | Score: 213 | Avg: -920 | Steps: 157 | Yield: 11626.4 kg/ha | N: 1080.0 kg/ha | Water: 672.0 mm | LR: 6.61e-02
Episode 544/2000 | Score: -44 | Avg: -920 | Steps: 160 | Yield: 12023.7 kg/ha | N: 1360.0 kg/ha | Water: 750.0 mm | LR: 6.58e-02
Episode 545/2000 | Score: -42 | Avg: -920 | Steps: 167 | Yield: 11496.8 kg/ha | N: 1200.0 kg/ha | Water: 816.0 mm | LR: 6.54e-02
Episode 546/2000 | Score: 355 | Avg: -920 | Steps: 156 | Yield: 10110.7 kg/ha | N: 680.0 kg/ha | Water: 612.0 mm | LR: 6.51e-02
Episode 547/2000 | Score: -393 | Avg: -920 | Steps: 153 | Yield: 10796.6 kg/ha | N: 1120.0 kg/ha | Water: 1104.0 mm | LR: 6.48e-02
Episode 548/2000 | Score: 392 | Avg: -920 | Steps: 161 | Yield: 12398.8 kg/ha | N: 1000.0 kg/ha | Water: 678.0 mm | LR: 6.45e-02
Episode 549/2000 | Score: 173 | Avg: -920 | Steps: 153 | Yield: 10292.7 kg/ha | N: 720.0 kg/ha | Water: 804.0 mm | LR: 6.41e-02
Episode 550/2000 | Score: -36 | Avg: -920 | Steps: 165 | Yield: 9639.8 kg/ha | N: 1080.0 kg/ha | Water: 642.0 mm | LR: 6.38e-02
Episode 551/2000 | Score: 392 | Avg: -920 | Steps: 168 | Yield: 11001.8 kg/ha | N: 760.0 kg/ha | Water: 678.0 mm | LR: 6.35e-02
Episode 552/2000 | Score: 44 | Avg: -920 | Steps: 167 | Yield: 8348.3 kg/ha | N: 720.0 kg/ha | Water: 642.0 mm | LR: 6.32e-02
Episode 553/2000 | Score: 14 | Avg: -920 | Steps: 159 | Yield: 11162.3 kg/ha | N: 1120.0 kg/ha | Water: 786.0 mm | LR: 6.29e-02
Episode 554/2000 | Score: 70 | Avg: -920 | Steps: 160 | Yield: 10273.6 kg/ha | N: 880.0 kg/ha | Water: 780.0 mm | LR: 6.25e-02
Episode 555/2000 | Score: -9 | Avg: -920 | Steps: 161 | Yield: 11568.8 kg/ha | N: 1080.0 kg/ha | Water: 894.0 mm | LR: 6.22e-02
Episode 556/2000 | Score: -8 | Avg: -920 | Steps: 153 | Yield: 10516.2 kg/ha | N: 960.0 kg/ha | Water: 828.0 mm | LR: 6.19e-02
Episode 557/2000 | Score: -124 | Avg: -920 | Steps: 159 | Yield: 12771.4 kg/ha | N: 1600.0 kg/ha | Water: 798.0 mm | LR: 6.16e-02
Episode 558/2000 | Score: 160 | Avg: -920 | Steps: 155 | Yield: 9724.6 kg/ha | N: 640.0 kg/ha | Water: 792.0 mm | LR: 6.13e-02
Episode 559/2000 | Score: 184 | Avg: -920 | Steps: 163 | Yield: 10132.4 kg/ha | N: 800.0 kg/ha | Water: 714.0 mm | LR: 6.10e-02
Episode 560/2000 | Score: 339 | Avg: -920 | Steps: 154 | Yield: 10592.0 kg/ha | N: 720.0 kg/ha | Water: 696.0 mm | LR: 6.07e-02
Episode 561/2000 | Score: 96 | Avg: -920 | Steps: 151 | Yield: 11732.6 kg/ha | N: 1040.0 kg/ha | Water: 810.0 mm | LR: 6.04e-02
Episode 562/2000 | Score: -171 | Avg: -920 | Steps: 164 | Yield: 8621.7 kg/ha | N: 1080.0 kg/ha | Water: 618.0 mm | LR: 6.01e-02
Episode 563/2000 | Score: 178 | Avg: -920 | Steps: 168 | Yield: 9549.6 kg/ha | N: 640.0 kg/ha | Water: 750.0 mm | LR: 5.98e-02
Episode 564/2000 | Score: 260 | Avg: -920 | Steps: 160 | Yield: 11282.3 kg/ha | N: 1000.0 kg/ha | Water: 666.0 mm | LR: 5.95e-02
Episode 565/2000 | Score: 500 | Avg: -920 | Steps: 165 | Yield: 12691.2 kg/ha | N: 1120.0 kg/ha | Water: 564.0 mm | LR: 5.92e-02
Episode 566/2000 | Score: 524 | Avg: -920 | Steps: 153 | Yield: 11186.0 kg/ha | N: 680.0 kg/ha | Water: 630.0 mm | LR: 5.89e-02
Episode 567/2000 | Score: 260 | Avg: -920 | Steps: 164 | Yield: 8497.0 kg/ha | N: 520.0 kg/ha | Water: 582.0 mm | LR: 5.86e-02
Episode 568/2000 | Score: 455 | Avg: -920 | Steps: 162 | Yield: 10271.4 kg/ha | N: 720.0 kg/ha | Water: 516.0 mm | LR: 5.83e-02
Episode 569/2000 | Score: -184 | Avg: -920 | Steps: 162 | Yield: 9599.2 kg/ha | N: 1000.0 kg/ha | Water: 828.0 mm | LR: 5.80e-02
Episode 570/2000 | Score: 374 | Avg: -920 | Steps: 166 | Yield: 11588.6 kg/ha | N: 800.0 kg/ha | Water: 738.0 mm | LR: 5.77e-02
Episode 571/2000 | Score: 324 | Avg: -920 | Steps: 157 | Yield: 11335.4 kg/ha | N: 720.0 kg/ha | Water: 696.0 mm | LR: 5.74e-02
Episode 572/2000 | Score: 207 | Avg: -920 | Steps: 160 | Yield: 9220.6 kg/ha | N: 680.0 kg/ha | Water: 648.0 mm | LR: 5.71e-02
Episode 573/2000 | Score: 167 | Avg: -920 | Steps: 156 | Yield: 8145.5 kg/ha | N: 440.0 kg/ha | Water: 696.0 mm | LR: 5.69e-02
Episode 574/2000 | Score: 201 | Avg: -920 | Steps: 156 | Yield: 11787.0 kg/ha | N: 960.0 kg/ha | Water: 768.0 mm | LR: 5.66e-02
Episode 575/2000 | Score: 28 | Avg: -920 | Steps: 156 | Yield: 8848.0 kg/ha | N: 840.0 kg/ha | Water: 642.0 mm | LR: 5.63e-02
Episode 576/2000 | Score: 387 | Avg: -920 | Steps: 156 | Yield: 9932.2 kg/ha | N: 720.0 kg/ha | Water: 546.0 mm | LR: 5.60e-02
Episode 577/2000 | Score: 395 | Avg: -920 | Steps: 161 | Yield: 9670.0 kg/ha | N: 640.0 kg/ha | Water: 570.0 mm | LR: 5.57e-02
Episode 578/2000 | Score: -1137 | Avg: -920 | Steps: 153 | Yield: 7611.7 kg/ha | N: 1760.0 kg/ha | Water: 822.0 mm | LR: 5.55e-02
Episode 579/2000 | Score: -23 | Avg: -920 | Steps: 156 | Yield: 10030.4 kg/ha | N: 800.0 kg/ha | Water: 840.0 mm | LR: 5.52e-02
Episode 580/2000 | Score: 349 | Avg: -920 | Steps: 160 | Yield: 8777.6 kg/ha | N: 520.0 kg/ha | Water: 570.0 mm | LR: 5.49e-02
Episode 581/2000 | Score: 61 | Avg: -920 | Steps: 159 | Yield: 7904.7 kg/ha | N: 640.0 kg/ha | Water: 534.0 mm | LR: 5.46e-02
Episode 582/2000 | Score: 551 | Avg: -920 | Steps: 156 | Yield: 9823.4 kg/ha | N: 440.0 kg/ha | Water: 594.0 mm | LR: 5.44e-02
Episode 583/2000 | Score: 292 | Avg: -920 | Steps: 162 | Yield: 11094.5 kg/ha | N: 880.0 kg/ha | Water: 684.0 mm | LR: 5.41e-02
Episode 584/2000 | Score: 440 | Avg: -920 | Steps: 161 | Yield: 11136.7 kg/ha | N: 680.0 kg/ha | Water: 636.0 mm | LR: 5.38e-02
Episode 585/2000 | Score: -201 | Avg: -920 | Steps: 163 | Yield: 10745.3 kg/ha | N: 1200.0 kg/ha | Water: 864.0 mm | LR: 5.35e-02
Episode 586/2000 | Score: -66 | Avg: -920 | Steps: 161 | Yield: 9335.7 kg/ha | N: 1120.0 kg/ha | Water: 486.0 mm | LR: 5.33e-02
Episode 587/2000 | Score: -62 | Avg: -920 | Steps: 167 | Yield: 9514.2 kg/ha | N: 720.0 kg/ha | Water: 906.0 mm | LR: 5.30e-02
Episode 588/2000 | Score: 246 | Avg: -920 | Steps: 162 | Yield: 10384.2 kg/ha | N: 680.0 kg/ha | Water: 780.0 mm | LR: 5.27e-02
Episode 589/2000 | Score: 328 | Avg: -920 | Steps: 164 | Yield: 10325.1 kg/ha | N: 480.0 kg/ha | Water: 828.0 mm | LR: 5.25e-02
Episode 590/2000 | Score: 129 | Avg: -920 | Steps: 160 | Yield: 10557.1 kg/ha | N: 920.0 kg/ha | Water: 738.0 mm | LR: 5.22e-02
Episode 591/2000 | Score: 216 | Avg: -920 | Steps: 160 | Yield: 9297.2 kg/ha | N: 600.0 kg/ha | Water: 708.0 mm | LR: 5.20e-02
Episode 592/2000 | Score: 253 | Avg: -920 | Steps: 152 | Yield: 9472.6 kg/ha | N: 1000.0 kg/ha | Water: 372.0 mm | LR: 5.17e-02
Episode 593/2000 | Score: 468 | Avg: -920 | Steps: 159 | Yield: 12559.3 kg/ha | N: 1000.0 kg/ha | Water: 648.0 mm | LR: 5.14e-02
Episode 594/2000 | Score: -1976 | Avg: -920 | Steps: 165 | Yield: 3964.7 kg/ha | N: 1360.0 kg/ha | Water: 1332.0 mm | LR: 5.12e-02
Episode 595/2000 | Score: -511 | Avg: -920 | Steps: 161 | Yield: 9052.7 kg/ha | N: 1480.0 kg/ha | Water: 702.0 mm | LR: 5.09e-02
Episode 596/2000 | Score: 224 | Avg: -920 | Steps: 155 | Yield: 9377.1 kg/ha | N: 560.0 kg/ha | Water: 672.0 mm | LR: 5.07e-02
Episode 597/2000 | Score: 282 | Avg: -920 | Steps: 156 | Yield: 11246.6 kg/ha | N: 840.0 kg/ha | Water: 750.0 mm | LR: 5.04e-02
Episode 598/2000 | Score: 10 | Avg: -920 | Steps: 163 | Yield: 10386.3 kg/ha | N: 920.0 kg/ha | Water: 816.0 mm | LR: 5.02e-02
Episode 599/2000 | Score: -21 | Avg: -920 | Steps: 156 | Yield: 9642.8 kg/ha | N: 1120.0 kg/ha | Water: 600.0 mm | LR: 4.99e-02
Episode 600/2000 | Score: 236 | Avg: -920 | Steps: 156 | Yield: 9745.5 kg/ha | N: 640.0 kg/ha | Water: 720.0 mm | LR: 4.97e-02
Episode 601/2000 | Score: 212 | Avg: -920 | Steps: 170 | Yield: 9791.9 kg/ha | N: 520.0 kg/ha | Water: 834.0 mm | LR: 4.94e-02
Episode 602/2000 | Score: -2673 | Avg: -920 | Steps: 154 | Yield: 7821.8 kg/ha | N: 3160.0 kg/ha | Water: 1284.0 mm | LR: 4.92e-02
Episode 603/2000 | Score: 352 | Avg: -920 | Steps: 163 | Yield: 8632.2 kg/ha | N: 520.0 kg/ha | Water: 546.0 mm | LR: 4.89e-02
Episode 604/2000 | Score: 396 | Avg: -920 | Steps: 157 | Yield: 10204.8 kg/ha | N: 520.0 kg/ha | Water: 732.0 mm | LR: 4.87e-02
Episode 605/2000 | Score: 343 | Avg: -920 | Steps: 158 | Yield: 9001.8 kg/ha | N: 480.0 kg/ha | Water: 636.0 mm | LR: 4.84e-02
Episode 606/2000 | Score: 465 | Avg: -920 | Steps: 163 | Yield: 8421.7 kg/ha | N: 360.0 kg/ha | Water: 522.0 mm | LR: 4.82e-02
Episode 607/2000 | Score: 362 | Avg: -920 | Steps: 163 | Yield: 10377.5 kg/ha | N: 840.0 kg/ha | Water: 552.0 mm | LR: 4.79e-02
Episode 608/2000 | Score: 340 | Avg: -920 | Steps: 161 | Yield: 8921.9 kg/ha | N: 360.0 kg/ha | Water: 714.0 mm | LR: 4.77e-02
Episode 609/2000 | Score: 436 | Avg: -920 | Steps: 160 | Yield: 12371.0 kg/ha | N: 1120.0 kg/ha | Water: 576.0 mm | LR: 4.75e-02
Episode 610/2000 | Score: 427 | Avg: -920 | Steps: 156 | Yield: 10085.8 kg/ha | N: 600.0 kg/ha | Water: 630.0 mm | LR: 4.72e-02
Episode 611/2000 | Score: -204 | Avg: -920 | Steps: 157 | Yield: 6624.5 kg/ha | N: 480.0 kg/ha | Water: 780.0 mm | LR: 4.70e-02
Episode 612/2000 | Score: 582 | Avg: -920 | Steps: 155 | Yield: 9959.5 kg/ha | N: 520.0 kg/ha | Water: 528.0 mm | LR: 4.68e-02
Episode 613/2000 | Score: -1119 | Avg: -920 | Steps: 162 | Yield: 6007.7 kg/ha | N: 1640.0 kg/ha | Water: 702.0 mm | LR: 4.65e-02
Episode 614/2000 | Score: 242 | Avg: -920 | Steps: 159 | Yield: 10918.6 kg/ha | N: 1000.0 kg/ha | Water: 630.0 mm | LR: 4.63e-02
Episode 615/2000 | Score: -1048 | Avg: -920 | Steps: 163 | Yield: 3925.7 kg/ha | N: 760.0 kg/ha | Water: 942.0 mm | LR: 4.61e-02
Episode 616/2000 | Score: 451 | Avg: -920 | Steps: 167 | Yield: 12040.5 kg/ha | N: 920.0 kg/ha | Water: 612.0 mm | LR: 4.58e-02
Episode 617/2000 | Score: 467 | Avg: -920 | Steps: 163 | Yield: 8951.6 kg/ha | N: 640.0 kg/ha | Water: 402.0 mm | LR: 4.56e-02
Episode 618/2000 | Score: 292 | Avg: -920 | Steps: 158 | Yield: 10672.8 kg/ha | N: 720.0 kg/ha | Water: 738.0 mm | LR: 4.54e-02
Episode 619/2000 | Score: 281 | Avg: -920 | Steps: 161 | Yield: 9708.7 kg/ha | N: 600.0 kg/ha | Water: 708.0 mm | LR: 4.52e-02
Episode 620/2000 | Score: 503 | Avg: -920 | Steps: 161 | Yield: 10897.2 kg/ha | N: 640.0 kg/ha | Water: 636.0 mm | LR: 4.49e-02
Episode 621/2000 | Score: 445 | Avg: -920 | Steps: 156 | Yield: 10284.0 kg/ha | N: 800.0 kg/ha | Water: 498.0 mm | LR: 4.47e-02
Episode 622/2000 | Score: 719 | Avg: -920 | Steps: 169 | Yield: 10008.4 kg/ha | N: 440.0 kg/ha | Water: 468.0 mm | LR: 4.45e-02
Episode 623/2000 | Score: 208 | Avg: -920 | Steps: 160 | Yield: 9665.5 kg/ha | N: 600.0 kg/ha | Water: 756.0 mm | LR: 4.43e-02
Episode 624/2000 | Score: 381 | Avg: -920 | Steps: 157 | Yield: 8370.6 kg/ha | N: 440.0 kg/ha | Water: 540.0 mm | LR: 4.40e-02
Episode 625/2000 | Score: -1391 | Avg: -920 | Steps: 155 | Yield: 2121.3 kg/ha | N: 1560.0 kg/ha | Water: 420.0 mm | LR: 4.38e-02
Episode 626/2000 | Score: 18 | Avg: -920 | Steps: 160 | Yield: 9865.4 kg/ha | N: 1040.0 kg/ha | Water: 654.0 mm | LR: 4.36e-02
Episode 627/2000 | Score: 132 | Avg: -920 | Steps: 160 | Yield: 6732.8 kg/ha | N: 720.0 kg/ha | Water: 330.0 mm | LR: 4.34e-02
Episode 628/2000 | Score: 581 | Avg: -920 | Steps: 167 | Yield: 10661.8 kg/ha | N: 520.0 kg/ha | Water: 630.0 mm | LR: 4.32e-02
Episode 629/2000 | Score: 425 | Avg: -920 | Steps: 171 | Yield: 10929.3 kg/ha | N: 720.0 kg/ha | Water: 654.0 mm | LR: 4.29e-02
Episode 630/2000 | Score: 237 | Avg: -920 | Steps: 161 | Yield: 9913.0 kg/ha | N: 880.0 kg/ha | Water: 576.0 mm | LR: 4.27e-02
Episode 631/2000 | Score: 93 | Avg: -920 | Steps: 159 | Yield: 9320.8 kg/ha | N: 760.0 kg/ha | Water: 708.0 mm | LR: 4.25e-02
Episode 632/2000 | Score: 439 | Avg: -920 | Steps: 159 | Yield: 9685.8 kg/ha | N: 320.0 kg/ha | Water: 762.0 mm | LR: 4.23e-02
Episode 633/2000 | Score: 98 | Avg: -920 | Steps: 168 | Yield: 8800.2 kg/ha | N: 600.0 kg/ha | Water: 738.0 mm | LR: 4.21e-02
Episode 634/2000 | Score: 145 | Avg: -920 | Steps: 159 | Yield: 8881.8 kg/ha | N: 640.0 kg/ha | Water: 684.0 mm | LR: 4.19e-02
Episode 635/2000 | Score: -182 | Avg: -920 | Steps: 155 | Yield: 6181.2 kg/ha | N: 280.0 kg/ha | Water: 852.0 mm | LR: 4.17e-02
Episode 636/2000 | Score: 425 | Avg: -920 | Steps: 158 | Yield: 9134.1 kg/ha | N: 320.0 kg/ha | Water: 696.0 mm | LR: 4.15e-02
Episode 637/2000 | Score: -55 | Avg: -920 | Steps: 162 | Yield: 12044.7 kg/ha | N: 1200.0 kg/ha | Water: 918.0 mm | LR: 4.13e-02
Episode 638/2000 | Score: 447 | Avg: -920 | Steps: 163 | Yield: 9881.8 kg/ha | N: 400.0 kg/ha | Water: 726.0 mm | LR: 4.10e-02
Episode 639/2000 | Score: 389 | Avg: -920 | Steps: 161 | Yield: 10403.3 kg/ha | N: 920.0 kg/ha | Water: 480.0 mm | LR: 4.08e-02
Episode 640/2000 | Score: 650 | Avg: -920 | Steps: 159 | Yield: 10925.0 kg/ha | N: 560.0 kg/ha | Water: 576.0 mm | LR: 4.06e-02
Episode 641/2000 | Score: 579 | Avg: -920 | Steps: 165 | Yield: 10733.2 kg/ha | N: 480.0 kg/ha | Water: 630.0 mm | LR: 4.04e-02
Episode 642/2000 | Score: 621 | Avg: -920 | Steps: 154 | Yield: 11251.4 kg/ha | N: 480.0 kg/ha | Water: 666.0 mm | LR: 4.02e-02
Episode 643/2000 | Score: 242 | Avg: -920 | Steps: 160 | Yield: 9996.4 kg/ha | N: 640.0 kg/ha | Water: 756.0 mm | LR: 4.00e-02
Episode 644/2000 | Score: 373 | Avg: -920 | Steps: 158 | Yield: 10474.7 kg/ha | N: 520.0 kg/ha | Water: 792.0 mm | LR: 3.98e-02
Episode 645/2000 | Score: 372 | Avg: -920 | Steps: 164 | Yield: 10654.1 kg/ha | N: 800.0 kg/ha | Water: 618.0 mm | LR: 3.96e-02
Episode 646/2000 | Score: -249 | Avg: -920 | Steps: 157 | Yield: 5696.4 kg/ha | N: 360.0 kg/ha | Water: 786.0 mm | LR: 3.94e-02
Episode 647/2000 | Score: 473 | Avg: -920 | Steps: 161 | Yield: 11885.7 kg/ha | N: 600.0 kg/ha | Water: 846.0 mm | LR: 3.92e-02
Episode 648/2000 | Score: -1611 | Avg: -920 | Steps: 156 | Yield: 4364.9 kg/ha | N: 1320.0 kg/ha | Water: 1086.0 mm | LR: 3.90e-02
Episode 649/2000 | Score: -344 | Avg: -920 | Steps: 160 | Yield: 7707.6 kg/ha | N: 640.0 kg/ha | Water: 960.0 mm | LR: 3.88e-02
Episode 650/2000 | Score: 95 | Avg: -920 | Steps: 158 | Yield: 11450.7 kg/ha | N: 800.0 kg/ha | Water: 972.0 mm | LR: 3.87e-02
Episode 651/2000 | Score: 160 | Avg: -920 | Steps: 163 | Yield: 8605.3 kg/ha | N: 600.0 kg/ha | Water: 660.0 mm | LR: 3.85e-02
Episode 652/2000 | Score: 185 | Avg: -920 | Steps: 156 | Yield: 9116.1 kg/ha | N: 520.0 kg/ha | Water: 768.0 mm | LR: 3.83e-02
Episode 653/2000 | Score: -3615 | Avg: -920 | Steps: 166 | Yield: 2210.4 kg/ha | N: 3160.0 kg/ha | Water: 1236.0 mm | LR: 3.81e-02
Episode 654/2000 | Score: 535 | Avg: -920 | Steps: 160 | Yield: 9530.3 kg/ha | N: 360.0 kg/ha | Water: 624.0 mm | LR: 3.79e-02
Episode 655/2000 | Score: 464 | Avg: -920 | Steps: 151 | Yield: 9545.4 kg/ha | N: 520.0 kg/ha | Water: 576.0 mm | LR: 3.77e-02
Episode 656/2000 | Score: 225 | Avg: -920 | Steps: 157 | Yield: 11861.9 kg/ha | N: 600.0 kg/ha | Water: 1056.0 mm | LR: 3.75e-02
Episode 657/2000 | Score: 546 | Avg: -920 | Steps: 162 | Yield: 10684.5 kg/ha | N: 560.0 kg/ha | Water: 636.0 mm | LR: 3.73e-02
Episode 658/2000 | Score: -614 | Avg: -920 | Steps: 160 | Yield: 5459.5 kg/ha | N: 440.0 kg/ha | Water: 1026.0 mm | LR: 3.71e-02
Episode 659/2000 | Score: 69 | Avg: -920 | Steps: 157 | Yield: 8538.0 kg/ha | N: 760.0 kg/ha | Water: 618.0 mm | LR: 3.69e-02
Episode 660/2000 | Score: 364 | Avg: -920 | Steps: 157 | Yield: 7224.7 kg/ha | N: 400.0 kg/ha | Water: 420.0 mm | LR: 3.68e-02
Episode 661/2000 | Score: 330 | Avg: -920 | Steps: 163 | Yield: 10275.0 kg/ha | N: 560.0 kg/ha | Water: 774.0 mm | LR: 3.66e-02
Episode 662/2000 | Score: -401 | Avg: -920 | Steps: 159 | Yield: 5912.6 kg/ha | N: 320.0 kg/ha | Water: 984.0 mm | LR: 3.64e-02
Episode 663/2000 | Score: -514 | Avg: -920 | Steps: 163 | Yield: 5048.0 kg/ha | N: 440.0 kg/ha | Water: 876.0 mm | LR: 3.62e-02
Episode 664/2000 | Score: 312 | Avg: -920 | Steps: 162 | Yield: 9145.7 kg/ha | N: 440.0 kg/ha | Water: 714.0 mm | LR: 3.60e-02
Episode 665/2000 | Score: 486 | Avg: -920 | Steps: 164 | Yield: 10838.6 kg/ha | N: 600.0 kg/ha | Water: 684.0 mm | LR: 3.59e-02
Episode 666/2000 | Score: -238 | Avg: -920 | Steps: 164 | Yield: 8370.6 kg/ha | N: 1040.0 kg/ha | Water: 666.0 mm | LR: 3.57e-02
Episode 667/2000 | Score: -179 | Avg: -920 | Steps: 52 | Yield: 0.0 kg/ha | N: 160.0 kg/ha | Water: 48.0 mm | LR: 3.55e-02
Episode 668/2000 | Score: -336 | Avg: -920 | Steps: 166 | Yield: 6934.4 kg/ha | N: 400.0 kg/ha | Water: 1014.0 mm | LR: 3.53e-02
Episode 669/2000 | Score: -536 | Avg: -920 | Steps: 156 | Yield: 4258.2 kg/ha | N: 360.0 kg/ha | Water: 840.0 mm | LR: 3.51e-02
Episode 670/2000 | Score: 402 | Avg: -920 | Steps: 160 | Yield: 8236.4 kg/ha | N: 320.0 kg/ha | Water: 588.0 mm | LR: 3.50e-02
Episode 671/2000 | Score: 599 | Avg: -920 | Steps: 163 | Yield: 9225.0 kg/ha | N: 360.0 kg/ha | Water: 522.0 mm | LR: 3.48e-02
Episode 672/2000 | Score: 152 | Avg: -920 | Steps: 160 | Yield: 8823.1 kg/ha | N: 720.0 kg/ha | Water: 612.0 mm | LR: 3.46e-02
Episode 673/2000 | Score: -1181 | Avg: -920 | Steps: 159 | Yield: 3657.6 kg/ha | N: 760.0 kg/ha | Water: 996.0 mm | LR: 3.44e-02
Episode 674/2000 | Score: 242 | Avg: -920 | Steps: 165 | Yield: 10115.4 kg/ha | N: 480.0 kg/ha | Water: 888.0 mm | LR: 3.43e-02
Episode 675/2000 | Score: 591 | Avg: -920 | Steps: 155 | Yield: 11212.1 kg/ha | N: 600.0 kg/ha | Water: 642.0 mm | LR: 3.41e-02
Episode 676/2000 | Score: 505 | Avg: -920 | Steps: 164 | Yield: 10366.9 kg/ha | N: 440.0 kg/ha | Water: 714.0 mm | LR: 3.39e-02
Episode 677/2000 | Score: 268 | Avg: -920 | Steps: 154 | Yield: 10377.5 kg/ha | N: 800.0 kg/ha | Water: 672.0 mm | LR: 3.38e-02
Episode 678/2000 | Score: 336 | Avg: -920 | Steps: 158 | Yield: 9112.5 kg/ha | N: 520.0 kg/ha | Water: 630.0 mm | LR: 3.36e-02
Episode 679/2000 | Score: -161 | Avg: -920 | Steps: 163 | Yield: 4667.7 kg/ha | N: 360.0 kg/ha | Water: 558.0 mm | LR: 3.34e-02
Episode 680/2000 | Score: 465 | Avg: -920 | Steps: 158 | Yield: 10639.6 kg/ha | N: 520.0 kg/ha | Water: 720.0 mm | LR: 3.33e-02
Episode 681/2000 | Score: -297 | Avg: -920 | Steps: 153 | Yield: 6239.0 kg/ha | N: 320.0 kg/ha | Water: 936.0 mm | LR: 3.31e-02
Episode 682/2000 | Score: -626 | Avg: -920 | Steps: 163 | Yield: 4599.4 kg/ha | N: 200.0 kg/ha | Water: 1086.0 mm | LR: 3.29e-02
Episode 683/2000 | Score: 256 | Avg: -920 | Steps: 160 | Yield: 10507.7 kg/ha | N: 600.0 kg/ha | Water: 846.0 mm | LR: 3.28e-02
Episode 684/2000 | Score: -780 | Avg: -920 | Steps: 155 | Yield: 1937.1 kg/ha | N: 280.0 kg/ha | Water: 786.0 mm | LR: 3.26e-02
Episode 685/2000 | Score: 112 | Avg: -920 | Steps: 162 | Yield: 5368.1 kg/ha | N: 80.0 kg/ha | Water: 612.0 mm | LR: 3.24e-02
Episode 686/2000 | Score: 297 | Avg: -920 | Steps: 164 | Yield: 8129.5 kg/ha | N: 440.0 kg/ha | Water: 582.0 mm | LR: 3.23e-02
Episode 687/2000 | Score: -2844 | Avg: -920 | Steps: 160 | Yield: 750.7 kg/ha | N: 2240.0 kg/ha | Water: 1056.0 mm | LR: 3.21e-02
Episode 688/2000 | Score: -54 | Avg: -920 | Steps: 155 | Yield: 11045.9 kg/ha | N: 1400.0 kg/ha | Water: 624.0 mm | LR: 3.19e-02
Episode 689/2000 | Score: -95 | Avg: -920 | Steps: 162 | Yield: 9378.3 kg/ha | N: 1160.0 kg/ha | Water: 594.0 mm | LR: 3.18e-02
Episode 690/2000 | Score: -888 | Avg: -920 | Steps: 166 | Yield: 3128.7 kg/ha | N: 280.0 kg/ha | Water: 1056.0 mm | LR: 3.16e-02
Episode 691/2000 | Score: -1461 | Avg: -920 | Steps: 156 | Yield: 4622.4 kg/ha | N: 1680.0 kg/ha | Water: 786.0 mm | LR: 3.15e-02
Episode 692/2000 | Score: 121 | Avg: -920 | Steps: 160 | Yield: 7769.6 kg/ha | N: 440.0 kg/ha | Water: 690.0 mm | LR: 3.13e-02
Episode 693/2000 | Score: 315 | Avg: -920 | Steps: 162 | Yield: 10214.4 kg/ha | N: 600.0 kg/ha | Water: 750.0 mm | LR: 3.12e-02
Episode 694/2000 | Score: -239 | Avg: -920 | Steps: 165 | Yield: 6749.4 kg/ha | N: 240.0 kg/ha | Water: 1014.0 mm | LR: 3.10e-02
Episode 695/2000 | Score: 174 | Avg: -920 | Steps: 158 | Yield: 6419.7 kg/ha | N: 320.0 kg/ha | Water: 534.0 mm | LR: 3.08e-02
Episode 696/2000 | Score: -749 | Avg: -920 | Steps: 162 | Yield: 5152.6 kg/ha | N: 960.0 kg/ha | Water: 732.0 mm | LR: 3.07e-02
Episode 697/2000 | Score: 205 | Avg: -920 | Steps: 163 | Yield: 6431.5 kg/ha | N: 400.0 kg/ha | Water: 450.0 mm | LR: 3.05e-02
Episode 698/2000 | Score: 485 | Avg: -920 | Steps: 160 | Yield: 9688.6 kg/ha | N: 280.0 kg/ha | Water: 750.0 mm | LR: 3.04e-02
Episode 699/2000 | Score: -164 | Avg: -920 | Steps: 153 | Yield: 7766.1 kg/ha | N: 600.0 kg/ha | Water: 834.0 mm | LR: 3.02e-02
Episode 700/2000 | Score: 378 | Avg: -920 | Steps: 163 | Yield: 8628.9 kg/ha | N: 320.0 kg/ha | Water: 666.0 mm | LR: 3.01e-02
Episode 701/2000 | Score: 387 | Avg: -920 | Steps: 169 | Yield: 9265.7 kg/ha | N: 520.0 kg/ha | Water: 606.0 mm | LR: 2.99e-02
Episode 702/2000 | Score: -578 | Avg: -920 | Steps: 159 | Yield: 3667.5 kg/ha | N: 320.0 kg/ha | Water: 822.0 mm | LR: 2.98e-02
Episode 703/2000 | Score: -1497 | Avg: -920 | Steps: 163 | Yield: 716.6 kg/ha | N: 1280.0 kg/ha | Water: 516.0 mm | LR: 2.96e-02
Episode 704/2000 | Score: 364 | Avg: -920 | Steps: 163 | Yield: 9197.3 kg/ha | N: 360.0 kg/ha | Water: 732.0 mm | LR: 2.95e-02
Episode 705/2000 | Score: -132 | Avg: -920 | Steps: 162 | Yield: 7229.1 kg/ha | N: 560.0 kg/ha | Water: 756.0 mm | LR: 2.93e-02
Episode 706/2000 | Score: 380 | Avg: -920 | Steps: 165 | Yield: 10534.3 kg/ha | N: 840.0 kg/ha | Water: 564.0 mm | LR: 2.92e-02
Episode 707/2000 | Score: -907 | Avg: -920 | Steps: 161 | Yield: 2444.3 kg/ha | N: 400.0 kg/ha | Water: 888.0 mm | LR: 2.90e-02
Episode 708/2000 | Score: -94 | Avg: -920 | Steps: 161 | Yield: 5575.4 kg/ha | N: 240.0 kg/ha | Water: 714.0 mm | LR: 2.89e-02
Episode 709/2000 | Score: 323 | Avg: -920 | Steps: 164 | Yield: 8530.4 kg/ha | N: 320.0 kg/ha | Water: 702.0 mm | LR: 2.88e-02
Episode 710/2000 | Score: -70 | Avg: -920 | Steps: 161 | Yield: 4669.7 kg/ha | N: 120.0 kg/ha | Water: 648.0 mm | LR: 2.86e-02
Episode 711/2000 | Score: 56 | Avg: -920 | Steps: 154 | Yield: 7383.5 kg/ha | N: 520.0 kg/ha | Water: 636.0 mm | LR: 2.85e-02
Episode 712/2000 | Score: 340 | Avg: -920 | Steps: 164 | Yield: 8507.9 kg/ha | N: 360.0 kg/ha | Water: 654.0 mm | LR: 2.83e-02
Episode 713/2000 | Score: -628 | Avg: -920 | Steps: 166 | Yield: 3071.8 kg/ha | N: 440.0 kg/ha | Water: 696.0 mm | LR: 2.82e-02
Episode 714/2000 | Score: 71 | Avg: -920 | Steps: 168 | Yield: 7046.0 kg/ha | N: 400.0 kg/ha | Water: 660.0 mm | LR: 2.80e-02
Episode 715/2000 | Score: 166 | Avg: -920 | Steps: 162 | Yield: 10872.2 kg/ha | N: 720.0 kg/ha | Water: 894.0 mm | LR: 2.79e-02
Episode 716/2000 | Score: 318 | Avg: -920 | Steps: 157 | Yield: 11368.3 kg/ha | N: 760.0 kg/ha | Water: 786.0 mm | LR: 2.78e-02
Episode 717/2000 | Score: 319 | Avg: -920 | Steps: 162 | Yield: 11735.2 kg/ha | N: 640.0 kg/ha | Water: 936.0 mm | LR: 2.76e-02
Episode 718/2000 | Score: 150 | Avg: -920 | Steps: 157 | Yield: 7286.0 kg/ha | N: 440.0 kg/ha | Water: 594.0 mm | LR: 2.75e-02
Episode 719/2000 | Score: -593 | Avg: -920 | Steps: 161 | Yield: 4602.0 kg/ha | N: 560.0 kg/ha | Water: 798.0 mm | LR: 2.74e-02
Episode 720/2000 | Score: 213 | Avg: -920 | Steps: 161 | Yield: 9673.1 kg/ha | N: 520.0 kg/ha | Water: 822.0 mm | LR: 2.72e-02
Episode 721/2000 | Score: -547 | Avg: -920 | Steps: 161 | Yield: 4024.7 kg/ha | N: 520.0 kg/ha | Water: 702.0 mm | LR: 2.71e-02
Episode 722/2000 | Score: -1480 | Avg: -920 | Steps: 155 | Yield: 713.6 kg/ha | N: 1040.0 kg/ha | Water: 672.0 mm | LR: 2.69e-02
Episode 723/2000 | Score: -37 | Avg: -920 | Steps: 160 | Yield: 8544.4 kg/ha | N: 720.0 kg/ha | Water: 744.0 mm | LR: 2.68e-02
Episode 724/2000 | Score: -460 | Avg: -920 | Steps: 158 | Yield: 2485.3 kg/ha | N: 160.0 kg/ha | Water: 660.0 mm | LR: 2.67e-02
Episode 725/2000 | Score: 456 | Avg: -920 | Steps: 165 | Yield: 11232.0 kg/ha | N: 600.0 kg/ha | Water: 768.0 mm | LR: 2.65e-02
Episode 726/2000 | Score: -28 | Avg: -920 | Steps: 164 | Yield: 5996.2 kg/ha | N: 240.0 kg/ha | Water: 714.0 mm | LR: 2.64e-02
Episode 727/2000 | Score: 6 | Avg: -920 | Steps: 169 | Yield: 7607.7 kg/ha | N: 520.0 kg/ha | Water: 714.0 mm | LR: 2.63e-02
Episode 728/2000 | Score: -253 | Avg: -920 | Steps: 163 | Yield: 5232.2 kg/ha | N: 640.0 kg/ha | Water: 522.0 mm | LR: 2.61e-02
Episode 729/2000 | Score: 409 | Avg: -920 | Steps: 160 | Yield: 8775.4 kg/ha | N: 360.0 kg/ha | Water: 630.0 mm | LR: 2.60e-02
Episode 730/2000 | Score: -365 | Avg: -920 | Steps: 164 | Yield: 6341.7 kg/ha | N: 360.0 kg/ha | Water: 984.0 mm | LR: 2.59e-02
Episode 731/2000 | Score: -696 | Avg: -920 | Steps: 163 | Yield: 1261.7 kg/ha | N: 240.0 kg/ha | Water: 642.0 mm | LR: 2.58e-02
Episode 732/2000 | Score: 167 | Avg: -920 | Steps: 159 | Yield: 5963.6 kg/ha | N: 480.0 kg/ha | Water: 360.0 mm | LR: 2.56e-02
Episode 733/2000 | Score: 280 | Avg: -920 | Steps: 160 | Yield: 5568.4 kg/ha | N: 200.0 kg/ha | Water: 402.0 mm | LR: 2.55e-02
Episode 734/2000 | Score: -9 | Avg: -920 | Steps: 157 | Yield: 8197.6 kg/ha | N: 840.0 kg/ha | Water: 582.0 mm | LR: 2.54e-02
Episode 735/2000 | Score: -4 | Avg: -920 | Steps: 159 | Yield: 7028.8 kg/ha | N: 400.0 kg/ha | Water: 726.0 mm | LR: 2.52e-02
Episode 736/2000 | Score: -787 | Avg: -920 | Steps: 157 | Yield: 1634.8 kg/ha | N: 480.0 kg/ha | Water: 606.0 mm | LR: 2.51e-02
Episode 737/2000 | Score: -365 | Avg: -920 | Steps: 163 | Yield: 3917.9 kg/ha | N: 160.0 kg/ha | Water: 780.0 mm | LR: 2.50e-02
Episode 738/2000 | Score: -390 | Avg: -920 | Steps: 163 | Yield: 3778.7 kg/ha | N: 280.0 kg/ha | Water: 696.0 mm | LR: 2.49e-02
Episode 739/2000 | Score: -896 | Avg: -920 | Steps: 159 | Yield: 1176.3 kg/ha | N: 200.0 kg/ha | Water: 840.0 mm | LR: 2.47e-02
Episode 740/2000 | Score: -561 | Avg: -920 | Steps: 162 | Yield: 2560.2 kg/ha | N: 120.0 kg/ha | Water: 792.0 mm | LR: 2.46e-02
Episode 741/2000 | Score: -785 | Avg: -920 | Steps: 161 | Yield: 1643.7 kg/ha | N: 120.0 kg/ha | Water: 864.0 mm | LR: 2.45e-02
Episode 742/2000 | Score: -179 | Avg: -920 | Steps: 163 | Yield: 5555.8 kg/ha | N: 360.0 kg/ha | Water: 702.0 mm | LR: 2.44e-02
Episode 743/2000 | Score: 415 | Avg: -920 | Steps: 160 | Yield: 6069.2 kg/ha | N: 120.0 kg/ha | Water: 408.0 mm | LR: 2.43e-02
Episode 744/2000 | Score: 160 | Avg: -920 | Steps: 166 | Yield: 7242.9 kg/ha | N: 360.0 kg/ha | Water: 636.0 mm | LR: 2.41e-02
Episode 745/2000 | Score: 697 | Avg: -920 | Steps: 162 | Yield: 11410.8 kg/ha | N: 280.0 kg/ha | Water: 798.0 mm | LR: 2.40e-02
Episode 746/2000 | Score: -28 | Avg: -920 | Steps: 163 | Yield: 5527.0 kg/ha | N: 280.0 kg/ha | Water: 618.0 mm | LR: 2.39e-02
Episode 747/2000 | Score: -101 | Avg: -920 | Steps: 160 | Yield: 4720.8 kg/ha | N: 120.0 kg/ha | Water: 684.0 mm | LR: 2.38e-02
Episode 748/2000 | Score: 249 | Avg: -920 | Steps: 157 | Yield: 7262.4 kg/ha | N: 160.0 kg/ha | Water: 702.0 mm | LR: 2.37e-02
Episode 749/2000 | Score: 524 | Avg: -920 | Steps: 159 | Yield: 9685.5 kg/ha | N: 280.0 kg/ha | Water: 714.0 mm | LR: 2.35e-02
Episode 750/2000 | Score: 611 | Avg: -920 | Steps: 162 | Yield: 10115.4 kg/ha | N: 280.0 kg/ha | Water: 696.0 mm | LR: 2.34e-02
Episode 751/2000 | Score: -261 | Avg: -920 | Steps: 149 | Yield: 5397.0 kg/ha | N: 240.0 kg/ha | Water: 840.0 mm | LR: 2.33e-02
Episode 752/2000 | Score: 89 | Avg: -920 | Steps: 161 | Yield: 8560.1 kg/ha | N: 560.0 kg/ha | Water: 660.0 mm | LR: 2.32e-02
Episode 753/2000 | Score: 714 | Avg: -920 | Steps: 160 | Yield: 11320.9 kg/ha | N: 400.0 kg/ha | Water: 690.0 mm | LR: 2.31e-02
Episode 754/2000 | Score: 189 | Avg: -920 | Steps: 155 | Yield: 5806.9 kg/ha | N: 120.0 kg/ha | Water: 576.0 mm | LR: 2.29e-02
Episode 755/2000 | Score: 404 | Avg: -920 | Steps: 154 | Yield: 9268.9 kg/ha | N: 400.0 kg/ha | Water: 624.0 mm | LR: 2.28e-02
Episode 756/2000 | Score: -415 | Avg: -920 | Steps: 162 | Yield: 5694.5 kg/ha | N: 640.0 kg/ha | Water: 660.0 mm | LR: 2.27e-02
Episode 757/2000 | Score: -565 | Avg: -920 | Steps: 160 | Yield: 902.0 kg/ha | N: 160.0 kg/ha | Water: 528.0 mm | LR: 2.26e-02
Episode 758/2000 | Score: -653 | Avg: -920 | Steps: 158 | Yield: 759.7 kg/ha | N: 160.0 kg/ha | Water: 588.0 mm | LR: 2.25e-02
Episode 759/2000 | Score: -129 | Avg: -920 | Steps: 159 | Yield: 4735.7 kg/ha | N: 200.0 kg/ha | Water: 654.0 mm | LR: 2.24e-02
Episode 760/2000 | Score: 552 | Avg: -920 | Steps: 164 | Yield: 10490.5 kg/ha | N: 480.0 kg/ha | Water: 660.0 mm | LR: 2.23e-02
Episode 761/2000 | Score: -427 | Avg: -920 | Steps: 167 | Yield: 2873.7 kg/ha | N: 280.0 kg/ha | Water: 600.0 mm | LR: 2.22e-02
Episode 762/2000 | Score: -515 | Avg: -920 | Steps: 165 | Yield: 582.1 kg/ha | N: 0.0 kg/ha | Water: 552.0 mm | LR: 2.20e-02
Episode 763/2000 | Score: -33 | Avg: -920 | Steps: 158 | Yield: 7403.0 kg/ha | N: 520.0 kg/ha | Water: 720.0 mm | LR: 2.19e-02
Episode 764/2000 | Score: 110 | Avg: -920 | Steps: 161 | Yield: 9405.1 kg/ha | N: 840.0 kg/ha | Water: 648.0 mm | LR: 2.18e-02
Episode 765/2000 | Score: 795 | Avg: -920 | Steps: 155 | Yield: 11582.7 kg/ha | N: 400.0 kg/ha | Water: 654.0 mm | LR: 2.17e-02
Episode 766/2000 | Score: 322 | Avg: -920 | Steps: 161 | Yield: 8880.7 kg/ha | N: 400.0 kg/ha | Water: 696.0 mm | LR: 2.16e-02
Episode 767/2000 | Score: -151 | Avg: -920 | Steps: 157 | Yield: 4521.2 kg/ha | N: 160.0 kg/ha | Water: 672.0 mm | LR: 2.15e-02
Episode 768/2000 | Score: 248 | Avg: -920 | Steps: 163 | Yield: 8402.9 kg/ha | N: 280.0 kg/ha | Water: 780.0 mm | LR: 2.14e-02
Episode 769/2000 | Score: -316 | Avg: -920 | Steps: 158 | Yield: 4242.9 kg/ha | N: 280.0 kg/ha | Water: 696.0 mm | LR: 2.13e-02
Episode 770/2000 | Score: 388 | Avg: -920 | Steps: 162 | Yield: 7932.9 kg/ha | N: 360.0 kg/ha | Water: 528.0 mm | LR: 2.12e-02
Episode 771/2000 | Score: 537 | Avg: -920 | Steps: 156 | Yield: 10468.5 kg/ha | N: 520.0 kg/ha | Water: 642.0 mm | LR: 2.11e-02
Episode 772/2000 | Score: -156 | Avg: -920 | Steps: 160 | Yield: 8748.7 kg/ha | N: 560.0 kg/ha | Water: 984.0 mm | LR: 2.10e-02
Episode 773/2000 | Score: 225 | Avg: -920 | Steps: 161 | Yield: 6407.4 kg/ha | N: 120.0 kg/ha | Water: 630.0 mm | LR: 2.09e-02
Episode 774/2000 | Score: 433 | Avg: -920 | Steps: 160 | Yield: 10744.1 kg/ha | N: 640.0 kg/ha | Water: 690.0 mm | LR: 2.08e-02
Episode 775/2000 | Score: 231 | Avg: -920 | Steps: 156 | Yield: 6878.4 kg/ha | N: 240.0 kg/ha | Water: 606.0 mm | LR: 2.07e-02
Episode 776/2000 | Score: -366 | Avg: -920 | Steps: 159 | Yield: 2909.3 kg/ha | N: 160.0 kg/ha | Water: 636.0 mm | LR: 2.06e-02
Episode 777/2000 | Score: -398 | Avg: -920 | Steps: 157 | Yield: 1907.2 kg/ha | N: 200.0 kg/ha | Water: 492.0 mm | LR: 2.05e-02
Episode 778/2000 | Score: 183 | Avg: -920 | Steps: 163 | Yield: 10391.3 kg/ha | N: 1120.0 kg/ha | Water: 522.0 mm | LR: 2.03e-02
Episode 779/2000 | Score: 381 | Avg: -920 | Steps: 157 | Yield: 8353.7 kg/ha | N: 320.0 kg/ha | Water: 624.0 mm | LR: 2.02e-02
Episode 780/2000 | Score: 453 | Avg: -920 | Steps: 158 | Yield: 10942.3 kg/ha | N: 520.0 kg/ha | Water: 786.0 mm | LR: 2.01e-02
Episode 781/2000 | Score: 300 | Avg: -920 | Steps: 164 | Yield: 7162.0 kg/ha | N: 200.0 kg/ha | Water: 612.0 mm | LR: 2.00e-02
Episode 782/2000 | Score: -297 | Avg: -920 | Steps: 163 | Yield: 4427.7 kg/ha | N: 200.0 kg/ha | Water: 762.0 mm | LR: 1.99e-02
Episode 783/2000 | Score: -714 | Avg: -920 | Steps: 159 | Yield: 763.5 kg/ha | N: 280.0 kg/ha | Water: 558.0 mm | LR: 1.98e-02
Episode 784/2000 | Score: 261 | Avg: -920 | Steps: 158 | Yield: 9160.3 kg/ha | N: 600.0 kg/ha | Water: 648.0 mm | LR: 1.97e-02
Episode 785/2000 | Score: 68 | Avg: -920 | Steps: 157 | Yield: 10767.8 kg/ha | N: 1040.0 kg/ha | Water: 738.0 mm | LR: 1.96e-02
Episode 786/2000 | Score: 583 | Avg: -920 | Steps: 162 | Yield: 11116.3 kg/ha | N: 600.0 kg/ha | Water: 636.0 mm | LR: 1.95e-02
Episode 787/2000 | Score: 155 | Avg: -920 | Steps: 164 | Yield: 7159.6 kg/ha | N: 400.0 kg/ha | Water: 600.0 mm | LR: 1.95e-02
Episode 788/2000 | Score: -427 | Avg: -920 | Steps: 153 | Yield: 896.9 kg/ha | N: 160.0 kg/ha | Water: 402.0 mm | LR: 1.94e-02
Episode 789/2000 | Score: 327 | Avg: -920 | Steps: 163 | Yield: 12225.8 kg/ha | N: 920.0 kg/ha | Water: 780.0 mm | LR: 1.93e-02
Episode 790/2000 | Score: 515 | Avg: -920 | Steps: 159 | Yield: 10475.5 kg/ha | N: 600.0 kg/ha | Water: 606.0 mm | LR: 1.92e-02
Episode 791/2000 | Score: -379 | Avg: -920 | Steps: 160 | Yield: 859.0 kg/ha | N: 0.0 kg/ha | Water: 468.0 mm | LR: 1.91e-02
Episode 792/2000 | Score: 130 | Avg: -920 | Steps: 161 | Yield: 10481.4 kg/ha | N: 720.0 kg/ha | Water: 870.0 mm | LR: 1.90e-02
Episode 793/2000 | Score: -40 | Avg: -920 | Steps: 160 | Yield: 7263.9 kg/ha | N: 200.0 kg/ha | Water: 930.0 mm | LR: 1.89e-02
Episode 794/2000 | Score: 608 | Avg: -920 | Steps: 155 | Yield: 8689.4 kg/ha | N: 200.0 kg/ha | Water: 552.0 mm | LR: 1.88e-02
Episode 795/2000 | Score: 218 | Avg: -920 | Steps: 162 | Yield: 6477.8 kg/ha | N: 360.0 kg/ha | Water: 474.0 mm | LR: 1.87e-02
Episode 796/2000 | Score: -96 | Avg: -920 | Steps: 154 | Yield: 3542.3 kg/ha | N: 120.0 kg/ha | Water: 510.0 mm | LR: 1.86e-02
Episode 797/2000 | Score: -466 | Avg: -920 | Steps: 161 | Yield: 794.1 kg/ha | N: 80.0 kg/ha | Water: 480.0 mm | LR: 1.85e-02
Episode 798/2000 | Score: 437 | Avg: -920 | Steps: 162 | Yield: 10189.1 kg/ha | N: 440.0 kg/ha | Water: 750.0 mm | LR: 1.84e-02
Episode 799/2000 | Score: -473 | Avg: -920 | Steps: 159 | Yield: 913.8 kg/ha | N: 80.0 kg/ha | Water: 504.0 mm | LR: 1.83e-02
Episode 800/2000 | Score: 368 | Avg: -920 | Steps: 163 | Yield: 8791.3 kg/ha | N: 240.0 kg/ha | Water: 756.0 mm | LR: 1.82e-02
Episode 801/2000 | Score: 353 | Avg: -920 | Steps: 165 | Yield: 7378.4 kg/ha | N: 360.0 kg/ha | Water: 480.0 mm | LR: 1.81e-02
Episode 802/2000 | Score: -424 | Avg: -920 | Steps: 156 | Yield: 1188.1 kg/ha | N: 200.0 kg/ha | Water: 360.0 mm | LR: 1.80e-02
Episode 803/2000 | Score: -457 | Avg: -920 | Steps: 154 | Yield: 890.8 kg/ha | N: 80.0 kg/ha | Water: 486.0 mm | LR: 1.80e-02
Episode 804/2000 | Score: -396 | Avg: -920 | Steps: 167 | Yield: 1067.1 kg/ha | N: 80.0 kg/ha | Water: 456.0 mm | LR: 1.79e-02
Episode 805/2000 | Score: 362 | Avg: -920 | Steps: 161 | Yield: 11231.7 kg/ha | N: 760.0 kg/ha | Water: 738.0 mm | LR: 1.78e-02
Episode 806/2000 | Score: 243 | Avg: -920 | Steps: 159 | Yield: 8619.2 kg/ha | N: 480.0 kg/ha | Water: 672.0 mm | LR: 1.77e-02
Episode 807/2000 | Score: 693 | Avg: -920 | Steps: 166 | Yield: 10980.9 kg/ha | N: 400.0 kg/ha | Water: 660.0 mm | LR: 1.76e-02
Episode 808/2000 | Score: -108 | Avg: -920 | Steps: 159 | Yield: 4713.8 kg/ha | N: 160.0 kg/ha | Water: 660.0 mm | LR: 1.75e-02
Episode 809/2000 | Score: -374 | Avg: -920 | Steps: 152 | Yield: 951.0 kg/ha | N: 120.0 kg/ha | Water: 390.0 mm | LR: 1.74e-02
Episode 810/2000 | Score: -294 | Avg: -920 | Steps: 160 | Yield: 2859.8 kg/ha | N: 200.0 kg/ha | Water: 534.0 mm | LR: 1.73e-02
Episode 811/2000 | Score: -127 | Avg: -920 | Steps: 157 | Yield: 1077.9 kg/ha | N: 0.0 kg/ha | Water: 264.0 mm | LR: 1.72e-02
Episode 812/2000 | Score: 537 | Avg: -920 | Steps: 164 | Yield: 9772.5 kg/ha | N: 280.0 kg/ha | Water: 708.0 mm | LR: 1.72e-02
Episode 813/2000 | Score: 488 | Avg: -920 | Steps: 163 | Yield: 10174.6 kg/ha | N: 440.0 kg/ha | Water: 702.0 mm | LR: 1.71e-02
Episode 814/2000 | Score: 866 | Avg: -920 | Steps: 164 | Yield: 12078.5 kg/ha | N: 400.0 kg/ha | Water: 660.0 mm | LR: 1.70e-02
Episode 815/2000 | Score: -74 | Avg: -920 | Steps: 164 | Yield: 6218.5 kg/ha | N: 360.0 kg/ha | Water: 702.0 mm | LR: 1.69e-02
Episode 816/2000 | Score: -80 | Avg: -920 | Steps: 163 | Yield: 7451.3 kg/ha | N: 480.0 kg/ha | Water: 798.0 mm | LR: 1.68e-02
Episode 817/2000 | Score: 389 | Avg: -920 | Steps: 161 | Yield: 8143.0 kg/ha | N: 200.0 kg/ha | Water: 672.0 mm | LR: 1.67e-02
Episode 818/2000 | Score: -192 | Avg: -920 | Steps: 153 | Yield: 1038.6 kg/ha | N: 0.0 kg/ha | Water: 318.0 mm | LR: 1.67e-02
Episode 819/2000 | Score: -675 | Avg: -920 | Steps: 162 | Yield: 1177.8 kg/ha | N: 280.0 kg/ha | Water: 582.0 mm | LR: 1.66e-02
Episode 820/2000 | Score: -270 | Avg: -920 | Steps: 165 | Yield: 722.0 kg/ha | N: 160.0 kg/ha | Water: 228.0 mm | LR: 1.65e-02
Episode 821/2000 | Score: 352 | Avg: -920 | Steps: 157 | Yield: 7498.0 kg/ha | N: 160.0 kg/ha | Water: 642.0 mm | LR: 1.64e-02
Episode 822/2000 | Score: 241 | Avg: -920 | Steps: 157 | Yield: 9681.0 kg/ha | N: 520.0 kg/ha | Water: 798.0 mm | LR: 1.63e-02
Episode 823/2000 | Score: -597 | Avg: -920 | Steps: 163 | Yield: 3338.2 kg/ha | N: 320.0 kg/ha | Water: 792.0 mm | LR: 1.62e-02
Episode 824/2000 | Score: 436 | Avg: -920 | Steps: 160 | Yield: 8272.6 kg/ha | N: 200.0 kg/ha | Water: 648.0 mm | LR: 1.62e-02
Episode 825/2000 | Score: 338 | Avg: -920 | Steps: 164 | Yield: 8546.6 kg/ha | N: 480.0 kg/ha | Water: 576.0 mm | LR: 1.61e-02
Episode 826/2000 | Score: -375 | Avg: -920 | Steps: 161 | Yield: 5023.1 kg/ha | N: 360.0 kg/ha | Water: 798.0 mm | LR: 1.60e-02
Episode 827/2000 | Score: -85 | Avg: -920 | Steps: 156 | Yield: 920.9 kg/ha | N: 0.0 kg/ha | Water: 204.0 mm | LR: 1.59e-02
Episode 828/2000 | Score: 626 | Avg: -920 | Steps: 156 | Yield: 10588.0 kg/ha | N: 320.0 kg/ha | Water: 612.0 mm | LR: 1.58e-02
Episode 829/2000 | Score: 712 | Avg: -920 | Steps: 154 | Yield: 10375.0 kg/ha | N: 280.0 kg/ha | Water: 642.0 mm | LR: 1.58e-02
Episode 830/2000 | Score: 590 | Avg: -920 | Steps: 158 | Yield: 8666.5 kg/ha | N: 160.0 kg/ha | Water: 594.0 mm | LR: 1.57e-02
Episode 831/2000 | Score: 166 | Avg: -920 | Steps: 158 | Yield: 2631.6 kg/ha | N: 40.0 kg/ha | Water: 198.0 mm | LR: 1.56e-02
Episode 832/2000 | Score: 281 | Avg: -920 | Steps: 170 | Yield: 3809.7 kg/ha | N: 80.0 kg/ha | Water: 234.0 mm | LR: 1.55e-02
Episode 833/2000 | Score: -253 | Avg: -920 | Steps: 157 | Yield: 5964.0 kg/ha | N: 280.0 kg/ha | Water: 822.0 mm | LR: 1.54e-02
Episode 834/2000 | Score: 410 | Avg: -920 | Steps: 166 | Yield: 10397.5 kg/ha | N: 480.0 kg/ha | Water: 678.0 mm | LR: 1.54e-02
Episode 835/2000 | Score: -584 | Avg: -920 | Steps: 163 | Yield: 1284.6 kg/ha | N: 320.0 kg/ha | Water: 486.0 mm | LR: 1.53e-02
Episode 836/2000 | Score: -75 | Avg: -920 | Steps: 159 | Yield: 3904.6 kg/ha | N: 600.0 kg/ha | Water: 198.0 mm | LR: 1.52e-02
Episode 837/2000 | Score: 79 | Avg: -920 | Steps: 158 | Yield: 6504.2 kg/ha | N: 240.0 kg/ha | Water: 690.0 mm | LR: 1.51e-02
Episode 838/2000 | Score: 139 | Avg: -920 | Steps: 159 | Yield: 8062.6 kg/ha | N: 400.0 kg/ha | Water: 744.0 mm | LR: 1.51e-02
Episode 839/2000 | Score: 497 | Avg: -920 | Steps: 160 | Yield: 7133.4 kg/ha | N: 280.0 kg/ha | Water: 372.0 mm | LR: 1.50e-02
Episode 840/2000 | Score: -221 | Avg: -920 | Steps: 160 | Yield: 1149.1 kg/ha | N: 0.0 kg/ha | Water: 366.0 mm | LR: 1.49e-02
Episode 841/2000 | Score: 447 | Avg: -920 | Steps: 164 | Yield: 11531.4 kg/ha | N: 880.0 kg/ha | Water: 618.0 mm | LR: 1.48e-02
Episode 842/2000 | Score: 70 | Avg: -920 | Steps: 149 | Yield: 6632.7 kg/ha | N: 160.0 kg/ha | Water: 774.0 mm | LR: 1.48e-02
Episode 843/2000 | Score: -278 | Avg: -920 | Steps: 158 | Yield: 3173.1 kg/ha | N: 160.0 kg/ha | Water: 594.0 mm | LR: 1.47e-02
Episode 844/2000 | Score: 479 | Avg: -920 | Steps: 157 | Yield: 9394.9 kg/ha | N: 320.0 kg/ha | Water: 684.0 mm | LR: 1.46e-02
Episode 845/2000 | Score: 680 | Avg: -920 | Steps: 160 | Yield: 11415.1 kg/ha | N: 520.0 kg/ha | Water: 648.0 mm | LR: 1.45e-02
Episode 846/2000 | Score: 547 | Avg: -920 | Steps: 163 | Yield: 7806.4 kg/ha | N: 200.0 kg/ha | Water: 480.0 mm | LR: 1.45e-02
Episode 847/2000 | Score: 232 | Avg: -920 | Steps: 163 | Yield: 7962.4 kg/ha | N: 280.0 kg/ha | Water: 732.0 mm | LR: 1.44e-02
Episode 848/2000 | Score: -405 | Avg: -920 | Steps: 157 | Yield: 1912.7 kg/ha | N: 160.0 kg/ha | Water: 528.0 mm | LR: 1.43e-02
Episode 849/2000 | Score: 157 | Avg: -920 | Steps: 158 | Yield: 6726.7 kg/ha | N: 520.0 kg/ha | Water: 450.0 mm | LR: 1.43e-02
Episode 850/2000 | Score: -50 | Avg: -920 | Steps: 165 | Yield: 1398.7 kg/ha | N: 0.0 kg/ha | Water: 240.0 mm | LR: 1.42e-02
Episode 851/2000 | Score: -147 | Avg: -920 | Steps: 156 | Yield: 962.5 kg/ha | N: 120.0 kg/ha | Water: 180.0 mm | LR: 1.41e-02
Episode 852/2000 | Score: -32 | Avg: -920 | Steps: 67 | Yield: 0.0 kg/ha | N: 40.0 kg/ha | Water: 0.0 mm | LR: 1.40e-02
Episode 853/2000 | Score: 396 | Avg: -920 | Steps: 156 | Yield: 7116.9 kg/ha | N: 120.0 kg/ha | Water: 576.0 mm | LR: 1.40e-02
Episode 854/2000 | Score: -58 | Avg: -920 | Steps: 158 | Yield: 1600.1 kg/ha | N: 160.0 kg/ha | Water: 162.0 mm | LR: 1.39e-02
Episode 855/2000 | Score: 304 | Avg: -920 | Steps: 155 | Yield: 6610.0 kg/ha | N: 160.0 kg/ha | Water: 558.0 mm | LR: 1.38e-02
Episode 856/2000 | Score: 394 | Avg: -920 | Steps: 160 | Yield: 5477.2 kg/ha | N: 120.0 kg/ha | Water: 342.0 mm | LR: 1.38e-02
Episode 857/2000 | Score: 424 | Avg: -920 | Steps: 152 | Yield: 4594.5 kg/ha | N: 40.0 kg/ha | Water: 246.0 mm | LR: 1.37e-02
Episode 858/2000 | Score: 398 | Avg: -920 | Steps: 153 | Yield: 8470.4 kg/ha | N: 280.0 kg/ha | Water: 654.0 mm | LR: 1.36e-02
Episode 859/2000 | Score: -66 | Avg: -920 | Steps: 163 | Yield: 5680.9 kg/ha | N: 200.0 kg/ha | Water: 732.0 mm | LR: 1.36e-02
Episode 860/2000 | Score: -410 | Avg: -920 | Steps: 165 | Yield: 2727.7 kg/ha | N: 120.0 kg/ha | Water: 678.0 mm | LR: 1.35e-02
Episode 861/2000 | Score: 175 | Avg: -920 | Steps: 165 | Yield: 5010.4 kg/ha | N: 120.0 kg/ha | Water: 474.0 mm | LR: 1.34e-02
Episode 862/2000 | Score: 520 | Avg: -920 | Steps: 156 | Yield: 7607.6 kg/ha | N: 120.0 kg/ha | Water: 534.0 mm | LR: 1.34e-02
Episode 863/2000 | Score: -194 | Avg: -920 | Steps: 163 | Yield: 1132.5 kg/ha | N: 280.0 kg/ha | Water: 132.0 mm | LR: 1.33e-02
Episode 864/2000 | Score: 286 | Avg: -920 | Steps: 161 | Yield: 10836.9 kg/ha | N: 920.0 kg/ha | Water: 630.0 mm | LR: 1.32e-02
Episode 865/2000 | Score: 430 | Avg: -920 | Steps: 158 | Yield: 5921.2 kg/ha | N: 80.0 kg/ha | Water: 402.0 mm | LR: 1.32e-02
Episode 866/2000 | Score: 8 | Avg: -920 | Steps: 159 | Yield: 1013.4 kg/ha | N: 0.0 kg/ha | Water: 132.0 mm | LR: 1.31e-02
Episode 867/2000 | Score: -84 | Avg: -920 | Steps: 161 | Yield: 703.8 kg/ha | N: 80.0 kg/ha | Water: 120.0 mm | LR: 1.30e-02
Episode 868/2000 | Score: 461 | Avg: -920 | Steps: 157 | Yield: 8479.8 kg/ha | N: 360.0 kg/ha | Water: 540.0 mm | LR: 1.30e-02
Episode 869/2000 | Score: 128 | Avg: -920 | Steps: 163 | Yield: 6789.3 kg/ha | N: 160.0 kg/ha | Water: 744.0 mm | LR: 1.29e-02
Episode 870/2000 | Score: 170 | Avg: -920 | Steps: 164 | Yield: 6155.8 kg/ha | N: 440.0 kg/ha | Water: 414.0 mm | LR: 1.28e-02
Episode 871/2000 | Score: 559 | Avg: -920 | Steps: 158 | Yield: 9773.9 kg/ha | N: 320.0 kg/ha | Water: 666.0 mm | LR: 1.28e-02
Episode 872/2000 | Score: -202 | Avg: -920 | Steps: 160 | Yield: 3460.1 kg/ha | N: 120.0 kg/ha | Water: 594.0 mm | LR: 1.27e-02
Episode 873/2000 | Score: -692 | Avg: -920 | Steps: 162 | Yield: 3494.9 kg/ha | N: 480.0 kg/ha | Water: 786.0 mm | LR: 1.26e-02
Episode 874/2000 | Score: 385 | Avg: -920 | Steps: 157 | Yield: 7135.7 kg/ha | N: 280.0 kg/ha | Water: 474.0 mm | LR: 1.26e-02
Episode 875/2000 | Score: 409 | Avg: -920 | Steps: 156 | Yield: 6988.2 kg/ha | N: 120.0 kg/ha | Water: 546.0 mm | LR: 1.25e-02
Episode 876/2000 | Score: -11 | Avg: -920 | Steps: 163 | Yield: 934.1 kg/ha | N: 0.0 kg/ha | Water: 144.0 mm | LR: 1.25e-02
Episode 877/2000 | Score: 302 | Avg: -920 | Steps: 163 | Yield: 8332.3 kg/ha | N: 240.0 kg/ha | Water: 750.0 mm | LR: 1.24e-02
Episode 878/2000 | Score: -467 | Avg: -920 | Steps: 160 | Yield: 2614.1 kg/ha | N: 120.0 kg/ha | Water: 714.0 mm | LR: 1.23e-02
Episode 879/2000 | Score: -452 | Avg: -920 | Steps: 153 | Yield: 2064.1 kg/ha | N: 200.0 kg/ha | Water: 564.0 mm | LR: 1.23e-02
Episode 880/2000 | Score: -169 | Avg: -920 | Steps: 161 | Yield: 992.5 kg/ha | N: 120.0 kg/ha | Water: 204.0 mm | LR: 1.22e-02
Episode 881/2000 | Score: 438 | Avg: -920 | Steps: 160 | Yield: 9579.0 kg/ha | N: 400.0 kg/ha | Water: 690.0 mm | LR: 1.21e-02
Episode 882/2000 | Score: -49 | Avg: -920 | Steps: 160 | Yield: 652.9 kg/ha | N: 0.0 kg/ha | Water: 138.0 mm | LR: 1.21e-02
Episode 883/2000 | Score: -50 | Avg: -920 | Steps: 159 | Yield: 605.4 kg/ha | N: 0.0 kg/ha | Water: 132.0 mm | LR: 1.20e-02
Episode 884/2000 | Score: -378 | Avg: -920 | Steps: 155 | Yield: 562.1 kg/ha | N: 240.0 kg/ha | Water: 246.0 mm | LR: 1.20e-02
Episode 885/2000 | Score: 182 | Avg: -920 | Steps: 159 | Yield: 6183.4 kg/ha | N: 280.0 kg/ha | Water: 522.0 mm | LR: 1.19e-02
Episode 886/2000 | Score: 439 | Avg: -920 | Steps: 162 | Yield: 5534.1 kg/ha | N: 200.0 kg/ha | Water: 252.0 mm | LR: 1.18e-02
Episode 887/2000 | Score: -249 | Avg: -920 | Steps: 164 | Yield: 1372.8 kg/ha | N: 80.0 kg/ha | Water: 366.0 mm | LR: 1.18e-02
Episode 888/2000 | Score: -77 | Avg: -920 | Steps: 159 | Yield: 1239.1 kg/ha | N: 120.0 kg/ha | Water: 156.0 mm | LR: 1.17e-02
Episode 889/2000 | Score: 523 | Avg: -920 | Steps: 163 | Yield: 9462.8 kg/ha | N: 320.0 kg/ha | Water: 654.0 mm | LR: 1.17e-02
Episode 890/2000 | Score: 350 | Avg: -920 | Steps: 161 | Yield: 5353.8 kg/ha | N: 160.0 kg/ha | Water: 336.0 mm | LR: 1.16e-02
Episode 891/2000 | Score: 501 | Avg: -920 | Steps: 159 | Yield: 7857.0 kg/ha | N: 160.0 kg/ha | Water: 558.0 mm | LR: 1.15e-02
Episode 892/2000 | Score: -38 | Avg: -920 | Steps: 159 | Yield: 1516.4 kg/ha | N: 0.0 kg/ha | Water: 246.0 mm | LR: 1.15e-02
Episode 893/2000 | Score: 135 | Avg: -920 | Steps: 161 | Yield: 8597.1 kg/ha | N: 680.0 kg/ha | Water: 624.0 mm | LR: 1.14e-02
Episode 894/2000 | Score: -414 | Avg: -920 | Steps: 160 | Yield: 2795.1 kg/ha | N: 80.0 kg/ha | Water: 720.0 mm | LR: 1.14e-02
Episode 895/2000 | Score: -299 | Avg: -920 | Steps: 157 | Yield: 3101.7 kg/ha | N: 80.0 kg/ha | Water: 660.0 mm | LR: 1.13e-02
Episode 896/2000 | Score: -415 | Avg: -920 | Steps: 161 | Yield: 1565.8 kg/ha | N: 120.0 kg/ha | Water: 516.0 mm | LR: 1.13e-02
Episode 897/2000 | Score: 169 | Avg: -920 | Steps: 165 | Yield: 6789.4 kg/ha | N: 200.0 kg/ha | Water: 678.0 mm | LR: 1.12e-02
Episode 898/2000 | Score: 46 | Avg: -920 | Steps: 159 | Yield: 1250.5 kg/ha | N: 0.0 kg/ha | Water: 132.0 mm | LR: 1.12e-02
Episode 899/2000 | Score: -95 | Avg: -920 | Steps: 163 | Yield: 482.6 kg/ha | N: 0.0 kg/ha | Water: 156.0 mm | LR: 1.11e-02
Episode 900/2000 | Score: -23 | Avg: -920 | Steps: 166 | Yield: 2559.6 kg/ha | N: 240.0 kg/ha | Water: 216.0 mm | LR: 1.10e-02
Episode 901/2000 | Score: 268 | Avg: -920 | Steps: 160 | Yield: 7433.8 kg/ha | N: 320.0 kg/ha | Water: 594.0 mm | LR: 1.10e-02
Episode 902/2000 | Score: -282 | Avg: -920 | Steps: 164 | Yield: 2740.9 kg/ha | N: 120.0 kg/ha | Water: 564.0 mm | LR: 1.09e-02
Episode 903/2000 | Score: 448 | Avg: -920 | Steps: 159 | Yield: 10994.2 kg/ha | N: 520.0 kg/ha | Water: 780.0 mm | LR: 1.09e-02
Episode 904/2000 | Score: 40 | Avg: -920 | Steps: 156 | Yield: 1422.4 kg/ha | N: 0.0 kg/ha | Water: 162.0 mm | LR: 1.08e-02
Episode 905/2000 | Score: 582 | Avg: -920 | Steps: 156 | Yield: 6564.8 kg/ha | N: 200.0 kg/ha | Water: 270.0 mm | LR: 1.08e-02
Episode 906/2000 | Score: -133 | Avg: -920 | Steps: 159 | Yield: 495.0 kg/ha | N: 0.0 kg/ha | Water: 186.0 mm | LR: 1.07e-02
Episode 907/2000 | Score: 495 | Avg: -920 | Steps: 159 | Yield: 10205.1 kg/ha | N: 520.0 kg/ha | Water: 642.0 mm | LR: 1.07e-02
Episode 908/2000 | Score: 324 | Avg: -920 | Steps: 160 | Yield: 11801.8 kg/ha | N: 1040.0 kg/ha | Water: 636.0 mm | LR: 1.06e-02
Episode 909/2000 | Score: -56 | Avg: -920 | Steps: 161 | Yield: 8236.2 kg/ha | N: 640.0 kg/ha | Water: 768.0 mm | LR: 1.06e-02
Episode 910/2000 | Score: -17 | Avg: -920 | Steps: 168 | Yield: 1146.8 kg/ha | N: 0.0 kg/ha | Water: 174.0 mm | LR: 1.05e-02
Episode 911/2000 | Score: 541 | Avg: -920 | Steps: 165 | Yield: 9520.4 kg/ha | N: 400.0 kg/ha | Water: 582.0 mm | LR: 1.04e-02
Episode 912/2000 | Score: 269 | Avg: -920 | Steps: 159 | Yield: 12721.5 kg/ha | N: 600.0 kg/ha | Water: 1134.0 mm | LR: 1.04e-02
Episode 913/2000 | Score: 272 | Avg: -920 | Steps: 161 | Yield: 2016.4 kg/ha | N: 0.0 kg/ha | Water: 42.0 mm | LR: 1.03e-02
Episode 914/2000 | Score: 218 | Avg: -920 | Steps: 162 | Yield: 8571.3 kg/ha | N: 360.0 kg/ha | Water: 768.0 mm | LR: 1.03e-02
Episode 915/2000 | Score: 420 | Avg: -920 | Steps: 163 | Yield: 10599.0 kg/ha | N: 560.0 kg/ha | Water: 738.0 mm | LR: 1.02e-02
Episode 916/2000 | Score: 455 | Avg: -920 | Steps: 156 | Yield: 11192.8 kg/ha | N: 560.0 kg/ha | Water: 774.0 mm | LR: 1.02e-02
Episode 917/2000 | Score: 51 | Avg: -920 | Steps: 156 | Yield: 1073.2 kg/ha | N: 0.0 kg/ha | Water: 108.0 mm | LR: 1.01e-02
Episode 918/2000 | Score: -130 | Avg: -920 | Steps: 163 | Yield: 1405.8 kg/ha | N: 120.0 kg/ha | Water: 228.0 mm | LR: 1.01e-02
Episode 919/2000 | Score: 54 | Avg: -920 | Steps: 159 | Yield: 6877.0 kg/ha | N: 400.0 kg/ha | Water: 570.0 mm | LR: 1.00e-02
Episode 920/2000 | Score: 604 | Avg: -920 | Steps: 165 | Yield: 10660.2 kg/ha | N: 440.0 kg/ha | Water: 648.0 mm | LR: 1.00e-02
Episode 921/2000 | Score: -166 | Avg: -920 | Steps: 160 | Yield: 2647.8 kg/ha | N: 640.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 922/2000 | Score: -223 | Avg: -920 | Steps: 159 | Yield: 1150.9 kg/ha | N: 120.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 923/2000 | Score: 368 | Avg: -920 | Steps: 164 | Yield: 9742.9 kg/ha | N: 840.0 kg/ha | Water: 444.0 mm | LR: 1.00e-02
Episode 924/2000 | Score: 126 | Avg: -920 | Steps: 167 | Yield: 7108.4 kg/ha | N: 360.0 kg/ha | Water: 642.0 mm | LR: 1.00e-02
Episode 925/2000 | Score: 135 | Avg: -920 | Steps: 162 | Yield: 1707.5 kg/ha | N: 120.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 926/2000 | Score: 135 | Avg: -920 | Steps: 161 | Yield: 6643.1 kg/ha | N: 280.0 kg/ha | Water: 630.0 mm | LR: 1.00e-02
Episode 927/2000 | Score: -147 | Avg: -920 | Steps: 166 | Yield: 1383.9 kg/ha | N: 120.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 928/2000 | Score: 125 | Avg: -920 | Steps: 163 | Yield: 1376.3 kg/ha | N: 0.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 929/2000 | Score: -404 | Avg: -920 | Steps: 163 | Yield: 2169.4 kg/ha | N: 160.0 kg/ha | Water: 564.0 mm | LR: 1.00e-02
Episode 930/2000 | Score: 33 | Avg: -920 | Steps: 159 | Yield: 876.0 kg/ha | N: 0.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 931/2000 | Score: -369 | Avg: -920 | Steps: 159 | Yield: 1121.4 kg/ha | N: 240.0 kg/ha | Water: 324.0 mm | LR: 1.00e-02
Episode 932/2000 | Score: 464 | Avg: -920 | Steps: 153 | Yield: 8239.8 kg/ha | N: 200.0 kg/ha | Water: 618.0 mm | LR: 1.00e-02
Episode 933/2000 | Score: 807 | Avg: -920 | Steps: 156 | Yield: 9724.1 kg/ha | N: 280.0 kg/ha | Water: 462.0 mm | LR: 1.00e-02
Episode 934/2000 | Score: -31 | Avg: -920 | Steps: 160 | Yield: 840.0 kg/ha | N: 40.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 935/2000 | Score: 66 | Avg: -920 | Steps: 162 | Yield: 5699.9 kg/ha | N: 280.0 kg/ha | Water: 558.0 mm | LR: 1.00e-02
Episode 936/2000 | Score: 368 | Avg: -920 | Steps: 159 | Yield: 4535.3 kg/ha | N: 40.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 937/2000 | Score: -79 | Avg: -920 | Steps: 163 | Yield: 1044.1 kg/ha | N: 0.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 938/2000 | Score: -43 | Avg: -920 | Steps: 157 | Yield: 1396.1 kg/ha | N: 0.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 939/2000 | Score: 261 | Avg: -920 | Steps: 163 | Yield: 2646.6 kg/ha | N: 40.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 940/2000 | Score: 438 | Avg: -920 | Steps: 165 | Yield: 7807.8 kg/ha | N: 280.0 kg/ha | Water: 522.0 mm | LR: 1.00e-02
Episode 941/2000 | Score: -20 | Avg: -920 | Steps: 167 | Yield: 877.0 kg/ha | N: 0.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 942/2000 | Score: 313 | Avg: -920 | Steps: 165 | Yield: 6715.3 kg/ha | N: 320.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 943/2000 | Score: 83 | Avg: -920 | Steps: 164 | Yield: 1316.9 kg/ha | N: 0.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 944/2000 | Score: 208 | Avg: -920 | Steps: 166 | Yield: 10930.3 kg/ha | N: 360.0 kg/ha | Water: 1122.0 mm | LR: 1.00e-02
Episode 945/2000 | Score: 563 | Avg: -920 | Steps: 159 | Yield: 9222.4 kg/ha | N: 280.0 kg/ha | Water: 612.0 mm | LR: 1.00e-02
Episode 946/2000 | Score: 93 | Avg: -920 | Steps: 158 | Yield: 8225.9 kg/ha | N: 800.0 kg/ha | Water: 522.0 mm | LR: 1.00e-02
Episode 947/2000 | Score: 580 | Avg: -920 | Steps: 150 | Yield: 9388.6 kg/ha | N: 400.0 kg/ha | Water: 528.0 mm | LR: 1.00e-02
Episode 948/2000 | Score: 346 | Avg: -920 | Steps: 160 | Yield: 7982.3 kg/ha | N: 440.0 kg/ha | Water: 516.0 mm | LR: 1.00e-02
Episode 949/2000 | Score: 816 | Avg: -920 | Steps: 166 | Yield: 10313.6 kg/ha | N: 320.0 kg/ha | Water: 492.0 mm | LR: 1.00e-02
Episode 950/2000 | Score: -230 | Avg: -920 | Steps: 153 | Yield: 1181.5 kg/ha | N: 160.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 951/2000 | Score: -110 | Avg: -920 | Steps: 163 | Yield: 977.2 kg/ha | N: 0.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 952/2000 | Score: 547 | Avg: -920 | Steps: 161 | Yield: 7895.2 kg/ha | N: 360.0 kg/ha | Water: 378.0 mm | LR: 1.00e-02
Episode 953/2000 | Score: 352 | Avg: -920 | Steps: 161 | Yield: 3865.3 kg/ha | N: 160.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 954/2000 | Score: 726 | Avg: -920 | Steps: 157 | Yield: 10171.3 kg/ha | N: 280.0 kg/ha | Water: 600.0 mm | LR: 1.00e-02
Episode 955/2000 | Score: 469 | Avg: -920 | Steps: 163 | Yield: 7686.0 kg/ha | N: 200.0 kg/ha | Water: 534.0 mm | LR: 1.00e-02
Episode 956/2000 | Score: 615 | Avg: -920 | Steps: 158 | Yield: 7875.6 kg/ha | N: 120.0 kg/ha | Water: 486.0 mm | LR: 1.00e-02
Episode 957/2000 | Score: 502 | Avg: -920 | Steps: 156 | Yield: 6712.3 kg/ha | N: 80.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 958/2000 | Score: 567 | Avg: -920 | Steps: 156 | Yield: 9054.2 kg/ha | N: 400.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 959/2000 | Score: 119 | Avg: -920 | Steps: 159 | Yield: 3193.4 kg/ha | N: 120.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 960/2000 | Score: 286 | Avg: -920 | Steps: 156 | Yield: 8524.8 kg/ha | N: 600.0 kg/ha | Water: 534.0 mm | LR: 1.00e-02
Episode 961/2000 | Score: 75 | Avg: -920 | Steps: 160 | Yield: 5935.4 kg/ha | N: 240.0 kg/ha | Water: 612.0 mm | LR: 1.00e-02
Episode 962/2000 | Score: 131 | Avg: -920 | Steps: 163 | Yield: 4123.2 kg/ha | N: 40.0 kg/ha | Water: 444.0 mm | LR: 1.00e-02
Episode 963/2000 | Score: 157 | Avg: -920 | Steps: 158 | Yield: 3744.1 kg/ha | N: 40.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 964/2000 | Score: 385 | Avg: -920 | Steps: 165 | Yield: 5805.4 kg/ha | N: 80.0 kg/ha | Water: 426.0 mm | LR: 1.00e-02
Episode 965/2000 | Score: 242 | Avg: -920 | Steps: 163 | Yield: 7936.1 kg/ha | N: 520.0 kg/ha | Water: 546.0 mm | LR: 1.00e-02
Episode 966/2000 | Score: 381 | Avg: -920 | Steps: 162 | Yield: 7417.4 kg/ha | N: 400.0 kg/ha | Water: 432.0 mm | LR: 1.00e-02
Episode 967/2000 | Score: 194 | Avg: -920 | Steps: 154 | Yield: 8290.1 kg/ha | N: 360.0 kg/ha | Water: 756.0 mm | LR: 1.00e-02
Episode 968/2000 | Score: 410 | Avg: -920 | Steps: 162 | Yield: 3878.9 kg/ha | N: 40.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 969/2000 | Score: 339 | Avg: -920 | Steps: 151 | Yield: 4770.4 kg/ha | N: 40.0 kg/ha | Water: 348.0 mm | LR: 1.00e-02
Episode 970/2000 | Score: 255 | Avg: -920 | Steps: 154 | Yield: 6775.1 kg/ha | N: 240.0 kg/ha | Water: 540.0 mm | LR: 1.00e-02
Episode 971/2000 | Score: 565 | Avg: -920 | Steps: 164 | Yield: 8823.9 kg/ha | N: 640.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 972/2000 | Score: 385 | Avg: -920 | Steps: 158 | Yield: 7544.6 kg/ha | N: 440.0 kg/ha | Water: 348.0 mm | LR: 1.00e-02
Episode 973/2000 | Score: 469 | Avg: -920 | Steps: 159 | Yield: 7139.3 kg/ha | N: 400.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 974/2000 | Score: 395 | Avg: -920 | Steps: 160 | Yield: 5002.6 kg/ha | N: 200.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 975/2000 | Score: 365 | Avg: -920 | Steps: 164 | Yield: 10073.7 kg/ha | N: 400.0 kg/ha | Water: 822.0 mm | LR: 1.00e-02
Episode 976/2000 | Score: -770 | Avg: -920 | Steps: 162 | Yield: 844.6 kg/ha | N: 960.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 977/2000 | Score: 345 | Avg: -920 | Steps: 158 | Yield: 7455.5 kg/ha | N: 720.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 978/2000 | Score: 616 | Avg: -920 | Steps: 163 | Yield: 8243.6 kg/ha | N: 200.0 kg/ha | Water: 474.0 mm | LR: 1.00e-02
Episode 979/2000 | Score: 781 | Avg: -920 | Steps: 157 | Yield: 10393.3 kg/ha | N: 480.0 kg/ha | Water: 438.0 mm | LR: 1.00e-02
Episode 980/2000 | Score: -128 | Avg: -920 | Steps: 163 | Yield: 3699.5 kg/ha | N: 760.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 981/2000 | Score: 757 | Avg: -920 | Steps: 157 | Yield: 9813.7 kg/ha | N: 320.0 kg/ha | Water: 492.0 mm | LR: 1.00e-02
Episode 982/2000 | Score: 573 | Avg: -920 | Steps: 148 | Yield: 7853.7 kg/ha | N: 520.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 983/2000 | Score: 464 | Avg: -920 | Steps: 160 | Yield: 7624.6 kg/ha | N: 160.0 kg/ha | Water: 558.0 mm | LR: 1.00e-02
Episode 984/2000 | Score: 837 | Avg: -920 | Steps: 160 | Yield: 9640.6 kg/ha | N: 200.0 kg/ha | Water: 480.0 mm | LR: 1.00e-02
Episode 985/2000 | Score: 28 | Avg: -920 | Steps: 161 | Yield: 3110.5 kg/ha | N: 520.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 986/2000 | Score: 679 | Avg: -920 | Steps: 152 | Yield: 9821.3 kg/ha | N: 320.0 kg/ha | Water: 564.0 mm | LR: 1.00e-02
Episode 987/2000 | Score: 535 | Avg: -920 | Steps: 160 | Yield: 11041.8 kg/ha | N: 520.0 kg/ha | Water: 726.0 mm | LR: 1.00e-02
Episode 988/2000 | Score: 468 | Avg: -920 | Steps: 166 | Yield: 7409.5 kg/ha | N: 280.0 kg/ha | Water: 438.0 mm | LR: 1.00e-02
Episode 989/2000 | Score: 757 | Avg: -920 | Steps: 160 | Yield: 11296.2 kg/ha | N: 440.0 kg/ha | Water: 618.0 mm | LR: 1.00e-02
Episode 990/2000 | Score: 295 | Avg: -920 | Steps: 160 | Yield: 7516.7 kg/ha | N: 520.0 kg/ha | Water: 432.0 mm | LR: 1.00e-02
Episode 991/2000 | Score: 468 | Avg: -920 | Steps: 163 | Yield: 10599.6 kg/ha | N: 560.0 kg/ha | Water: 660.0 mm | LR: 1.00e-02
Episode 992/2000 | Score: -91 | Avg: -920 | Steps: 161 | Yield: 3791.0 kg/ha | N: 640.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 993/2000 | Score: 681 | Avg: -920 | Steps: 160 | Yield: 9433.0 kg/ha | N: 440.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 994/2000 | Score: 717 | Avg: -920 | Steps: 161 | Yield: 10579.8 kg/ha | N: 640.0 kg/ha | Water: 396.0 mm | LR: 1.00e-02
Episode 995/2000 | Score: 698 | Avg: -920 | Steps: 159 | Yield: 11246.5 kg/ha | N: 480.0 kg/ha | Water: 630.0 mm | LR: 1.00e-02
Episode 996/2000 | Score: 541 | Avg: -920 | Steps: 158 | Yield: 10678.2 kg/ha | N: 640.0 kg/ha | Water: 564.0 mm | LR: 1.00e-02
Episode 997/2000 | Score: 277 | Avg: -920 | Steps: 160 | Yield: 7227.2 kg/ha | N: 360.0 kg/ha | Water: 522.0 mm | LR: 1.00e-02
Episode 998/2000 | Score: 302 | Avg: -920 | Steps: 166 | Yield: 11145.5 kg/ha | N: 560.0 kg/ha | Water: 906.0 mm | LR: 1.00e-02
Episode 999/2000 | Score: 642 | Avg: -920 | Steps: 161 | Yield: 7504.2 kg/ha | N: 120.0 kg/ha | Water: 408.0 mm | LR: 1.00e-02
Episode 1000/2000 | Score: 450 | Avg: -920 | Steps: 160 | Yield: 11185.9 kg/ha | N: 640.0 kg/ha | Water: 720.0 mm | LR: 1.00e-02
Episode 1001/2000 | Score: 274 | Avg: -920 | Steps: 156 | Yield: 10852.0 kg/ha | N: 520.0 kg/ha | Water: 936.0 mm | LR: 1.00e-02
Episode 1002/2000 | Score: 480 | Avg: -920 | Steps: 163 | Yield: 9939.5 kg/ha | N: 520.0 kg/ha | Water: 600.0 mm | LR: 1.00e-02
Episode 1003/2000 | Score: 630 | Avg: -920 | Steps: 164 | Yield: 9658.5 kg/ha | N: 600.0 kg/ha | Water: 384.0 mm | LR: 1.00e-02
Episode 1004/2000 | Score: 618 | Avg: -920 | Steps: 161 | Yield: 8498.7 kg/ha | N: 400.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1005/2000 | Score: 201 | Avg: -920 | Steps: 164 | Yield: 6447.3 kg/ha | N: 600.0 kg/ha | Water: 312.0 mm | LR: 1.00e-02
Episode 1006/2000 | Score: 587 | Avg: -920 | Steps: 156 | Yield: 8289.9 kg/ha | N: 280.0 kg/ha | Water: 456.0 mm | LR: 1.00e-02
Episode 1007/2000 | Score: 723 | Avg: -920 | Steps: 159 | Yield: 10334.7 kg/ha | N: 400.0 kg/ha | Water: 534.0 mm | LR: 1.00e-02
Episode 1008/2000 | Score: 993 | Avg: -920 | Steps: 155 | Yield: 11387.7 kg/ha | N: 360.0 kg/ha | Water: 474.0 mm | LR: 1.00e-02
Episode 1009/2000 | Score: 364 | Avg: -920 | Steps: 160 | Yield: 10551.1 kg/ha | N: 680.0 kg/ha | Water: 690.0 mm | LR: 1.00e-02
Episode 1010/2000 | Score: 552 | Avg: -920 | Steps: 162 | Yield: 8116.8 kg/ha | N: 240.0 kg/ha | Water: 492.0 mm | LR: 1.00e-02
Episode 1011/2000 | Score: 632 | Avg: -920 | Steps: 157 | Yield: 10639.5 kg/ha | N: 400.0 kg/ha | Water: 666.0 mm | LR: 1.00e-02
Episode 1012/2000 | Score: 227 | Avg: -920 | Steps: 165 | Yield: 5730.9 kg/ha | N: 600.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1013/2000 | Score: 638 | Avg: -920 | Steps: 161 | Yield: 10123.2 kg/ha | N: 440.0 kg/ha | Water: 540.0 mm | LR: 1.00e-02
Episode 1014/2000 | Score: 787 | Avg: -920 | Steps: 162 | Yield: 10616.8 kg/ha | N: 400.0 kg/ha | Water: 522.0 mm | LR: 1.00e-02
Episode 1015/2000 | Score: 631 | Avg: -920 | Steps: 164 | Yield: 9543.3 kg/ha | N: 400.0 kg/ha | Water: 504.0 mm | LR: 1.00e-02
Episode 1016/2000 | Score: 600 | Avg: -920 | Steps: 156 | Yield: 8282.8 kg/ha | N: 320.0 kg/ha | Water: 414.0 mm | LR: 1.00e-02
Episode 1017/2000 | Score: 494 | Avg: -920 | Steps: 158 | Yield: 9673.9 kg/ha | N: 440.0 kg/ha | Water: 606.0 mm | LR: 1.00e-02
Episode 1018/2000 | Score: 551 | Avg: -920 | Steps: 157 | Yield: 11013.5 kg/ha | N: 520.0 kg/ha | Water: 708.0 mm | LR: 1.00e-02
Episode 1019/2000 | Score: 783 | Avg: -920 | Steps: 165 | Yield: 9509.6 kg/ha | N: 560.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1020/2000 | Score: 472 | Avg: -920 | Steps: 154 | Yield: 8194.2 kg/ha | N: 640.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1021/2000 | Score: 306 | Avg: -920 | Steps: 163 | Yield: 7731.1 kg/ha | N: 440.0 kg/ha | Water: 510.0 mm | LR: 1.00e-02
Episode 1022/2000 | Score: 779 | Avg: -920 | Steps: 154 | Yield: 10404.3 kg/ha | N: 360.0 kg/ha | Water: 522.0 mm | LR: 1.00e-02
Episode 1023/2000 | Score: 727 | Avg: -920 | Steps: 158 | Yield: 10109.1 kg/ha | N: 400.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 1024/2000 | Score: 921 | Avg: -920 | Steps: 162 | Yield: 11289.1 kg/ha | N: 440.0 kg/ha | Water: 462.0 mm | LR: 1.00e-02
Episode 1025/2000 | Score: 413 | Avg: -920 | Steps: 162 | Yield: 7822.5 kg/ha | N: 240.0 kg/ha | Water: 570.0 mm | LR: 1.00e-02
Episode 1026/2000 | Score: 654 | Avg: -920 | Steps: 155 | Yield: 9706.7 kg/ha | N: 320.0 kg/ha | Water: 570.0 mm | LR: 1.00e-02
Episode 1027/2000 | Score: 583 | Avg: -920 | Steps: 158 | Yield: 8054.6 kg/ha | N: 280.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1028/2000 | Score: 796 | Avg: -920 | Steps: 159 | Yield: 9406.9 kg/ha | N: 280.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1029/2000 | Score: 745 | Avg: -920 | Steps: 159 | Yield: 9842.3 kg/ha | N: 440.0 kg/ha | Water: 414.0 mm | LR: 1.00e-02
Episode 1030/2000 | Score: 597 | Avg: -920 | Steps: 154 | Yield: 10791.0 kg/ha | N: 600.0 kg/ha | Water: 558.0 mm | LR: 1.00e-02
Episode 1031/2000 | Score: 98 | Avg: -920 | Steps: 159 | Yield: 5289.3 kg/ha | N: 600.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1032/2000 | Score: 816 | Avg: -920 | Steps: 159 | Yield: 10254.8 kg/ha | N: 400.0 kg/ha | Water: 438.0 mm | LR: 1.00e-02
Episode 1033/2000 | Score: 543 | Avg: -920 | Steps: 154 | Yield: 9114.2 kg/ha | N: 400.0 kg/ha | Water: 510.0 mm | LR: 1.00e-02
Episode 1034/2000 | Score: 917 | Avg: -920 | Steps: 157 | Yield: 11380.7 kg/ha | N: 480.0 kg/ha | Water: 456.0 mm | LR: 1.00e-02
Episode 1035/2000 | Score: 835 | Avg: -920 | Steps: 164 | Yield: 11987.8 kg/ha | N: 480.0 kg/ha | Water: 618.0 mm | LR: 1.00e-02
Episode 1036/2000 | Score: 399 | Avg: -920 | Steps: 160 | Yield: 8095.9 kg/ha | N: 680.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1037/2000 | Score: 588 | Avg: -920 | Steps: 154 | Yield: 8234.0 kg/ha | N: 360.0 kg/ha | Water: 390.0 mm | LR: 1.00e-02
Episode 1038/2000 | Score: 1037 | Avg: -920 | Steps: 160 | Yield: 11118.2 kg/ha | N: 360.0 kg/ha | Water: 396.0 mm | LR: 1.00e-02
Episode 1039/2000 | Score: 876 | Avg: -920 | Steps: 162 | Yield: 10405.4 kg/ha | N: 320.0 kg/ha | Water: 468.0 mm | LR: 1.00e-02
Episode 1040/2000 | Score: 704 | Avg: -920 | Steps: 155 | Yield: 11092.6 kg/ha | N: 600.0 kg/ha | Water: 516.0 mm | LR: 1.00e-02
Episode 1041/2000 | Score: 403 | Avg: -920 | Steps: 160 | Yield: 8209.1 kg/ha | N: 280.0 kg/ha | Water: 612.0 mm | LR: 1.00e-02
Episode 1042/2000 | Score: -257 | Avg: -920 | Steps: 163 | Yield: 1094.8 kg/ha | N: 360.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1043/2000 | Score: 235 | Avg: -920 | Steps: 159 | Yield: 5309.1 kg/ha | N: 280.0 kg/ha | Water: 348.0 mm | LR: 1.00e-02
Episode 1044/2000 | Score: 139 | Avg: -920 | Steps: 161 | Yield: 6041.1 kg/ha | N: 280.0 kg/ha | Water: 540.0 mm | LR: 1.00e-02
Episode 1045/2000 | Score: 339 | Avg: -920 | Steps: 160 | Yield: 7976.4 kg/ha | N: 640.0 kg/ha | Water: 372.0 mm | LR: 1.00e-02
Episode 1046/2000 | Score: 738 | Avg: -920 | Steps: 158 | Yield: 11682.5 kg/ha | N: 600.0 kg/ha | Water: 570.0 mm | LR: 1.00e-02
Episode 1047/2000 | Score: 690 | Avg: -920 | Steps: 163 | Yield: 9614.4 kg/ha | N: 440.0 kg/ha | Water: 432.0 mm | LR: 1.00e-02
Episode 1048/2000 | Score: 684 | Avg: -920 | Steps: 162 | Yield: 9518.5 kg/ha | N: 520.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1049/2000 | Score: 277 | Avg: -920 | Steps: 158 | Yield: 11531.6 kg/ha | N: 920.0 kg/ha | Water: 744.0 mm | LR: 1.00e-02
Episode 1050/2000 | Score: 313 | Avg: -920 | Steps: 159 | Yield: 5768.4 kg/ha | N: 440.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1051/2000 | Score: 780 | Avg: -920 | Steps: 167 | Yield: 9776.8 kg/ha | N: 400.0 kg/ha | Water: 402.0 mm | LR: 1.00e-02
Episode 1052/2000 | Score: 843 | Avg: -920 | Steps: 157 | Yield: 10408.8 kg/ha | N: 480.0 kg/ha | Water: 378.0 mm | LR: 1.00e-02
Episode 1053/2000 | Score: 699 | Avg: -920 | Steps: 165 | Yield: 8403.8 kg/ha | N: 320.0 kg/ha | Water: 342.0 mm | LR: 1.00e-02
Episode 1054/2000 | Score: 586 | Avg: -920 | Steps: 163 | Yield: 9193.0 kg/ha | N: 520.0 kg/ha | Water: 408.0 mm | LR: 1.00e-02
Episode 1055/2000 | Score: 434 | Avg: -920 | Steps: 155 | Yield: 7813.0 kg/ha | N: 320.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 1056/2000 | Score: 558 | Avg: -920 | Steps: 161 | Yield: 7693.9 kg/ha | N: 440.0 kg/ha | Water: 270.0 mm | LR: 1.00e-02
Episode 1057/2000 | Score: 761 | Avg: -920 | Steps: 165 | Yield: 12070.5 kg/ha | N: 640.0 kg/ha | Water: 576.0 mm | LR: 1.00e-02
Episode 1058/2000 | Score: 958 | Avg: -920 | Steps: 166 | Yield: 11267.7 kg/ha | N: 280.0 kg/ha | Water: 528.0 mm | LR: 1.00e-02
Episode 1059/2000 | Score: 792 | Avg: -920 | Steps: 159 | Yield: 8868.3 kg/ha | N: 320.0 kg/ha | Water: 324.0 mm | LR: 1.00e-02
Episode 1060/2000 | Score: 622 | Avg: -920 | Steps: 158 | Yield: 9706.7 kg/ha | N: 720.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1061/2000 | Score: 888 | Avg: -920 | Steps: 159 | Yield: 11369.8 kg/ha | N: 440.0 kg/ha | Water: 510.0 mm | LR: 1.00e-02
Episode 1062/2000 | Score: 894 | Avg: -920 | Steps: 155 | Yield: 10893.3 kg/ha | N: 320.0 kg/ha | Water: 516.0 mm | LR: 1.00e-02
Episode 1063/2000 | Score: -207 | Avg: -920 | Steps: 162 | Yield: 2652.0 kg/ha | N: 400.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1064/2000 | Score: 839 | Avg: -920 | Steps: 158 | Yield: 8896.5 kg/ha | N: 400.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1065/2000 | Score: 985 | Avg: -920 | Steps: 157 | Yield: 10965.8 kg/ha | N: 320.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 1066/2000 | Score: 860 | Avg: -920 | Steps: 165 | Yield: 11059.5 kg/ha | N: 480.0 kg/ha | Water: 456.0 mm | LR: 1.00e-02
Episode 1067/2000 | Score: 771 | Avg: -920 | Steps: 160 | Yield: 11073.7 kg/ha | N: 320.0 kg/ha | Water: 642.0 mm | LR: 1.00e-02
Episode 1068/2000 | Score: 418 | Avg: -920 | Steps: 163 | Yield: 5599.4 kg/ha | N: 240.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1069/2000 | Score: 511 | Avg: -920 | Steps: 155 | Yield: 8874.7 kg/ha | N: 360.0 kg/ha | Water: 546.0 mm | LR: 1.00e-02
Episode 1070/2000 | Score: 600 | Avg: -920 | Steps: 158 | Yield: 8372.1 kg/ha | N: 280.0 kg/ha | Water: 456.0 mm | LR: 1.00e-02
Episode 1071/2000 | Score: 720 | Avg: -920 | Steps: 165 | Yield: 10795.8 kg/ha | N: 880.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1072/2000 | Score: 90 | Avg: -920 | Steps: 159 | Yield: 5183.2 kg/ha | N: 280.0 kg/ha | Water: 462.0 mm | LR: 1.00e-02
Episode 1073/2000 | Score: 831 | Avg: -920 | Steps: 158 | Yield: 11001.6 kg/ha | N: 480.0 kg/ha | Water: 480.0 mm | LR: 1.00e-02
Episode 1074/2000 | Score: 911 | Avg: -920 | Steps: 162 | Yield: 11025.1 kg/ha | N: 400.0 kg/ha | Water: 468.0 mm | LR: 1.00e-02
Episode 1075/2000 | Score: 704 | Avg: -920 | Steps: 160 | Yield: 9683.8 kg/ha | N: 360.0 kg/ha | Water: 486.0 mm | LR: 1.00e-02
Episode 1076/2000 | Score: 82 | Avg: -920 | Steps: 160 | Yield: 10763.2 kg/ha | N: 1280.0 kg/ha | Water: 540.0 mm | LR: 1.00e-02
Episode 1077/2000 | Score: 876 | Avg: -920 | Steps: 164 | Yield: 10854.1 kg/ha | N: 360.0 kg/ha | Water: 486.0 mm | LR: 1.00e-02
Episode 1078/2000 | Score: 692 | Avg: -920 | Steps: 158 | Yield: 9046.7 kg/ha | N: 440.0 kg/ha | Water: 336.0 mm | LR: 1.00e-02
Episode 1079/2000 | Score: 764 | Avg: -920 | Steps: 158 | Yield: 11195.5 kg/ha | N: 720.0 kg/ha | Water: 396.0 mm | LR: 1.00e-02
Episode 1080/2000 | Score: 721 | Avg: -920 | Steps: 154 | Yield: 10772.2 kg/ha | N: 440.0 kg/ha | Water: 576.0 mm | LR: 1.00e-02
Episode 1081/2000 | Score: 770 | Avg: -920 | Steps: 157 | Yield: 9721.1 kg/ha | N: 360.0 kg/ha | Water: 438.0 mm | LR: 1.00e-02
Episode 1082/2000 | Score: 660 | Avg: -920 | Steps: 162 | Yield: 8524.7 kg/ha | N: 360.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1083/2000 | Score: 125 | Avg: -920 | Steps: 157 | Yield: 4533.7 kg/ha | N: 440.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 1084/2000 | Score: -105 | Avg: -920 | Steps: 158 | Yield: 2930.6 kg/ha | N: 360.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1085/2000 | Score: -57 | Avg: -920 | Steps: 162 | Yield: 2593.5 kg/ha | N: 240.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1086/2000 | Score: -106 | Avg: -920 | Steps: 154 | Yield: 3132.7 kg/ha | N: 360.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 1087/2000 | Score: 188 | Avg: -920 | Steps: 163 | Yield: 3982.2 kg/ha | N: 400.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1088/2000 | Score: 951 | Avg: -920 | Steps: 163 | Yield: 10543.4 kg/ha | N: 320.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1089/2000 | Score: 310 | Avg: -920 | Steps: 160 | Yield: 4814.0 kg/ha | N: 320.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1090/2000 | Score: -176 | Avg: -920 | Steps: 159 | Yield: 2189.5 kg/ha | N: 560.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1091/2000 | Score: -177 | Avg: -920 | Steps: 155 | Yield: 3135.6 kg/ha | N: 600.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1092/2000 | Score: 525 | Avg: -920 | Steps: 163 | Yield: 6978.8 kg/ha | N: 280.0 kg/ha | Water: 324.0 mm | LR: 1.00e-02
Episode 1093/2000 | Score: 702 | Avg: -920 | Steps: 156 | Yield: 9591.2 kg/ha | N: 520.0 kg/ha | Water: 360.0 mm | LR: 1.00e-02
Episode 1094/2000 | Score: 896 | Avg: -920 | Steps: 167 | Yield: 11354.7 kg/ha | N: 360.0 kg/ha | Water: 540.0 mm | LR: 1.00e-02
Episode 1095/2000 | Score: 184 | Avg: -920 | Steps: 157 | Yield: 3458.7 kg/ha | N: 400.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1096/2000 | Score: 276 | Avg: -920 | Steps: 162 | Yield: 5304.0 kg/ha | N: 360.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1097/2000 | Score: 590 | Avg: -920 | Steps: 162 | Yield: 7286.8 kg/ha | N: 360.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1098/2000 | Score: -39 | Avg: -920 | Steps: 157 | Yield: 3115.1 kg/ha | N: 280.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1099/2000 | Score: 471 | Avg: -920 | Steps: 157 | Yield: 9824.2 kg/ha | N: 400.0 kg/ha | Water: 690.0 mm | LR: 1.00e-02
Episode 1100/2000 | Score: 954 | Avg: -920 | Steps: 159 | Yield: 10088.0 kg/ha | N: 400.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 1101/2000 | Score: 814 | Avg: -920 | Steps: 161 | Yield: 11087.9 kg/ha | N: 360.0 kg/ha | Water: 594.0 mm | LR: 1.00e-02
Episode 1102/2000 | Score: -237 | Avg: -920 | Steps: 163 | Yield: 1808.7 kg/ha | N: 520.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1103/2000 | Score: 853 | Avg: -920 | Steps: 159 | Yield: 9204.6 kg/ha | N: 360.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1104/2000 | Score: -198 | Avg: -920 | Steps: 132 | Yield: 0.0 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1105/2000 | Score: 150 | Avg: -920 | Steps: 165 | Yield: 4162.6 kg/ha | N: 400.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1106/2000 | Score: 991 | Avg: -920 | Steps: 167 | Yield: 11403.7 kg/ha | N: 400.0 kg/ha | Water: 444.0 mm | LR: 1.00e-02
Episode 1107/2000 | Score: -370 | Avg: -920 | Steps: 159 | Yield: 1345.3 kg/ha | N: 520.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1108/2000 | Score: 482 | Avg: -920 | Steps: 163 | Yield: 7981.4 kg/ha | N: 560.0 kg/ha | Water: 300.0 mm | LR: 1.00e-02
Episode 1109/2000 | Score: 756 | Avg: -920 | Steps: 158 | Yield: 9088.4 kg/ha | N: 360.0 kg/ha | Water: 354.0 mm | LR: 1.00e-02
Episode 1110/2000 | Score: -295 | Avg: -920 | Steps: 165 | Yield: 2231.1 kg/ha | N: 360.0 kg/ha | Water: 324.0 mm | LR: 1.00e-02
Episode 1111/2000 | Score: -341 | Avg: -920 | Steps: 156 | Yield: 3822.6 kg/ha | N: 720.0 kg/ha | Water: 336.0 mm | LR: 1.00e-02
Episode 1112/2000 | Score: 697 | Avg: -920 | Steps: 159 | Yield: 9050.7 kg/ha | N: 560.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1113/2000 | Score: -380 | Avg: -920 | Steps: 153 | Yield: 1250.7 kg/ha | N: 480.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1114/2000 | Score: 702 | Avg: -920 | Steps: 169 | Yield: 11763.0 kg/ha | N: 520.0 kg/ha | Water: 678.0 mm | LR: 1.00e-02
Episode 1115/2000 | Score: 238 | Avg: -920 | Steps: 161 | Yield: 5984.2 kg/ha | N: 520.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 1116/2000 | Score: 505 | Avg: -920 | Steps: 158 | Yield: 10708.3 kg/ha | N: 600.0 kg/ha | Water: 624.0 mm | LR: 1.00e-02
Episode 1117/2000 | Score: 686 | Avg: -920 | Steps: 158 | Yield: 11047.4 kg/ha | N: 760.0 kg/ha | Water: 360.0 mm | LR: 1.00e-02
Episode 1118/2000 | Score: 331 | Avg: -920 | Steps: 157 | Yield: 6557.0 kg/ha | N: 400.0 kg/ha | Water: 348.0 mm | LR: 1.00e-02
Episode 1119/2000 | Score: 668 | Avg: -920 | Steps: 160 | Yield: 7947.1 kg/ha | N: 360.0 kg/ha | Water: 270.0 mm | LR: 1.00e-02
Episode 1120/2000 | Score: 1024 | Avg: -920 | Steps: 167 | Yield: 9499.7 kg/ha | N: 320.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1121/2000 | Score: -489 | Avg: -920 | Steps: 154 | Yield: 1723.1 kg/ha | N: 680.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1122/2000 | Score: 1073 | Avg: -920 | Steps: 156 | Yield: 11623.5 kg/ha | N: 440.0 kg/ha | Water: 378.0 mm | LR: 1.00e-02
Episode 1123/2000 | Score: 256 | Avg: -920 | Steps: 162 | Yield: 10678.6 kg/ha | N: 960.0 kg/ha | Water: 606.0 mm | LR: 1.00e-02
Episode 1124/2000 | Score: 765 | Avg: -920 | Steps: 162 | Yield: 10164.7 kg/ha | N: 480.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1125/2000 | Score: -123 | Avg: -920 | Steps: 159 | Yield: 1320.7 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1126/2000 | Score: 65 | Avg: -920 | Steps: 160 | Yield: 4243.1 kg/ha | N: 440.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1127/2000 | Score: -225 | Avg: -920 | Steps: 156 | Yield: 3154.7 kg/ha | N: 640.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1128/2000 | Score: 777 | Avg: -920 | Steps: 163 | Yield: 8779.7 kg/ha | N: 280.0 kg/ha | Water: 348.0 mm | LR: 1.00e-02
Episode 1129/2000 | Score: 461 | Avg: -920 | Steps: 161 | Yield: 7867.1 kg/ha | N: 480.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1130/2000 | Score: 607 | Avg: -920 | Steps: 163 | Yield: 9108.8 kg/ha | N: 360.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 1131/2000 | Score: -286 | Avg: -920 | Steps: 162 | Yield: 1693.1 kg/ha | N: 400.0 kg/ha | Water: 210.0 mm | LR: 1.00e-02
Episode 1132/2000 | Score: 598 | Avg: -920 | Steps: 159 | Yield: 10887.7 kg/ha | N: 560.0 kg/ha | Water: 600.0 mm | LR: 1.00e-02
Episode 1133/2000 | Score: 839 | Avg: -920 | Steps: 160 | Yield: 9260.5 kg/ha | N: 280.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1134/2000 | Score: 986 | Avg: -920 | Steps: 168 | Yield: 10432.3 kg/ha | N: 320.0 kg/ha | Water: 372.0 mm | LR: 1.00e-02
Episode 1135/2000 | Score: 928 | Avg: -920 | Steps: 158 | Yield: 10135.3 kg/ha | N: 360.0 kg/ha | Water: 336.0 mm | LR: 1.00e-02
Episode 1136/2000 | Score: -90 | Avg: -920 | Steps: 156 | Yield: 1825.3 kg/ha | N: 320.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1137/2000 | Score: 655 | Avg: -920 | Steps: 156 | Yield: 7882.5 kg/ha | N: 440.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1138/2000 | Score: 1143 | Avg: -920 | Steps: 157 | Yield: 10588.7 kg/ha | N: 320.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1139/2000 | Score: 177 | Avg: -920 | Steps: 161 | Yield: 4454.3 kg/ha | N: 400.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1140/2000 | Score: -72 | Avg: -920 | Steps: 163 | Yield: 5575.4 kg/ha | N: 800.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1141/2000 | Score: 748 | Avg: -920 | Steps: 161 | Yield: 9044.0 kg/ha | N: 520.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1142/2000 | Score: 790 | Avg: -920 | Steps: 165 | Yield: 8551.4 kg/ha | N: 360.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1143/2000 | Score: 776 | Avg: -920 | Steps: 162 | Yield: 8179.9 kg/ha | N: 320.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1144/2000 | Score: 525 | Avg: -920 | Steps: 159 | Yield: 6290.5 kg/ha | N: 360.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1145/2000 | Score: -237 | Avg: -920 | Steps: 164 | Yield: 2633.3 kg/ha | N: 320.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1146/2000 | Score: -441 | Avg: -920 | Steps: 164 | Yield: 925.8 kg/ha | N: 560.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1147/2000 | Score: 711 | Avg: -920 | Steps: 171 | Yield: 8735.4 kg/ha | N: 480.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1148/2000 | Score: 67 | Avg: -920 | Steps: 160 | Yield: 2967.7 kg/ha | N: 400.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1149/2000 | Score: -299 | Avg: -920 | Steps: 159 | Yield: 2685.3 kg/ha | N: 640.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1150/2000 | Score: -102 | Avg: -920 | Steps: 155 | Yield: 1928.6 kg/ha | N: 440.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1151/2000 | Score: 594 | Avg: -920 | Steps: 150 | Yield: 8046.6 kg/ha | N: 440.0 kg/ha | Water: 300.0 mm | LR: 1.00e-02
Episode 1152/2000 | Score: 232 | Avg: -920 | Steps: 160 | Yield: 7502.1 kg/ha | N: 680.0 kg/ha | Water: 372.0 mm | LR: 1.00e-02
Episode 1153/2000 | Score: 374 | Avg: -920 | Steps: 159 | Yield: 9718.9 kg/ha | N: 1120.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1154/2000 | Score: -37 | Avg: -920 | Steps: 164 | Yield: 2625.4 kg/ha | N: 480.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1155/2000 | Score: 973 | Avg: -920 | Steps: 165 | Yield: 10072.9 kg/ha | N: 240.0 kg/ha | Water: 390.0 mm | LR: 1.00e-02
Episode 1156/2000 | Score: 348 | Avg: -920 | Steps: 169 | Yield: 4405.0 kg/ha | N: 240.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1157/2000 | Score: 189 | Avg: -920 | Steps: 161 | Yield: 3484.4 kg/ha | N: 240.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1158/2000 | Score: 679 | Avg: -920 | Steps: 161 | Yield: 9492.0 kg/ha | N: 320.0 kg/ha | Water: 516.0 mm | LR: 1.00e-02
Episode 1159/2000 | Score: 1149 | Avg: -920 | Steps: 167 | Yield: 10657.5 kg/ha | N: 360.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1160/2000 | Score: 1046 | Avg: -920 | Steps: 161 | Yield: 11510.4 kg/ha | N: 360.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1161/2000 | Score: 708 | Avg: -920 | Steps: 157 | Yield: 7286.7 kg/ha | N: 360.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1162/2000 | Score: 40 | Avg: -920 | Steps: 154 | Yield: 2231.6 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1163/2000 | Score: 873 | Avg: -920 | Steps: 160 | Yield: 9679.0 kg/ha | N: 480.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1164/2000 | Score: 155 | Avg: -920 | Steps: 158 | Yield: 3891.2 kg/ha | N: 440.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1165/2000 | Score: 197 | Avg: -920 | Steps: 159 | Yield: 4150.5 kg/ha | N: 480.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1166/2000 | Score: 885 | Avg: -920 | Steps: 157 | Yield: 8526.6 kg/ha | N: 360.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1167/2000 | Score: -13 | Avg: -920 | Steps: 154 | Yield: 2142.8 kg/ha | N: 320.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1168/2000 | Score: 948 | Avg: -920 | Steps: 158 | Yield: 9970.1 kg/ha | N: 560.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1169/2000 | Score: 403 | Avg: -920 | Steps: 162 | Yield: 4533.4 kg/ha | N: 280.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1170/2000 | Score: 863 | Avg: -920 | Steps: 164 | Yield: 8902.6 kg/ha | N: 320.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1171/2000 | Score: 277 | Avg: -920 | Steps: 162 | Yield: 3812.3 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1172/2000 | Score: 961 | Avg: -920 | Steps: 160 | Yield: 8147.8 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1173/2000 | Score: 288 | Avg: -920 | Steps: 159 | Yield: 3642.0 kg/ha | N: 280.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1174/2000 | Score: 723 | Avg: -920 | Steps: 164 | Yield: 8657.7 kg/ha | N: 440.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 1175/2000 | Score: 770 | Avg: -920 | Steps: 158 | Yield: 8570.7 kg/ha | N: 280.0 kg/ha | Water: 330.0 mm | LR: 1.00e-02
Episode 1176/2000 | Score: 416 | Avg: -920 | Steps: 160 | Yield: 4418.5 kg/ha | N: 240.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1177/2000 | Score: 698 | Avg: -920 | Steps: 156 | Yield: 7989.0 kg/ha | N: 280.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1178/2000 | Score: 534 | Avg: -920 | Steps: 159 | Yield: 5361.6 kg/ha | N: 280.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1179/2000 | Score: -353 | Avg: -920 | Steps: 157 | Yield: 876.0 kg/ha | N: 480.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1180/2000 | Score: 602 | Avg: -920 | Steps: 158 | Yield: 7230.9 kg/ha | N: 400.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1181/2000 | Score: -588 | Avg: -920 | Steps: 158 | Yield: 11399.4 kg/ha | N: 2280.0 kg/ha | Water: 510.0 mm | LR: 1.00e-02
Episode 1182/2000 | Score: 753 | Avg: -920 | Steps: 154 | Yield: 11428.0 kg/ha | N: 480.0 kg/ha | Water: 612.0 mm | LR: 1.00e-02
Episode 1183/2000 | Score: -138 | Avg: -920 | Steps: 160 | Yield: 1868.7 kg/ha | N: 440.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1184/2000 | Score: 226 | Avg: -920 | Steps: 165 | Yield: 4758.1 kg/ha | N: 440.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1185/2000 | Score: -179 | Avg: -920 | Steps: 162 | Yield: 3695.8 kg/ha | N: 640.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1186/2000 | Score: -316 | Avg: -920 | Steps: 156 | Yield: 978.2 kg/ha | N: 520.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1187/2000 | Score: 977 | Avg: -920 | Steps: 164 | Yield: 9745.5 kg/ha | N: 320.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 1188/2000 | Score: 761 | Avg: -920 | Steps: 160 | Yield: 7302.2 kg/ha | N: 280.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1189/2000 | Score: -159 | Avg: -920 | Steps: 155 | Yield: 1181.1 kg/ha | N: 320.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1190/2000 | Score: -59 | Avg: -920 | Steps: 162 | Yield: 2565.9 kg/ha | N: 320.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1191/2000 | Score: 567 | Avg: -920 | Steps: 153 | Yield: 6781.7 kg/ha | N: 480.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1192/2000 | Score: 240 | Avg: -920 | Steps: 165 | Yield: 5699.5 kg/ha | N: 560.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1193/2000 | Score: 590 | Avg: -920 | Steps: 167 | Yield: 6125.2 kg/ha | N: 320.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1194/2000 | Score: 353 | Avg: -920 | Steps: 164 | Yield: 7319.8 kg/ha | N: 600.0 kg/ha | Water: 300.0 mm | LR: 1.00e-02
Episode 1195/2000 | Score: 553 | Avg: -920 | Steps: 167 | Yield: 9816.0 kg/ha | N: 720.0 kg/ha | Water: 384.0 mm | LR: 1.00e-02
Episode 1196/2000 | Score: 257 | Avg: -920 | Steps: 163 | Yield: 4410.3 kg/ha | N: 440.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1197/2000 | Score: 552 | Avg: -920 | Steps: 165 | Yield: 11051.0 kg/ha | N: 960.0 kg/ha | Water: 390.0 mm | LR: 1.00e-02
Episode 1198/2000 | Score: 252 | Avg: -920 | Steps: 158 | Yield: 3914.5 kg/ha | N: 280.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1199/2000 | Score: 906 | Avg: -920 | Steps: 156 | Yield: 9464.6 kg/ha | N: 320.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1200/2000 | Score: 686 | Avg: -920 | Steps: 166 | Yield: 8279.3 kg/ha | N: 520.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1201/2000 | Score: 927 | Avg: -920 | Steps: 163 | Yield: 9406.9 kg/ha | N: 240.0 kg/ha | Water: 336.0 mm | LR: 1.00e-02
Episode 1202/2000 | Score: 123 | Avg: -920 | Steps: 159 | Yield: 2814.7 kg/ha | N: 240.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1203/2000 | Score: 633 | Avg: -920 | Steps: 165 | Yield: 8141.1 kg/ha | N: 560.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1204/2000 | Score: 1195 | Avg: -920 | Steps: 159 | Yield: 12252.5 kg/ha | N: 520.0 kg/ha | Water: 300.0 mm | LR: 1.00e-02
Episode 1205/2000 | Score: -152 | Avg: -920 | Steps: 163 | Yield: 847.6 kg/ha | N: 320.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1206/2000 | Score: 780 | Avg: -920 | Steps: 160 | Yield: 8401.1 kg/ha | N: 400.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1207/2000 | Score: 429 | Avg: -920 | Steps: 155 | Yield: 6028.0 kg/ha | N: 320.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1208/2000 | Score: 526 | Avg: -920 | Steps: 155 | Yield: 5475.3 kg/ha | N: 320.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1209/2000 | Score: 743 | Avg: -920 | Steps: 162 | Yield: 6803.0 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1210/2000 | Score: 933 | Avg: -920 | Steps: 165 | Yield: 9324.5 kg/ha | N: 400.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1211/2000 | Score: 759 | Avg: -920 | Steps: 159 | Yield: 8400.5 kg/ha | N: 360.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1212/2000 | Score: 997 | Avg: -920 | Steps: 159 | Yield: 10500.6 kg/ha | N: 320.0 kg/ha | Water: 366.0 mm | LR: 1.00e-02
Episode 1213/2000 | Score: -241 | Avg: -920 | Steps: 162 | Yield: 1258.7 kg/ha | N: 440.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1214/2000 | Score: 249 | Avg: -920 | Steps: 157 | Yield: 4577.4 kg/ha | N: 400.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1215/2000 | Score: 163 | Avg: -920 | Steps: 163 | Yield: 3699.5 kg/ha | N: 400.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1216/2000 | Score: 986 | Avg: -920 | Steps: 157 | Yield: 9430.5 kg/ha | N: 320.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1217/2000 | Score: 1271 | Avg: -920 | Steps: 160 | Yield: 10420.7 kg/ha | N: 200.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1218/2000 | Score: -29 | Avg: -920 | Steps: 161 | Yield: 2636.0 kg/ha | N: 480.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1219/2000 | Score: 266 | Avg: -920 | Steps: 157 | Yield: 5135.5 kg/ha | N: 440.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1220/2000 | Score: 69 | Avg: -920 | Steps: 161 | Yield: 3020.1 kg/ha | N: 400.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1221/2000 | Score: -921 | Avg: -920 | Steps: 160 | Yield: 1633.1 kg/ha | N: 1200.0 kg/ha | Water: 210.0 mm | LR: 1.00e-02
Episode 1222/2000 | Score: -210 | Avg: -920 | Steps: 153 | Yield: 2365.3 kg/ha | N: 480.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1223/2000 | Score: 96 | Avg: -920 | Steps: 163 | Yield: 4606.4 kg/ha | N: 440.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1224/2000 | Score: 148 | Avg: -920 | Steps: 163 | Yield: 3778.3 kg/ha | N: 360.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1225/2000 | Score: -36 | Avg: -920 | Steps: 159 | Yield: 2808.2 kg/ha | N: 440.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1226/2000 | Score: 314 | Avg: -920 | Steps: 158 | Yield: 4224.1 kg/ha | N: 280.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1227/2000 | Score: 587 | Avg: -920 | Steps: 158 | Yield: 7164.9 kg/ha | N: 280.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1228/2000 | Score: 410 | Avg: -920 | Steps: 162 | Yield: 6446.7 kg/ha | N: 520.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1229/2000 | Score: 66 | Avg: -920 | Steps: 159 | Yield: 4263.6 kg/ha | N: 360.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1230/2000 | Score: 667 | Avg: -920 | Steps: 156 | Yield: 8935.1 kg/ha | N: 600.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1231/2000 | Score: 46 | Avg: -920 | Steps: 160 | Yield: 3430.6 kg/ha | N: 360.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1232/2000 | Score: 649 | Avg: -920 | Steps: 161 | Yield: 7868.9 kg/ha | N: 560.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1233/2000 | Score: 82 | Avg: -920 | Steps: 155 | Yield: 2986.6 kg/ha | N: 360.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1234/2000 | Score: 310 | Avg: -920 | Steps: 164 | Yield: 5715.5 kg/ha | N: 400.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1235/2000 | Score: -168 | Avg: -920 | Steps: 157 | Yield: 996.4 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1236/2000 | Score: -195 | Avg: -920 | Steps: 159 | Yield: 865.1 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1237/2000 | Score: 389 | Avg: -920 | Steps: 160 | Yield: 5633.8 kg/ha | N: 400.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1238/2000 | Score: 467 | Avg: -920 | Steps: 161 | Yield: 6651.8 kg/ha | N: 480.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1239/2000 | Score: 396 | Avg: -920 | Steps: 159 | Yield: 4866.1 kg/ha | N: 280.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1240/2000 | Score: 321 | Avg: -920 | Steps: 156 | Yield: 4144.0 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1241/2000 | Score: 826 | Avg: -920 | Steps: 165 | Yield: 8154.5 kg/ha | N: 360.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1242/2000 | Score: -105 | Avg: -920 | Steps: 165 | Yield: 1526.3 kg/ha | N: 280.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1243/2000 | Score: -63 | Avg: -920 | Steps: 161 | Yield: 3280.4 kg/ha | N: 560.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1244/2000 | Score: -229 | Avg: -920 | Steps: 166 | Yield: 1345.8 kg/ha | N: 400.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1245/2000 | Score: 192 | Avg: -920 | Steps: 161 | Yield: 3306.0 kg/ha | N: 360.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1246/2000 | Score: 97 | Avg: -920 | Steps: 161 | Yield: 3577.8 kg/ha | N: 400.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1247/2000 | Score: -128 | Avg: -920 | Steps: 158 | Yield: 1291.5 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1248/2000 | Score: 216 | Avg: -920 | Steps: 158 | Yield: 4084.7 kg/ha | N: 360.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1249/2000 | Score: -141 | Avg: -920 | Steps: 159 | Yield: 2676.8 kg/ha | N: 480.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1250/2000 | Score: 112 | Avg: -920 | Steps: 154 | Yield: 9504.9 kg/ha | N: 640.0 kg/ha | Water: 780.0 mm | LR: 1.00e-02
Episode 1251/2000 | Score: 192 | Avg: -920 | Steps: 163 | Yield: 3244.0 kg/ha | N: 280.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1252/2000 | Score: 430 | Avg: -920 | Steps: 157 | Yield: 5785.8 kg/ha | N: 320.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1253/2000 | Score: 584 | Avg: -920 | Steps: 159 | Yield: 5974.6 kg/ha | N: 280.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1254/2000 | Score: 554 | Avg: -920 | Steps: 160 | Yield: 10042.4 kg/ha | N: 680.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 1255/2000 | Score: 334 | Avg: -920 | Steps: 163 | Yield: 4143.4 kg/ha | N: 280.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1256/2000 | Score: 828 | Avg: -920 | Steps: 163 | Yield: 11949.9 kg/ha | N: 640.0 kg/ha | Water: 504.0 mm | LR: 1.00e-02
Episode 1257/2000 | Score: -145 | Avg: -920 | Steps: 157 | Yield: 1334.5 kg/ha | N: 400.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1258/2000 | Score: 93 | Avg: -920 | Steps: 153 | Yield: 5494.4 kg/ha | N: 680.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 1259/2000 | Score: 680 | Avg: -920 | Steps: 154 | Yield: 6811.5 kg/ha | N: 360.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1260/2000 | Score: 564 | Avg: -920 | Steps: 160 | Yield: 7349.1 kg/ha | N: 480.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1261/2000 | Score: 925 | Avg: -920 | Steps: 169 | Yield: 10592.1 kg/ha | N: 480.0 kg/ha | Water: 330.0 mm | LR: 1.00e-02
Episode 1262/2000 | Score: 844 | Avg: -920 | Steps: 160 | Yield: 7569.8 kg/ha | N: 320.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1263/2000 | Score: 61 | Avg: -920 | Steps: 167 | Yield: 2171.2 kg/ha | N: 240.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1264/2000 | Score: 397 | Avg: -920 | Steps: 162 | Yield: 6074.8 kg/ha | N: 320.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1265/2000 | Score: 158 | Avg: -920 | Steps: 161 | Yield: 3309.4 kg/ha | N: 320.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1266/2000 | Score: 822 | Avg: -920 | Steps: 158 | Yield: 8611.6 kg/ha | N: 440.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1267/2000 | Score: 443 | Avg: -920 | Steps: 164 | Yield: 12345.5 kg/ha | N: 1040.0 kg/ha | Water: 624.0 mm | LR: 1.00e-02
Episode 1268/2000 | Score: 603 | Avg: -920 | Steps: 157 | Yield: 6274.5 kg/ha | N: 400.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1269/2000 | Score: 138 | Avg: -920 | Steps: 161 | Yield: 2940.2 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1270/2000 | Score: 222 | Avg: -920 | Steps: 158 | Yield: 4334.8 kg/ha | N: 360.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1271/2000 | Score: -87 | Avg: -920 | Steps: 164 | Yield: 1343.5 kg/ha | N: 320.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1272/2000 | Score: 238 | Avg: -920 | Steps: 160 | Yield: 4478.4 kg/ha | N: 360.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1273/2000 | Score: 357 | Avg: -920 | Steps: 164 | Yield: 4378.1 kg/ha | N: 240.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1274/2000 | Score: -83 | Avg: -920 | Steps: 163 | Yield: 1676.4 kg/ha | N: 240.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1275/2000 | Score: -99 | Avg: -920 | Steps: 157 | Yield: 1898.4 kg/ha | N: 280.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1276/2000 | Score: 812 | Avg: -920 | Steps: 167 | Yield: 7750.6 kg/ha | N: 280.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1277/2000 | Score: 353 | Avg: -920 | Steps: 166 | Yield: 6281.6 kg/ha | N: 400.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1278/2000 | Score: 46 | Avg: -920 | Steps: 162 | Yield: 2403.2 kg/ha | N: 280.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1279/2000 | Score: 235 | Avg: -920 | Steps: 162 | Yield: 3672.8 kg/ha | N: 320.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1280/2000 | Score: 820 | Avg: -920 | Steps: 164 | Yield: 8775.1 kg/ha | N: 400.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1281/2000 | Score: 343 | Avg: -920 | Steps: 158 | Yield: 5665.5 kg/ha | N: 440.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1282/2000 | Score: -162 | Avg: -920 | Steps: 163 | Yield: 2134.3 kg/ha | N: 440.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1283/2000 | Score: 908 | Avg: -920 | Steps: 167 | Yield: 10794.3 kg/ha | N: 400.0 kg/ha | Water: 420.0 mm | LR: 1.00e-02
Episode 1284/2000 | Score: 196 | Avg: -920 | Steps: 162 | Yield: 4024.5 kg/ha | N: 440.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1285/2000 | Score: 13 | Avg: -920 | Steps: 161 | Yield: 2747.7 kg/ha | N: 400.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1286/2000 | Score: 470 | Avg: -920 | Steps: 158 | Yield: 5473.4 kg/ha | N: 400.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1287/2000 | Score: 671 | Avg: -920 | Steps: 166 | Yield: 7492.1 kg/ha | N: 440.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1288/2000 | Score: 94 | Avg: -920 | Steps: 163 | Yield: 3511.9 kg/ha | N: 400.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1289/2000 | Score: -62 | Avg: -920 | Steps: 158 | Yield: 2510.0 kg/ha | N: 480.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1290/2000 | Score: -80 | Avg: -920 | Steps: 165 | Yield: 2446.9 kg/ha | N: 440.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1291/2000 | Score: 154 | Avg: -920 | Steps: 155 | Yield: 3561.2 kg/ha | N: 400.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1292/2000 | Score: 783 | Avg: -920 | Steps: 161 | Yield: 8392.1 kg/ha | N: 320.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1293/2000 | Score: 750 | Avg: -920 | Steps: 160 | Yield: 6781.3 kg/ha | N: 240.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1294/2000 | Score: 334 | Avg: -920 | Steps: 159 | Yield: 3803.9 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1295/2000 | Score: 332 | Avg: -920 | Steps: 162 | Yield: 3845.3 kg/ha | N: 240.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1296/2000 | Score: 148 | Avg: -920 | Steps: 163 | Yield: 2377.1 kg/ha | N: 280.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1297/2000 | Score: 320 | Avg: -920 | Steps: 157 | Yield: 3686.0 kg/ha | N: 240.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1298/2000 | Score: -117 | Avg: -920 | Steps: 162 | Yield: 1080.2 kg/ha | N: 280.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1299/2000 | Score: 479 | Avg: -920 | Steps: 158 | Yield: 6776.0 kg/ha | N: 640.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1300/2000 | Score: -75 | Avg: -920 | Steps: 162 | Yield: 1391.9 kg/ha | N: 240.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1301/2000 | Score: -173 | Avg: -920 | Steps: 159 | Yield: 772.2 kg/ha | N: 240.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1302/2000 | Score: 224 | Avg: -920 | Steps: 163 | Yield: 3145.8 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1303/2000 | Score: 139 | Avg: -920 | Steps: 161 | Yield: 2691.5 kg/ha | N: 320.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1304/2000 | Score: 873 | Avg: -920 | Steps: 161 | Yield: 7427.3 kg/ha | N: 280.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1305/2000 | Score: 364 | Avg: -920 | Steps: 157 | Yield: 4027.5 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1306/2000 | Score: -160 | Avg: -920 | Steps: 164 | Yield: 1360.3 kg/ha | N: 400.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1307/2000 | Score: 573 | Avg: -920 | Steps: 154 | Yield: 8369.9 kg/ha | N: 640.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1308/2000 | Score: 58 | Avg: -920 | Steps: 161 | Yield: 2343.2 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1309/2000 | Score: -25 | Avg: -920 | Steps: 161 | Yield: 1585.8 kg/ha | N: 240.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1310/2000 | Score: 277 | Avg: -920 | Steps: 154 | Yield: 3442.6 kg/ha | N: 280.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1311/2000 | Score: 128 | Avg: -920 | Steps: 161 | Yield: 4513.6 kg/ha | N: 600.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1312/2000 | Score: 906 | Avg: -920 | Steps: 164 | Yield: 7459.3 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1313/2000 | Score: 516 | Avg: -920 | Steps: 159 | Yield: 5409.0 kg/ha | N: 320.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1314/2000 | Score: 779 | Avg: -920 | Steps: 152 | Yield: 7037.5 kg/ha | N: 280.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1315/2000 | Score: 1259 | Avg: -920 | Steps: 161 | Yield: 10538.7 kg/ha | N: 280.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1316/2000 | Score: 101 | Avg: -920 | Steps: 161 | Yield: 3711.5 kg/ha | N: 280.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1317/2000 | Score: 885 | Avg: -920 | Steps: 155 | Yield: 8887.1 kg/ha | N: 440.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1318/2000 | Score: -14 | Avg: -920 | Steps: 159 | Yield: 2371.5 kg/ha | N: 400.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1319/2000 | Score: -215 | Avg: -920 | Steps: 167 | Yield: 1522.5 kg/ha | N: 360.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1320/2000 | Score: -63 | Avg: -920 | Steps: 160 | Yield: 1461.1 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1321/2000 | Score: 534 | Avg: -920 | Steps: 161 | Yield: 5323.4 kg/ha | N: 280.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1322/2000 | Score: 120 | Avg: -920 | Steps: 154 | Yield: 2892.6 kg/ha | N: 360.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1323/2000 | Score: 197 | Avg: -920 | Steps: 161 | Yield: 3062.6 kg/ha | N: 280.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1324/2000 | Score: 688 | Avg: -920 | Steps: 160 | Yield: 6421.0 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1325/2000 | Score: -112 | Avg: -920 | Steps: 160 | Yield: 2944.7 kg/ha | N: 480.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1326/2000 | Score: -133 | Avg: -920 | Steps: 165 | Yield: 1409.7 kg/ha | N: 400.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1327/2000 | Score: 611 | Avg: -920 | Steps: 160 | Yield: 6105.1 kg/ha | N: 280.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1328/2000 | Score: -177 | Avg: -920 | Steps: 162 | Yield: 4732.1 kg/ha | N: 920.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1329/2000 | Score: 713 | Avg: -920 | Steps: 156 | Yield: 6779.7 kg/ha | N: 320.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1330/2000 | Score: 545 | Avg: -920 | Steps: 162 | Yield: 5423.5 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1331/2000 | Score: 486 | Avg: -920 | Steps: 165 | Yield: 6021.6 kg/ha | N: 480.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1332/2000 | Score: 267 | Avg: -920 | Steps: 160 | Yield: 4149.3 kg/ha | N: 400.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1333/2000 | Score: 779 | Avg: -920 | Steps: 159 | Yield: 7763.5 kg/ha | N: 400.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1334/2000 | Score: -83 | Avg: -920 | Steps: 164 | Yield: 1490.0 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1335/2000 | Score: 1051 | Avg: -920 | Steps: 166 | Yield: 9663.5 kg/ha | N: 360.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1336/2000 | Score: 448 | Avg: -920 | Steps: 156 | Yield: 5046.3 kg/ha | N: 400.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1337/2000 | Score: 925 | Avg: -920 | Steps: 167 | Yield: 7806.8 kg/ha | N: 240.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1338/2000 | Score: 667 | Avg: -920 | Steps: 165 | Yield: 6595.6 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1339/2000 | Score: -18 | Avg: -920 | Steps: 162 | Yield: 1472.7 kg/ha | N: 200.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1340/2000 | Score: 567 | Avg: -920 | Steps: 154 | Yield: 8551.9 kg/ha | N: 600.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1341/2000 | Score: 607 | Avg: -920 | Steps: 161 | Yield: 5858.7 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1342/2000 | Score: 889 | Avg: -920 | Steps: 159 | Yield: 10357.0 kg/ha | N: 320.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 1343/2000 | Score: -68 | Avg: -920 | Steps: 156 | Yield: 1781.2 kg/ha | N: 400.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1344/2000 | Score: 966 | Avg: -920 | Steps: 161 | Yield: 9596.9 kg/ha | N: 480.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1345/2000 | Score: 386 | Avg: -920 | Steps: 163 | Yield: 5606.1 kg/ha | N: 440.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1346/2000 | Score: 924 | Avg: -920 | Steps: 165 | Yield: 9782.4 kg/ha | N: 520.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1347/2000 | Score: 290 | Avg: -920 | Steps: 159 | Yield: 3841.9 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1348/2000 | Score: 300 | Avg: -920 | Steps: 160 | Yield: 3957.3 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1349/2000 | Score: 145 | Avg: -920 | Steps: 160 | Yield: 2779.0 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1350/2000 | Score: 355 | Avg: -920 | Steps: 164 | Yield: 4350.3 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1351/2000 | Score: 1244 | Avg: -920 | Steps: 167 | Yield: 10067.2 kg/ha | N: 280.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1352/2000 | Score: 504 | Avg: -920 | Steps: 158 | Yield: 6807.8 kg/ha | N: 440.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1353/2000 | Score: 166 | Avg: -920 | Steps: 157 | Yield: 2774.7 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1354/2000 | Score: 846 | Avg: -920 | Steps: 155 | Yield: 7382.1 kg/ha | N: 280.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1355/2000 | Score: -266 | Avg: -920 | Steps: 150 | Yield: 810.6 kg/ha | N: 440.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1356/2000 | Score: 564 | Avg: -920 | Steps: 159 | Yield: 5635.1 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1357/2000 | Score: 651 | Avg: -920 | Steps: 159 | Yield: 6461.7 kg/ha | N: 360.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1358/2000 | Score: 125 | Avg: -920 | Steps: 158 | Yield: 3607.5 kg/ha | N: 480.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1359/2000 | Score: 586 | Avg: -920 | Steps: 161 | Yield: 5777.8 kg/ha | N: 280.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1360/2000 | Score: 809 | Avg: -920 | Steps: 160 | Yield: 7358.5 kg/ha | N: 280.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1361/2000 | Score: 717 | Avg: -920 | Steps: 159 | Yield: 8549.2 kg/ha | N: 360.0 kg/ha | Water: 318.0 mm | LR: 1.00e-02
Episode 1362/2000 | Score: -126 | Avg: -920 | Steps: 158 | Yield: 1727.5 kg/ha | N: 480.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1363/2000 | Score: -23 | Avg: -920 | Steps: 148 | Yield: 2905.5 kg/ha | N: 560.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1364/2000 | Score: 3 | Avg: -920 | Steps: 160 | Yield: 2867.9 kg/ha | N: 520.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1365/2000 | Score: 116 | Avg: -920 | Steps: 158 | Yield: 2498.6 kg/ha | N: 320.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1366/2000 | Score: 170 | Avg: -920 | Steps: 158 | Yield: 2936.4 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1367/2000 | Score: 82 | Avg: -920 | Steps: 155 | Yield: 2298.0 kg/ha | N: 280.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1368/2000 | Score: 292 | Avg: -920 | Steps: 158 | Yield: 4555.8 kg/ha | N: 400.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1369/2000 | Score: 451 | Avg: -920 | Steps: 163 | Yield: 4870.5 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1370/2000 | Score: 408 | Avg: -920 | Steps: 164 | Yield: 4194.2 kg/ha | N: 280.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1371/2000 | Score: 648 | Avg: -920 | Steps: 158 | Yield: 6158.7 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1372/2000 | Score: 227 | Avg: -920 | Steps: 157 | Yield: 3286.7 kg/ha | N: 320.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1373/2000 | Score: -40 | Avg: -920 | Steps: 157 | Yield: 2272.7 kg/ha | N: 480.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1374/2000 | Score: -141 | Avg: -920 | Steps: 156 | Yield: 1769.4 kg/ha | N: 440.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1375/2000 | Score: 1010 | Avg: -920 | Steps: 157 | Yield: 8371.1 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1376/2000 | Score: 821 | Avg: -920 | Steps: 156 | Yield: 8398.3 kg/ha | N: 440.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1377/2000 | Score: 327 | Avg: -920 | Steps: 155 | Yield: 3720.6 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1378/2000 | Score: 473 | Avg: -920 | Steps: 166 | Yield: 4852.6 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1379/2000 | Score: 511 | Avg: -920 | Steps: 164 | Yield: 6383.6 kg/ha | N: 480.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1380/2000 | Score: 364 | Avg: -920 | Steps: 161 | Yield: 5246.5 kg/ha | N: 480.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1381/2000 | Score: 470 | Avg: -920 | Steps: 164 | Yield: 6125.3 kg/ha | N: 480.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1382/2000 | Score: 304 | Avg: -920 | Steps: 159 | Yield: 3660.0 kg/ha | N: 280.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1383/2000 | Score: 38 | Avg: -920 | Steps: 155 | Yield: 2213.8 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1384/2000 | Score: 742 | Avg: -920 | Steps: 161 | Yield: 8865.1 kg/ha | N: 600.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1385/2000 | Score: 595 | Avg: -920 | Steps: 166 | Yield: 5967.5 kg/ha | N: 240.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1386/2000 | Score: 279 | Avg: -920 | Steps: 157 | Yield: 3698.9 kg/ha | N: 320.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1387/2000 | Score: 144 | Avg: -920 | Steps: 167 | Yield: 2730.4 kg/ha | N: 280.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1388/2000 | Score: 466 | Avg: -920 | Steps: 168 | Yield: 6319.4 kg/ha | N: 440.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1389/2000 | Score: 991 | Avg: -920 | Steps: 165 | Yield: 9141.4 kg/ha | N: 440.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1390/2000 | Score: 283 | Avg: -920 | Steps: 159 | Yield: 4377.4 kg/ha | N: 400.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1391/2000 | Score: 229 | Avg: -920 | Steps: 160 | Yield: 3301.4 kg/ha | N: 320.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1392/2000 | Score: 1008 | Avg: -920 | Steps: 156 | Yield: 8637.1 kg/ha | N: 360.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1393/2000 | Score: 264 | Avg: -920 | Steps: 160 | Yield: 5972.5 kg/ha | N: 760.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1394/2000 | Score: 14 | Avg: -920 | Steps: 159 | Yield: 1950.8 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1395/2000 | Score: 425 | Avg: -920 | Steps: 159 | Yield: 4541.3 kg/ha | N: 320.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1396/2000 | Score: 706 | Avg: -920 | Steps: 163 | Yield: 6811.0 kg/ha | N: 360.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1397/2000 | Score: 82 | Avg: -920 | Steps: 156 | Yield: 2243.9 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1398/2000 | Score: 610 | Avg: -920 | Steps: 165 | Yield: 7715.9 kg/ha | N: 520.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1399/2000 | Score: 318 | Avg: -920 | Steps: 164 | Yield: 4446.1 kg/ha | N: 320.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1400/2000 | Score: 674 | Avg: -920 | Steps: 162 | Yield: 9023.1 kg/ha | N: 600.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1401/2000 | Score: 625 | Avg: -920 | Steps: 161 | Yield: 6958.4 kg/ha | N: 400.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1402/2000 | Score: -51 | Avg: -920 | Steps: 160 | Yield: 2781.4 kg/ha | N: 520.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1403/2000 | Score: 574 | Avg: -920 | Steps: 160 | Yield: 9350.1 kg/ha | N: 600.0 kg/ha | Water: 390.0 mm | LR: 1.00e-02
Episode 1404/2000 | Score: 490 | Avg: -920 | Steps: 157 | Yield: 4665.3 kg/ha | N: 280.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1405/2000 | Score: 238 | Avg: -920 | Steps: 156 | Yield: 3597.4 kg/ha | N: 360.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1406/2000 | Score: 294 | Avg: -920 | Steps: 163 | Yield: 5066.4 kg/ha | N: 440.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1407/2000 | Score: 456 | Avg: -920 | Steps: 161 | Yield: 6439.6 kg/ha | N: 560.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1408/2000 | Score: -89 | Avg: -920 | Steps: 157 | Yield: 1519.7 kg/ha | N: 400.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1409/2000 | Score: 875 | Avg: -920 | Steps: 161 | Yield: 8763.9 kg/ha | N: 320.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1410/2000 | Score: 927 | Avg: -920 | Steps: 164 | Yield: 10125.6 kg/ha | N: 400.0 kg/ha | Water: 318.0 mm | LR: 1.00e-02
Episode 1411/2000 | Score: 657 | Avg: -920 | Steps: 155 | Yield: 5884.5 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1412/2000 | Score: -147 | Avg: -920 | Steps: 155 | Yield: 1561.8 kg/ha | N: 440.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1413/2000 | Score: 279 | Avg: -920 | Steps: 168 | Yield: 3948.5 kg/ha | N: 320.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1414/2000 | Score: 52 | Avg: -920 | Steps: 156 | Yield: 2777.3 kg/ha | N: 440.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1415/2000 | Score: 144 | Avg: -920 | Steps: 153 | Yield: 2645.4 kg/ha | N: 280.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1416/2000 | Score: 188 | Avg: -920 | Steps: 160 | Yield: 4016.7 kg/ha | N: 440.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1417/2000 | Score: 727 | Avg: -920 | Steps: 161 | Yield: 6705.4 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1418/2000 | Score: 761 | Avg: -920 | Steps: 164 | Yield: 10123.1 kg/ha | N: 560.0 kg/ha | Water: 354.0 mm | LR: 1.00e-02
Episode 1419/2000 | Score: 508 | Avg: -920 | Steps: 159 | Yield: 5233.1 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1420/2000 | Score: 73 | Avg: -920 | Steps: 161 | Yield: 2848.6 kg/ha | N: 360.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1421/2000 | Score: 408 | Avg: -920 | Steps: 164 | Yield: 8294.4 kg/ha | N: 800.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1422/2000 | Score: -7 | Avg: -920 | Steps: 162 | Yield: 1679.8 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1423/2000 | Score: 692 | Avg: -920 | Steps: 157 | Yield: 7492.1 kg/ha | N: 280.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1424/2000 | Score: 544 | Avg: -920 | Steps: 161 | Yield: 5042.3 kg/ha | N: 320.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1425/2000 | Score: 541 | Avg: -920 | Steps: 161 | Yield: 5904.1 kg/ha | N: 320.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1426/2000 | Score: 290 | Avg: -920 | Steps: 157 | Yield: 5721.7 kg/ha | N: 560.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1427/2000 | Score: 304 | Avg: -920 | Steps: 158 | Yield: 4315.7 kg/ha | N: 320.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1428/2000 | Score: 939 | Avg: -920 | Steps: 159 | Yield: 9015.5 kg/ha | N: 280.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1429/2000 | Score: 826 | Avg: -920 | Steps: 165 | Yield: 7579.9 kg/ha | N: 320.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1430/2000 | Score: 384 | Avg: -920 | Steps: 166 | Yield: 4887.8 kg/ha | N: 400.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1431/2000 | Score: 392 | Avg: -920 | Steps: 157 | Yield: 4617.9 kg/ha | N: 360.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1432/2000 | Score: 660 | Avg: -920 | Steps: 166 | Yield: 5869.5 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1433/2000 | Score: 83 | Avg: -920 | Steps: 162 | Yield: 4013.5 kg/ha | N: 480.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1434/2000 | Score: 272 | Avg: -920 | Steps: 157 | Yield: 5103.8 kg/ha | N: 560.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1435/2000 | Score: 282 | Avg: -920 | Steps: 161 | Yield: 4223.1 kg/ha | N: 320.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1436/2000 | Score: 503 | Avg: -920 | Steps: 162 | Yield: 6514.5 kg/ha | N: 440.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1437/2000 | Score: 487 | Avg: -920 | Steps: 157 | Yield: 5826.5 kg/ha | N: 440.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1438/2000 | Score: 4 | Avg: -920 | Steps: 155 | Yield: 2624.1 kg/ha | N: 520.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1439/2000 | Score: 237 | Avg: -920 | Steps: 164 | Yield: 3528.5 kg/ha | N: 280.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1440/2000 | Score: 466 | Avg: -920 | Steps: 160 | Yield: 6360.3 kg/ha | N: 440.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1441/2000 | Score: 906 | Avg: -920 | Steps: 157 | Yield: 7675.3 kg/ha | N: 280.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1442/2000 | Score: 572 | Avg: -920 | Steps: 157 | Yield: 5218.5 kg/ha | N: 320.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1443/2000 | Score: 781 | Avg: -920 | Steps: 160 | Yield: 8906.5 kg/ha | N: 400.0 kg/ha | Water: 270.0 mm | LR: 1.00e-02
Episode 1444/2000 | Score: 722 | Avg: -920 | Steps: 160 | Yield: 8513.8 kg/ha | N: 480.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1445/2000 | Score: 103 | Avg: -920 | Steps: 153 | Yield: 2638.7 kg/ha | N: 280.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1446/2000 | Score: -13 | Avg: -920 | Steps: 133 | Yield: 1641.2 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1447/2000 | Score: 562 | Avg: -920 | Steps: 159 | Yield: 5931.8 kg/ha | N: 400.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1448/2000 | Score: -153 | Avg: -920 | Steps: 154 | Yield: 1408.8 kg/ha | N: 400.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1449/2000 | Score: 275 | Avg: -920 | Steps: 155 | Yield: 3703.1 kg/ha | N: 200.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1450/2000 | Score: 198 | Avg: -920 | Steps: 161 | Yield: 3027.6 kg/ha | N: 280.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1451/2000 | Score: -216 | Avg: -920 | Steps: 164 | Yield: 2275.2 kg/ha | N: 520.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1452/2000 | Score: 249 | Avg: -920 | Steps: 158 | Yield: 3183.2 kg/ha | N: 280.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1453/2000 | Score: 610 | Avg: -920 | Steps: 161 | Yield: 6263.4 kg/ha | N: 280.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1454/2000 | Score: 488 | Avg: -920 | Steps: 156 | Yield: 5140.4 kg/ha | N: 360.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1455/2000 | Score: 668 | Avg: -920 | Steps: 166 | Yield: 7196.4 kg/ha | N: 360.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1456/2000 | Score: 180 | Avg: -920 | Steps: 159 | Yield: 2823.1 kg/ha | N: 320.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1457/2000 | Score: -605 | Avg: -920 | Steps: 164 | Yield: 677.6 kg/ha | N: 600.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 1458/2000 | Score: 259 | Avg: -920 | Steps: 154 | Yield: 3983.1 kg/ha | N: 360.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1459/2000 | Score: 297 | Avg: -920 | Steps: 157 | Yield: 5441.3 kg/ha | N: 520.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1460/2000 | Score: -85 | Avg: -920 | Steps: 163 | Yield: 1824.5 kg/ha | N: 280.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1461/2000 | Score: 407 | Avg: -920 | Steps: 158 | Yield: 5149.1 kg/ha | N: 440.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1462/2000 | Score: 131 | Avg: -920 | Steps: 159 | Yield: 3121.6 kg/ha | N: 400.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1463/2000 | Score: 1221 | Avg: -920 | Steps: 156 | Yield: 10495.0 kg/ha | N: 320.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1464/2000 | Score: 190 | Avg: -920 | Steps: 163 | Yield: 3101.8 kg/ha | N: 280.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1465/2000 | Score: 426 | Avg: -920 | Steps: 159 | Yield: 4704.1 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1466/2000 | Score: 219 | Avg: -920 | Steps: 163 | Yield: 3247.5 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1467/2000 | Score: 1002 | Avg: -920 | Steps: 165 | Yield: 8027.4 kg/ha | N: 320.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1468/2000 | Score: -56 | Avg: -920 | Steps: 157 | Yield: 1769.7 kg/ha | N: 400.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1469/2000 | Score: 628 | Avg: -920 | Steps: 157 | Yield: 5658.9 kg/ha | N: 320.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1470/2000 | Score: -103 | Avg: -920 | Steps: 151 | Yield: 2417.6 kg/ha | N: 480.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1471/2000 | Score: 1031 | Avg: -920 | Steps: 160 | Yield: 9136.5 kg/ha | N: 280.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1472/2000 | Score: 459 | Avg: -920 | Steps: 154 | Yield: 5487.0 kg/ha | N: 400.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1473/2000 | Score: 50 | Avg: -920 | Steps: 157 | Yield: 2326.8 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1474/2000 | Score: 855 | Avg: -920 | Steps: 159 | Yield: 7032.0 kg/ha | N: 240.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1475/2000 | Score: 254 | Avg: -920 | Steps: 160 | Yield: 3909.0 kg/ha | N: 360.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1476/2000 | Score: 127 | Avg: -920 | Steps: 157 | Yield: 3494.7 kg/ha | N: 480.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1477/2000 | Score: -259 | Avg: -920 | Steps: 163 | Yield: 1136.9 kg/ha | N: 480.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1478/2000 | Score: -159 | Avg: -920 | Steps: 163 | Yield: 1664.6 kg/ha | N: 400.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1479/2000 | Score: 316 | Avg: -920 | Steps: 159 | Yield: 4098.4 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1480/2000 | Score: 301 | Avg: -920 | Steps: 162 | Yield: 4732.6 kg/ha | N: 440.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1481/2000 | Score: 285 | Avg: -920 | Steps: 159 | Yield: 4274.8 kg/ha | N: 360.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1482/2000 | Score: 260 | Avg: -920 | Steps: 161 | Yield: 3695.6 kg/ha | N: 360.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1483/2000 | Score: 65 | Avg: -920 | Steps: 156 | Yield: 2100.9 kg/ha | N: 280.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1484/2000 | Score: 518 | Avg: -920 | Steps: 167 | Yield: 5220.0 kg/ha | N: 280.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1485/2000 | Score: 303 | Avg: -920 | Steps: 164 | Yield: 5469.5 kg/ha | N: 560.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1486/2000 | Score: 569 | Avg: -920 | Steps: 167 | Yield: 5983.0 kg/ha | N: 360.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1487/2000 | Score: 790 | Avg: -920 | Steps: 161 | Yield: 6766.9 kg/ha | N: 320.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1488/2000 | Score: 339 | Avg: -920 | Steps: 156 | Yield: 3787.9 kg/ha | N: 320.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1489/2000 | Score: -125 | Avg: -920 | Steps: 151 | Yield: 1377.7 kg/ha | N: 400.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1490/2000 | Score: 82 | Avg: -920 | Steps: 159 | Yield: 3703.9 kg/ha | N: 520.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1491/2000 | Score: 173 | Avg: -920 | Steps: 151 | Yield: 3121.5 kg/ha | N: 280.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1492/2000 | Score: -19 | Avg: -920 | Steps: 163 | Yield: 1887.9 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1493/2000 | Score: 85 | Avg: -920 | Steps: 155 | Yield: 2549.5 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1494/2000 | Score: 552 | Avg: -920 | Steps: 158 | Yield: 5822.0 kg/ha | N: 440.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1495/2000 | Score: 486 | Avg: -920 | Steps: 167 | Yield: 4798.9 kg/ha | N: 320.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1496/2000 | Score: 535 | Avg: -920 | Steps: 167 | Yield: 6253.1 kg/ha | N: 440.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1497/2000 | Score: 434 | Avg: -920 | Steps: 155 | Yield: 4441.8 kg/ha | N: 280.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1498/2000 | Score: 926 | Avg: -920 | Steps: 159 | Yield: 7722.7 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1499/2000 | Score: 360 | Avg: -920 | Steps: 155 | Yield: 4567.9 kg/ha | N: 400.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1500/2000 | Score: 804 | Avg: -920 | Steps: 157 | Yield: 6170.6 kg/ha | N: 200.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1501/2000 | Score: -90 | Avg: -920 | Steps: 163 | Yield: 1558.4 kg/ha | N: 400.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1502/2000 | Score: 871 | Avg: -920 | Steps: 162 | Yield: 7862.1 kg/ha | N: 320.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1503/2000 | Score: 478 | Avg: -920 | Steps: 158 | Yield: 4552.2 kg/ha | N: 280.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1504/2000 | Score: 120 | Avg: -920 | Steps: 160 | Yield: 2566.5 kg/ha | N: 320.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1505/2000 | Score: 349 | Avg: -920 | Steps: 157 | Yield: 3489.6 kg/ha | N: 240.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1506/2000 | Score: -22 | Avg: -920 | Steps: 155 | Yield: 1762.6 kg/ha | N: 280.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1507/2000 | Score: 709 | Avg: -920 | Steps: 162 | Yield: 7254.9 kg/ha | N: 320.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1508/2000 | Score: -54 | Avg: -920 | Steps: 154 | Yield: 1223.3 kg/ha | N: 280.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1509/2000 | Score: 638 | Avg: -920 | Steps: 161 | Yield: 6178.5 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1510/2000 | Score: -191 | Avg: -920 | Steps: 158 | Yield: 2267.5 kg/ha | N: 520.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1511/2000 | Score: 654 | Avg: -920 | Steps: 164 | Yield: 5907.0 kg/ha | N: 320.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1512/2000 | Score: 907 | Avg: -920 | Steps: 162 | Yield: 8301.2 kg/ha | N: 320.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1513/2000 | Score: 796 | Avg: -920 | Steps: 165 | Yield: 8209.4 kg/ha | N: 400.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1514/2000 | Score: 175 | Avg: -920 | Steps: 158 | Yield: 3167.9 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1515/2000 | Score: -473 | Avg: -920 | Steps: 167 | Yield: 1455.2 kg/ha | N: 480.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1516/2000 | Score: 181 | Avg: -920 | Steps: 162 | Yield: 3945.9 kg/ha | N: 360.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1517/2000 | Score: 951 | Avg: -920 | Steps: 161 | Yield: 8597.5 kg/ha | N: 240.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1518/2000 | Score: 14 | Avg: -920 | Steps: 158 | Yield: 1414.9 kg/ha | N: 240.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1519/2000 | Score: 399 | Avg: -920 | Steps: 162 | Yield: 5381.7 kg/ha | N: 480.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1520/2000 | Score: 331 | Avg: -920 | Steps: 164 | Yield: 3334.6 kg/ha | N: 240.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1521/2000 | Score: 759 | Avg: -920 | Steps: 165 | Yield: 6273.4 kg/ha | N: 160.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1522/2000 | Score: 752 | Avg: -920 | Steps: 163 | Yield: 7548.3 kg/ha | N: 240.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1523/2000 | Score: 373 | Avg: -920 | Steps: 163 | Yield: 4013.8 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1524/2000 | Score: 921 | Avg: -920 | Steps: 159 | Yield: 8364.5 kg/ha | N: 240.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1525/2000 | Score: 792 | Avg: -920 | Steps: 155 | Yield: 8129.7 kg/ha | N: 440.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1526/2000 | Score: 1071 | Avg: -920 | Steps: 160 | Yield: 8982.8 kg/ha | N: 240.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1527/2000 | Score: 837 | Avg: -920 | Steps: 166 | Yield: 10077.2 kg/ha | N: 680.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1528/2000 | Score: 160 | Avg: -920 | Steps: 156 | Yield: 4579.7 kg/ha | N: 480.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1529/2000 | Score: 422 | Avg: -920 | Steps: 159 | Yield: 5332.8 kg/ha | N: 440.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1530/2000 | Score: 19 | Avg: -920 | Steps: 158 | Yield: 1773.8 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1531/2000 | Score: -37 | Avg: -920 | Steps: 159 | Yield: 1250.3 kg/ha | N: 280.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1532/2000 | Score: 27 | Avg: -920 | Steps: 159 | Yield: 1653.3 kg/ha | N: 280.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1533/2000 | Score: 878 | Avg: -920 | Steps: 158 | Yield: 6358.5 kg/ha | N: 160.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1534/2000 | Score: 147 | Avg: -920 | Steps: 166 | Yield: 2705.1 kg/ha | N: 280.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1535/2000 | Score: 336 | Avg: -920 | Steps: 158 | Yield: 3737.3 kg/ha | N: 280.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1536/2000 | Score: 488 | Avg: -920 | Steps: 162 | Yield: 4371.7 kg/ha | N: 240.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1537/2000 | Score: 497 | Avg: -920 | Steps: 164 | Yield: 5068.5 kg/ha | N: 360.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1538/2000 | Score: 60 | Avg: -920 | Steps: 160 | Yield: 2352.7 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1539/2000 | Score: 1125 | Avg: -920 | Steps: 160 | Yield: 10927.0 kg/ha | N: 360.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1540/2000 | Score: -110 | Avg: -920 | Steps: 154 | Yield: 1823.5 kg/ha | N: 320.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1541/2000 | Score: 136 | Avg: -920 | Steps: 156 | Yield: 2626.4 kg/ha | N: 320.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1542/2000 | Score: 374 | Avg: -920 | Steps: 158 | Yield: 4450.2 kg/ha | N: 400.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1543/2000 | Score: 181 | Avg: -920 | Steps: 163 | Yield: 2355.2 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1544/2000 | Score: 70 | Avg: -920 | Steps: 157 | Yield: 2272.3 kg/ha | N: 240.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1545/2000 | Score: 676 | Avg: -920 | Steps: 166 | Yield: 6298.3 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1546/2000 | Score: 534 | Avg: -920 | Steps: 158 | Yield: 4791.4 kg/ha | N: 240.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1547/2000 | Score: -9 | Avg: -920 | Steps: 149 | Yield: 2493.7 kg/ha | N: 360.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1548/2000 | Score: 433 | Avg: -920 | Steps: 171 | Yield: 4791.2 kg/ha | N: 360.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1549/2000 | Score: 935 | Avg: -920 | Steps: 160 | Yield: 12307.3 kg/ha | N: 560.0 kg/ha | Water: 510.0 mm | LR: 1.00e-02
Episode 1550/2000 | Score: 433 | Avg: -920 | Steps: 161 | Yield: 5635.1 kg/ha | N: 320.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1551/2000 | Score: 496 | Avg: -920 | Steps: 163 | Yield: 5523.6 kg/ha | N: 360.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1552/2000 | Score: 540 | Avg: -920 | Steps: 160 | Yield: 4961.0 kg/ha | N: 200.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1553/2000 | Score: 210 | Avg: -920 | Steps: 162 | Yield: 1326.2 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1554/2000 | Score: 680 | Avg: -920 | Steps: 153 | Yield: 8495.6 kg/ha | N: 680.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1555/2000 | Score: 278 | Avg: -920 | Steps: 163 | Yield: 6220.3 kg/ha | N: 800.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1556/2000 | Score: 289 | Avg: -920 | Steps: 160 | Yield: 2903.9 kg/ha | N: 40.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1557/2000 | Score: -51 | Avg: -920 | Steps: 160 | Yield: 846.2 kg/ha | N: 200.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1558/2000 | Score: 1083 | Avg: -920 | Steps: 156 | Yield: 9223.9 kg/ha | N: 240.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1559/2000 | Score: 249 | Avg: -920 | Steps: 163 | Yield: 3419.2 kg/ha | N: 360.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1560/2000 | Score: -36 | Avg: -920 | Steps: 152 | Yield: 980.9 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1561/2000 | Score: 790 | Avg: -920 | Steps: 158 | Yield: 6825.9 kg/ha | N: 240.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1562/2000 | Score: 1017 | Avg: -920 | Steps: 162 | Yield: 8080.1 kg/ha | N: 120.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1563/2000 | Score: 212 | Avg: -920 | Steps: 157 | Yield: 1869.4 kg/ha | N: 80.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1564/2000 | Score: 469 | Avg: -920 | Steps: 159 | Yield: 4429.5 kg/ha | N: 200.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1565/2000 | Score: 1098 | Avg: -920 | Steps: 158 | Yield: 8931.8 kg/ha | N: 280.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1566/2000 | Score: 792 | Avg: -920 | Steps: 156 | Yield: 5413.0 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1567/2000 | Score: 259 | Avg: -920 | Steps: 158 | Yield: 3120.0 kg/ha | N: 280.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1568/2000 | Score: 372 | Avg: -920 | Steps: 160 | Yield: 5246.3 kg/ha | N: 320.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1569/2000 | Score: 267 | Avg: -920 | Steps: 160 | Yield: 2897.3 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1570/2000 | Score: 1004 | Avg: -920 | Steps: 163 | Yield: 8415.5 kg/ha | N: 320.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1571/2000 | Score: 799 | Avg: -920 | Steps: 160 | Yield: 6391.7 kg/ha | N: 200.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1572/2000 | Score: 988 | Avg: -920 | Steps: 165 | Yield: 7922.5 kg/ha | N: 200.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1573/2000 | Score: 462 | Avg: -920 | Steps: 163 | Yield: 6444.4 kg/ha | N: 520.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1574/2000 | Score: 638 | Avg: -920 | Steps: 161 | Yield: 6790.8 kg/ha | N: 200.0 kg/ha | Water: 252.0 mm | LR: 1.00e-02
Episode 1575/2000 | Score: 4 | Avg: -920 | Steps: 155 | Yield: 1431.1 kg/ha | N: 240.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1576/2000 | Score: 530 | Avg: -920 | Steps: 154 | Yield: 4890.1 kg/ha | N: 240.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1577/2000 | Score: 483 | Avg: -920 | Steps: 160 | Yield: 4096.3 kg/ha | N: 200.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1578/2000 | Score: 338 | Avg: -920 | Steps: 161 | Yield: 3608.7 kg/ha | N: 160.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1579/2000 | Score: 873 | Avg: -920 | Steps: 166 | Yield: 7176.9 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1580/2000 | Score: -10 | Avg: -920 | Steps: 161 | Yield: 1264.2 kg/ha | N: 240.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1581/2000 | Score: 819 | Avg: -920 | Steps: 163 | Yield: 6635.5 kg/ha | N: 240.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1582/2000 | Score: 648 | Avg: -920 | Steps: 158 | Yield: 5026.9 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1583/2000 | Score: -274 | Avg: -920 | Steps: 155 | Yield: 1873.0 kg/ha | N: 680.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1584/2000 | Score: 711 | Avg: -920 | Steps: 156 | Yield: 5935.5 kg/ha | N: 120.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1585/2000 | Score: 125 | Avg: -920 | Steps: 155 | Yield: 2684.4 kg/ha | N: 240.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1586/2000 | Score: 363 | Avg: -920 | Steps: 164 | Yield: 3588.8 kg/ha | N: 200.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1587/2000 | Score: 233 | Avg: -920 | Steps: 158 | Yield: 5428.5 kg/ha | N: 640.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1588/2000 | Score: 881 | Avg: -920 | Steps: 158 | Yield: 7357.6 kg/ha | N: 240.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1589/2000 | Score: 866 | Avg: -920 | Steps: 164 | Yield: 10361.5 kg/ha | N: 400.0 kg/ha | Water: 408.0 mm | LR: 1.00e-02
Episode 1590/2000 | Score: 754 | Avg: -920 | Steps: 160 | Yield: 6300.0 kg/ha | N: 280.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1591/2000 | Score: 399 | Avg: -920 | Steps: 158 | Yield: 4950.2 kg/ha | N: 360.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1592/2000 | Score: 249 | Avg: -920 | Steps: 158 | Yield: 4113.0 kg/ha | N: 440.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1593/2000 | Score: 543 | Avg: -920 | Steps: 154 | Yield: 4886.4 kg/ha | N: 240.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1594/2000 | Score: 817 | Avg: -920 | Steps: 159 | Yield: 8699.3 kg/ha | N: 280.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1595/2000 | Score: 200 | Avg: -920 | Steps: 159 | Yield: 2400.3 kg/ha | N: 160.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1596/2000 | Score: 101 | Avg: -920 | Steps: 155 | Yield: 1403.7 kg/ha | N: 120.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1597/2000 | Score: 165 | Avg: -920 | Steps: 161 | Yield: 3422.9 kg/ha | N: 400.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1598/2000 | Score: 332 | Avg: -920 | Steps: 155 | Yield: 3067.4 kg/ha | N: 160.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1599/2000 | Score: 157 | Avg: -920 | Steps: 164 | Yield: 1492.7 kg/ha | N: 0.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1600/2000 | Score: 546 | Avg: -920 | Steps: 161 | Yield: 4299.8 kg/ha | N: 160.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1601/2000 | Score: -134 | Avg: -920 | Steps: 159 | Yield: 961.5 kg/ha | N: 320.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1602/2000 | Score: 293 | Avg: -920 | Steps: 160 | Yield: 1855.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1603/2000 | Score: 1260 | Avg: -920 | Steps: 165 | Yield: 10686.5 kg/ha | N: 200.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1604/2000 | Score: 1081 | Avg: -920 | Steps: 166 | Yield: 8802.0 kg/ha | N: 200.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1605/2000 | Score: 385 | Avg: -920 | Steps: 157 | Yield: 5065.8 kg/ha | N: 400.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1606/2000 | Score: 524 | Avg: -920 | Steps: 157 | Yield: 4164.9 kg/ha | N: 120.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1607/2000 | Score: 201 | Avg: -920 | Steps: 157 | Yield: 1353.0 kg/ha | N: 0.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1608/2000 | Score: -54 | Avg: -920 | Steps: 159 | Yield: 1511.5 kg/ha | N: 320.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1609/2000 | Score: 371 | Avg: -920 | Steps: 167 | Yield: 4350.6 kg/ha | N: 200.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1610/2000 | Score: -7419 | Avg: -920 | Steps: 164 | Yield: 5422.8 kg/ha | N: 9600.0 kg/ha | Water: 600.0 mm | LR: 1.00e-02
Episode 1611/2000 | Score: 998 | Avg: -920 | Steps: 165 | Yield: 9326.1 kg/ha | N: 360.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1612/2000 | Score: 406 | Avg: -920 | Steps: 157 | Yield: 2892.1 kg/ha | N: 40.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1613/2000 | Score: 346 | Avg: -920 | Steps: 161 | Yield: 2513.8 kg/ha | N: 40.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1614/2000 | Score: 233 | Avg: -920 | Steps: 161 | Yield: 1474.4 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1615/2000 | Score: 673 | Avg: -920 | Steps: 164 | Yield: 5067.6 kg/ha | N: 120.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1616/2000 | Score: 89 | Avg: -920 | Steps: 163 | Yield: 1614.1 kg/ha | N: 160.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1617/2000 | Score: 653 | Avg: -920 | Steps: 159 | Yield: 7151.8 kg/ha | N: 320.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1618/2000 | Score: 728 | Avg: -920 | Steps: 166 | Yield: 5296.9 kg/ha | N: 80.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1619/2000 | Score: 719 | Avg: -920 | Steps: 160 | Yield: 6082.3 kg/ha | N: 240.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1620/2000 | Score: 744 | Avg: -920 | Steps: 156 | Yield: 6411.7 kg/ha | N: 240.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1621/2000 | Score: -61 | Avg: -920 | Steps: 163 | Yield: 656.2 kg/ha | N: 200.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1622/2000 | Score: 229 | Avg: -920 | Steps: 159 | Yield: 2255.1 kg/ha | N: 120.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1623/2000 | Score: 49 | Avg: -920 | Steps: 164 | Yield: 1193.2 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1624/2000 | Score: -77 | Avg: -920 | Steps: 160 | Yield: 760.8 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1625/2000 | Score: -94 | Avg: -920 | Steps: 154 | Yield: 1497.9 kg/ha | N: 160.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1626/2000 | Score: 116 | Avg: -920 | Steps: 162 | Yield: 2712.1 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1627/2000 | Score: 114 | Avg: -920 | Steps: 162 | Yield: 2458.6 kg/ha | N: 280.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1628/2000 | Score: 893 | Avg: -920 | Steps: 162 | Yield: 7510.5 kg/ha | N: 280.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1629/2000 | Score: 459 | Avg: -920 | Steps: 162 | Yield: 4271.1 kg/ha | N: 240.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1630/2000 | Score: 1154 | Avg: -920 | Steps: 158 | Yield: 9716.7 kg/ha | N: 240.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1631/2000 | Score: 1195 | Avg: -920 | Steps: 162 | Yield: 10155.4 kg/ha | N: 360.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1632/2000 | Score: 915 | Avg: -920 | Steps: 158 | Yield: 9241.4 kg/ha | N: 440.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1633/2000 | Score: 727 | Avg: -920 | Steps: 165 | Yield: 6251.9 kg/ha | N: 280.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1634/2000 | Score: -2599 | Avg: -920 | Steps: 158 | Yield: 8276.9 kg/ha | N: 4680.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1635/2000 | Score: 877 | Avg: -920 | Steps: 164 | Yield: 7593.3 kg/ha | N: 200.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1636/2000 | Score: 716 | Avg: -920 | Steps: 163 | Yield: 7733.2 kg/ha | N: 440.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1637/2000 | Score: 226 | Avg: -920 | Steps: 163 | Yield: 1430.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1638/2000 | Score: 447 | Avg: -920 | Steps: 151 | Yield: 4069.3 kg/ha | N: 240.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1639/2000 | Score: -78 | Avg: -920 | Steps: 162 | Yield: 874.9 kg/ha | N: 240.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1640/2000 | Score: 77 | Avg: -920 | Steps: 158 | Yield: 1580.6 kg/ha | N: 160.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1641/2000 | Score: 425 | Avg: -920 | Steps: 167 | Yield: 4749.1 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1642/2000 | Score: 203 | Avg: -920 | Steps: 163 | Yield: 2049.6 kg/ha | N: 120.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1643/2000 | Score: 923 | Avg: -920 | Steps: 161 | Yield: 9815.8 kg/ha | N: 360.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1644/2000 | Score: 589 | Avg: -920 | Steps: 158 | Yield: 4737.2 kg/ha | N: 160.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1645/2000 | Score: 188 | Avg: -920 | Steps: 160 | Yield: 1189.4 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1646/2000 | Score: 656 | Avg: -920 | Steps: 163 | Yield: 7765.5 kg/ha | N: 440.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1647/2000 | Score: 399 | Avg: -920 | Steps: 161 | Yield: 3783.1 kg/ha | N: 160.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1648/2000 | Score: 591 | Avg: -920 | Steps: 159 | Yield: 6256.4 kg/ha | N: 320.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1649/2000 | Score: 1135 | Avg: -920 | Steps: 154 | Yield: 9285.0 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1650/2000 | Score: 422 | Avg: -920 | Steps: 160 | Yield: 4645.2 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1651/2000 | Score: 52 | Avg: -920 | Steps: 160 | Yield: 1581.6 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1652/2000 | Score: 390 | Avg: -920 | Steps: 156 | Yield: 3675.8 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1653/2000 | Score: 580 | Avg: -920 | Steps: 156 | Yield: 4038.5 kg/ha | N: 40.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1654/2000 | Score: 612 | Avg: -920 | Steps: 155 | Yield: 5805.9 kg/ha | N: 320.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1655/2000 | Score: 142 | Avg: -920 | Steps: 160 | Yield: 1183.2 kg/ha | N: 40.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1656/2000 | Score: 575 | Avg: -920 | Steps: 166 | Yield: 10908.5 kg/ha | N: 1120.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1657/2000 | Score: 451 | Avg: -920 | Steps: 161 | Yield: 8137.2 kg/ha | N: 680.0 kg/ha | Water: 258.0 mm | LR: 1.00e-02
Episode 1658/2000 | Score: 88 | Avg: -920 | Steps: 166 | Yield: 2249.1 kg/ha | N: 280.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1659/2000 | Score: 515 | Avg: -920 | Steps: 159 | Yield: 5467.3 kg/ha | N: 400.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1660/2000 | Score: 623 | Avg: -920 | Steps: 163 | Yield: 7436.4 kg/ha | N: 440.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1661/2000 | Score: 684 | Avg: -920 | Steps: 155 | Yield: 6174.0 kg/ha | N: 160.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1662/2000 | Score: 1 | Avg: -920 | Steps: 168 | Yield: 835.8 kg/ha | N: 40.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1663/2000 | Score: 163 | Avg: -920 | Steps: 160 | Yield: 6435.1 kg/ha | N: 680.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 1664/2000 | Score: 76 | Avg: -920 | Steps: 161 | Yield: 1733.6 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1665/2000 | Score: -51 | Avg: -920 | Steps: 160 | Yield: 1915.6 kg/ha | N: 80.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 1666/2000 | Score: -63 | Avg: -920 | Steps: 161 | Yield: 8920.8 kg/ha | N: 1440.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1667/2000 | Score: 268 | Avg: -920 | Steps: 159 | Yield: 8648.3 kg/ha | N: 1040.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1668/2000 | Score: 568 | Avg: -920 | Steps: 162 | Yield: 8237.9 kg/ha | N: 720.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1669/2000 | Score: 506 | Avg: -920 | Steps: 160 | Yield: 5840.9 kg/ha | N: 360.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1670/2000 | Score: 219 | Avg: -920 | Steps: 159 | Yield: 1389.0 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1671/2000 | Score: -102 | Avg: -920 | Steps: 156 | Yield: 528.2 kg/ha | N: 160.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1672/2000 | Score: 1084 | Avg: -920 | Steps: 153 | Yield: 8487.2 kg/ha | N: 200.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1673/2000 | Score: -132 | Avg: -920 | Steps: 156 | Yield: 2400.5 kg/ha | N: 480.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1674/2000 | Score: 252 | Avg: -920 | Steps: 158 | Yield: 1595.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1675/2000 | Score: 124 | Avg: -920 | Steps: 159 | Yield: 1108.7 kg/ha | N: 40.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1676/2000 | Score: 694 | Avg: -920 | Steps: 159 | Yield: 4916.9 kg/ha | N: 80.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1677/2000 | Score: 653 | Avg: -920 | Steps: 159 | Yield: 5674.2 kg/ha | N: 200.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1678/2000 | Score: 130 | Avg: -920 | Steps: 158 | Yield: 1959.1 kg/ha | N: 160.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1679/2000 | Score: 327 | Avg: -920 | Steps: 153 | Yield: 3711.2 kg/ha | N: 320.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1680/2000 | Score: 848 | Avg: -920 | Steps: 160 | Yield: 7842.7 kg/ha | N: 320.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1681/2000 | Score: 204 | Avg: -920 | Steps: 157 | Yield: 8860.2 kg/ha | N: 1000.0 kg/ha | Water: 306.0 mm | LR: 1.00e-02
Episode 1682/2000 | Score: -4078 | Avg: -920 | Steps: 168 | Yield: 1510.8 kg/ha | N: 5080.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1683/2000 | Score: -83 | Avg: -920 | Steps: 158 | Yield: 2084.5 kg/ha | N: 480.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1684/2000 | Score: 377 | Avg: -920 | Steps: 160 | Yield: 4136.7 kg/ha | N: 200.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1685/2000 | Score: -121 | Avg: -920 | Steps: 158 | Yield: 2449.9 kg/ha | N: 560.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1686/2000 | Score: 745 | Avg: -920 | Steps: 162 | Yield: 5317.3 kg/ha | N: 120.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1687/2000 | Score: -154 | Avg: -920 | Steps: 155 | Yield: 1349.3 kg/ha | N: 440.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1688/2000 | Score: 108 | Avg: -920 | Steps: 154 | Yield: 2662.2 kg/ha | N: 320.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1689/2000 | Score: 578 | Avg: -920 | Steps: 160 | Yield: 8295.6 kg/ha | N: 760.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1690/2000 | Score: 799 | Avg: -920 | Steps: 156 | Yield: 7133.5 kg/ha | N: 240.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1691/2000 | Score: 899 | Avg: -920 | Steps: 166 | Yield: 8931.5 kg/ha | N: 440.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1692/2000 | Score: -9 | Avg: -920 | Steps: 160 | Yield: 1024.7 kg/ha | N: 200.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1693/2000 | Score: 916 | Avg: -920 | Steps: 161 | Yield: 9011.0 kg/ha | N: 400.0 kg/ha | Water: 174.0 mm | LR: 1.00e-02
Episode 1694/2000 | Score: 963 | Avg: -920 | Steps: 153 | Yield: 8022.6 kg/ha | N: 360.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1695/2000 | Score: 155 | Avg: -920 | Steps: 160 | Yield: 1025.7 kg/ha | N: 0.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1696/2000 | Score: 349 | Avg: -920 | Steps: 164 | Yield: 4253.6 kg/ha | N: 200.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1697/2000 | Score: 50 | Avg: -920 | Steps: 161 | Yield: 1157.4 kg/ha | N: 160.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1698/2000 | Score: 251 | Avg: -920 | Steps: 153 | Yield: 1590.3 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1699/2000 | Score: 1017 | Avg: -920 | Steps: 158 | Yield: 9342.5 kg/ha | N: 280.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 1700/2000 | Score: 622 | Avg: -920 | Steps: 158 | Yield: 7611.7 kg/ha | N: 560.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1701/2000 | Score: 644 | Avg: -920 | Steps: 161 | Yield: 7007.6 kg/ha | N: 160.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1702/2000 | Score: 26 | Avg: -920 | Steps: 163 | Yield: 899.8 kg/ha | N: 80.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1703/2000 | Score: -98 | Avg: -920 | Steps: 153 | Yield: 1111.0 kg/ha | N: 280.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1704/2000 | Score: 236 | Avg: -920 | Steps: 158 | Yield: 1619.7 kg/ha | N: 0.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1705/2000 | Score: 25 | Avg: -920 | Steps: 159 | Yield: 1641.3 kg/ha | N: 280.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1706/2000 | Score: 108 | Avg: -920 | Steps: 163 | Yield: 852.4 kg/ha | N: 0.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1707/2000 | Score: 733 | Avg: -920 | Steps: 161 | Yield: 6715.3 kg/ha | N: 240.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1708/2000 | Score: 238 | Avg: -920 | Steps: 161 | Yield: 2440.5 kg/ha | N: 120.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1709/2000 | Score: 531 | Avg: -920 | Steps: 168 | Yield: 5997.9 kg/ha | N: 160.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 1710/2000 | Score: 167 | Avg: -920 | Steps: 160 | Yield: 3435.4 kg/ha | N: 200.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1711/2000 | Score: -27 | Avg: -920 | Steps: 156 | Yield: 831.7 kg/ha | N: 200.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1712/2000 | Score: 116 | Avg: -920 | Steps: 159 | Yield: 3970.6 kg/ha | N: 480.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1713/2000 | Score: 282 | Avg: -920 | Steps: 160 | Yield: 3088.5 kg/ha | N: 160.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1714/2000 | Score: 228 | Avg: -920 | Steps: 162 | Yield: 3567.4 kg/ha | N: 400.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1715/2000 | Score: 857 | Avg: -920 | Steps: 161 | Yield: 10539.1 kg/ha | N: 680.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1716/2000 | Score: 96 | Avg: -920 | Steps: 161 | Yield: 1132.8 kg/ha | N: 80.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1717/2000 | Score: 434 | Avg: -920 | Steps: 157 | Yield: 3933.1 kg/ha | N: 120.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1718/2000 | Score: 638 | Avg: -920 | Steps: 162 | Yield: 7525.4 kg/ha | N: 480.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1719/2000 | Score: 558 | Avg: -920 | Steps: 159 | Yield: 5050.7 kg/ha | N: 120.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1720/2000 | Score: 644 | Avg: -920 | Steps: 167 | Yield: 6600.4 kg/ha | N: 120.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1721/2000 | Score: 760 | Avg: -920 | Steps: 159 | Yield: 8135.8 kg/ha | N: 440.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1722/2000 | Score: 231 | Avg: -920 | Steps: 157 | Yield: 4112.8 kg/ha | N: 480.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1723/2000 | Score: 997 | Avg: -920 | Steps: 160 | Yield: 9639.2 kg/ha | N: 240.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1724/2000 | Score: 875 | Avg: -920 | Steps: 162 | Yield: 8147.3 kg/ha | N: 280.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1725/2000 | Score: -37 | Avg: -920 | Steps: 161 | Yield: 1534.9 kg/ha | N: 320.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1726/2000 | Score: -107 | Avg: -920 | Steps: 162 | Yield: 1005.2 kg/ha | N: 320.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1727/2000 | Score: 1057 | Avg: -920 | Steps: 166 | Yield: 9963.1 kg/ha | N: 320.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1728/2000 | Score: -63 | Avg: -920 | Steps: 56 | Yield: 0.0 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1729/2000 | Score: -117 | Avg: -920 | Steps: 162 | Yield: 1499.5 kg/ha | N: 400.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1730/2000 | Score: 946 | Avg: -920 | Steps: 159 | Yield: 9224.8 kg/ha | N: 480.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1731/2000 | Score: 617 | Avg: -920 | Steps: 162 | Yield: 9622.7 kg/ha | N: 960.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1732/2000 | Score: 10 | Avg: -920 | Steps: 153 | Yield: 914.0 kg/ha | N: 120.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1733/2000 | Score: 364 | Avg: -920 | Steps: 162 | Yield: 4104.6 kg/ha | N: 160.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1734/2000 | Score: 1001 | Avg: -920 | Steps: 160 | Yield: 8845.2 kg/ha | N: 160.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1735/2000 | Score: 583 | Avg: -920 | Steps: 158 | Yield: 4617.5 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1736/2000 | Score: 691 | Avg: -920 | Steps: 165 | Yield: 5358.4 kg/ha | N: 80.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1737/2000 | Score: 738 | Avg: -920 | Steps: 159 | Yield: 6105.4 kg/ha | N: 120.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1738/2000 | Score: 1051 | Avg: -920 | Steps: 160 | Yield: 8729.5 kg/ha | N: 240.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1739/2000 | Score: 869 | Avg: -920 | Steps: 162 | Yield: 10461.4 kg/ha | N: 600.0 kg/ha | Water: 282.0 mm | LR: 1.00e-02
Episode 1740/2000 | Score: 46 | Avg: -920 | Steps: 165 | Yield: 3386.6 kg/ha | N: 560.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1741/2000 | Score: 857 | Avg: -920 | Steps: 163 | Yield: 7699.3 kg/ha | N: 280.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1742/2000 | Score: 938 | Avg: -920 | Steps: 164 | Yield: 7778.2 kg/ha | N: 160.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1743/2000 | Score: 370 | Avg: -920 | Steps: 163 | Yield: 10902.1 kg/ha | N: 1320.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1744/2000 | Score: 242 | Avg: -920 | Steps: 165 | Yield: 3934.8 kg/ha | N: 280.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1745/2000 | Score: 341 | Avg: -920 | Steps: 156 | Yield: 3629.1 kg/ha | N: 160.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1746/2000 | Score: 806 | Avg: -920 | Steps: 158 | Yield: 7626.3 kg/ha | N: 280.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1747/2000 | Score: 880 | Avg: -920 | Steps: 163 | Yield: 9156.8 kg/ha | N: 400.0 kg/ha | Water: 216.0 mm | LR: 1.00e-02
Episode 1748/2000 | Score: 24 | Avg: -920 | Steps: 160 | Yield: 1752.2 kg/ha | N: 320.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1749/2000 | Score: 497 | Avg: -920 | Steps: 163 | Yield: 6224.4 kg/ha | N: 440.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1750/2000 | Score: 205 | Avg: -920 | Steps: 163 | Yield: 1383.1 kg/ha | N: 0.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1751/2000 | Score: 901 | Avg: -920 | Steps: 158 | Yield: 7088.2 kg/ha | N: 160.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1752/2000 | Score: 89 | Avg: -920 | Steps: 159 | Yield: 2896.6 kg/ha | N: 400.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1753/2000 | Score: 75 | Avg: -920 | Steps: 162 | Yield: 2656.5 kg/ha | N: 320.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1754/2000 | Score: 592 | Avg: -920 | Steps: 155 | Yield: 4312.0 kg/ha | N: 80.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1755/2000 | Score: -87 | Avg: -920 | Steps: 163 | Yield: 2028.3 kg/ha | N: 440.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1756/2000 | Score: 664 | Avg: -920 | Steps: 158 | Yield: 8549.2 kg/ha | N: 560.0 kg/ha | Water: 210.0 mm | LR: 1.00e-02
Episode 1757/2000 | Score: 790 | Avg: -920 | Steps: 155 | Yield: 5690.0 kg/ha | N: 80.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1758/2000 | Score: 251 | Avg: -920 | Steps: 154 | Yield: 4221.7 kg/ha | N: 360.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1759/2000 | Score: 949 | Avg: -920 | Steps: 159 | Yield: 11645.3 kg/ha | N: 560.0 kg/ha | Water: 396.0 mm | LR: 1.00e-02
Episode 1760/2000 | Score: -584 | Avg: -920 | Steps: 154 | Yield: 2314.7 kg/ha | N: 1120.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1761/2000 | Score: 611 | Avg: -920 | Steps: 165 | Yield: 6980.8 kg/ha | N: 80.0 kg/ha | Water: 390.0 mm | LR: 1.00e-02
Episode 1762/2000 | Score: 126 | Avg: -920 | Steps: 161 | Yield: 795.6 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1763/2000 | Score: 96 | Avg: -920 | Steps: 167 | Yield: 1619.0 kg/ha | N: 160.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1764/2000 | Score: 45 | Avg: -920 | Steps: 163 | Yield: 1747.0 kg/ha | N: 200.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1765/2000 | Score: 238 | Avg: -920 | Steps: 158 | Yield: 5583.4 kg/ha | N: 640.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1766/2000 | Score: 1194 | Avg: -920 | Steps: 156 | Yield: 10778.3 kg/ha | N: 360.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1767/2000 | Score: 233 | Avg: -920 | Steps: 165 | Yield: 11320.2 kg/ha | N: 800.0 kg/ha | Water: 828.0 mm | LR: 1.00e-02
Episode 1768/2000 | Score: 756 | Avg: -920 | Steps: 162 | Yield: 5921.8 kg/ha | N: 160.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1769/2000 | Score: 496 | Avg: -920 | Steps: 160 | Yield: 4847.4 kg/ha | N: 200.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1770/2000 | Score: 946 | Avg: -920 | Steps: 162 | Yield: 11194.0 kg/ha | N: 640.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1771/2000 | Score: 299 | Avg: -920 | Steps: 153 | Yield: 2290.7 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1772/2000 | Score: 320 | Avg: -920 | Steps: 161 | Yield: 6946.5 kg/ha | N: 600.0 kg/ha | Water: 270.0 mm | LR: 1.00e-02
Episode 1773/2000 | Score: 422 | Avg: -920 | Steps: 163 | Yield: 4761.4 kg/ha | N: 360.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1774/2000 | Score: 209 | Avg: -920 | Steps: 161 | Yield: 5191.9 kg/ha | N: 640.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1775/2000 | Score: 25 | Avg: -920 | Steps: 162 | Yield: 2041.5 kg/ha | N: 360.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1776/2000 | Score: 822 | Avg: -920 | Steps: 166 | Yield: 8390.1 kg/ha | N: 320.0 kg/ha | Water: 228.0 mm | LR: 1.00e-02
Episode 1777/2000 | Score: 230 | Avg: -920 | Steps: 158 | Yield: 1857.5 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1778/2000 | Score: 932 | Avg: -920 | Steps: 156 | Yield: 8338.7 kg/ha | N: 120.0 kg/ha | Water: 264.0 mm | LR: 1.00e-02
Episode 1779/2000 | Score: 923 | Avg: -920 | Steps: 161 | Yield: 8478.9 kg/ha | N: 360.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1780/2000 | Score: 410 | Avg: -920 | Steps: 162 | Yield: 10434.8 kg/ha | N: 1000.0 kg/ha | Water: 408.0 mm | LR: 1.00e-02
Episode 1781/2000 | Score: 979 | Avg: -920 | Steps: 163 | Yield: 9402.4 kg/ha | N: 600.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1782/2000 | Score: 366 | Avg: -920 | Steps: 160 | Yield: 5921.6 kg/ha | N: 520.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1783/2000 | Score: 813 | Avg: -920 | Steps: 152 | Yield: 6895.3 kg/ha | N: 200.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1784/2000 | Score: -112 | Avg: -920 | Steps: 161 | Yield: 1340.5 kg/ha | N: 360.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1785/2000 | Score: -77 | Avg: -920 | Steps: 156 | Yield: 653.8 kg/ha | N: 120.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1786/2000 | Score: 36 | Avg: -920 | Steps: 153 | Yield: 1152.1 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1787/2000 | Score: 1200 | Avg: -920 | Steps: 163 | Yield: 10447.3 kg/ha | N: 320.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1788/2000 | Score: 722 | Avg: -920 | Steps: 156 | Yield: 5219.3 kg/ha | N: 80.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1789/2000 | Score: 41 | Avg: -920 | Steps: 155 | Yield: 3026.8 kg/ha | N: 520.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1790/2000 | Score: 384 | Avg: -920 | Steps: 158 | Yield: 2995.3 kg/ha | N: 80.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1791/2000 | Score: 977 | Avg: -920 | Steps: 163 | Yield: 8738.1 kg/ha | N: 360.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1792/2000 | Score: 375 | Avg: -920 | Steps: 158 | Yield: 4752.3 kg/ha | N: 400.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1793/2000 | Score: 49 | Avg: -920 | Steps: 158 | Yield: 1145.2 kg/ha | N: 0.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1794/2000 | Score: 700 | Avg: -920 | Steps: 163 | Yield: 6424.7 kg/ha | N: 240.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1795/2000 | Score: 914 | Avg: -920 | Steps: 164 | Yield: 7974.7 kg/ha | N: 120.0 kg/ha | Water: 222.0 mm | LR: 1.00e-02
Episode 1796/2000 | Score: 925 | Avg: -920 | Steps: 159 | Yield: 10409.5 kg/ha | N: 560.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1797/2000 | Score: -40 | Avg: -920 | Steps: 156 | Yield: 850.3 kg/ha | N: 120.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1798/2000 | Score: 691 | Avg: -920 | Steps: 162 | Yield: 6533.6 kg/ha | N: 240.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1799/2000 | Score: 337 | Avg: -920 | Steps: 155 | Yield: 4004.4 kg/ha | N: 240.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1800/2000 | Score: 201 | Avg: -920 | Steps: 158 | Yield: 1275.2 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1801/2000 | Score: 633 | Avg: -920 | Steps: 159 | Yield: 4826.3 kg/ha | N: 80.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1802/2000 | Score: 720 | Avg: -920 | Steps: 161 | Yield: 8405.4 kg/ha | N: 160.0 kg/ha | Water: 438.0 mm | LR: 1.00e-02
Episode 1803/2000 | Score: 898 | Avg: -920 | Steps: 154 | Yield: 8326.8 kg/ha | N: 320.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1804/2000 | Score: 386 | Avg: -920 | Steps: 163 | Yield: 3043.1 kg/ha | N: 120.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1805/2000 | Score: 7 | Avg: -920 | Steps: 151 | Yield: 1250.8 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1806/2000 | Score: 220 | Avg: -920 | Steps: 153 | Yield: 2603.1 kg/ha | N: 200.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1807/2000 | Score: 1116 | Avg: -920 | Steps: 164 | Yield: 8625.4 kg/ha | N: 120.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1808/2000 | Score: 168 | Avg: -920 | Steps: 158 | Yield: 1227.5 kg/ha | N: 0.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1809/2000 | Score: 878 | Avg: -920 | Steps: 165 | Yield: 7207.4 kg/ha | N: 80.0 kg/ha | Water: 180.0 mm | LR: 1.00e-02
Episode 1810/2000 | Score: 972 | Avg: -920 | Steps: 156 | Yield: 7432.0 kg/ha | N: 80.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1811/2000 | Score: 371 | Avg: -920 | Steps: 164 | Yield: 3294.1 kg/ha | N: 80.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1812/2000 | Score: 126 | Avg: -920 | Steps: 160 | Yield: 1395.4 kg/ha | N: 120.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1813/2000 | Score: 543 | Avg: -920 | Steps: 162 | Yield: 5595.1 kg/ha | N: 240.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1814/2000 | Score: 196 | Avg: -920 | Steps: 158 | Yield: 1932.6 kg/ha | N: 80.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1815/2000 | Score: 133 | Avg: -920 | Steps: 163 | Yield: 965.7 kg/ha | N: 0.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1816/2000 | Score: 721 | Avg: -920 | Steps: 162 | Yield: 9030.4 kg/ha | N: 200.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 1817/2000 | Score: 2 | Avg: -920 | Steps: 156 | Yield: 412.0 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1818/2000 | Score: 582 | Avg: -920 | Steps: 162 | Yield: 4615.9 kg/ha | N: 120.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1819/2000 | Score: 217 | Avg: -920 | Steps: 162 | Yield: 1696.3 kg/ha | N: 40.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1820/2000 | Score: 255 | Avg: -920 | Steps: 162 | Yield: 4054.0 kg/ha | N: 480.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1821/2000 | Score: 122 | Avg: -920 | Steps: 162 | Yield: 1021.6 kg/ha | N: 0.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1822/2000 | Score: 842 | Avg: -920 | Steps: 161 | Yield: 5938.1 kg/ha | N: 80.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1823/2000 | Score: 41 | Avg: -920 | Steps: 154 | Yield: 859.0 kg/ha | N: 120.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1824/2000 | Score: 157 | Avg: -920 | Steps: 161 | Yield: 993.8 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1825/2000 | Score: 963 | Avg: -920 | Steps: 161 | Yield: 7245.9 kg/ha | N: 80.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1826/2000 | Score: 148 | Avg: -920 | Steps: 167 | Yield: 977.0 kg/ha | N: 0.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1827/2000 | Score: 146 | Avg: -920 | Steps: 154 | Yield: 1809.3 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1828/2000 | Score: 898 | Avg: -920 | Steps: 158 | Yield: 8805.8 kg/ha | N: 240.0 kg/ha | Water: 276.0 mm | LR: 1.00e-02
Episode 1829/2000 | Score: 814 | Avg: -920 | Steps: 160 | Yield: 7281.7 kg/ha | N: 200.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1830/2000 | Score: 607 | Avg: -920 | Steps: 156 | Yield: 6328.0 kg/ha | N: 280.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1831/2000 | Score: 771 | Avg: -920 | Steps: 155 | Yield: 6989.8 kg/ha | N: 80.0 kg/ha | Water: 246.0 mm | LR: 1.00e-02
Episode 1832/2000 | Score: 237 | Avg: -920 | Steps: 160 | Yield: 1499.3 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1833/2000 | Score: 1098 | Avg: -920 | Steps: 157 | Yield: 8930.3 kg/ha | N: 120.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1834/2000 | Score: 870 | Avg: -920 | Steps: 158 | Yield: 6648.9 kg/ha | N: 120.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1835/2000 | Score: 10 | Avg: -920 | Steps: 158 | Yield: 945.6 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1836/2000 | Score: 123 | Avg: -920 | Steps: 160 | Yield: 1873.5 kg/ha | N: 160.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1837/2000 | Score: 368 | Avg: -920 | Steps: 160 | Yield: 2329.7 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1838/2000 | Score: 129 | Avg: -920 | Steps: 160 | Yield: 1739.3 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1839/2000 | Score: 766 | Avg: -920 | Steps: 159 | Yield: 7093.7 kg/ha | N: 240.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1840/2000 | Score: 208 | Avg: -920 | Steps: 159 | Yield: 1441.5 kg/ha | N: 0.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1841/2000 | Score: 297 | Avg: -920 | Steps: 160 | Yield: 3327.4 kg/ha | N: 240.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1842/2000 | Score: 421 | Avg: -920 | Steps: 165 | Yield: 4266.0 kg/ha | N: 120.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1843/2000 | Score: 149 | Avg: -920 | Steps: 164 | Yield: 2717.6 kg/ha | N: 280.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1844/2000 | Score: 1036 | Avg: -920 | Steps: 162 | Yield: 7950.5 kg/ha | N: 120.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1845/2000 | Score: 369 | Avg: -920 | Steps: 162 | Yield: 6557.7 kg/ha | N: 560.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1846/2000 | Score: 691 | Avg: -920 | Steps: 164 | Yield: 8861.6 kg/ha | N: 680.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1847/2000 | Score: 996 | Avg: -920 | Steps: 155 | Yield: 7374.3 kg/ha | N: 80.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1848/2000 | Score: 129 | Avg: -920 | Steps: 159 | Yield: 816.4 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1849/2000 | Score: 1020 | Avg: -920 | Steps: 161 | Yield: 8764.2 kg/ha | N: 320.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1850/2000 | Score: 203 | Avg: -920 | Steps: 163 | Yield: 1286.8 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1851/2000 | Score: 213 | Avg: -920 | Steps: 160 | Yield: 3241.8 kg/ha | N: 320.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1852/2000 | Score: 162 | Avg: -920 | Steps: 159 | Yield: 1024.7 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1853/2000 | Score: 31 | Avg: -920 | Steps: 159 | Yield: 1123.1 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1854/2000 | Score: 106 | Avg: -920 | Steps: 161 | Yield: 880.9 kg/ha | N: 0.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1855/2000 | Score: 207 | Avg: -920 | Steps: 158 | Yield: 1711.5 kg/ha | N: 80.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1856/2000 | Score: 605 | Avg: -920 | Steps: 166 | Yield: 4491.1 kg/ha | N: 40.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1857/2000 | Score: 188 | Avg: -920 | Steps: 160 | Yield: 1190.2 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1858/2000 | Score: -58 | Avg: -920 | Steps: 165 | Yield: 881.5 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1859/2000 | Score: 201 | Avg: -920 | Steps: 154 | Yield: 1273.3 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1860/2000 | Score: 205 | Avg: -920 | Steps: 162 | Yield: 1298.5 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1861/2000 | Score: 432 | Avg: -920 | Steps: 154 | Yield: 3550.1 kg/ha | N: 80.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1862/2000 | Score: 619 | Avg: -920 | Steps: 163 | Yield: 6446.4 kg/ha | N: 280.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1863/2000 | Score: 713 | Avg: -920 | Steps: 168 | Yield: 6237.4 kg/ha | N: 120.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1864/2000 | Score: 943 | Avg: -920 | Steps: 164 | Yield: 7119.7 kg/ha | N: 80.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1865/2000 | Score: 277 | Avg: -920 | Steps: 158 | Yield: 2163.2 kg/ha | N: 40.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1866/2000 | Score: 878 | Avg: -920 | Steps: 157 | Yield: 8638.6 kg/ha | N: 240.0 kg/ha | Water: 270.0 mm | LR: 1.00e-02
Episode 1867/2000 | Score: 793 | Avg: -920 | Steps: 155 | Yield: 6538.5 kg/ha | N: 120.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1868/2000 | Score: 188 | Avg: -920 | Steps: 160 | Yield: 3126.6 kg/ha | N: 120.0 kg/ha | Water: 192.0 mm | LR: 1.00e-02
Episode 1869/2000 | Score: 937 | Avg: -920 | Steps: 154 | Yield: 7400.4 kg/ha | N: 160.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1870/2000 | Score: 59 | Avg: -920 | Steps: 157 | Yield: 1301.8 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1871/2000 | Score: 939 | Avg: -920 | Steps: 163 | Yield: 8093.5 kg/ha | N: 280.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1872/2000 | Score: 966 | Avg: -920 | Steps: 163 | Yield: 11174.1 kg/ha | N: 360.0 kg/ha | Water: 456.0 mm | LR: 1.00e-02
Episode 1873/2000 | Score: 945 | Avg: -920 | Steps: 157 | Yield: 7158.2 kg/ha | N: 160.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1874/2000 | Score: 544 | Avg: -920 | Steps: 158 | Yield: 4323.5 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1875/2000 | Score: 223 | Avg: -920 | Steps: 153 | Yield: 1413.6 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1876/2000 | Score: 39 | Avg: -920 | Steps: 161 | Yield: 930.9 kg/ha | N: 120.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1877/2000 | Score: 151 | Avg: -920 | Steps: 158 | Yield: 996.4 kg/ha | N: 0.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1878/2000 | Score: 274 | Avg: -920 | Steps: 160 | Yield: 2419.8 kg/ha | N: 120.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1879/2000 | Score: 915 | Avg: -920 | Steps: 155 | Yield: 8104.0 kg/ha | N: 320.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1880/2000 | Score: 628 | Avg: -920 | Steps: 160 | Yield: 8474.0 kg/ha | N: 440.0 kg/ha | Water: 330.0 mm | LR: 1.00e-02
Episode 1881/2000 | Score: 173 | Avg: -920 | Steps: 165 | Yield: 1092.2 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1882/2000 | Score: 984 | Avg: -920 | Steps: 162 | Yield: 10934.8 kg/ha | N: 240.0 kg/ha | Water: 498.0 mm | LR: 1.00e-02
Episode 1883/2000 | Score: 122 | Avg: -920 | Steps: 165 | Yield: 772.6 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1884/2000 | Score: 304 | Avg: -920 | Steps: 156 | Yield: 1923.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1885/2000 | Score: 763 | Avg: -920 | Steps: 161 | Yield: 6516.2 kg/ha | N: 120.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1886/2000 | Score: 137 | Avg: -920 | Steps: 164 | Yield: 3496.4 kg/ha | N: 200.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1887/2000 | Score: 759 | Avg: -920 | Steps: 161 | Yield: 6366.1 kg/ha | N: 120.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1888/2000 | Score: 876 | Avg: -920 | Steps: 167 | Yield: 7230.2 kg/ha | N: 120.0 kg/ha | Water: 156.0 mm | LR: 1.00e-02
Episode 1889/2000 | Score: 1068 | Avg: -920 | Steps: 158 | Yield: 9456.5 kg/ha | N: 280.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1890/2000 | Score: 57 | Avg: -920 | Steps: 164 | Yield: 1446.9 kg/ha | N: 200.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1891/2000 | Score: -1 | Avg: -920 | Steps: 155 | Yield: 433.1 kg/ha | N: 80.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1892/2000 | Score: 411 | Avg: -920 | Steps: 160 | Yield: 3703.4 kg/ha | N: 120.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1893/2000 | Score: 496 | Avg: -920 | Steps: 158 | Yield: 4408.0 kg/ha | N: 120.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1894/2000 | Score: 199 | Avg: -920 | Steps: 157 | Yield: 2152.3 kg/ha | N: 120.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1895/2000 | Score: 450 | Avg: -920 | Steps: 157 | Yield: 3822.6 kg/ha | N: 120.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1896/2000 | Score: 718 | Avg: -920 | Steps: 153 | Yield: 9435.3 kg/ha | N: 560.0 kg/ha | Water: 288.0 mm | LR: 1.00e-02
Episode 1897/2000 | Score: 179 | Avg: -920 | Steps: 158 | Yield: 1299.5 kg/ha | N: 0.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1898/2000 | Score: 325 | Avg: -920 | Steps: 163 | Yield: 2663.2 kg/ha | N: 80.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1899/2000 | Score: 484 | Avg: -920 | Steps: 159 | Yield: 4090.4 kg/ha | N: 80.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1900/2000 | Score: 687 | Avg: -920 | Steps: 160 | Yield: 5530.7 kg/ha | N: 120.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1901/2000 | Score: 141 | Avg: -920 | Steps: 156 | Yield: 894.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1902/2000 | Score: 306 | Avg: -920 | Steps: 163 | Yield: 3529.0 kg/ha | N: 160.0 kg/ha | Water: 114.0 mm | LR: 1.00e-02
Episode 1903/2000 | Score: 582 | Avg: -920 | Steps: 154 | Yield: 5691.0 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1904/2000 | Score: -318 | Avg: -920 | Steps: 153 | Yield: 2721.7 kg/ha | N: 880.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1905/2000 | Score: 1042 | Avg: -920 | Steps: 156 | Yield: 8818.2 kg/ha | N: 160.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1906/2000 | Score: 826 | Avg: -920 | Steps: 154 | Yield: 6940.2 kg/ha | N: 200.0 kg/ha | Water: 102.0 mm | LR: 1.00e-02
Episode 1907/2000 | Score: 639 | Avg: -920 | Steps: 159 | Yield: 6020.9 kg/ha | N: 120.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1908/2000 | Score: 918 | Avg: -920 | Steps: 157 | Yield: 7731.2 kg/ha | N: 200.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1909/2000 | Score: 864 | Avg: -920 | Steps: 163 | Yield: 9204.4 kg/ha | N: 480.0 kg/ha | Water: 186.0 mm | LR: 1.00e-02
Episode 1910/2000 | Score: -804 | Avg: -920 | Steps: 162 | Yield: 1038.0 kg/ha | N: 1000.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1911/2000 | Score: 71 | Avg: -920 | Steps: 157 | Yield: 2284.8 kg/ha | N: 200.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1912/2000 | Score: 1047 | Avg: -920 | Steps: 163 | Yield: 8655.9 kg/ha | N: 280.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1913/2000 | Score: 446 | Avg: -920 | Steps: 164 | Yield: 6093.8 kg/ha | N: 520.0 kg/ha | Water: 96.0 mm | LR: 1.00e-02
Episode 1914/2000 | Score: 96 | Avg: -920 | Steps: 156 | Yield: 1214.5 kg/ha | N: 80.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1915/2000 | Score: 757 | Avg: -920 | Steps: 163 | Yield: 6617.1 kg/ha | N: 240.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1916/2000 | Score: 367 | Avg: -920 | Steps: 160 | Yield: 3833.5 kg/ha | N: 160.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1917/2000 | Score: 418 | Avg: -920 | Steps: 157 | Yield: 6375.0 kg/ha | N: 320.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1918/2000 | Score: 777 | Avg: -920 | Steps: 160 | Yield: 7427.6 kg/ha | N: 160.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1919/2000 | Score: -101 | Avg: -920 | Steps: 161 | Yield: 654.9 kg/ha | N: 200.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1920/2000 | Score: 802 | Avg: -920 | Steps: 162 | Yield: 6103.1 kg/ha | N: 80.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1921/2000 | Score: 883 | Avg: -920 | Steps: 165 | Yield: 9440.0 kg/ha | N: 160.0 kg/ha | Water: 426.0 mm | LR: 1.00e-02
Episode 1922/2000 | Score: 233 | Avg: -920 | Steps: 160 | Yield: 3578.1 kg/ha | N: 320.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1923/2000 | Score: 87 | Avg: -920 | Steps: 159 | Yield: 1049.6 kg/ha | N: 0.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1924/2000 | Score: -7 | Avg: -920 | Steps: 160 | Yield: 929.9 kg/ha | N: 120.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1925/2000 | Score: 550 | Avg: -920 | Steps: 150 | Yield: 4209.2 kg/ha | N: 120.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1926/2000 | Score: 525 | Avg: -920 | Steps: 158 | Yield: 4549.2 kg/ha | N: 120.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1927/2000 | Score: 398 | Avg: -920 | Steps: 162 | Yield: 2763.2 kg/ha | N: 40.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1928/2000 | Score: 107 | Avg: -920 | Steps: 156 | Yield: 1083.6 kg/ha | N: 40.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1929/2000 | Score: 398 | Avg: -920 | Steps: 156 | Yield: 3408.8 kg/ha | N: 120.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1930/2000 | Score: 565 | Avg: -920 | Steps: 156 | Yield: 5055.7 kg/ha | N: 120.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1931/2000 | Score: 74 | Avg: -920 | Steps: 164 | Yield: 1437.9 kg/ha | N: 160.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1932/2000 | Score: 294 | Avg: -920 | Steps: 162 | Yield: 4406.7 kg/ha | N: 400.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1933/2000 | Score: 710 | Avg: -920 | Steps: 154 | Yield: 5238.9 kg/ha | N: 40.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1934/2000 | Score: 465 | Avg: -920 | Steps: 157 | Yield: 6813.2 kg/ha | N: 440.0 kg/ha | Water: 240.0 mm | LR: 1.00e-02
Episode 1935/2000 | Score: 260 | Avg: -920 | Steps: 165 | Yield: 2461.8 kg/ha | N: 80.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1936/2000 | Score: 61 | Avg: -920 | Steps: 156 | Yield: 1561.5 kg/ha | N: 160.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1937/2000 | Score: 462 | Avg: -920 | Steps: 158 | Yield: 4750.7 kg/ha | N: 240.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1938/2000 | Score: 93 | Avg: -920 | Steps: 159 | Yield: 1450.0 kg/ha | N: 80.0 kg/ha | Water: 66.0 mm | LR: 1.00e-02
Episode 1939/2000 | Score: 24 | Avg: -920 | Steps: 159 | Yield: 1693.7 kg/ha | N: 200.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1940/2000 | Score: 129 | Avg: -920 | Steps: 162 | Yield: 1598.3 kg/ha | N: 40.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1941/2000 | Score: 223 | Avg: -920 | Steps: 156 | Yield: 1414.2 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1942/2000 | Score: 222 | Avg: -920 | Steps: 164 | Yield: 1404.3 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1943/2000 | Score: -487 | Avg: -920 | Steps: 165 | Yield: 1939.0 kg/ha | N: 720.0 kg/ha | Water: 204.0 mm | LR: 1.00e-02
Episode 1944/2000 | Score: 86 | Avg: -920 | Steps: 153 | Yield: 1467.6 kg/ha | N: 160.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
Episode 1945/2000 | Score: 214 | Avg: -920 | Steps: 161 | Yield: 1393.3 kg/ha | N: 0.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1946/2000 | Score: 88 | Avg: -920 | Steps: 168 | Yield: 554.9 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1947/2000 | Score: 1356 | Avg: -920 | Steps: 161 | Yield: 11014.2 kg/ha | N: 160.0 kg/ha | Water: 234.0 mm | LR: 1.00e-02
Episode 1948/2000 | Score: 64 | Avg: -920 | Steps: 161 | Yield: 1205.3 kg/ha | N: 160.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1949/2000 | Score: 113 | Avg: -920 | Steps: 165 | Yield: 1889.8 kg/ha | N: 160.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1950/2000 | Score: 53 | Avg: -920 | Steps: 159 | Yield: 1017.8 kg/ha | N: 120.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1951/2000 | Score: 734 | Avg: -920 | Steps: 154 | Yield: 8821.5 kg/ha | N: 560.0 kg/ha | Water: 198.0 mm | LR: 1.00e-02
Episode 1952/2000 | Score: 705 | Avg: -920 | Steps: 162 | Yield: 7866.7 kg/ha | N: 440.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1953/2000 | Score: 516 | Avg: -920 | Steps: 162 | Yield: 5328.1 kg/ha | N: 320.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1954/2000 | Score: 619 | Avg: -920 | Steps: 162 | Yield: 5268.4 kg/ha | N: 120.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1955/2000 | Score: 571 | Avg: -920 | Steps: 172 | Yield: 9430.3 kg/ha | N: 520.0 kg/ha | Water: 450.0 mm | LR: 1.00e-02
Episode 1956/2000 | Score: 456 | Avg: -920 | Steps: 158 | Yield: 4247.0 kg/ha | N: 80.0 kg/ha | Water: 138.0 mm | LR: 1.00e-02
Episode 1957/2000 | Score: 244 | Avg: -920 | Steps: 158 | Yield: 1542.5 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1958/2000 | Score: 69 | Avg: -920 | Steps: 159 | Yield: 517.2 kg/ha | N: 0.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1959/2000 | Score: 78 | Avg: -920 | Steps: 160 | Yield: 1918.7 kg/ha | N: 160.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1960/2000 | Score: 292 | Avg: -920 | Steps: 164 | Yield: 2675.1 kg/ha | N: 40.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 1961/2000 | Score: 546 | Avg: -920 | Steps: 163 | Yield: 4843.1 kg/ha | N: 160.0 kg/ha | Water: 84.0 mm | LR: 1.00e-02
Episode 1962/2000 | Score: 934 | Avg: -920 | Steps: 166 | Yield: 8631.9 kg/ha | N: 360.0 kg/ha | Water: 132.0 mm | LR: 1.00e-02
Episode 1963/2000 | Score: 158 | Avg: -920 | Steps: 164 | Yield: 2567.1 kg/ha | N: 80.0 kg/ha | Water: 168.0 mm | LR: 1.00e-02
Episode 1964/2000 | Score: 296 | Avg: -920 | Steps: 158 | Yield: 2282.0 kg/ha | N: 40.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1965/2000 | Score: 254 | Avg: -920 | Steps: 163 | Yield: 3225.6 kg/ha | N: 240.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1966/2000 | Score: 81 | Avg: -920 | Steps: 163 | Yield: 846.4 kg/ha | N: 0.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1967/2000 | Score: 218 | Avg: -920 | Steps: 162 | Yield: 1883.2 kg/ha | N: 0.0 kg/ha | Water: 72.0 mm | LR: 1.00e-02
Episode 1968/2000 | Score: 798 | Avg: -920 | Steps: 160 | Yield: 6654.1 kg/ha | N: 120.0 kg/ha | Water: 144.0 mm | LR: 1.00e-02
Episode 1969/2000 | Score: -18 | Avg: -920 | Steps: 160 | Yield: 818.3 kg/ha | N: 120.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1970/2000 | Score: 1022 | Avg: -920 | Steps: 157 | Yield: 10006.2 kg/ha | N: 240.0 kg/ha | Water: 336.0 mm | LR: 1.00e-02
Episode 1971/2000 | Score: 129 | Avg: -920 | Steps: 164 | Yield: 4308.4 kg/ha | N: 400.0 kg/ha | Water: 162.0 mm | LR: 1.00e-02
Episode 1972/2000 | Score: 157 | Avg: -920 | Steps: 164 | Yield: 1435.8 kg/ha | N: 80.0 kg/ha | Water: 6.0 mm | LR: 1.00e-02
Episode 1973/2000 | Score: 129 | Avg: -920 | Steps: 154 | Yield: 816.0 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1974/2000 | Score: 37 | Avg: -920 | Steps: 162 | Yield: 1003.5 kg/ha | N: 120.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1975/2000 | Score: 497 | Avg: -920 | Steps: 162 | Yield: 5023.4 kg/ha | N: 200.0 kg/ha | Water: 126.0 mm | LR: 1.00e-02
Episode 1976/2000 | Score: 74 | Avg: -920 | Steps: 155 | Yield: 1434.1 kg/ha | N: 160.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1977/2000 | Score: 774 | Avg: -920 | Steps: 154 | Yield: 5506.3 kg/ha | N: 80.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1978/2000 | Score: -31 | Avg: -920 | Steps: 162 | Yield: 1219.0 kg/ha | N: 200.0 kg/ha | Water: 60.0 mm | LR: 1.00e-02
Episode 1979/2000 | Score: 372 | Avg: -920 | Steps: 156 | Yield: 3240.4 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1980/2000 | Score: 62 | Avg: -920 | Steps: 159 | Yield: 394.7 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1981/2000 | Score: 60 | Avg: -920 | Steps: 163 | Yield: 4970.3 kg/ha | N: 760.0 kg/ha | Water: 108.0 mm | LR: 1.00e-02
Episode 1982/2000 | Score: 402 | Avg: -920 | Steps: 161 | Yield: 4555.8 kg/ha | N: 360.0 kg/ha | Water: 30.0 mm | LR: 1.00e-02
Episode 1983/2000 | Score: 300 | Avg: -920 | Steps: 156 | Yield: 2780.9 kg/ha | N: 160.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1984/2000 | Score: 816 | Avg: -920 | Steps: 159 | Yield: 11618.3 kg/ha | N: 840.0 kg/ha | Water: 318.0 mm | LR: 1.00e-02
Episode 1985/2000 | Score: 43 | Avg: -920 | Steps: 156 | Yield: 754.1 kg/ha | N: 80.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1986/2000 | Score: 938 | Avg: -920 | Steps: 162 | Yield: 7430.6 kg/ha | N: 240.0 kg/ha | Water: 42.0 mm | LR: 1.00e-02
Episode 1987/2000 | Score: 40 | Avg: -920 | Steps: 162 | Yield: 1785.9 kg/ha | N: 240.0 kg/ha | Water: 48.0 mm | LR: 1.00e-02
Episode 1988/2000 | Score: 276 | Avg: -920 | Steps: 163 | Yield: 2998.5 kg/ha | N: 200.0 kg/ha | Water: 36.0 mm | LR: 1.00e-02
Episode 1989/2000 | Score: 1078 | Avg: -920 | Steps: 158 | Yield: 8665.7 kg/ha | N: 160.0 kg/ha | Water: 150.0 mm | LR: 1.00e-02
Episode 1990/2000 | Score: 61 | Avg: -920 | Steps: 159 | Yield: 868.0 kg/ha | N: 80.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1991/2000 | Score: 309 | Avg: -920 | Steps: 159 | Yield: 2533.8 kg/ha | N: 40.0 kg/ha | Water: 54.0 mm | LR: 1.00e-02
Episode 1992/2000 | Score: 215 | Avg: -920 | Steps: 163 | Yield: 1361.5 kg/ha | N: 0.0 kg/ha | Water: 0.0 mm | LR: 1.00e-02
Episode 1993/2000 | Score: 159 | Avg: -920 | Steps: 160 | Yield: 1288.1 kg/ha | N: 40.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1994/2000 | Score: 308 | Avg: -920 | Steps: 151 | Yield: 4694.4 kg/ha | N: 440.0 kg/ha | Water: 78.0 mm | LR: 1.00e-02
Episode 1995/2000 | Score: 1179 | Avg: -920 | Steps: 162 | Yield: 10748.2 kg/ha | N: 240.0 kg/ha | Water: 294.0 mm | LR: 1.00e-02
Episode 1996/2000 | Score: 799 | Avg: -920 | Steps: 156 | Yield: 7294.4 kg/ha | N: 280.0 kg/ha | Water: 120.0 mm | LR: 1.00e-02
Episode 1997/2000 | Score: 342 | Avg: -920 | Steps: 158 | Yield: 2933.9 kg/ha | N: 120.0 kg/ha | Water: 24.0 mm | LR: 1.00e-02
Episode 1998/2000 | Score: 97 | Avg: -920 | Steps: 156 | Yield: 899.0 kg/ha | N: 40.0 kg/ha | Water: 12.0 mm | LR: 1.00e-02
Episode 1999/2000 | Score: 714 | Avg: -920 | Steps: 167 | Yield: 5744.8 kg/ha | N: 120.0 kg/ha | Water: 90.0 mm | LR: 1.00e-02
Episode 2000/2000 | Score: 35 | Avg: -920 | Steps: 158 | Yield: 544.5 kg/ha | N: 40.0 kg/ha | Water: 18.0 mm | LR: 1.00e-02
"""

# ============================================================================
#                              绘图函数
# ============================================================================

def save_plots(df: pd.DataFrame):
    """保存所有图表 (参考原代码的_save_plots方法)"""
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    episodes = df['episode'].values
    
    # ==================== 图1: 3x3多子图 ====================
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'{OUTPUT_PREFIX} Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. 最终产量
    ax = axes[0, 0]
    ax.plot(episodes, df['yield_val'], 'b-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Final Yield (kg/ha)', fontsize=10)
    ax.set_title('Final Yield', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # 添加平均线
    ax.axhline(y=df['yield_val'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["yield_val"].mean():.0f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 2. 灌溉量
    ax = axes[0, 1]
    ax.plot(episodes, df['water_amount'], 'g-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Irrigation (mm)', fontsize=10)
    ax.set_title('Irrigation Amount', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['water_amount'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["water_amount"].mean():.0f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 3. 施肥量
    ax = axes[0, 2]
    ax.plot(episodes, df['n_amount'], 'r-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Fertilizer (kg/ha)', fontsize=10)
    ax.set_title('Fertilizer Amount', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['n_amount'].mean(), color='b', linestyle='--', alpha=0.5, label=f'Mean: {df["n_amount"].mean():.0f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 4. WUE (水分利用效率)
    ax = axes[1, 0]
    ax.plot(episodes, df['wue'], 'c-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('WUE (kg/mm)', fontsize=10)
    ax.set_title('Water Use Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['wue'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["wue"].mean():.2f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 5. NUE (氮肥利用效率)
    ax = axes[1, 1]
    ax.plot(episodes, df['nue'], 'm-', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('NUE (kg/kg)', fontsize=10)
    ax.set_title('Nitrogen Use Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['nue'].mean(), color='b', linestyle='--', alpha=0.5, label=f'Mean: {df["nue"].mean():.2f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 6. 回报曲线
    ax = axes[1, 2]
    ax.plot(episodes, df['score'], 'b-', alpha=0.5, linewidth=0.8, label='Episode Score')
    ax.plot(episodes, df['rolling_avg_reward'], 'r-', linewidth=2, label='Rolling Avg (10-ep)')
    ax.axhline(y=EXPERT_PERFORMANCE_THRESHOLD, color='g', linestyle='--', alpha=0.7, label='Expert Threshold')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Reward', fontsize=10)
    ax.set_title('Reward Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 7. 步数
    ax = axes[2, 0]
    ax.plot(episodes, df['steps'], 'orange', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Steps', fontsize=10)
    ax.set_title('Episode Steps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=df['steps'].mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {df["steps"].mean():.0f}')
    ax.legend(loc='upper right', fontsize=8)
    
    # 8. 学习率曲线
    ax = axes[2, 1]
    ax.plot(episodes, df['lr'], 'purple', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 9. 累积步数
    ax = axes[2, 2]
    ax.plot(episodes, df['total_steps'], 'brown', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Total Steps', fontsize=10)
    ax.set_title('Cumulative Steps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_metrics_plot.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filepath}")
    
    # ==================== 图2: 详细回报曲线 ====================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # 主曲线
    ax2.plot(episodes, df['score'], 'b-', alpha=0.4, linewidth=0.8, label='Episode Score')
    ax2.plot(episodes, df['rolling_avg_reward'], 'r-', linewidth=2.5, label='Rolling Avg Reward (10-ep window)')
    
    # 标记样本效率
    sample_eff_ep = df['sample_efficiency_ep'].iloc[0]
    if pd.notna(sample_eff_ep):
        ax2.axvline(x=sample_eff_ep, color='g', linestyle='--', linewidth=2, 
                   label=f'Sample Efficiency (Ep {int(sample_eff_ep)})')
    
    # 标记收敛速度
    conv_ep = df['convergence_ep'].iloc[0]
    if pd.notna(conv_ep):
        ax2.axvline(x=conv_ep, color='purple', linestyle='--', linewidth=2,
                   label=f'Convergence (Ep {int(conv_ep)})')
    
    # 专家阈值
    ax2.axhline(y=EXPERT_PERFORMANCE_THRESHOLD, color='orange', linestyle=':', linewidth=2, 
               label=f'Expert Threshold ({EXPERT_PERFORMANCE_THRESHOLD})')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Training Reward Curve with Key Milestones', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加文本注释
    max_reward = df['score'].max()
    max_reward_ep = df.loc[df['score'].idxmax(), 'episode']
    ax2.annotate(f'Max: {max_reward:.0f} (Ep {int(max_reward_ep)})', 
                xy=(max_reward_ep, max_reward), 
                xytext=(max_reward_ep + 100, max_reward + 200),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    filepath2 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_reward_curve.png")
    plt.savefig(filepath2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reward curve saved: {filepath2}")
    
    # ==================== 图3: 资源利用效率 ====================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Resource Use Efficiency Analysis', fontsize=14, fontweight='bold')
    
    # WUE随训练变化
    ax = axes3[0]
    scatter = ax.scatter(df['water_amount'], df['yield_val'], c=df['episode'], 
                        cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel('Irrigation (mm)', fontsize=11)
    ax.set_ylabel('Final Yield (kg/ha)', fontsize=11)
    ax.set_title('Yield vs Irrigation (colored by episode)', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Episode')
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(df['water_amount'], df['yield_val'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['water_amount'].min(), df['water_amount'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
    ax.legend(loc='upper right')
    
    # NUE随训练变化
    ax = axes3[1]
    scatter2 = ax.scatter(df['n_amount'], df['yield_val'], c=df['episode'], 
                         cmap='plasma', alpha=0.6, s=30)
    ax.set_xlabel('Fertilizer (kg/ha)', fontsize=11)
    ax.set_ylabel('Final Yield (kg/ha)', fontsize=11)
    ax.set_title('Yield vs Fertilizer (colored by episode)', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax)
    cbar2.set_label('Episode')
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z2 = np.polyfit(df['n_amount'], df['yield_val'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(df['n_amount'].min(), df['n_amount'].max(), 100)
    ax.plot(x_line2, p2(x_line2), 'r--', alpha=0.7, linewidth=2, label='Trend')
    ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath3 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_efficiency_analysis.png")
    plt.savefig(filepath3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Efficiency analysis saved: {filepath3}")
    
    # ==================== 图4: 训练阶段分析 ====================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Training Phase Analysis', fontsize=14, fontweight='bold')
    
    # 按训练阶段分组分析
    n_episodes = len(df)
    early = df.iloc[:n_episodes//3]
    mid = df.iloc[n_episodes//3:2*n_episodes//3]
    late = df.iloc[2*n_episodes//3:]
    
    # 4.1 产量分布对比
    ax = axes4[0, 0]
    ax.hist(early['yield_val'], bins=20, alpha=0.5, label=f'Early (Ep 1-{n_episodes//3})', color='blue')
    ax.hist(mid['yield_val'], bins=20, alpha=0.5, label=f'Mid (Ep {n_episodes//3+1}-{2*n_episodes//3})', color='green')
    ax.hist(late['yield_val'], bins=20, alpha=0.5, label=f'Late (Ep {2*n_episodes//3+1}-{n_episodes})', color='red')
    ax.set_xlabel('Final Yield (kg/ha)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Yield Distribution by Training Phase', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4.2 回报分布对比
    ax = axes4[0, 1]
    ax.hist(early['score'], bins=20, alpha=0.5, label='Early Phase', color='blue')
    ax.hist(mid['score'], bins=20, alpha=0.5, label='Mid Phase', color='green')
    ax.hist(late['score'], bins=20, alpha=0.5, label='Late Phase', color='red')
    ax.set_xlabel('Episode Reward', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Reward Distribution by Training Phase', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4.3 WUE随阶段变化
    ax = axes4[1, 0]
    phases = ['Early', 'Mid', 'Late']
    wue_means = [early['wue'].mean(), mid['wue'].mean(), late['wue'].mean()]
    wue_stds = [early['wue'].std(), mid['wue'].std(), late['wue'].std()]
    bars = ax.bar(phases, wue_means, yerr=wue_stds, capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xlabel('Training Phase', fontsize=11)
    ax.set_ylabel('WUE (kg/mm)', fontsize=11)
    ax.set_title('Water Use Efficiency by Phase', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, mean in zip(bars, wue_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{mean:.1f}', 
               ha='center', va='bottom', fontsize=10)
    
    # 4.4 NUE随阶段变化
    ax = axes4[1, 1]
    nue_means = [early['nue'].mean(), mid['nue'].mean(), late['nue'].mean()]
    nue_stds = [early['nue'].std(), mid['nue'].std(), late['nue'].std()]
    bars = ax.bar(phases, nue_means, yerr=nue_stds, capsize=5, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xlabel('Training Phase', fontsize=11)
    ax.set_ylabel('NUE (kg/kg)', fontsize=11)
    ax.set_title('Nitrogen Use Efficiency by Phase', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, mean in zip(bars, nue_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{mean:.1f}', 
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath4 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_phase_analysis.png")
    plt.savefig(filepath4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Phase analysis saved: {filepath4}")
    
    # ==================== 图5: 学习进度热力图 ====================
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    
    # 创建滑动窗口统计
    window_size = 50
    n_windows = len(df) // window_size
    
    heat_data = []
    window_labels = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_df = df.iloc[start_idx:end_idx]
        
        heat_data.append([
            window_df['score'].mean(),
            window_df['yield_val'].mean(),
            window_df['wue'].mean(),
            window_df['nue'].mean(),
            window_df['water_amount'].mean(),
            window_df['n_amount'].mean()
        ])
        window_labels.append(f'{start_idx+1}-{end_idx}')
    
    heat_array = np.array(heat_data)
    
    # 标准化用于热力图显示
    heat_normalized = (heat_array - heat_array.min(axis=0)) / (heat_array.max(axis=0) - heat_array.min(axis=0) + 1e-8)
    
    im = ax5.imshow(heat_normalized.T, aspect='auto', cmap='RdYlGn')
    ax5.set_yticks(range(6))
    ax5.set_yticklabels(['Reward', 'Yield', 'WUE', 'NUE', 'Water', 'Fertilizer'])
    ax5.set_xticks(range(len(window_labels)))
    ax5.set_xticklabels(window_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_xlabel('Episode Window', fontsize=11)
    ax5.set_title('Training Progress Heatmap (Normalized)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Normalized Value (0=Min, 1=Max)')
    
    plt.tight_layout()
    filepath5 = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_progress_heatmap.png")
    plt.savefig(filepath5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Progress heatmap saved: {filepath5}")


def save_summary_to_excel(df: pd.DataFrame):
    """保存数据到Excel"""
    filepath = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_metrics.xlsx")
    
    # 整体统计
    summary_data = {
        'Metric': [
            'Total Episodes',
            'Total Steps',
            'Avg Final Yield (kg/ha)',
            'Max Final Yield (kg/ha)',
            'Min Final Yield (kg/ha)',
            'Avg Irrigation (mm)',
            'Avg Fertilizer (kg/ha)',
            'Avg WUE (kg/mm)',
            'Max WUE (kg/mm)',
            'Avg NUE (kg/kg)',
            'Max NUE (kg/kg)',
            'Avg Episode Reward',
            'Max Episode Reward',
            'Min Episode Reward',
            'Sample Efficiency Episode',
            'Convergence Episode'
        ],
        'Value': [
            len(df),
            df['total_steps'].iloc[-1],
            f"{df['yield_val'].mean():.2f}",
            f"{df['yield_val'].max():.2f}",
            f"{df['yield_val'].min():.2f}",
            f"{df['water_amount'].mean():.2f}",
            f"{df['n_amount'].mean():.2f}",
            f"{df['wue'].mean():.4f}",
            f"{df['wue'].max():.4f}",
            f"{df['nue'].mean():.4f}",
            f"{df['nue'].max():.4f}",
            f"{df['score'].mean():.2f}",
            f"{df['score'].max():.2f}",
            f"{df['score'].min():.2f}",
            str(int(df['sample_efficiency_ep'].iloc[0])) if pd.notna(df['sample_efficiency_ep'].iloc[0]) else 'N/A',
            str(int(df['convergence_ep'].iloc[0])) if pd.notna(df['convergence_ep'].iloc[0]) else 'N/A'
        ]
    }
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 原始数据
        df.to_excel(writer, sheet_name='All_Episodes', index=False)
        
        # 统计摘要
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # 每100轮汇总
        summary_100 = []
        for ep in range(100, len(df) + 1, 100):
            subset = df[df['episode'] <= ep]
            summary_100.append({
                'Episode': ep,
                'Avg_Reward': subset['score'].mean(),
                'Max_Reward': subset['score'].max(),
                'Avg_Yield': subset['yield_val'].mean(),
                'Max_Yield': subset['yield_val'].max(),
                'Avg_WUE': subset['wue'].mean(),
                'Avg_NUE': subset['nue'].mean(),
                'Avg_Water': subset['water_amount'].mean(),
                'Avg_Fertilizer': subset['n_amount'].mean()
            })
        pd.DataFrame(summary_100).to_excel(writer, sheet_name='Every_100_Episodes', index=False)
    
    print(f"Excel saved: {filepath}")


# ============================================================================
#                              主函数
# ============================================================================

def main():
    print("="*70)
    print(f"  {OUTPUT_PREFIX} Training Metrics Plotting Script")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # 1. 解析日志数据
    print("\n[1/4] Parsing log data...")
    records = parse_log_data(LOG_DATA)
    print(f"   Parsed {len(records)} episode records")
    
    # 2. 计算衍生指标
    print("\n[2/4] Calculating derived metrics...")
    df = calculate_metrics(records)
    print(f"   Calculated WUE, NUE, and other metrics")
    
    # 3. 生成图表
    print("\n[3/4] Generating plots...")
    save_plots(df)
    
    # 4. 保存Excel
    print("\n[4/4] Saving to Excel...")
    save_summary_to_excel(df)
    
    # 打印摘要
    print("\n" + "="*70)
    print("  Training Summary")
    print("="*70)
    print(f"\n  Total Episodes:      {len(df)}")
    print(f"  Total Steps:         {df['total_steps'].iloc[-1]}")
    print(f"\n  Yield Statistics:")
    print(f"    Average:           {df['yield_val'].mean():.2f} kg/ha")
    print(f"    Maximum:           {df['yield_val'].max():.2f} kg/ha")
    print(f"    Minimum:           {df['yield_val'].min():.2f} kg/ha")
    print(f"\n  Resource Efficiency:")
    print(f"    Avg WUE:           {df['wue'].mean():.4f} kg/mm")
    print(f"    Avg NUE:           {df['nue'].mean():.4f} kg/kg")
    print(f"\n  Reward Statistics:")
    print(f"    Average:           {df['score'].mean():.2f}")
    print(f"    Maximum:           {df['score'].max():.2f} (Episode {int(df.loc[df['score'].idxmax(), 'episode'])})")
    print(f"    Minimum:           {df['score'].min():.2f}")
    
    sample_eff = df['sample_efficiency_ep'].iloc[0]
    conv_ep = df['convergence_ep'].iloc[0]
    print(f"\n  Key Milestones:")
    print(f"    Sample Efficiency: Episode {int(sample_eff)}" if pd.notna(sample_eff) else "    Sample Efficiency: Not reached")
    print(f"    Convergence:       Episode {int(conv_ep)}" if pd.notna(conv_ep) else "    Convergence:       Not reached")
    
    print("\n" + "="*70)
    print("  All plots saved successfully!")
    print("="*70)


if __name__ == "__main__":
    main()