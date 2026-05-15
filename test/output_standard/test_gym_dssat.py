#!/usr/bin/env python3
"""测试gym-dssat环境是否正常工作"""

import gym
import os
import sys

def test_env():
    print("测试gym-dssat环境...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")
    
    # 确保日志目录存在
    os.makedirs('./logs', exist_ok=True)
    
    env_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'all',
        'seed': 123456,
        'random_weather': False,
    }
    
    print("正在创建gym环境...")
    try:
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        print("✓ 环境创建成功")
        
        print("正在重置环境...")
        state = env.reset()
        print(f"✓ 环境重置成功，状态维度: {len(state) if hasattr(state, '__len__') else 'unknown'}")
        
        print("测试一步动作...")
        action = {'anfer': 40, 'amir': 6}
        next_state, reward, done, info = env.step(action)
        print(f"✓ 动作执行成功，奖励: {reward}, 完成: {done}")
        
        env.close()
        print("✓ 环境测试完成")
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()