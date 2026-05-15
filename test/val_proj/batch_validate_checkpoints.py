#!/usr/bin/env python3
"""
批量验证脚本 - 遍历checkpoint文件夹，每个checkpoint验证10次，只保留第5轮结果

使用方法:
    /opt/gym_dssat_pdi/bin/python batch_validate_checkpoints.py

配置说明:
    - CHECKPOINT_DIR: checkpoint文件夹路径
    - N_EPISODES: 每次验证运行几轮（默认5轮）
    - N_REPEATS: 每个checkpoint重复验证多少次（默认10次）
    - 只记录每次验证的第5轮结果

输出字段（从验证脚本生成的Excel中直接读取）:
    - reward: 奖励分数
    - yield: 产量 (kg/ha)
    - irr: 灌溉量 (mm)
    - fert: 施肥量 (kg/ha)
    - wue: 水分利用效率 (kg/mm)
    - nue: 氮肥利用效率 (kg/kg)
"""

import os
import glob
import subprocess
import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import time
import shutil

# ==================== 配置区域 ====================
CHECKPOINT_DIR = "/home/wuyang/checkpoints/llama_ppo_rnd_checkpoints/best_0417"
SCRIPT_PATH = "/home/wuyang/test/val_proj/valodate_llama_ppo_rnd.py"
PYTHON_PATH = "/opt/gym_dssat_pdi/bin/python"
OUTPUT_DIR = "/home/wuyang/test/val_proj/batch_val_results_0417"
TEMP_OUTPUT_DIR = "/home/wuyang/test/val_proj/temp_val_output"  # 临时输出目录
N_EPISODES = 5      # 每次验证5轮
N_REPEATS = 10      # 每个checkpoint重复10次
# ==================================================


def find_checkpoints(checkpoint_dir):
    """查找所有checkpoint文件并按episode数排序"""
    pattern = os.path.join(checkpoint_dir, "model_ep*.pth")
    checkpoints = glob.glob(pattern)
    # 按episode数字排序
    checkpoints.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]))
    return checkpoints


def run_single_validation(checkpoint_path, repeat_idx, temp_dir):
    """
    运行单次验证，并从生成的Excel中读取第5轮结果
    
    根据源码，ValidationLogger 会生成 Excel 文件，包含 Episodes sheet:
    - Episode: 轮次编号
    - Reward: 奖励
    - Final_Yield_kg_ha: 产量 (kg/ha)
    - Irrigation_mm: 灌溉量 (mm)
    - Fertilizer_kg_ha: 施肥量 (kg/ha)
    - WUE_kg_mm: 水分利用效率 (kg/mm)
    - NUE_kg_kg: 氮肥利用效率 (kg/kg)
    - Steps: 步数
    
    Returns:
        dict: 包含第5轮结果的字典
    """
    # 为每次验证创建唯一的输出目录
    repeat_output_dir = os.path.join(temp_dir, f"repeat_{repeat_idx}")
    os.makedirs(repeat_output_dir, exist_ok=True)
    
    cmd = [
        PYTHON_PATH,
        SCRIPT_PATH,
        "--checkpoint", checkpoint_path,
        "--n_episodes", str(N_EPISODES),
        "--output_dir", repeat_output_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        # 查找生成的Excel文件
        # 根据 ValidationLogger，Excel文件名格式为: {checkpoint_name}_validation.xlsx
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        excel_pattern = os.path.join(repeat_output_dir, checkpoint_name, f"{checkpoint_name}_validation.xlsx")
        
        # 如果找不到，尝试其他可能的路径
        if not os.path.exists(excel_pattern):
            # 搜索所有可能的xlsx文件
            xlsx_files = glob.glob(os.path.join(repeat_output_dir, "**", "*.xlsx"), recursive=True)
            if xlsx_files:
                excel_pattern = xlsx_files[0]
            else:
                return {'reward': None, 'yield': None, 'irr': None, 'fert': None, 'wue': None, 'nue': None}
        
        # 读取Excel文件中的Episodes sheet
        df_episodes = pd.read_excel(excel_pattern, sheet_name='Episodes')
        
        # 获取第5轮的数据（Episode=5）
        episode_5 = df_episodes[df_episodes['Episode'] == 5]
        
        if len(episode_5) == 0:
            # 如果没有Episode=5，尝试获取最后一行
            episode_5 = df_episodes.iloc[[-1]]
        
        # 提取各字段
        result = {
            'reward': float(episode_5['Reward'].values[0]) if 'Reward' in episode_5.columns else None,
            'yield': float(episode_5['Final_Yield_kg_ha'].values[0]) if 'Final_Yield_kg_ha' in episode_5.columns else None,
            'irr': float(episode_5['Irrigation_mm'].values[0]) if 'Irrigation_mm' in episode_5.columns else None,
            'fert': float(episode_5['Fertilizer_kg_ha'].values[0]) if 'Fertilizer_kg_ha' in episode_5.columns else None,
            'wue': float(episode_5['WUE_kg_mm'].values[0]) if 'WUE_kg_mm' in episode_5.columns else None,
            'nue': float(episode_5['NUE_kg_kg'].values[0]) if 'NUE_kg_kg' in episode_5.columns else None,
        }
        
        return result
        
    except subprocess.TimeoutExpired:
        return {'reward': None, 'yield': None, 'irr': None, 'fert': None, 'wue': None, 'nue': None}
    except Exception as e:
        print(f"    ERROR reading Excel: {e}")
        return {'reward': None, 'yield': None, 'irr': None, 'fert': None, 'wue': None, 'nue': None}


def save_results(all_results, output_dir):
    """保存所有结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"batch_validation_results_{timestamp}.xlsx")
    csv_path = os.path.join(output_dir, f"batch_validation_results_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path}")
    
    # 创建汇总统计
    summary_data = []
    checkpoints = df['checkpoint'].unique()
    
    for ckpt in checkpoints:
        ckpt_df = df[df['checkpoint'] == ckpt]
        
        def safe_mean(series):
            vals = series.dropna()
            return vals.mean() if len(vals) > 0 else np.nan
        
        def safe_std(series):
            vals = series.dropna()
            return vals.std() if len(vals) > 0 else np.nan
        
        def safe_min(series):
            vals = series.dropna()
            return vals.min() if len(vals) > 0 else np.nan
        
        def safe_max(series):
            vals = series.dropna()
            return vals.max() if len(vals) > 0 else np.nan
        
        summary_data.append({
            'checkpoint': ckpt,
            'n_valid_runs': len(ckpt_df),
            'n_successful': ckpt_df['reward'].notna().sum(),
            
            # Reward
            'mean_reward': safe_mean(ckpt_df['reward']),
            'std_reward': safe_std(ckpt_df['reward']),
            'min_reward': safe_min(ckpt_df['reward']),
            'max_reward': safe_max(ckpt_df['reward']),
            
            # Yield
            'mean_yield': safe_mean(ckpt_df['yield']),
            'std_yield': safe_std(ckpt_df['yield']),
            'min_yield': safe_min(ckpt_df['yield']),
            'max_yield': safe_max(ckpt_df['yield']),
            
            # Irrigation
            'mean_irr': safe_mean(ckpt_df['irr']),
            'std_irr': safe_std(ckpt_df['irr']),
            
            # Fertilizer
            'mean_fert': safe_mean(ckpt_df['fert']),
            'std_fert': safe_std(ckpt_df['fert']),
            
            # WUE
            'mean_wue': safe_mean(ckpt_df['wue']),
            'std_wue': safe_std(ckpt_df['wue']),
            
            # NUE
            'mean_nue': safe_mean(ckpt_df['nue']),
            'std_nue': safe_std(ckpt_df['nue']),
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到Excel（多个sheet）
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Results', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Excel saved: {excel_path}")
    
    # 绘制对比图
    plot_results(df_summary, output_dir, timestamp)
    
    return df, df_summary


def plot_results(df_summary, output_dir, timestamp):
    """绘制对比图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Checkpoint Validation Results (Episode 5 only, 10 repeats each)', fontsize=16)
    
    checkpoints = df_summary['checkpoint'].tolist()
    x = range(len(checkpoints))
    labels = [c.replace('model_ep', 'ep') for c in checkpoints]
    
    def safe_bar(ax, data, std_data, ylabel, title, color):
        """安全绘制柱状图，处理NaN值"""
        valid_mask = ~np.isnan(data)
        if not any(valid_mask):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        ax.bar(x, np.where(valid_mask, data, 0), 
               yerr=np.where(valid_mask, std_data, 0), 
               capsize=3, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # 1. 平均奖励
    ax = axes[0, 0]
    safe_bar(ax, df_summary['mean_reward'].values, df_summary['std_reward'].values, 
             'Reward', 'Mean Reward', 'steelblue')
    
    # 2. 平均产量
    ax = axes[0, 1]
    safe_bar(ax, df_summary['mean_yield'].values, df_summary['std_yield'].values,
             'Yield (kg/ha)', 'Mean Yield', 'forestgreen')
    
    # 3. 平均灌溉量
    ax = axes[0, 2]
    safe_bar(ax, df_summary['mean_irr'].values, df_summary['std_irr'].values,
             'Irrigation (mm)', 'Mean Irrigation', 'deepskyblue')
    
    # 4. 平均施肥量
    ax = axes[0, 3]
    safe_bar(ax, df_summary['mean_fert'].values, df_summary['std_fert'].values,
             'Fertilizer (kg/ha)', 'Mean Fertilizer', 'orange')
    
    # 5. 平均WUE
    ax = axes[1, 0]
    safe_bar(ax, df_summary['mean_wue'].values, df_summary['std_wue'].values,
             'WUE (kg/mm)', 'Mean WUE', 'coral')
    
    # 6. 平均NUE
    ax = axes[1, 1]
    safe_bar(ax, df_summary['mean_nue'].values, df_summary['std_nue'].values,
             'NUE (kg/kg)', 'Mean NUE', 'purple')
    
    # 7. Reward vs Yield
    ax = axes[1, 2]
    valid_mask = ~np.isnan(df_summary['mean_reward']) & ~np.isnan(df_summary['mean_yield'])
    if any(valid_mask):
        ax.scatter(df_summary['mean_reward'][valid_mask], df_summary['mean_yield'][valid_mask], 
                   s=100, c=range(sum(valid_mask)), cmap='viridis', alpha=0.8)
        for i, (r, y, label) in enumerate(zip(df_summary['mean_reward'][valid_mask], 
                                               df_summary['mean_yield'][valid_mask], 
                                               [labels[j] for j in range(len(labels)) if valid_mask.iloc[j]])):
            ax.annotate(label, (r, y), fontsize=6, ha='left')
    ax.set_xlabel('Mean Reward')
    ax.set_ylabel('Mean Yield (kg/ha)')
    ax.set_title('Reward vs Yield')
    ax.grid(True, alpha=0.3)
    
    # 8. WUE vs NUE
    ax = axes[1, 3]
    valid_mask = ~np.isnan(df_summary['mean_wue']) & ~np.isnan(df_summary['mean_nue'])
    if any(valid_mask):
        ax.scatter(df_summary['mean_wue'][valid_mask], df_summary['mean_nue'][valid_mask], 
                   s=100, c=range(sum(valid_mask)), cmap='coolwarm', alpha=0.8)
        for i, (w, n, label) in enumerate(zip(df_summary['mean_wue'][valid_mask], 
                                               df_summary['mean_nue'][valid_mask], 
                                               [labels[j] for j in range(len(labels)) if valid_mask.iloc[j]])):
            ax.annotate(label, (w, n), fontsize=6, ha='left')
    ax.set_xlabel('WUE (kg/mm)')
    ax.set_ylabel('NUE (kg/kg)')
    ax.set_title('WUE vs NUE')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"validation_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")


def print_checkpoint_status(checkpoint_name, current_results, repeat_idx, total_repeats):
    """打印当前checkpoint的实时统计状态"""
    if not current_results:
        return
    
    # 提取有效结果
    valid_results = [r for r in current_results if r['reward'] is not None]
    
    if not valid_results:
        print(f"\n    [{checkpoint_name}] Repeat {repeat_idx}/{total_repeats} - No valid results yet")
        return
    
    # 辅助函数
    def get_stats(values):
        if not values or all(v is None for v in values):
            return None, None, None, None
        vals = [v for v in values if v is not None]
        if not vals:
            return None, None, None, None
        return np.mean(vals), np.std(vals), min(vals), max(vals)
    
    rewards = [r['reward'] for r in valid_results]
    yields = [r['yield'] for r in valid_results]
    irrs = [r['irr'] for r in valid_results]
    ferts = [r['fert'] for r in valid_results]
    wues = [r['wue'] for r in valid_results]
    nues = [r['nue'] for r in valid_results]
    
    r_mean, r_std, r_min, r_max = get_stats(rewards)
    y_mean, y_std, y_min, y_max = get_stats(yields)
    i_mean, i_std, i_min, i_max = get_stats(irrs)
    f_mean, f_std, f_min, f_max = get_stats(ferts)
    w_mean, w_std, w_min, w_max = get_stats(wues)
    n_mean, n_std, n_min, n_max = get_stats(nues)
    
    print(f"\n    ╔{'─'*86}╗")
    print(f"    ║  Checkpoint: {checkpoint_name:<72}║")
    print(f"    ╠{'─'*86}╣")
    print(f"    ║  Progress: {repeat_idx}/{total_repeats} completed ({len(valid_results)} valid)                                      ║")
    print(f"    ╠{'─'*86}╣")
    print(f"    ║  │ Metric       │     Mean ± Std     │       Min       │       Max       │ Unit     ║")
    print(f"    ╠{'─'*86}╣")
    
    if r_mean is not None:
        print(f"    ║  │ Reward       │ {r_mean:>9.1f} ± {r_std:<7.1f} │ {r_min:>13.1f} │ {r_max:>13.1f} │ -        ║")
    if y_mean is not None:
        print(f"    ║  │ Yield        │ {y_mean:>9.0f} ± {y_std:<7.0f} │ {y_min:>13.0f} │ {y_max:>13.0f} │ kg/ha    ║")
    if i_mean is not None:
        print(f"    ║  │ Irrigation   │ {i_mean:>9.1f} ± {i_std:<7.1f} │ {i_min:>13.1f} │ {i_max:>13.1f} │ mm       ║")
    else:
        print(f"    ║  │ Irrigation   │         N/A         │        -        │        -        │ mm       ║")
    if f_mean is not None:
        print(f"    ║  │ Fertilizer   │ {f_mean:>9.1f} ± {f_std:<7.1f} │ {f_min:>13.1f} │ {f_max:>13.1f} │ kg/ha    ║")
    else:
        print(f"    ║  │ Fertilizer   │         N/A         │        -        │        -        │ kg/ha    ║")
    if w_mean is not None:
        print(f"    ║  │ WUE          │ {w_mean:>9.2f} ± {w_std:<7.2f} │ {w_min:>13.2f} │ {w_max:>13.2f} │ kg/mm    ║")
    else:
        print(f"    ║  │ WUE          │         N/A         │        -        │        -        │ kg/mm    ║")
    if n_mean is not None:
        print(f"    ║  │ NUE          │ {n_mean:>9.2f} ± {n_std:<7.2f} │ {n_min:>13.2f} │ {n_max:>13.2f} │ kg/kg    ║")
    else:
        print(f"    ║  │ NUE          │         N/A         │        -        │        -        │ kg/kg    ║")
    
    print(f"    ╚{'─'*86}╝")


def main():
    print("\n" + "=" * 88)
    print("  Batch Checkpoint Validation")
    print("=" * 88)
    print(f"  Checkpoint Directory: {CHECKPOINT_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Temp Output Directory: {TEMP_OUTPUT_DIR}")
    print(f"  Repeats per checkpoint: {N_REPEATS}")
    print(f"  Episodes per run: {N_EPISODES} (only episode 5 recorded)")
    print("=" * 88)
    
    # 创建临时输出目录
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    
    # 查找所有checkpoint
    checkpoints = find_checkpoints(CHECKPOINT_DIR)
    
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {CHECKPOINT_DIR}")
        return
    
    n_checkpoints = len(checkpoints)
    total_validations = n_checkpoints * N_REPEATS
    
    print(f"\n✓ Found {n_checkpoints} checkpoints:")
    for i, ckpt in enumerate(checkpoints, 1):
        ckpt_name = os.path.basename(ckpt)
        ckpt_size = os.path.getsize(ckpt) / (1024 * 1024)  # MB
        print(f"    {i:2d}. {ckpt_name} ({ckpt_size:.1f} MB)")
    
    print(f"\n✓ Total validations to run: {total_validations}")
    print("=" * 88)
    
    # 收集所有结果
    all_results = []
    
    # 用于存储每个checkpoint的实时结果
    checkpoint_results = {}
    
    # 计算开始时间
    start_time = time.time()
    
    # ==================== 主进度条 ====================
    main_pbar = tqdm(
        total=total_validations,
        desc="Overall Progress",
        unit="val",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # 遍历每个checkpoint
    for ckpt_idx, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
        
        # 初始化该checkpoint的结果列表
        checkpoint_results[checkpoint_name] = []
        
        # 为这个checkpoint创建临时目录
        ckpt_temp_dir = os.path.join(TEMP_OUTPUT_DIR, checkpoint_name)
        os.makedirs(ckpt_temp_dir, exist_ok=True)
        
        # 打印checkpoint开始信息
        print(f"\n{'='*88}")
        print(f"  [{ckpt_idx+1}/{n_checkpoints}] Validating: {checkpoint_name}")
        print(f"{'='*88}")
        
        # 更新进度条描述
        main_pbar.set_description(f"Overall [{ckpt_idx+1}/{n_checkpoints}] {checkpoint_name}")
        
        # 对每个checkpoint运行N_REPEATS次验证
        for repeat_idx in range(N_REPEATS):
            # 运行单次验证
            result = run_single_validation(checkpoint_path, repeat_idx, ckpt_temp_dir)
            
            # 记录结果
            result['checkpoint'] = checkpoint_name
            result['repeat'] = repeat_idx + 1
            all_results.append(result)
            checkpoint_results[checkpoint_name].append(result)
            
            # 更新进度条后缀
            if result['reward'] is not None:
                postfix = {'rwd': f"{result['reward']:.0f}", 'yld': f"{result['yield']:.0f}"}
                if result['wue'] is not None:
                    postfix['wue'] = f"{result['wue']:.1f}"
                if result['nue'] is not None:
                    postfix['nue'] = f"{result['nue']:.1f}"
                main_pbar.set_postfix(postfix)
            else:
                main_pbar.set_postfix({'status': 'FAILED'})
            
            # 更新进度条
            main_pbar.update(1)
            
            # 每2次验证后打印当前状态
            if (repeat_idx + 1) % 5 == 0 or repeat_idx == N_REPEATS - 1:
                print_checkpoint_status(checkpoint_name, checkpoint_results[checkpoint_name], 
                                       repeat_idx + 1, N_REPEATS)
    
    main_pbar.close()
    # ================================================
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n✓ Validation completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # 清理临时目录
    try:
        shutil.rmtree(TEMP_OUTPUT_DIR)
        print(f"✓ Cleaned up temp directory: {TEMP_OUTPUT_DIR}")
    except:
        pass
    
    # 保存结果
    df, df_summary = save_results(all_results, OUTPUT_DIR)
    
    # 打印最终汇总
    print("\n" + "=" * 110)
    print("  FINAL SUMMARY")
    print("=" * 110)
    
    # 格式化打印汇总表
    header = f"\n  {'Checkpoint':<14} {'Reward':>12} {'Yield':>12} {'Irr':>10} {'Fert':>10} {'WUE':>10} {'NUE':>10} {'Success':>8}"
    print(header)
    print("  " + "-" * 106)
    
    for _, row in df_summary.iterrows():
        ckpt_short = row['checkpoint'].replace('model_ep', 'ep')
        
        # 格式化各项指标，处理NaN
        def fmt_val(mean, std, fmt):
            if pd.isna(mean):
                return "     N/A    "
            if pd.isna(std):
                std = 0
            return f"{mean:{fmt}}±{std:{fmt.split('.')[0]}.{fmt.split('.')[-1] if '.' in fmt else '0'}}"
        
        reward_str = fmt_val(row['mean_reward'], row['std_reward'], "7.1f")
        yield_str = fmt_val(row['mean_yield'], row['std_yield'], "7.0f")
        irr_str = fmt_val(row['mean_irr'], row['std_irr'], "7.1f")
        fert_str = fmt_val(row['mean_fert'], row['std_fert'], "7.1f")
        wue_str = fmt_val(row['mean_wue'], row['std_wue'], "7.2f")
        nue_str = fmt_val(row['mean_nue'], row['std_nue'], "7.2f")
        
        print(f"  {ckpt_short:<14} {reward_str} {yield_str} {irr_str} {fert_str} {wue_str} {nue_str} {row['n_successful']:>5.0f}/10")
    
    print("  " + "-" * 106)
    
    # 打印最佳checkpoint
    valid_summary = df_summary.dropna(subset=['mean_reward'])
    if len(valid_summary) > 0:
        best_reward = valid_summary.loc[valid_summary['mean_reward'].idxmax()]
        print(f"\n  🏆 Best by Reward:    {best_reward['checkpoint']:<20} {best_reward['mean_reward']:.1f}")
    
    valid_summary = df_summary.dropna(subset=['mean_yield'])
    if len(valid_summary) > 0:
        best_yield = valid_summary.loc[valid_summary['mean_yield'].idxmax()]
        print(f"  🏆 Best by Yield:     {best_yield['checkpoint']:<20} {best_yield['mean_yield']:.0f} kg/ha")
    
    valid_summary = df_summary.dropna(subset=['mean_wue'])
    if len(valid_summary) > 0:
        best_wue = valid_summary.loc[valid_summary['mean_wue'].idxmax()]
        print(f"  🏆 Best by WUE:       {best_wue['checkpoint']:<20} {best_wue['mean_wue']:.2f} kg/mm")
    
    valid_summary = df_summary.dropna(subset=['mean_nue'])
    if len(valid_summary) > 0:
        best_nue = valid_summary.loc[valid_summary['mean_nue'].idxmax()]
        print(f"  🏆 Best by NUE:       {best_nue['checkpoint']:<20} {best_nue['mean_nue']:.2f} kg/kg")
    
    print(f"\n✓ Total validations: {len(all_results)}")
    print(f"✓ Successful: {sum(1 for r in all_results if r['reward'] is not None)}")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()