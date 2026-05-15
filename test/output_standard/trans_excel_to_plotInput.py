# 1. 安装依赖：执行 pip install pandas openpyxl
import pandas as pd

# -------------------------- 配置项（可根据你的文件修改）--------------------------
# Excel文件路径（请替换为你的实际文件路径，如 "data.xlsx"）
prefix = "bert_dqn"
EXCEL_FILE_PATH = f'/home/gymusr/{prefix}_results/results/{prefix}_results.xlsx'
SAVE_TXT_PATH = f'/home/gymusr/{prefix}_results/{prefix}_results.txt'
# 固定总轮数（日志中固定显示 /3000）
TOTAL_EPISODES = 2000
# --------------------------------------------------------------------------------

def excel_to_log_string(excel_path):
    """
    读取Excel文件，转换为指定格式的日志字符串
    :param excel_path: Excel文件路径
    :return: 格式化后的LOG_DATA字符串
    """
    # 读取Excel数据
    df = pd.read_excel(excel_path)
    
    # 计算全局平均Score（所有Return的平均值，取整）
    avg_score = round(df["Return"].mean())
    
    # 存储每一行日志
    log_lines = []
    
    # 遍历每一行数据，格式化输出
    for _, row in df.iterrows():
        # 提取字段
        episode = int(row["Episode"])
        score = round(row["Return"])        # Score：Return取整
        steps = int(row["Episode_Length"])  # Steps：周期长度
        yield_val = row["Yield"]            # 产量
        n_amount = row["N_Amount"]         # 氮肥量
        w_amount = row["W_Amount"]         # 水量
        lr = row["Epsilon"]                # 学习率（Epsilon）
        
        # 严格按照要求格式化每一行
        line = (
            f"Episode {episode}/{TOTAL_EPISODES} | Score: {score} | Avg: {avg_score} | Steps: {steps} | "
            f"Yield: {yield_val:.1f} kg/ha | N: {n_amount:.1f} kg/ha | Water: {w_amount:.1f} mm | LR: {lr:.2e}"
        )
        log_lines.append(line)
    
    # 拼接为最终的LOG_DATA格式（三引号包裹，换行分隔）
    log_data = 'LOG_DATA = """\n' + '\n'.join(log_lines) + '\n"""'
    return log_data

# 执行转换并打印结果
if __name__ == "__main__":
    result = excel_to_log_string(EXCEL_FILE_PATH)
    # -------------------------- 新增：保存为TXT文件 --------------------------
    with open(SAVE_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"\n✅ 文件已成功保存至：{SAVE_TXT_PATH}")
    # ------------------------------------------------------------------------