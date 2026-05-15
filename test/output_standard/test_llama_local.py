import requests
import json
import time
import re

# 全局配置
API_URL = "http://10.111.5.16:10096/v1/chat/completions"
MODEL_NAME = "qwen3-llm"
API_KEY = ""

def call_qwen_model(user_prompt, temperature=0.7, max_tokens=512):
    """
    调用本地 Qwen 模型进行单次问答
    
    Args:
        user_prompt (str): 用户输入的问题
        temperature (float): 控制随机性，默认 0.7
        max_tokens (int): 限制回复生成的最大 Token 数，默认 512
    
    Returns:
        str: 模型的回复内容。如果出错或超时，返回空字符串。
    """
    # 记录开始时间
    start_time = time.time()
    
    # 构造请求数据
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user", 
                "content": '/no_think' + user_prompt
            }
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_think": False
    }

    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        # 发送 POST 请求，设置5秒超时
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=5)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 检查 HTTP 状态码
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                reply_content = result["choices"][0]["message"]["content"]
                
                # 移除 <think> 标签及其内容
                reply_content = re.sub(r'<think>.*?</think>', '', reply_content, flags=re.DOTALL)
                
                # 打印耗时信息
                print(f"调用成功 | 耗时: {elapsed_time:.2f}秒")
                return reply_content
            else:
                print(f"响应格式异常 | 耗时: {elapsed_time:.2f}秒")
                print("原始响应:", result)
                return ""
        else:
            print(f"请求失败 (状态码: {response.status_code}) | 耗时: {elapsed_time:.2f}秒")
            print("错误信息:", response.text)
            return ""
            
    except requests.exceptions.Timeout as e:
        elapsed_time = time.time() - start_time
        print(f"请求超时: {e} | 耗时: {elapsed_time:.2f}秒")
        return ""
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        print(f"网络请求错误: {e} | 耗时: {elapsed_time:.2f}秒")
        return ""
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"未知错误: {e} | 耗时: {elapsed_time:.2f}秒")
        return ""

# --- 测试调用 ---
if __name__ == "__main__":
    question = '''
决策结果与决策输出如下：

======================================================================
【决策输出】 第10轮 第163步 (生长第164天) [回合结束]
======================================================================

🎯 【决策结果】
--------------------------------------------------
   动作编号: 23
   施氮量: 120 公斤/公顷
   灌溉量: 24 毫米
   置信度: 85%

📊 【推理解释】
--------------------------------------------------
【一、数据概览】
当前为玉米生长周期第164天，处于成熟期。籽粒成熟，准备收获。
产量预测值：10634 公斤/公顷。

【二、土壤水分状况分析】
土壤水分指数为1602，水分充足。可适当减少灌溉量，注意田间排水防止涝害。

【三、遥感数据分析】
1. 植被指数（NDVI）：0.574
   植被覆盖度良好，生长状态正常
2. 叶面积指数（LAI）：3.08
   叶面积指数偏高（参考范围1.5-3.0），可能存在徒长风险
3. 遥感土壤湿度：0.297 立方米/立方米
   土壤水分适宜，利于作物生长
4. 叶绿素含量：41.4 微克/平方厘米
   叶绿素含量较高，氮素营养充足
遥感数据综合评估：作物长势正常，继续保持常规管理。

【四、农事记录分析】
记录日期：2024年9月14日
操作类型：监测
备注：玉米生长第164天，处于成熟期

【五、生长阶段需水需肥分析】
当前成熟期水肥需求特点：籽粒成熟，准备收获。
氮肥建议：停止施用。
灌溉建议：减少供水。

【六、决策理由】
氮肥决策：选择施氮120公斤/公顷。
  理由：根据土壤肥力和作物需肥规律，补充适量氮肥。

灌溉决策：选择灌溉24毫米。
  理由：根据土壤墒情和作物需水规律，补充适量水分。

【七、风险评估】
当前风险等级较低，作物生长状态正常。

【八、决策置信度】
本次决策置信度：85%
依据：综合分析了生长阶段、土壤状态、遥感数据、农事记录等信息。


🔍 【关键决策因素】
   1. 生长阶段：成熟期
   2. 土壤水分：1602
   3. 植被指数：0.574
   4. 叶绿素：41.4
   5. 上次操作：监测
======================================================================

请根据【一、数据概览】、【二、土壤水分状况分析】、【五、生长阶段需水需肥分析】等情况，分析【决策结果】输出是否合理，直观、具体、精炼返回你的结论
'''
    
    print(f"用户提问: {question}\n")
    
    # 调用函数
    answer = call_qwen_model(question, max_tokens=1024)
    
    # 打印结果
    if answer:
        print("\n模型回复:")
        print("-" * 50)
        print(answer)
        print("-" * 50)