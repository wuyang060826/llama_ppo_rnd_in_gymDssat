import sys
import platform
import subprocess
import os

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_system_info():
    print_section("1. 系统基础信息")
    print(f"操作系统  : {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"Python路径: {sys.executable}")

def check_pytorch():
    print_section("2. PyTorch 安装与版本")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"编译时的CUDA版本: {torch.version.cuda}")
        return True
    except ImportError:
        print("错误: 未检测到 PyTorch，请先安装 PyTorch。")
        return False

def check_cuda_driver():
    print_section("3. CUDA 驱动与运行时")
    try:
        import torch
        if not torch.cuda.is_available():
            print("警告: PyTorch 检测不到 CUDA 设备 (torch.cuda.is_available() = False)")
            print("可能原因:")
            print("1. 机器没有 NVIDIA GPU")
            print("2. 安装的是 CPU 版本的 PyTorch")
            print("3. NVIDIA 驱动版本过低或未正确安装")
            return False
        
        print(f"CUDA 是否可用: 是")
        print(f"当前 CUDA 设备索引: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"驱动 API 版本: {torch.version.cuda or 'N/A'}") # PyTorch编译的CUDA版本
        print(f"运行时 CUDA 版本: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        
        # 尝试获取系统驱动版本 (通过 nvidia-smi)
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                driver_version = "N/A"
                for line in lines:
                    if "Driver Version" in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            driver_info = parts[1].strip()
                            driver_version = driver_info.split()[0]
                            break
                print(f"系统 NVIDIA 驱动版本: {driver_version}")
            else:
                print("无法运行 nvidia-smi 检查驱动版本")
        except FileNotFoundError:
            print("未找到 nvidia-smi 工具 (可能未安装驱动或不在PATH中)")
            
        return True
    except Exception as e:
        print(f"检查 CUDA 时发生错误: {e}")
        return False

def check_cudnn():
    print_section("4. cuDNN 版本信息")
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA 不可用，跳过 cuDNN 检查")
            return

        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN 是否启用: {torch.backends.cudnn.enabled}")
        print(f"cuDNN Benchmark 模式: {torch.backends.cudnn.benchmark}")
    except Exception as e:
        print(f"检查 cuDNN 时发生错误: {e}")

def run_tensor_test():
    print_section("5. GPU 张量计算测试")
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA 不可用，跳过计算测试")
            return

        print("正在创建张量并传输到 GPU...")
        x = torch.rand(5000, 5000).cuda()
        y = torch.rand(5000, 5000).cuda()
        
        print("正在执行矩阵乘法 (C = A * B)...")
        # 预热，避免首次运行的初始化开销影响计时
        for _ in range(5):
            _ = torch.mm(x, y)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        import time
        start = time.time()
        for _ in range(10):
            z = torch.mm(x, y)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        print(f"计算完成。")
        print(f"10次 5000x5000 矩阵乘法耗时: {(end - start)*1000:.2f} ms")
        print(f"结果张量设备: {z.device}")
        print("✅ GPU 计算功能正常")
        
    except Exception as e:
        print(f"❌ 计算测试失败: {e}")

def check_other_libs():
    print_section("6. 其他常用大模型库检查")
    libs = {
        'transformers': 'Hugging Face Transformers',
        'datasets': 'Hugging Face Datasets',
        'accelerate': 'Hugging Face Accelerate',
        'peft': 'PEFT (LoRA等)',
        'bitsandbytes': 'BitsAndBytes (量化加载)',
        'flash_attn': 'Flash Attention 2',
        'xformers': 'xFormers (内存优化注意力机制)'
    }
    
    for module, name in libs.items():
        try:
            mod = __import__(module)
            # 尝试获取版本号
            version = getattr(mod, '__version__', '未知版本')
            print(f"✅ {name:30s} : {version}")
        except ImportError:
            print(f"⚪ {name:30s} : 未安装")

if __name__ == "__main__":
    check_system_info()
    if check_pytorch():
        check_cuda_driver()
        check_cudnn()
        run_tensor_test()
    check_other_libs()
    print("\n" + "="*60)
    print("  检查结束")
    print("="*60)