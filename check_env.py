# check_env.py - 验证 PyTorch + GPU 环境
import torch

# 基础信息
print("===== 深度学习环境验证 =====")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Python 环境: ml_env (Python 3.10)")

# GPU 专属验证（RTX 4060）
print("\n===== GPU 信息 =====")
if torch.cuda.is_available():
    print(f"CUDA 可用: ✅ True")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    # 测试 GPU 张量运算
    gpu_tensor = torch.rand(5, 3).cuda()
    print(f"GPU 张量示例:\n{gpu_tensor}")
else:
    print("CUDA 可用: ❌ False (当前为 CPU 版本)")

# CPU 张量备用验证
cpu_tensor = torch.rand(5, 3)
print("\n===== CPU 张量示例 =====")
print(cpu_tensor)