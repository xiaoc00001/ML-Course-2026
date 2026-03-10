# 导入 PyTorch 库
import torch

# 打印 PyTorch 版本（验证安装）
print("PyTorch 版本：", torch.__version__)
# 检查 CUDA 是否可用（GPU 版本显示 True，CPU 版本显示 False）
print("CUDA 是否可用：", torch.cuda.is_available())
# 创建一个 5行3列 的随机张量（验证 PyTorch 功能）
test_tensor = torch.rand(5, 3)
print("随机张量示例：")
print(test_tensor)