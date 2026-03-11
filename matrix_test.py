# matrix_test.py - 矩阵乘法运算（支持 CPU/GPU）
import torch

# 定义矩阵 A 和 B（浮点型张量，避免整数运算精度问题）
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# ===== CPU 矩阵乘法 =====
C_cpu = torch.matmul(A, B)
print("===== CPU 矩阵运算 =====")
print("矩阵 A:\n", A)
print("矩阵 B:\n", B)
print("运算结果 C = A * B (CPU):\n", C_cpu)

# ===== GPU 矩阵乘法（利用 RTX 4060 算力）=====
if torch.cuda.is_available():
    # 将矩阵移到 GPU 上
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    # GPU 上执行矩阵乘法
    C_gpu = torch.matmul(A_gpu, B_gpu)
    print("\n===== GPU 矩阵运算（RTX 4060） =====")
    print("矩阵 A (GPU):\n", A_gpu)
    print("矩阵 B (GPU):\n", B_gpu)
    print("运算结果 C = A * B (GPU):\n", C_gpu)
else:
    print("\n未检测到 GPU，仅执行 CPU 运算")