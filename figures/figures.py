import matplotlib.pyplot as plt
import subprocess
import os
import re

# Data extracted from the user input
test_num = 64
sizes = [(i+1)*256 for i in range(test_num)]

pattern = re.compile(r'AVG Performance = \s*(\d+\.\d+) Gflops')

# 执行 cublas gemm
os.chdir("../cublas")
result = subprocess.run(["./a.out"], capture_output=True, text=True)
matches = pattern.findall(result.stdout)
cublas_performances = [float(performance) for performance in matches]

# 执行 gemm_1 下面的 a.out
os.chdir("../gemm_1")
result = subprocess.run(["./a.out"], capture_output=True, text=True)
matches = pattern.findall(result.stdout)
v1_performances = [float(performance) for performance in matches]

# # 执行 gemm_2 下面的 a.out
os.chdir("../gemm_2")
result = subprocess.run(["./a.out"], capture_output=True, text=True)
matches = pattern.findall(result.stdout)
v2_performances = [float(performance) for performance in matches]

# 执行 gemm_3 下面的 a.out
os.chdir("../gemm_3")
result = subprocess.run(["./a.out"], capture_output=True, text=True)
matches = pattern.findall(result.stdout)
v3_performances = [float(performance) for performance in matches]

# 执行 gemm_4 下面的 a.out
os.chdir("../gemm_4")
result = subprocess.run(["./a.out"], capture_output=True, text=True)
matches = pattern.findall(result.stdout)
v4_performances = [float(performance) for performance in matches]

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(sizes, cublas_performances, marker='o', label='Cublas')
plt.plot(sizes, v1_performances, marker='x', label='CUTE-GEMM-V1')
plt.plot(sizes, v2_performances, marker='x', label='CUTE-GEMM-V2')
plt.plot(sizes, v3_performances, marker='x', label='CUTE-GEMM-V3')
plt.plot(sizes, v4_performances, marker='x', label='CUTE-GEMM-V4')
plt.title('Performance by Matrix Size')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Performance (Gflops)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Performance.png')