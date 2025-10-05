import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
models = {
    "MEMIT":       [0.65, 0.55, 0.51, 0.35, 0, 0, 0, 0, 0],
    "GLAME":       [0.65, 0.29, 0, 0, 0, 0, 0, 0, 0],
    "PMET":        [0.65, 0.59, 0.52, 0.43, 0.37, 0.25, 0.13, 0, 0],
    "AlphaEdit":   [0.65, 0.58, 0.55, 0.52, 0.47, 0.45, 0.43, 0.4, 0.35],
    "CoT2Edit":    [0.65, 0.62, 0.63, 0.62, 0.62, 0.61, 0.62, 0.61, 0.6]
}

# 颜色（与图像中类似）
colors = {
    "MEMIT": "#ff6e9b",
    "GLAME": "#6e4fa2",
    "PMET": "#4c9db2",
    "AlphaEdit": "#f5a623",
    "CoT2Edit": "#f8e71c"
}

# 图形风格设置
plt.figure(figsize=(8, 5))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

for model, y in models.items():
    plt.plot(x, y, label=model, marker='o', linewidth=2.5,
             color=colors.get(model, None))

# 坐标轴 & 标签设置
plt.xlabel("Number of Facts", fontsize=16, fontweight='bold')
plt.ylabel("F1 Score (MMLU)", fontsize=16, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, 0.7)
plt.xlim(0, 4100)

# 图例设置
plt.legend(fontsize=11, frameon=False)

# 去除顶部和右侧边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图
plt.tight_layout()
plt.savefig("general_number.png", dpi=300, bbox_inches='tight')
plt.show()
