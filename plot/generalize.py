import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
df = pd.read_csv('general.csv')
models = df['Model'].tolist()
edit_numbers = df.columns[1:].tolist()
data = {model: df[df['Model'] == model].iloc[:, 1:].values.flatten().tolist() for model in models}

# 颜色 & 标记
colors = {
    'GLAME': '#ffdd55',       # 道奇蓝
    'MEMIT': '#55aaff',       # 紫罗兰
    'PMET': '#cc66ff',        # 中海绿
    'AlphaEdit': '#77cc77',   # 浅海蓝
    'CoT2Edit': '#FFA07A'     # 浅鲑红
}
markers = {
    'RoME': 'o',
    'GLAME': 'o',
    'MEMIT': 'o',
    'PMET': 'o',
    'AlphaEdit': 'o',
    'CoT2Edit': 'o'
}

# 设置字体
plt.rc('font', family='Comic Sans MS')

# 创建雷达图（比例缩小以适配论文版面）
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))  # 适合单栏图

# 设置角度 & 闭合
angles = [n / float(len(edit_numbers)) * 2 * np.pi for n in range(len(edit_numbers))]
angles += angles[:1]

# 绘图
for model in data:
    values = [v / 100 for v in data[model]]
    values += values[:1]
    ax.plot(angles, values, label=model, color=colors[model], marker=markers[model],
            linewidth=2.5, markersize=6)
    ax.fill(angles, values, alpha=0.25, color=colors[model])

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels('', fontsize=14, fontweight='bold')

# 设置半径标签（百分比）
yticks = np.arange(0.2, 0.9, 0.2)
ax.set_yticks(yticks)
ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in yticks], fontsize=12)
ax.set_ylim(0, 0.88)

# 网格线 & 背景色
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
ax.set_facecolor('#e6f2ff')  # 浅蓝色背景 (#e6f2ff)

for label, angle in zip(edit_numbers, angles[:-1]):
    x = angle
    if label == 'MRPC' or label == 'CoLA':
        y = ax.get_rmax() + 0.18 # 径向坐标，越大离圆心越远（可微调）
    else:
        y = ax.get_rmax() + 0.1
    ax.text(x, y, label, size=18, weight='bold', ha='center', va='center')

# 图例（右侧紧凑）
legend = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.43),ncol=2, fontsize=12)
# legend.get_frame().set_alpha(0.8)

# 标题
plt.title('General ability', fontsize=20, fontweight='bold', pad=20)

# 紧凑布局 & 显示
plt.tight_layout()
plt.savefig("general.png", dpi=300, bbox_inches='tight')
plt.show()
