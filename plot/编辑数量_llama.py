import pandas as pd
import matplotlib.pyplot as plt

# CSV文件路径列表
csv_files = ['en_la.csv', 'en_fa.csv', 'en_qw.csv']
names = ["Llama3-8B", "Falcon3-10B", "Qwen3-14B"]

# 设置图表字体
plt.rc('font', family='Comic Sans MS')

# 创建1x3子图布局（缩小整体图形尺寸）
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))  # 更适合LaTeX文档中插图使用

# 定义颜色与样式
colors = ['#ffdd55', '#55aaff', '#cc66ff', '#77cc77', '#ffaa55', "#ff6666"]
markers = ['o', 'o', 'o', "o", "o"]

# 遍历每个CSV文件和子图
for ax, file, name in zip(axes, csv_files, names):
    df = pd.read_csv(file)
    x_values = df.columns[1:].astype(int)
    models = df['Model']

    for i, model in enumerate(models):
        model_data = df.loc[df['Model'] == model].drop('Model', axis=1).values.flatten()
        ax.plot(x_values, model_data, marker=markers[i % len(markers)],
                label=model, color=colors[i % len(colors)],
                linestyle='-', markersize=6, linewidth=2.5)

    ax.set_title(f'{name}', fontsize=18, weight='bold')
    ax.set_xlabel('Fact numbers', fontsize=18, weight='bold')
    ax.set_ylabel('Performance', fontsize=18, weight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)

    custom_xticks = [2000, 6000, 8000, 10000, 15000, 20000]
    custom_xtick_labels = ['2k', '6k', '8k', '10k', '15k', '20k']
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xtick_labels, fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=18)

# 获取图例信息
handles, labels = axes[0].get_legend_handles_labels()

# 添加统一图例（改为底部更适合排版）
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.08),
           fancybox=True, shadow=False, ncol=len(labels), fontsize=18)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # 给底部图例留空间

# 显示与保存图像
plt.show()
fig.savefig("edit_number.pdf", dpi=300, bbox_inches='tight')