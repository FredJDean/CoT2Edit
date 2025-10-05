import matplotlib.pyplot as plt
import numpy as np


def add_growth_arrow(start_idx, end_idx, x_move, y_move,  color='gray'):
    ax.annotate('', xy=(x[end_idx], actor_acc[end_idx]), xytext=(x[start_idx], actor_acc[start_idx]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    diff = actor_acc[end_idx] - actor_acc[start_idx]
    ax.text((x[start_idx] + x[end_idx]) / 2-x_move, actor_acc[end_idx] - y_move,
            f'+{diff:.1f}%', ha='center', fontsize=16, color=color, weight='bold')

plt.rcParams['font.family'] = 'serif'

methods = [
    "RoME", "MEMIT", "PMET", "GLAME", "AlphaEdit", "CoT2Edit"
]

x = np.arange(len(methods))
actor_acc = [15.34, 18.49, 23.69, 20.28, 32.55, 73.13]
bar_width = 0.35

hatches = ['///', 'xx', '++', '///', 'xx', '++']
colors = ['red', 'blue', 'green', 'cyan', 'deepskyblue', 'purple']

fig, ax = plt.subplots(figsize=(6, 5))
plt.rc('font', family='Comic Sans MS')
# fig.patch.set_alpha(0.)
# 绘制空心柱状图
for i in range(len(x)):
    ax.bar(x[i], actor_acc[i], width=bar_width,
           facecolor='none', edgecolor=colors[i],
           hatch=hatches[i], linewidth=2)

ax.set_title('(III) Test of locality ability (ZsRE)', fontsize=16, fontweight='bold')
ax.set_ylabel('Locality ability (%)',fontsize=14, weight='bold')
# plt.tick_params(labelsize=14)
ax.set_ylim(0, 80)
ax.set_xticks(x)
ax.set_xticklabels([])

# add_growth_arrow(2, 3,0.2,3, color='gray')
# add_growth_arrow(1, 2, 0.2,3, color='red')
add_growth_arrow(4, 5, 0.2, 20,color='blue')
# 添加网格
ax.grid(True, linestyle='--', alpha=0.6)

# 添加每个柱顶上的数值标签
for i, v in enumerate(actor_acc):
    ax.text(x[i], v + 0.8, f"{v:.2f}", ha='center', fontsize=12)

# 添加箭头和注释强调 CoT2Edit 的优势
cot_index = 5
alpha_index = 3

# 图例放入图内空白处
handles = [plt.Rectangle((0, 0), 1, 1, facecolor='none',
                         edgecolor=colors[i], hatch=hatches[i], linewidth=2)
           for i in range(6)]

labels = methods
fig.legend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0.83),
           ncol=3, fontsize=13, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig('xovee.png', transparent=True)
plt.show()
