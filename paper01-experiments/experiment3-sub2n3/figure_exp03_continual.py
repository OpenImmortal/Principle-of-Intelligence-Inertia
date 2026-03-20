import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_continual_learning_analysis(json_path):
    # 1. 加载数据
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    # 提取 continual 实验数据
    data = full_data['continual']
    switch_epoch = 20

    # 2. 创建画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25)

    # 模型配置
    models = [
        ('regulated', 'Wrapper', 'blue', '-'),
        ('baseline', 'Exponential', 'orange', '--')
    ]

    for key, label, color, ls in models:
        raw_list = sorted(data[key], key=lambda x: x['epoch'])
        
        epochs = np.array([e['epoch'] for e in raw_list])
        loss_old = np.array([e['loss_old'] for e in raw_list])
        loss_full = np.array([e['loss_full'] for e in raw_list])
        lrs = np.array([e['lr'] for e in raw_list])

        # --- 左图: Loss 曲线 ---
        # 1. 绘制 loss_old (旧任务保持)
        ax1.plot(epochs, loss_old, color=color, linestyle=ls, linewidth=2.5, 
                 marker='o', markersize=4, label=f"{label} (Old Task)")
        
        # 2. 绘制 loss_full (后半段全量评估)
        # 筛选 epoch >= 20 的数据
        full_mask = epochs >= switch_epoch
        ax1.plot(epochs[full_mask], loss_full[full_mask], color=color, linestyle=':', 
                 linewidth=2, marker='s', markersize=4, alpha=0.7, label=f"{label} (Full Task)")

        # --- 右图: LR 曲线 ---
        ax2.plot(epochs, lrs, color=color, linestyle=ls, linewidth=2.5, 
                 marker='d', markersize=4, label=label)

    # --- 格式化图表 ---

    # 标注任务切换线
    for ax in [ax1, ax2]:
        ax.axvline(x=switch_epoch, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(switch_epoch + 0.5, ax.get_ylim()[1]*0.88, 'Task Switch (0-4 → 5-9)', 
                color='gray', fontsize=1.7*10)

    # 左图 (Loss) 设置
    ax1.set_xlabel("Epochs", fontsize=1.7*12)
    ax1.set_ylabel("Loss", fontsize=1.7*12)
    ax1.set_title("Loss Dynamic Response", fontsize=1.7*14)
    ax1.legend(fontsize=1.7*9, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # 右图 (LR) 设置
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1e-1) # 按照要求设置在 -1 到 -7 之间
    ax2.set_xlabel("Epochs", fontsize=1.7*12)
    ax2.set_ylabel("Step Length (LR)", fontsize=1.7*12)
    ax2.set_title("LR Dynamic Response", fontsize=1.7*14)
    ax2.grid(True, which="both", alpha=0.2)
    
    plt.suptitle("Inertial Barrier in Abrupt Task Transitions", 
                 fontsize=1.7*18, y=1.02)

    # 导出
    output_name = 'exp03_continual_learning_analysis'
    plt.tight_layout()
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    print(f"✅ Continual learning analysis plot saved as {output_name}.pdf")
    plt.show()

if __name__ == "__main__":
    # 执行绘图
    plot_continual_learning_analysis('exp03_final_comparison_composed_limit_results2_pool6.json')