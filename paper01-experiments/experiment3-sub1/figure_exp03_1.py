import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_experiment_03_grid(path_false, path_true):
    # 1. 加载数据
    with open(path_false, 'r') as f:
        data_f = json.load(f)
    with open(path_true, 'r') as f:
        data_t = json.load(f)

    # 定义 4 行的调度器分配
    rows_config = [
        [('cosine', 'Cosine Annealing'), ('cosine_restart', 'Cosine Restart')],
        [('plateau', 'Reduce On Plateau'), ('one_cycle', 'OneCycle Policy')],
        [('multistep', 'Multi-Step Decay'), ('cyclic', 'Cyclic LR')],
        [('exponential', 'Exponential'), ('polynomial', 'Polynomial')]
    ]

    # 设置 8 种高对比度颜色
    colors = plt.cm.get_cmap('tab10', 8)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 24), sharex=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # 提取纯 Inertia 数据用于背景参考
    pure_inertia = data_f['inertia_regulator']
    p_epochs = [e['epoch'] for e in pure_inertia]
    p_losses = [e['min_loss'] for e in pure_inertia]
    p_lrs = [e['lr'] for e in pure_inertia]

    color_idx = 0
    for row_idx, configs in enumerate(rows_config):
        ax_loss = axes[row_idx, 0]
        ax_lr = axes[row_idx, 1]

        # 在 Loss 图中绘制深灰色 Pure Inertia 基准
        ax_loss.plot(p_epochs, p_losses, color='#4F4F4F', linewidth=3, 
                     alpha=0.4, label='Pure Inertia (Wrapper Only)', zorder=1)
        
        # 在 LR 图中绘制基准
        ax_lr.plot(p_epochs, p_lrs, color='#4F4F4F', linewidth=2, alpha=0.4)

        for key_base, label_name in configs:
            color = colors(color_idx)
            key_f = f"{key_base}_baseline"
            
            # --- 绘制对照组 (False - 虚线) ---
            if key_f in data_f:
                df = data_f[key_f]
                ax_loss.plot([e['epoch'] for e in df], [e['min_loss'] for e in df], 
                             color=color, linestyle='--', linewidth=1.5, label=f'{label_name} (Base)')
                ax_lr.plot([e['epoch'] for e in df], [e['lr'] for e in df], 
                           color=color, linestyle='--', linewidth=1.5)

            # --- 绘制实验组 (True - 实线) ---
            if key_f in data_t:
                dt = data_t[key_f]
                ax_loss.plot([e['epoch'] for e in dt], [e['min_loss'] for e in dt], 
                             color=color, linestyle='-', linewidth=2.5, label=f'{label_name} + Wrapper')
                ax_lr.plot([e['epoch'] for e in dt], [e['lr'] for e in dt], 
                           color=color, linestyle='-', linewidth=2.5)
            
            color_idx += 1

        # --- 格式化左列 (Loss) ---
        ax_loss.set_ylim(1.200, 1.800)
        ax_loss.axvline(x=30, color='red', linestyle=':', alpha=0.6) # 判决线
        if row_idx == 0:
            ax_loss.set_title("Reachability Limit ($\mathcal{L}_{min}$)", fontsize=1.5*16)
        ax_loss.set_ylabel("Loss", fontsize=1.8*12)
        ax_loss.legend(loc='upper right', fontsize=1.8*9, framealpha=0.8)
        ax_loss.grid(True, alpha=0.2)

        # --- 格式化右列 (LR) ---
        ax_lr.set_yscale('log')
        ax_lr.set_ylim(1e-6, 1e-1)
        if row_idx == 0:
            ax_lr.set_title("Learning Rate Dynamics", fontsize=1.5*16)
        ax_lr.set_ylabel("LR (Log Scale)", fontsize=1.8*12)
        ax_lr.grid(True, which='both', alpha=0.2)
        
    # 设置底部 X 轴标签
    axes[3, 0].set_xlabel("Epochs", fontsize=1.8*14)
    axes[3, 1].set_xlabel("Epochs", fontsize=1.8*14)

    # 导出
    plt.suptitle("Universal Enhancement of Learning Dynamics", 
                 fontsize=1.4*22, y=0.92)
    
    print("Saving multi-panel adjudication plots...")
    plt.savefig('exp03_matrix_adjudication.png', dpi=300, bbox_inches='tight')
    plt.savefig('exp03_matrix_adjudication.pdf', format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_experiment_03_grid('experiment03_composed_false.json', 'experiment03_composed_true.json')