import json
import numpy as np
import matplotlib.pyplot as plt

def plot_shock_dynamics(json_path):
    # 1. 加载数据
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    # 提取 shock 实验数据
    data = full_data['shock']
    
    # 定义绘图对象
    # 左图: Loss 轨迹; 右图: LR 响应
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25)

    # 模型配置: (Key, Label, BaseColor)
    models = [
        ('regulated', 'Wrapper', 'blue'),
        ('baseline', 'Exponential', 'orange') # 对应您提到的指数收敛颜色
    ]

    for key, label, color in models:
        raw_list = data[key]
        # 排序确保连线正确
        raw_list = sorted(raw_list, key=lambda x: x['epoch'])
        
        epochs = np.array([e['epoch'] for e in raw_list])
        losses = np.array([e['loss'] for e in raw_list])
        lrs = np.array([e['LR'] if 'LR' in e else e['lr'] for e in raw_list])

        # --- 分类数据点 ---
        # 逻辑：从中点(20)开始，偶数epoch为噪音点
        noise_start = 20
        clean_mask = []
        noisy_mask = []
        
        for e in epochs:
            is_noisy_ep = (e >= noise_start) and (e % 2 == 0)
            clean_mask.append(not is_noisy_ep)
            noisy_mask.append(is_noisy_ep)
            
        clean_mask = np.array(clean_mask)
        noisy_mask = np.array(noisy_mask)

        # --- 绘制连线 ---
        # 1. 绘制正常路径 (Clean Path)
        ls = '-' if key == 'regulated' else '--'
        ax1.plot(epochs[clean_mask], losses[clean_mask], color=color, linestyle=ls, 
                 linewidth=2.5, marker='o', markersize=4, label=f"{label} (Clean)")
        ax2.plot(epochs[clean_mask], lrs[clean_mask], color=color, linestyle=ls, 
                 linewidth=2.5, marker='o', markersize=4, label=f"{label} (Clean)")

        # 2. 绘制噪音冲击路径 (Noisy Path)
        # 只有在噪音开始后才有数据
        if np.any(noisy_mask):
            # 为了让噪音线连起来，需要包含噪音开始前的最后一个点
            # 但为了清晰，我们直接绘制受灾点的趋势
            ax1.plot(epochs[noisy_mask], losses[noisy_mask], color='red', linestyle=ls, 
                     linewidth=1.5, marker='x', markersize=6, alpha=0.8, label=f"{label} (Noisy Step)")
            ax2.plot(epochs[noisy_mask], lrs[noisy_mask], color='red', linestyle=ls, 
                     linewidth=1.5, marker='x', markersize=6, alpha=0.8)

    # --- 格式化图表 ---
    
    # 标注噪音注入线
    ax1.axvline(x=20, color='gray', linestyle=':', alpha=0.7)
    ax1.text(20.5, ax1.get_ylim()[1]*0.83, 'Noise Injection Start', color='gray', fontsize=1.7*10, fontstyle='italic')
    
    # 左图 (Loss)
    # ax1.set_ylim(1.20, 2.50) # 根据实际调整，确保能看到2.3左右的起始点
    ax1.set_xlabel("Epochs", fontsize=1.7*12)
    ax1.set_ylabel("Epoch Loss", fontsize=1.7*12)
    ax1.set_title("Loss Dynamic Response", fontsize=1.7*14)
    ax1.legend(fontsize=1.7*9, loc='upper left')
    ax1.grid(True, alpha=0.2)

    # 右图 (LR)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-4, 1e-1)
    ax2.set_xlabel("Epochs", fontsize=1.7*12)
    ax2.set_ylabel("Step Length (LR - Log Scale)", fontsize=1.7*12)
    ax2.set_title("LR Dynamic Response", fontsize=1.7*14)
    ax2.grid(True, which="both", alpha=0.2)
    
    plt.suptitle("Logical Resilience and Protective Braking under Noise Shock", 
                 fontsize=1.7*18, y=1.02)

    # 导出
    output_name = 'exp03_shock_test_analysis_01'
    plt.tight_layout()
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    print(f"✅ Shock dynamics plot saved to {output_name}.pdf")
    plt.show()
def plot_velocity_zoom_comparison(json_path):
    # 1. 加载数据
    try:
        with open(json_path, 'r') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return

    data = full_data['shock']
    noise_start = 20

    # 2. 创建画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # 配置两组数据：key, label, axis, 颜色
    groups = [
        ('regulated', 'Experimental Group (Scheduler Wrapper)', ax1, 'blue'),
        ('baseline', 'Control Group (Exponential Scheduler)', ax2, 'orange')
    ]

    for key, label, ax, base_color in groups:
        # 排序确保连线逻辑正确
        raw_list = sorted(data[key], key=lambda x: x['epoch'])
        
        epochs = np.array([e['epoch'] for e in raw_list])
        v_ins = np.array([e['v_ins'] for e in raw_list])

        # --- 分类数据点 ---
        # 干净周期: 20轮前，或者20轮后的奇数轮
        # 噪音周期: 20轮后的偶数轮
        clean_mask = []
        noisy_mask = []
        
        for e in epochs:
            is_noisy_ep = (e >= noise_start) and (e % 2 == 0)
            clean_mask.append(not is_noisy_ep)
            noisy_mask.append(is_noisy_ep)
            
        clean_mask = np.array(clean_mask)
        noisy_mask = np.array(noisy_mask)

        # --- 绘图 ---
        # 1. 绘制干净周期轨迹 (Solid Line)
        ax.plot(epochs[clean_mask], v_ins[clean_mask], color=base_color, linestyle='-', 
                linewidth=2.5, marker='o', markersize=6, label='Clean Signal Phase')
        
        # 2. 绘制噪音冲击轨迹 (Red Dashed Line)
        if np.any(noisy_mask):
            ax.plot(epochs[noisy_mask], v_ins[noisy_mask], color='red', linestyle='--', 
                    linewidth=2, marker='x', markersize=8, alpha=0.9, label='Noise Shock Phase')

        # --- 细节装饰 ---
        # 注入线
        ax.axvline(x=noise_start, color='black', linestyle=':', alpha=0.5)
        
        ax.set_title(label, fontsize=1.7*14)
        ax.set_xlabel("Epochs", fontsize=1.7*12)
        
        # [核心修改] 设置 Y 轴范围 0.4 到 0.9
        ax.set_ylim(0.4, 1) 
        
        ax.grid(True, alpha=0.2, linestyle='--')
        
        if ax == ax1:
            ax.set_ylabel(r"Velocity $v$", fontsize=1.7*13)
        
        ax.legend(loc='upper left', fontsize=1.7*10, frameon=True, framealpha=0.9)

    # 3. 整体美化与导出
    plt.suptitle("Impact of Pulsed Noise on Velocity", 
                 fontsize=1.7*18, y=1.02)

    output_name = 'exp03_shock_test_analysis_02'
    plt.tight_layout()
    
    # 导出高清矢量图
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    
    print(f"✅ Zoomed velocity comparison plot saved as {output_name}.pdf")
    plt.show()

if __name__ == "__main__":
    # 使用您的文件名运行
    filename = 'exp03_final_comparison_composed_limit_results1_pool6.json'
    plot_shock_dynamics('exp03_final_comparison_composed_limit_results1_pool6.json')
    plot_velocity_zoom_comparison(filename)