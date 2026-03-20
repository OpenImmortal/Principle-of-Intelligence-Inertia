import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# ==========================================
# 1. 注入实验数据 (5M Params 完整版)
# ==========================================
results = [
    {"name": "MLP",  "A": 0, "B": 0, "v": 0.502, "L": 2.302, "flops": 5.8e11},
    {"name": "BN",  "A": 1, "B": 0, "v": 0.522, "L": 1.389, "flops": 6.87e13},
    {"name": "Res",  "A": 2, "B": 0, "v": 0.818, "L": 1.322, "flops": 1.17e14},
    {"name": "MLP-CNN",  "A": 0, "B": 1, "v": 0.179, "L": 1.485, "flops": 8.30e14},
    {"name": "MLP-MCNN", "A": 0, "B": 2, "v": 0.194, "L": 1.339, "flops": 3.67e15},
    {"name": "BN-CNN",  "A": 1, "B": 1, "v": 0.532, "L": 0.647, "flops": 6.64e14},
    {"name": "Res-CNN",  "A": 2, "B": 1, "v": 0.689, "L": 0.578, "flops": 8.63e14},
    {"name": "BN-MCNN", "A": 1, "B": 2, "v": 0.531, "L": 0.599, "flops": 2.48e15},
    {"name": "Res-MCNN", "A": 2, "B": 2, "v": 0.452, "L": 0.553, "flops": 2.09e15}
]

def plot_topography_and_phase(data):
    A = np.array([d['A'] for d in data])
    B = np.array([d['B'] for d in data])
    L = np.array([d['L'] for d in data])
    V = np.array([d['v'] for d in data])
    F = np.array([d['flops'] for d in data])
    names = [d['name'] for d in data]

    # 插值网格
    ai = np.linspace(0, 2, 100)
    bi = np.linspace(0, 2, 100)
    AI, BI = np.meshgrid(ai, bi)
    ZI = griddata((A, B), L, (AI, BI), method='cubic')

    # ==========================================
    # 图 1: 3D 可达性地形图 (标注具体 Loss)
    # ==========================================
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 表面
    surf = ax.plot_surface(AI, BI, ZI, cmap='terrain', edgecolor='none', alpha=0.7, antialiased=True)
    
    # 散点
    ax.scatter(A, B, L, color='red', s=120, edgecolors='k', alpha=1, depthshade=False)

    # 标注名称 + Loss
    for i in range(len(data)):
        label = f"{names[i]}\nL={L[i]:.3f}"
        ax.text(A[i], B[i], L[i] + 0.1, label, fontsize=1.5*7, fontweight='bold', ha='center')

    # 路径绘制
    zigzag_idx = [0, 3, 5, 7, 8] # MLP-A0 -> CNN-A0 -> CNN-A1 -> MCNN-A1 -> MCNN-A2
    ax.plot(A[zigzag_idx], B[zigzag_idx], L[zigzag_idx], 'b-o', linewidth=4, markersize=8, label='Optimal Evolutionary Path', zorder=20)
    
    ax.set_xlabel(r'Optimization ($R$)', fontsize=1.5*13)
    ax.set_ylabel(r'Optimization ($S_{ext}$)', fontsize=1.5*13)
    ax.set_zlabel(r'Reach. Limit ($\mathcal{L}_{min}$)', fontsize=1.5*13)
    ax.set_title('Reachability Topography', fontsize=1.5*16, pad=30)
    ax.view_init(elev=20, azim=140)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('exp02_topography_loss.pdf', format='pdf', bbox_inches='tight')
    print("Saved: exp02_topography_loss.pdf")
    plt.show()

    # ==========================================
    # 图 2: 3D 效率偏差地形图 (Velocity Deviation Topography)
    # Z轴 = |v - 0.5|, Label 包含 FLOPs
    # ==========================================
    fig2 = plt.figure(figsize=(14, 11))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # 计算偏差 Z 轴: |v - 0.5|
    Z_dev = np.abs(V - 0.5)
    
    # 建立插值网格
    ZI_dev = griddata((A, B), Z_dev, (AI, BI), method='cubic')
    
    # 绘制表面 (使用逆向色系，偏差小的地方显示为平原)
    surf2 = ax2.plot_surface(AI, BI, ZI_dev, cmap='YlOrRd_r', edgecolor='none', alpha=0.6, antialiased=True)
    
    # 绘制原始数据点
    ax2.scatter(A, B, Z_dev, color='blue', s=120, edgecolors='k', alpha=1, depthshade=False)

    # 标注名称 + FLOPs (单位: T)
    for i in range(len(data)):
        tflops = F[i] / 1e12
        # 标签内容：名称与总耗能
        label = f"{names[i]}\nFLOPs={tflops:.1f} T"
        # 调整 text 位置，防止重叠
        ax2.text(A[i], B[i], Z_dev[i] + 0.1, label, fontsize=1.5*7, fontweight='bold', ha='center')

    # 绘制最优进化路径 (Zig-Zag) - 保持蓝色实线
    # 按照数据索引: MLP-A0(0) -> CNN-A0(3) -> CNN-A1(5) -> MCNN-A1(7) -> MCNN-A2(8)
    zigzag_idx = [0, 3, 5, 7, 8]
    ax2.plot(A[zigzag_idx], B[zigzag_idx], Z_dev[zigzag_idx], 'b-o', 
             linewidth=5, markersize=10, label='Optimal Evolutionary Path', zorder=30)
    
    ax2.set_xlabel(r'Optimization ($R$)', fontsize=1.5*13)
    ax2.set_ylabel(r'Optimization ($S_{ext}$)', fontsize=1.5*13)
    ax2.set_zlabel('Vel. Dev. (|v - 0.5|)', fontsize=1.5*13)
    ax2.set_title('Velocity Deviation Topography', fontsize=1.5*16, pad=35)
    
    # 设置 Z 轴范围，突出“河谷”感
    ax2.set_zlim(0, 0.4)
    ax2.view_init(elev=20, azim=230) # 调整视角看清 A-B 平面的演化
    
    # 添加图例
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('exp02_v_deviation_3d.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('exp02_v_deviation_3d.png', dpi=300)
    print("Saved: exp02_v_deviation_3d.pdf")
    plt.show()

if __name__ == "__main__":
    plot_topography_and_phase(results)