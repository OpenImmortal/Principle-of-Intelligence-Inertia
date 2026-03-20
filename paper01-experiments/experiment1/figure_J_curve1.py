import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 注入数据
results = {
    'noise': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.962, 1.0], 
    'velocity': [0.5029751565811376, 0.6310214547051747, 0.714383781900109, 0.7547174121907853, 
                 0.8243982731010415, 0.8616751197467166, 0.8803923562567658, 0.8906682762239302, 
                 0.9173381437234738, 0.9206525786323375, 0.9248294050979938, 0.9250501835214924, 
                 0.9467885967656827, 0.9389990241862245, 0.9426493958265094], 
    'cost': [53, 85, 130, 132, 135, 161, 171, 213, 262, 227, 306, 246, 275, 272, 313]
}

############
# 鲁棒性（参考系无关性）检验 (Robustness Check)
############





def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def plot_robustness_check(results):
    v_raw = np.array(results['velocity'])
    c_raw = np.array(results['cost'])
    
    idx = np.argsort(v_raw)
    v = v_raw[idx]
    c = c_raw[idx]
    
    # 清洗 (去除 Cost 下降点)
    valid_mask = []
    max_c = -1
    for val in c:
        if val >= max_c:
            valid_mask.append(True)
            max_c = val
        else:
            valid_mask.append(False)
    v_cl = v[valid_mask]
    c_cl = c[valid_mask]
    
    v_start = np.min(v_cl)
    # v_cliff = np.max(v_cl) # 观测到的最大速度作为光速
    v_cliff = 1.0  # 使用理论光速作为极限速度 (1.0)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # X轴绘图范围
    x_plot = np.linspace(0.3, v_cliff + 0.02, 500)

    # ====================================================
    # 图 A: FIM 的脆弱性 (Shifted vs Absolute)
    # ====================================================
    ax1.scatter(v_cl, c_cl, s=150, c='red', edgecolors='k', zorder=10, label='Measured Data')
    
    rmse_fs = 0.0
    rmse_fa = 0.0
    
    # 1. FIM Absolute (绝对参考系 - 深绿粗实线)
    try:
        def fim_abs(x, k, b): return k * x**2 + b
        popt_fa, _ = curve_fit(fim_abs, v_cl, c_cl, maxfev=5000)
        y_pred_fa = fim_abs(v_cl, *popt_fa)
        rmse_fa = calc_rmse(c_cl, y_pred_fa)
        
        ax1.plot(x_plot, fim_abs(x_plot, *popt_fa), color='darkgreen', linestyle='-', linewidth=4, alpha=0.6, label=r'FIM-Absolute ($v^2$)')
    except: pass

    # 2. FIM Shifted (平移参考系 - 绿色虚线)
    try:
        def fim_shifted(x, k, b): return k * (x - v_start)**2 + b
        popt_fs, _ = curve_fit(fim_shifted, v_cl, c_cl, maxfev=5000)
        y_pred_fs = fim_shifted(v_cl, *popt_fs)
        rmse_fs = calc_rmse(c_cl, y_pred_fs)
        
        ax1.plot(x_plot, fim_shifted(x_plot, *popt_fs), color='limegreen', linestyle='--', linewidth=2.5, label=r'FIM-Shifted ($(v-v_0)^2$)')
        ax2.axvline(x=c_fit, color='gray', linestyle=':', label=f'Limit c={c_fit:.3f}')
    except: pass
    
    
    # 样式设置
    ax1.set_title("Arena 1: FIM Sensitivity to Frame", fontsize=2*14)
    ax1.set_xlabel('Velocity', fontsize=2*12)
    ax1.set_ylabel('Cost (Epochs)', fontsize=2*12)
    ax1.legend(loc='upper left', fontsize=2*11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(v_start - 0.03, v_cliff + 0.03)
    ax1.set_ylim(0, 450)
    
    # 添加 RMSE Box
    textstr_1 = '\n'.join((
        r'$\bf{Model\ Fit\ Error}$',
        f'RMSE (Absolute): {rmse_fa:.1f}',
        f'RMSE (Shifted): {rmse_fs:.1f}'))
    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.8, edgecolor='green')
    ax1.text(0.03, 0.65, textstr_1, transform=ax1.transAxes, fontsize=2*11,
            verticalalignment='top', bbox=props)

    # ====================================================
    # 图 B: UIT 的鲁棒性 (Shifted vs Absolute)
    # ====================================================
    ax2.scatter(v_cl, c_cl, s=150, c='red', edgecolors='k', zorder=10, label='Measured Data')
    
    rmse_ua = 0.0
    rmse_us = 0.0
    
    # 1. UIT Absolute (绝对参考系 - 蓝色粗实线)
    try:
        def uit_abs(x, k, c_lim, b):
            x_safe = np.clip(x, 0, c_lim * 0.99999)
            return k * (1/np.sqrt(1-(x_safe/c_lim)**2) - 1) + b
        
        bounds = ([0, v_cliff, -np.inf], [np.inf, v_cliff+0.05, np.inf])
        p0 = [100, v_cliff+0.01, min(c_cl)]
        popt_ua, _ = curve_fit(uit_abs, v_cl, c_cl, p0=p0, bounds=bounds, maxfev=5000)
        
        c_fit = popt_ua[1]
        x_uit = np.linspace(0.3, c_fit * 0.999, 500)
        y_pred_ua = uit_abs(v_cl, *popt_ua)
        rmse_ua = calc_rmse(c_cl, y_pred_ua)
        
        ax2.plot(x_uit, uit_abs(x_uit, *popt_ua), color='blue', linestyle='-', linewidth=4, alpha=0.6, label=r'Relativistic Mass-Absolute ($\gamma(v)$)')
        
    except: pass
    
    # 2. UIT Shifted (Lorentz Shift - 紫色虚线)
    try:
        def uit_shifted(x, k, c_abs, b):
            c_safe = max(c_abs, np.max(x) + 1e-5)
            numer = x - v_start
            denom = 1 - (x * v_start) / (c_safe**2)
            v_rel = numer / denom
            
            ratio = v_rel / c_safe
            ratio = np.clip(ratio, 0, 0.9999)
            return k * (1/np.sqrt(1 - ratio**2) - 1) + b
            
        popt_us, _ = curve_fit(uit_shifted, v_cl, c_cl, p0=p0, bounds=bounds, maxfev=10000)
        
        c_fit_s = popt_us[1]
        x_uit_s = np.linspace(0.3, c_fit_s * 0.999, 500)
        y_pred_us = uit_shifted(v_cl, *popt_us)
        rmse_us = calc_rmse(c_cl, y_pred_us)
        
        ax2.plot(x_uit_s, uit_shifted(x_uit_s, *popt_us), color='magenta', linestyle='--', linewidth=2.5, label=r'Relativistic Mass-Shifted ($\gamma(v_{rel})$)')
        
    except Exception as e: print(e)
    
    ax2.set_title("Arena 2: Inertia Model Frame Independence", fontsize=2*14)
    ax2.set_xlabel('Velocity', fontsize=2*12)
    ax2.legend(loc='upper left', fontsize=2*11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(v_start - 0.03, v_cliff + 0.03)
    ax2.set_ylim(0, 450)

    # 添加 RMSE Box
    textstr_2 = '\n'.join((
        r'$\bf{Model\ Fit\ Error}$',
        f'RMSE (Absolute): {rmse_ua:.1f}',
        f'RMSE (Shifted): {rmse_us:.1f}'))
    props_2 = dict(boxstyle='round', facecolor='lavender', alpha=0.8, edgecolor='blue')
    ax2.text(0.03, 0.65, textstr_2, transform=ax2.transAxes, fontsize=2*11,
            verticalalignment='top', bbox=props_2)

    plt.tight_layout()
    plt.savefig('j_curve_figure1.png', dpi=300)
    plt.savefig('j_curve_figure1.pdf')
    plt.show()

if __name__ == "__main__":
    plot_robustness_check(results)