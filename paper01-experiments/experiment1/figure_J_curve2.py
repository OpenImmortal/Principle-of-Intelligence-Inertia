import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# 数据注入
results = {
    'noise': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.962, 1.0], 
    'velocity': [0.5029751565811376, 0.6310214547051747, 0.714383781900109, 0.7547174121907853, 
                 0.8243982731010415, 0.8616751197467166, 0.8803923562567658, 0.8906682762239302, 
                 0.9173381437234738, 0.9206525786323375, 0.9248294050979938, 0.9250501835214924, 
                 0.9467885967656827, 0.9389990241862245, 0.9426493958265094], 
    'cost': [53, 85, 130, 132, 135, 161, 171, 213, 262, 227, 306, 246, 275, 272, 313]
}

def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_final_4_curves(results):
    v_raw = np.array(results['velocity'])
    c_raw = np.array(results['cost'])
    
    # 排序 & 清洗 (去除 Cost 下降点)
    idx = np.argsort(v_raw)
    v = v_raw[idx]
    c = c_raw[idx]
    
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
    v_cliff = 1.0 # 理论的最大速度作为光速
    
    # 设置画布：1行2列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 绘图用的 X 轴
    x_plot = np.linspace(v_start * 0.98, v_cliff + 0.015, 500)

    # =======================================================
    # 图一：经典参考系 (Galilean Frame: v' = v - v0)
    # =======================================================
    ax1.scatter(v_cl, c_cl, s=120, c='red', edgecolors='k', zorder=10, label='Measured Data')
    ax1.set_title("Arena 3: Classical ($v_{rel} = v - v_0$)", fontsize=2*14)
    ax1.set_xlabel(r'Velocity $v$', fontsize=2*12)
    ax1.set_ylabel('Cost (Epochs)', fontsize=2*12)
    
    rmse_f1 = 0.0
    rmse_u1 = 0.0
    
    # --- Curve 1: FIM (Galilean) ---
    try:
        def fim_gal(x, k, b):
            return k * (x - v_start)**2 + b
        popt_f1, _ = curve_fit(fim_gal, v_cl, c_cl, maxfev=5000)
        y_pred_f1 = fim_gal(v_cl, *popt_f1)
        rmse_f1 = calc_rmse(c_cl, y_pred_f1)
        ax1.plot(x_plot, fim_gal(x_plot, *popt_f1), color='limegreen', linestyle='--', linewidth=2.5, label=r'FIM')
    except: pass
    
    # --- Curve 2: UIT (Galilean) ---
    try:
        def uit_gal(x, k, c_lim, b):
            v_rel = x - v_start
            x_safe = np.clip(v_rel, 0, c_lim * 0.9999)
            return k * (1/np.sqrt(1-(x_safe/c_lim)**2) - 1) + b
        
        rel_cliff = v_cliff - v_start
        bounds = ([0, rel_cliff+1e-5, -np.inf], [np.inf, rel_cliff+0.05, np.inf])
        p0 = [100, rel_cliff+0.01, min(c_cl)]
        popt_u1, _ = curve_fit(uit_gal, v_cl, c_cl, p0=p0, bounds=bounds, maxfev=5000)
        
        y_pred_u1 = uit_gal(v_cl, *popt_u1)
        rmse_u1 = calc_rmse(c_cl, y_pred_u1)
        
        x_plot_rel = x_plot - v_start
        mask = x_plot_rel < popt_u1[1]
        ax1.plot(x_plot[mask], uit_gal(x_plot[mask], *popt_u1), color='blue', linestyle='-', linewidth=3, label=r'Relativistic Mass')
    except: pass
    
    # Box 1
    textstr_1 = '\n'.join((
        r'$\bf{Model\ Fit\ Error}$',
        f'FIM: RMSE={rmse_f1:.1f}',
        f'Relativistic Mass: RMSE={rmse_u1:.1f}'))
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.8, edgecolor='blue')
    ax1.text(0.03, 0.65, textstr_1, transform=ax1.transAxes, fontsize=2*11,
            verticalalignment='top', bbox=props)
    
    ax1.legend(loc='upper left', fontsize=2*11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(v_start - 0.02, v_cliff + 0.03)
    ax1.set_ylim(0, 450)

    # =======================================================
    # 图二：相对论参考系 (Lorentz Frame)
    # =======================================================
    ax2.scatter(v_cl, c_cl, s=120, c='red', edgecolors='k', zorder=10, label='Measured Data')
    ax2.set_title("Arena 4: Relativistic (Lorentz Transform)", fontsize=2*14)
    ax2.set_xlabel(r'Velocity $v$', fontsize=2*12)
    
    rmse_f2 = 0.0
    rmse_u2 = 0.0
    
    # --- Curve 3: Hybrid FIM (Quadratic on Lorentz Velocity) ---
    try:
        def fim_lorentz(x, k, c_abs, b):
            c_safe = max(c_abs, np.max(x) + 1e-5)
            numer = x - v_start
            denom = 1 - (x * v_start) / (c_safe**2)
            v_rel = numer / denom
            return k * v_rel**2 + b
        
        bounds_l = ([0, v_cliff+1e-5, -np.inf], [np.inf, v_cliff+0.05, np.inf])
        p0_l = [100, v_cliff+0.01, min(c_cl)]
        popt_f2, _ = curve_fit(fim_lorentz, v_cl, c_cl, p0=p0_l, bounds=bounds_l, maxfev=10000)
        
        y_pred_f2 = fim_lorentz(v_cl, *popt_f2)
        rmse_f2 = calc_rmse(c_cl, y_pred_f2)
        
        ax2.plot(x_plot, fim_lorentz(x_plot, *popt_f2), color='limegreen', linestyle='--', linewidth=2.5, label=r'Hybrid FIM')
    except Exception as e: print(e)
    
    # --- Curve 4: UIT Full (Relativistic) ---
    try:
        def uit_lorentz(x, k, c_abs, b):
            c_safe = max(c_abs, np.max(x) + 1e-5)
            numer = x - v_start
            denom = 1 - (x * v_start) / (c_safe**2)
            v_rel = numer / denom
            ratio = v_rel / c_safe
            ratio = np.clip(ratio, 0, 0.9999)
            gamma = 1.0 / np.sqrt(1 - ratio**2)
            return k * (gamma - 1) + b
        
        popt_u2, _ = curve_fit(uit_lorentz, v_cl, c_cl, p0=p0_l, bounds=bounds_l, maxfev=10000)
        
        y_pred_u2 = uit_lorentz(v_cl, *popt_u2)
        rmse_u2 = calc_rmse(c_cl, y_pred_u2)
        
        c_fit = popt_u2[1]
        x_plot_cut = x_plot[x_plot < c_fit]
        ax2.plot(x_plot_cut, uit_lorentz(x_plot_cut, *popt_u2), color='blue', linestyle='-', linewidth=3, label=r'Relativistic Mass')
        # ax2.axvline(x=c_fit, color='gray', linestyle=':', label=f'Limit c={c_fit:.3f}')
        
    except: pass

    # Box 2
    textstr_2 = '\n'.join((
        r'$\bf{Model\ Fit\ Error}$',
        f'Hybrid FIM: RMSE={rmse_f2:.1f}',
        f'Relativistic Mass: RMSE={rmse_u2:.1f}'))
    props_2 = dict(boxstyle='round', facecolor='lavender', alpha=0.8, edgecolor='blue')
    ax2.text(0.03, 0.65, textstr_2, transform=ax2.transAxes, fontsize=2*11,
            verticalalignment='top', bbox=props_2)

    ax2.legend(loc='upper left', fontsize=2*11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(v_start - 0.02, v_cliff + 0.03)
    ax2.set_ylim(0, 450)

    plt.tight_layout()
    plt.savefig('j_curve_figure2.png', dpi=300)
    plt.savefig('j_curve_figure2.pdf')
    plt.show()

if __name__ == "__main__":
    plot_final_4_curves(results)