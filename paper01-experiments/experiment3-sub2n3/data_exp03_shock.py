import json
import numpy as np

def generate_shock_test_table(json_path):
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    data = full_data['shock']
    noise_start = 20
    
    print(f"{'Group':<12} | {'Phase':<10} | {'Avg Loss':<10} | {'Avg v_ins':<10} | {'Avg LR':<12} | {'LR Brake Ratio'}")
    print("-" * 85)

    for key in ['baseline', 'regulated']:
        raw_list = sorted(data[key], key=lambda x: x['epoch'])
        
        # 1. 提取三个阶段的数据
        pre_shock = [e for e in raw_list if e['epoch'] < noise_start]
        post_clean = [e for e in raw_list if e['epoch'] >= noise_start and e['epoch'] % 2 != 0]
        post_noisy = [e for e in raw_list if e['epoch'] >= noise_start and e['epoch'] % 2 == 0]

        def get_stats(subset):
            if not subset: return 0, 0, 0
            losses = [e['loss'] for e in subset]
            v_ins = [e['v_ins'] for e in subset]
            lrs = [e['LR'] if 'LR' in e else e['lr'] for e in subset]
            return np.mean(losses), np.mean(v_ins), np.mean(lrs)

        # 计算各阶段统计量
        m_l_pre, m_v_pre, m_lr_pre = get_stats(pre_shock)
        m_l_cl, m_v_cl, m_lr_cl = get_stats(post_clean)
        m_l_ns, m_v_ns, m_lr_ns = get_stats(post_noisy)

        # 计算制动倍率 (Clean LR / Noisy LR)
        brake_ratio = m_lr_cl / (m_lr_ns + 1e-12)

        label = "Regulated" if key == 'regulated' else "Baseline"
        
        print(f"{label:<12} | {'Pre-Shock':<10} | {m_l_pre:<10.4f} | {m_v_pre:<10.4f} | {m_lr_pre:<12.6f} | {'N/A'}")
        print(f"{'':<12} | {'Post-Clean':<10} | {m_l_cl:<10.4f} | {m_v_cl:<10.4f} | {m_lr_cl:<12.6f} | {brake_ratio:>10.1f}x")
        print(f"{'':<12} | {'Post-Noisy':<10} | {m_l_ns:<10.4f} | {m_v_ns:<10.4f} | {m_lr_ns:<12.6f} |")
        print("-" * 85)

if __name__ == "__main__":
    generate_shock_test_table('exp03_final_comparison_composed_limit_results1_pool6.json')