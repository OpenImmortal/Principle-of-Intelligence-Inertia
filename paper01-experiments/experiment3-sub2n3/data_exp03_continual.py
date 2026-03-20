import json
import numpy as np

def generate_continual_learning_table(json_path):
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    data = full_data['continual']
    switch_epoch = 20
    final_epoch = 39

    print(f"{'Metric':<25} | {'Baseline':<12} | {'Regulated':<12} | {'Improvement'}")
    print("-" * 75)

    def get_val(group, epoch, key):
        entry = next((e for e in data[group] if e['epoch'] == epoch), None)
        if entry:
            return entry.get(key, 0)
        return 0

    # 1. 记忆保持评估 (Old Task Loss)
    l_old_start = get_val('regulated', switch_epoch - 1, 'loss_old') # 切换前夕
    l_old_base_end = get_val('baseline', final_epoch, 'loss_old')
    l_old_reg_end = get_val('regulated', final_epoch, 'loss_old')
    
    # 计算遗忘增量 (越小越好)
    forgetting_base = l_old_base_end - l_old_start
    forgetting_reg = l_old_reg_end - l_old_start
    retention_gain = (forgetting_base - forgetting_reg) / forgetting_base * 100

    # 2. 最终综合性能 (Full Task Loss)
    l_full_base = get_val('baseline', final_epoch, 'loss_full')
    l_full_reg = get_val('regulated', final_epoch, 'loss_full')
    accuracy_gain = (l_full_base - l_full_reg) / l_full_base * 100

    # 3. 物理响应 (LR Brake at Switch)
    lr_pre = get_val('regulated', switch_epoch - 1, 'lr')
    lr_switch = get_val('regulated', switch_epoch, 'lr')
    brake_magnitude = lr_pre / (lr_switch + 1e-12)

    # baseline 制动默认为 衰减率

    b_lr_pre = get_val('baseline', switch_epoch - 1, 'lr')
    b_lr_switch = get_val('baseline', switch_epoch, 'lr')
    b_brake_magnitude = b_lr_pre / (b_lr_switch + 1e-12)

    print(f"{'Old Task Loss (Ep 19)':<25} | {get_val('baseline', 19, 'loss_old'):<12.4f} | {l_old_start:<12.4f} | {'Baseline'}")
    print(f"{'Old Task Loss (Ep 39)':<25} | {l_old_base_end:<12.4f} | {l_old_reg_end:<12.4f} | {((l_old_base_end-l_old_reg_end)/l_old_base_end*100):>10.2f}% Lower")
    print(f"{'Retention Deficit (ΔL)':<25} | {forgetting_base:<12.4f} | {forgetting_reg:<12.4f} | {retention_gain:>10.2f}% Protected")
    print("-" * 75)
    print(f"{'Full Task Loss (Ep 39)':<25} | {l_full_base:<12.4f} | {l_full_reg:<12.4f} | {accuracy_gain:>10.2f}% Better")
    print(f"{'LR Brake at Switch':<25} | {b_brake_magnitude:<12.2f}x | {brake_magnitude:<12.2f}x | {'Inst. Freeze'}")

if __name__ == "__main__":
    generate_continual_learning_table('exp03_final_comparison_composed_limit_results2_pool6.json')