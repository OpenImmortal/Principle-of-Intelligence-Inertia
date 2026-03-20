import json
import numpy as np

def generate_experiment_03_detailed_table(path_false, path_true):
    # 1. 加载数据
    with open(path_false, 'r') as f:
        data_f = json.load(f)
    with open(path_true, 'r') as f:
        data_t = json.load(f)

    # 定义所有模型
    schedulers = [
        ('cosine', 'Cosine Annealing'),
        ('cosine_restart', 'Cosine Restart'),
        ('plateau', 'Reduce On Plateau'),
        ('one_cycle', 'OneCycle Policy'),
        ('multistep', 'Multi-Step Decay'),
        ('cyclic', 'Cyclic LR'),
        ('exponential', 'Exponential'),
        ('polynomial', 'Polynomial')
    ]

    # 获取初始状态（假设所有模型起始 Loss 约为 2.3026，即 ln(10)）
    # 或者从数据中提取第一个 epoch 的 loss
    L_start = 2.3026 

    print(f"{'Scheduler':<20} | {'Type':<10} | {'Loss@30':<8} |  {'Progress@30%':<9} |  {'Dyn.Acc%':<9} | {'L_min':<8} | {'Thermo.Gain%':<12} | {'Best Ep':<8}")
    print("-" * 105)

    # 处理纯 Inertia 标杆
    pure_ir = data_f['inertia_regulator']
    p_30 = next((e['min_loss'] for e in pure_ir if e['epoch'] == 30), pure_ir[-1]['min_loss'])
    p_min = min([e['min_loss'] for e in pure_ir])
    p_best_ep = next(e['epoch'] for e in pure_ir if e['min_loss'] == p_min)
    p_progress_30 = (L_start - p_30) / (L_start - p_min) * 100 if p_min < L_start else 0

    print(f"{'Pure Inertia (Ref)':<20} | {'Regulated':<10} | {p_30:<8.4f} | {p_progress_30:>7.2f}% | {'N/A':<9} | {p_min:<8.4f} | {'N/A':<12} | {p_best_ep:<8}")
    print("-" * 105)

    for key_base, label_name in schedulers:
        key = f"{key_base}_baseline"
        if key not in data_f or key not in data_t: continue

        # 对照组 (Base)
        df = data_f[key]
        f_30 = next((e['min_loss'] for e in df if e['epoch'] == 30), df[-1]['min_loss'])
        f_min = min([e['min_loss'] for e in df])
        f_best_ep = next(e['epoch'] for e in df if e['min_loss'] == f_min)
        f_progress_30 = (L_start - f_30) / (L_start - f_min) * 100 if f_min < L_start else 0

        # 实验组 (+Inertia)
        dt = data_t[key]
        t_30 = next((e['min_loss'] for e in dt if e['epoch'] == 30), dt[-1]['min_loss'])
        t_min = min([e['min_loss'] for e in dt])
        t_best_ep = next(e['epoch'] for e in dt if e['min_loss'] == t_min)
        t_progress_30 = (L_start - t_30) / (L_start - t_min) * 100 if t_min < L_start else 0

        # 1. 计算动力学加速率 (Dynamical Acceleration at Ep 30)
        # 定义为在 Ep30 处，由于惯性修正多降低的 Loss 比例
        # DA = (Loss_base_30 - Loss_reg_30) / (Loss_start - Loss_base_30)
        # 这里为了直观，直接使用 (L_base - L_reg) / L_base
        

        dyn_acc = (f_30 - t_30) / f_30 * 100



        # 2. 计算热力学下限增益 (Thermodynamic Floor Gain)
        # 定义为收敛极限的提升
        thermo_gain = (f_min - t_min) / f_min * 100

        # 输出格式
        print(f"{label_name:<20} | {'Base':<10} | {f_30:<8.4f} |  {f_progress_30:>7.2f}% |  {'-':<9} | {f_min:<8.4f} | {'-':<12} | {f_best_ep:<8}")
        print(f"{'':<20} | {'+Inertia':<10} | {t_30:<8.4f} | {t_progress_30:>7.2f}% |  {dyn_acc:>7.2f}% | {t_min:<8.4f} | {thermo_gain:>10.2f}% | {t_best_ep:<8}")
        print("-" * 105)

if __name__ == "__main__":
    generate_experiment_03_detailed_table('experiment03_composed_false.json', 'experiment03_composed_true.json')