import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from PIL import Image


# ================= 实验参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0] # 包含 1.0 极限
# NOISE_LEVELS = [ 1.0] # 包含 1.0 极限

DATA_SUBSET_RATIO = 0.1 
BATCH_SIZE = 128
MAX_EPOCHS = 1500           # 稍微增加轮数
WARMUP_EPOCHS = 3
LR = 0.05
WEIGHT_DECAY = 5e-4 
MOMENTUM = 0.9
TARGET_LOSS = 1.0
FAILURE_THRESHOLD = 1.0   # 稍微放宽，因为高噪声下 Loss 很高
PATIENCE = 50
WINDOW_SIZE = 10 
MAX_SAMPLING_NODES = 25

epsilon = 0.0001 # 用于渐近线夹逼
# ===========================================

class OrthogonalPhysicsTracker:
    """
    R-S 相对论追踪器 (最终物理版)
    -------------------------------------------------
    物理量定义:
    1. dR (Rule Path): 参数空间路程
    2. dS_ext (External State Shift): 环境有效增益
       dS_ext = (dR * cos * density) / ||D||
    3. ||D|| (Granularity): 解释性空间的最小尺度 (能耗系数)
    4. v (Velocity): dR / (dR + dS_ext)
    -------------------------------------------------
    """
    def __init__(self,model=None, manual_D=None):
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0 # 原始环境增益 (未缩放)
        
        # 全局积分
        self.total_dR = 0.0
        self.total_dS_ext = 0.0
        
        self.is_warmed_up = False
        
        # === [核心物理常数] ===
        # 如果 manual_D 为 None，则在预热结束时自动校准
        self.D = manual_D 
        
        # 预热期统计
        self.warmup_dR = 0.0
        self.warmup_gain = 0.0
        
    def step_batch(self, grad_train_vec, grad_val_vec, lr, val_loss):
        """
        [微观计算]
        """
        # 1. dR (努力值)
        norm_train = torch.norm(grad_train_vec).item()
        step_dR = lr * norm_train
        
        # 2. Raw Gain (原始增益: 方向 * 信息密度)
        dot_prod = torch.dot(grad_train_vec, grad_val_vec).item()
        norm_val = torch.norm(grad_val_vec).item()
        
        if norm_train > 0 and norm_val > 0:
            cosine = dot_prod / (norm_train * norm_val)
        else:
            cosine = 0.0
            
        # 信息密度 = 1 / 熵(Loss)
        entropy = val_loss if val_loss > 1e-4 else 1e-4
        density = 1.0 / entropy
        
        # 原始增益 = dR * 方向 * 密度
        # 注意：这里还没有除以 ||D||，因为 ||D|| 可能还没定出来
        step_raw_gain = step_dR * max(0.0, cosine) * density
        
        if not self.is_warmed_up:
            # 预热期累积数据用于定标
            self.warmup_dR += step_dR
            self.warmup_gain += step_raw_gain
        else:
            # 正式期记录
            self.epoch_dR += step_dR
            self.epoch_raw_gain += step_raw_gain

    def step_epoch(self, current_loss=None):
        if not self.is_warmed_up:
            return 0.0, 0.0, 0.0

        # 应用物理常数 ||D|| 计算 dS_ext
        # dS_ext = Raw_Gain / ||D||
        epoch_dS_ext = self.epoch_raw_gain / self.D
        
        # 累加到全局
        self.total_dR += self.epoch_dR
        self.total_dS_ext += epoch_dS_ext
        
        # 计算相对论速度
        # v = dR / (dR + dS_ext)
        denom = self.total_dR + self.total_dS_ext
        if denom == 0:
            velocity = 0.0
        else:
            velocity = self.total_dR / denom
        
        # 快照
        curr_dR = self.epoch_dR
        curr_dS_ext = epoch_dS_ext
        
        # 重置瞬时器
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0
        
        return velocity, curr_dR, curr_dS_ext

    def start_tracking(self):
        """
        预热结束，定标 ||D||
        """
        # 如果没有手动指定 D，则进行自动校准
        if self.D is None:
            # 自动校准逻辑：
            # 假设在预热期(通常是低噪声)，系统处于"非相对论区域"。
            # 我们设定目标基准速度 v_base = 0.5 (即 dS_ext =  dR) 
            # 意味着，在理想的学习状态（0% 噪声）下，系统的“内部做功”与“外部增益”在量级上应当是平衡的。实际上，对于神经网络，很难达到100%掌握训练数据，所以实际速度低于0.5.可以手动根据经验来设定
            # 公式: Gain / D = dR  ->  D = Gain / dR
            
            if self.warmup_dR > 0:
                target_ratio = 1.0 # 让 dS 是 dR 的 1 倍
                self.D = self.warmup_gain / (target_ratio * self.warmup_dR)
            else:
                self.D = 1.0 # Fallback
                
            print(f"       [Calibration] Auto-set ||D|| = {self.D:.4f} (based on v~0.5)")
        else:
            print(f"       [Calibration] Manual ||D|| = {self.D:.4f}")

        self.is_warmed_up = True
        # 清零积分，重新开始正式测量
        self.total_dR = 0.0
        self.total_dS_ext = 0.0
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0
        print(f"       [Tracker] Tracking Started.")


def get_grad_vector(model):
    """
    获取模型所有梯度的扁平化向量
    用于计算向量点积和模长
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    
    # 极少数情况下可能没有梯度，返回空tensor
    if not grads:
        return torch.tensor([]).to(DEVICE)
        
    return torch.cat(grads)

def get_data_loaders(noise_ratio, subset_ratio):
    """
    构建训练集和验证集
    关键：Train集注入噪声，Val集保持干净(Ground Truth)
    """
    # 标准 CIFAR-10 增强
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载原始数据
    full_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 1. 划分索引
    # 训练集大小
    train_len = int(len(full_ds) * subset_ratio)
    # 验证集大小 (固定 1000 张用于计算环境梯度，足够稳定且快)
    val_len = 1000
    
    # 简单的顺序切分，保证复现性
    all_indices = np.arange(len(full_ds))
    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len : train_len + val_len]
    
    # 2. 提取数据 (为了修改 target，必须从 Dataset 中把 data 和 targets 拿出来)
    # CIFAR10 对象有 .data (numpy) 和 .targets (list)
    train_data = full_ds.data[train_indices]
    train_targets = np.array(full_ds.targets)[train_indices]
    
    val_data = full_ds.data[val_indices]
    val_targets = np.array(full_ds.targets)[val_indices] # 验证集保持干净！
    
    # 3. 注入噪声 (仅对训练集)
    if noise_ratio > 0:
        n_noise = int(len(train_indices) * noise_ratio)
        # 固定随机种子
        rng = np.random.default_rng(42)
        noise_idx = rng.choice(np.arange(len(train_indices)), n_noise, replace=False)
        # 随机替换为 0-9
        train_targets[noise_idx] = rng.integers(0, 10, size=n_noise)
        
    # 4. 封装回 Dataset
    class FastDS(torch.utils.data.Dataset):
        def __init__(self, x, y, tf):
            self.x = x
            self.y = y
            self.tf = tf
        def __len__(self): return len(self.x)
        def __getitem__(self, idx):
            # 需要转为 PIL Image 才能应用 transforms
            img = Image.fromarray(self.x[idx])
            if self.tf: img = self.tf(img)
            return img, int(self.y[idx])

    train_ds = FastDS(train_data, train_targets, transform)
    val_ds = FastDS(val_data, val_targets, transform)
    
    # 5. 生成 Loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 验证集 Batch Size 可以大一点，梯度更稳
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    
    return train_dl, val_dl

def run_single_trial(noise, fixed_D=None):
    print(f"\n   >>> Training with Noise: {noise*100:.1f}% ...")

    """
    运行单个实验
    fixed_D: 如果不为 None，则强制使用该物理常数
    """
    print(f"\n   >>> Training with Noise: {noise*100:.1f}% ...")
    if fixed_D is not None:
        print(f"       [System] Using Fixed ||D|| = {fixed_D:.4f}")

    
    # 1. 获取 Loader (Train + Val)
    train_loader, val_loader = get_data_loaders(noise, DATA_SUBSET_RATIO)
    # 固定验证集 Batch 用于计算环境投影 (Ground Truth Signal)
    val_iter = iter(val_loader)
    fixed_val_batch = next(val_iter)
    
    model = resnet18(num_classes=10).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 使用正交物理追踪器 (Relativistic Projection: dS = dR + dS_ext)
    # === 关键修改 ===
    # 传入锁定的 D。如果是 None，Tracker 会自动计算。
    tracker = OrthogonalPhysicsTracker(manual_D=fixed_D)
    
    # 状态变量
    best_loss = float('inf')
    stall_counter = 0
    success_counter = 0
    loss_history = []
    base_loss = None 
    valid_epochs = 0

    # 窗口状态: (epoch, total_dR, total_dS_ext)
    window_states = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Step 1: Train Backward (计算参数更新方向) ---
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            
            # 保存训练梯度向量
            grad_train = get_grad_vector(model)
            
            # --- Step 2: Val Backward (Physics) ---
            # [关键修正] 什么时候计算物理量？
            # 1. 已经预热完成 (tracker.is_warmed_up)
            # 2. 或者：正在预热的最后一轮 (用于收集数据定标 D)
            # 这样确保 start_tracking 时有数据可用
            should_measure = tracker.is_warmed_up or (epoch == WARMUP_EPOCHS - 1)
            
            if should_measure:
                # 切换 Eval 计算验证梯度
                model.eval()
                optimizer.zero_grad()
                
                vx, vy = fixed_val_batch
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                v_out = model(vx)
                v_loss = criterion(v_out, vy)
                v_loss.backward()
                
                grad_val = get_grad_vector(model)
                val_loss_val = v_loss.item()
                
                # 恢复训练状态 & 梯度
                model.train()
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                
                # 记录物理量
                tracker.step_batch(grad_train, grad_val, LR, val_loss_val)
            
            # --- Step 3: Update ---
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # [控制] 预热逻辑
        if epoch < WARMUP_EPOCHS:
            if epoch == WARMUP_EPOCHS - 1:
                tracker.start_tracking()
                base_loss = avg_loss
            continue
            
        # [物理] 结算 Epoch
        # step_epoch 返回: v_global, curr_dR, curr_dS_ext
        v, curr_dR, curr_dS_ext = tracker.step_epoch(avg_loss)
        
        loss_history.append(avg_loss)
        valid_epochs += 1
        
        # === [新增] 记录窗口状态 (使用累积量) ===
        # window_states 记录 (valid_epochs, total_dR, total_dS_ext)
        window_states.append((valid_epochs, tracker.total_dR, tracker.total_dS_ext))
        if len(window_states) > WINDOW_SIZE:
            window_states.pop(0) 
        
        # 计算当前的窗口速度
        v_window = 0.0
        if len(window_states) > 1:
            # Delta = End - Start
            win_dR = window_states[-1][1] - window_states[0][1]
            win_dS_ext = window_states[-1][2] - window_states[0][2]
            
            # Relativistic Velocity v = dR / (dR + dS_ext)
            denom = win_dR + win_dS_ext
            if denom > 0:
                v_window = win_dR / denom

        if epoch % 10 == 0:
            print(f"       Ep {epoch}: Loss {avg_loss:.4f} | dR {curr_dR:.2f} | dS_ext {curr_dS_ext:.4f} | v_win {v_window:.3f}")

        # [控制] 成功判定
        if avg_loss < TARGET_LOSS:
            success_counter += 1
        else:
            success_counter = 0
        if success_counter >= 2:
            print(f"       [Success] Converged at Epoch {epoch}.")
            break
            
        # [控制] 僵死判定
        if avg_loss < best_loss - 0.005:
            best_loss = avg_loss
            stall_counter = 0
        else:
            stall_counter += 1
        if stall_counter >= PATIENCE:
            print(f"       [Stop] Loss stalled.")
            break

    # === 结果结算 ===
    final_loss = loss_history[-1] if loss_history else avg_loss
    
    if final_loss > FAILURE_THRESHOLD:
        print(f"   [WALL HIT] Final Loss {final_loss:.4f}. Failed.")
        return None 
        
    potential_gap = max(1e-4, base_loss - TARGET_LOSS)
    actual_drop = max(0, base_loss - final_loss)
    progress = max(0.01, min(1.0, actual_drop / potential_gap))
    
    proj_cost = valid_epochs if progress >= 1.0 else float('inf')
    
    # 使用窗口速度作为最终结果 (反映撞墙前的极限状态)
    v_final = v_window if v_window > 0 else v
    
    print(f"   [Result] v={v_final:.3f}, Cost={proj_cost:.1f} (Prog {progress*100:.1f}%)")
    return {'noise': noise, 'v': v_final, 'c': proj_cost, 'calibrated_D': tracker.D}

def main_adaptive_loop():
    results = {'noise': [], 'velocity': [], 'cost': []}
    
    # 确保 0.0 (基准) 在第一个，以便定标
    queue = sorted(list(set(NOISE_LEVELS))) 
    if 0.0 in queue:
        queue.remove(0.0)
        queue.insert(0, 0.0)
        
    last_success_noise = 0.0
    
    # 全局锁定的物理常数 D
    GLOBAL_D = None
    
    print(f"🚀 Experiment Started. Global D will be locked after first success.")
    
    # 增加最大采样次数防止死循环
    iteration = 0
    
    while len(queue) > 0 and iteration < MAX_SAMPLING_NODES:
        iteration += 1
        current_noise = queue.pop(0)
        
        # 传入 GLOBAL_D
        data = run_single_trial(current_noise, fixed_D=GLOBAL_D)
        
        if data is not None:
            # === 成功 ===
            # 锁定 D
            if GLOBAL_D is None:
                GLOBAL_D = data['calibrated_D']
                print(f"   [System] 🔒 Global Physics Constant Locked: ||D|| = {GLOBAL_D:.4f}")
            
            results['noise'].append(data['noise'])
            results['velocity'].append(data['v'])
            results['cost'].append(data['c'])
            last_success_noise = current_noise
        else:
            # === 撞墙 ===
            # 触发夹逼：在“上一次成功”和“这一次失败”之间找中点
            midpoint = round((last_success_noise + current_noise) / 2, 3)
            
            # 如果步长太小，说明已经摸到了悬崖边缘
            if abs(midpoint - last_success_noise) <= epsilon :
                print(f"   [Boundary Found] Precision limit reached at noise ~ {last_success_noise}")
                break
                
            print(f"   [Backtracking] Trying {midpoint}")
            queue.insert(0, midpoint)
            
    return results
def plot_final(results):
    if not results['velocity']: return
    
    # 1. 数据预处理
    v_raw = np.array(results['velocity'])
    c_raw = np.array(results['cost'])
    n_raw = np.array(results['noise'])
    
    # 排序
    idx = np.argsort(v_raw)
    v = v_raw[idx]
    c = c_raw[idx]
    
    # 数据清洗：去除 Cost 下降的异常点 (Non-monotonic filtering)
    # 物理上，速度越快成本应该越高。最后的下降点通常是测量崩坏或随机误差，会误导拟合。
    valid_mask = np.ones_like(c, dtype=bool)
    max_c_so_far = -np.inf
    for i in range(len(c)):
        if c[i] < max_c_so_far:
            valid_mask[i] = False # 标记为异常下降点（可选，为了图好看建议过滤）
        else:
            max_c_so_far = c[i]
    
    # 如果想保留所有点展示真实性，可以注释掉下面这两行
    # v = v[valid_mask]
    # c = c[valid_mask]
    
    # 2. 定义静止参考系 (Rest Frame)
    v_start = np.min(v) # 基准速度 (0% noise)
    v_cliff = np.max(v) # 悬崖边缘
    
    plt.figure(figsize=(10, 7))
    
    # 绘制实测点
    plt.scatter(v, c, s=150, c='red', edgecolors='k', zorder=10, label='Measured Data')
    
    # 生成绘图 X 轴
    x_plot = np.linspace(v_start, v_cliff + 0.005, 500)
    
    # === 3. FIM 拟合 (Quadratic) ===
    # 模型：y = a * (v - v_start)^2 + b
    # 同样进行平移，公平对比
    try:
        def fim_shifted(x, a, b):
            return a * (x - v_start)**2 + b
        popt_f, _ = curve_fit(fim_shifted, v, c, maxfev=5000)
        plt.plot(x_plot, fim_shifted(x_plot, *popt_f), 'g--', linewidth=2, label=r'FIM ($v^2$)')
    except: pass
    
    # === 4. UIT 拟合 (Relativistic Shifted) ===
    # 模型：基于相对速度的洛伦兹因子
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
        
        # 约束：绝对光速 c_abs 必须 > v_cliff (观测到的最大速度)
        # 且应该很接近 v_cliff (夹逼)
        epsilon = 0.02 # 允许的拟合裕度
        lower_bound = v_cliff + 1e-5
        upper_bound = v_cliff + epsilon
        
        # p0: k, c_abs, b
        p0 = [100, lower_bound + 0.001, min(c)]
        bounds = ([0, lower_bound, -np.inf], [np.inf, upper_bound, np.inf])
        
        popt_u, _ = curve_fit(uit_lorentz, v, c, p0=p0, bounds=bounds, maxfev=10000)
        
        # 绘制
        c_fit_abs = popt_u[1]
        # 只画到渐近线前
        x_plot_uit = np.linspace(v_start, c_fit_abs * 0.999, 500)
        
        plt.plot(x_plot_uit, uit_lorentz(x_plot_uit, *popt_u), 'b-', linewidth=3, label=r'UIT ($\gamma$)')
        
        # 画出墙
        plt.axvline(x=c_fit_abs, color='gray', linestyle=':', label=f'Limit $c={c_fit_abs:.3f}$')
        print(f"   [Fit] Relativistic Limit c = {c_fit_abs:.4f} (v_start={v_start:.3f})")
        
    except Exception as e:
        print(f"UIT Fit Failed: {e}")

    # 5. 视图优化
    # X 轴：只展示从 v_start 开始到墙后一点点
    plt.xlim(v_start - 0.02, v_cliff + 0.03)
    plt.ylim(0, max(c) * 1.3)

    plt.xlabel(r'Fisher Velocity $v$ (With Rest Frame Shift)', fontsize=12)
    plt.ylabel('Projected Cost (Epochs)', fontsize=12)
    plt.title(f'Adjudication: Shifted Relativistic Dynamics ($v_0={v_start:.2f}$)', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # === [新增] 导出 PDF ===
    print("   [Output] Saving plots to PDF/PNG...")
    plt.savefig('adjudication_final.png', dpi=300)
    plt.savefig('adjudication_final.pdf', format='pdf', bbox_inches='tight') # 矢量图
    
    plt.show()

if __name__ == "__main__":
    data = main_adaptive_loop()
    print("FINAL RESULTS:", data)
    plot_final(data)