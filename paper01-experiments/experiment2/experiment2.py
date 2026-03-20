import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from thop import profile # 引入 FLOPs 计算库，因为不同架构单步资源使用差异较大，FLOPs 更公平

# ================= 实验参数 (Experiment Config) =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 基础设置
TARGET_PARAMS = 5.0e6  
TOLERANCE = 0.05
DATA_SUBSET = 0.4      
BATCH_SIZE = 256

# 2. 压力设置
# 提高噪声到 40%，拉开架构差距
NOISE_RATIO = 0.0      
# 目标 Loss 
TARGET_VAL_LOSS = 0.1
MAX_EPOCHS_LIMIT = 200

# 3. 优化器
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3 
WARMUP_EPOCHS = 3
WINDOW_SIZE = 5
PATIENCE = 50
# ==============================================================

# --- 1. 物理追踪器 (完全复用您的版本) ---
class OrthogonalPhysicsTracker:
    """
    R-S 相对论追踪器 (标准版)
    用于 Experiment 1 & 2，确保物理量纲一致。
    """
    def __init__(self, model=None, manual_D=None):
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0 
        
        self.total_dR = 0.0
        self.total_dS_ext = 0.0
        
        self.is_warmed_up = False
        self.D = manual_D 
        
        self.warmup_dR = 0.0
        self.warmup_gain = 0.0
        
    def step_batch(self, grad_train_vec, grad_val_vec, lr, val_loss):
        # 1. dR (努力值: 参数空间路程)
        norm_train = torch.norm(grad_train_vec).item()
        step_dR = lr * norm_train
        
        # 2. Raw Gain (原始增益: 投影 * 密度)
        dot_prod = torch.dot(grad_train_vec, grad_val_vec).item()
        norm_val = torch.norm(grad_val_vec).item()
        
        cosine = 0.0
        if norm_train > 0 and norm_val > 0:
            cosine = dot_prod / (norm_train * norm_val)
            
        # 信息密度 = 1 / 熵(Loss)
        entropy = val_loss if val_loss > 1e-4 else 1e-4
        density = 1.0 / entropy
        
        step_raw_gain = step_dR * max(0.0, cosine) * density
        
        if not self.is_warmed_up:
            self.warmup_dR += step_dR
            self.warmup_gain += step_raw_gain
        else:
            self.epoch_dR += step_dR
            self.epoch_raw_gain += step_raw_gain

    def step_epoch(self, current_loss=None):
        if not self.is_warmed_up:
            return 0.0, 0.0, 0.0

        # 应用物理常数 ||D|| (从预热期定标获得)
        epoch_dS_ext = self.epoch_raw_gain / self.D
        
        # 累加全局积分
        self.total_dR += self.epoch_dR
        self.total_dS_ext += epoch_dS_ext
        
        # 计算全局速度
        denom = self.total_dR + self.total_dS_ext
        velocity = 0.0 if denom == 0 else self.total_dR / denom
        
        # 快照当前 Epoch 增量
        curr_dR = self.epoch_dR
        curr_dS_ext = epoch_dS_ext
        
        # 重置瞬时器
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0
        
        return velocity, curr_dR, curr_dS_ext

    def start_tracking(self):
        # 自动校准逻辑: 设定基准速度 v_base = 0.5
        # 这意味着在自然状态下，内部做功 dR 与外部增益 dS 量级相当
        # Gain / D = dR  =>  D = Gain / dR
        if self.D is None:
            if self.warmup_dR > 0:
                target_ratio = 1.0 
                self.D = self.warmup_gain / (target_ratio * self.warmup_dR)
            else:
                self.D = 1.0
            print(f"       [Calibration] Auto-set ||D|| = {self.D:.4f} (based on v~0.5)")
        else:
            print(f"       [Calibration] Manual ||D|| = {self.D:.4f}")

        self.is_warmed_up = True
        self.total_dR = 0.0
        self.total_dS_ext = 0.0
        self.epoch_dR = 0.0
        self.epoch_raw_gain = 0.0
        print(f"       [Tracker] Tracking Started.")

def get_grad_vector(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if not grads: return torch.tensor([]).to(DEVICE)
    return torch.cat(grads)





# --- 2. 模型工厂 (保持不变) ---
# ... 为节省篇幅，这里假设 make_mlp 等函数已定义 ...
# 必须包含: count_params, make_mlp, make_res_mlp, make_cnn, make_cnn_bn, make_resnet, make_mobilenet, auto_scale
# 请确保之前定义的 Model Factory 代码块在这里是可用的

# --- 2. 模块化模型工厂 (Grid System: 3x3 Matrix) ---
def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)




# ================= Axis B=1: Wide CNN (VGG-Style) =================
# 结构: [Conv-ReLU] -> Pool -> [Conv-ReLU] -> Pool -> [Conv-ReLU] -> Pool -> FC

# --- 1. 空间多尺度分支 (B2 的新内核：3x3 + 5x5) ---
class MultiScaleBranch(nn.Module):
    def __init__(self, in_c, out_c, a_level, stride=1):
        super().__init__()
        self.a_level = a_level
        # 分支 1: 3x3 标准卷积
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=(a_level==0)),
            nn.BatchNorm2d(out_c) if a_level >= 1 else nn.Identity()
        )
        # 分支 2: 5x5 卷积 (用两个 3x3 模拟，这是 SOTA 常用做法，更稳且能效高)
        # 5x5 能提供更大的感受野，捕获更宏观的 dS_ext
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=(a_level==0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=(a_level==0)),
            nn.BatchNorm2d(out_c) if a_level >= 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 融合 3x3 和 5x5 的特征
        return self.relu(self.branch3x3(x) + self.branch5x5(x))

# --- 2. 统一动力学块 (UnifiedBlock) ---
class UnifiedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, a_level, b_level, mode='conv', stride=1):
        super().__init__()
        self.a_level = a_level
        self.mode = mode
        
        # 决定是否在该块应用残差连接：仅限 linear 模式且 a_level == 2
        self.apply_residual = (a_level == 2 and mode == 'linear')

        if mode == 'conv':
            # 卷积层：无论 A2 还是 A1，都执行相同的结构（不再受残差影响）
            if b_level < 2:
                # B1 等级：标准单路径卷积
                self.layer = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=(a_level==0)),
                    nn.BatchNorm2d(out_dim) if a_level >= 1 else nn.Identity(),
                    nn.ReLU(inplace=True)
                )
            else:
                # B2 等级：多尺度并行分支 (Invariance 增强)
                self.layer = MultiScaleBranch(in_dim, out_dim, a_level, stride)
        else:
            # Linear 模式 (用于后端蓄水池)
            if self.apply_residual:
                # A2 线性层：主路不加 ReLU，留到残差相加后
                self.layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=False),
                    nn.LayerNorm(out_dim)
                )
            else:
                # A0, A1 线性层
                self.layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=(a_level==0)),
                    nn.LayerNorm(out_dim) if a_level >= 1 else nn.Identity(),
                    nn.ReLU(inplace=True)
                )

        # 构建残差路径
        self.shortcut = nn.Identity()
        self.final_relu = nn.Identity() # 默认不激活

        if self.apply_residual:
            self.final_relu = nn.ReLU(inplace=True)
            if in_dim != out_dim:
                self.shortcut = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=False),
                    nn.LayerNorm(out_dim)
                )

    def forward(self, x):
        if not self.apply_residual:
            # A0, A1 或 卷积层的 A2：直接走层前向
            return self.layer(x)
        else:
            # 仅线性层的 A2：执行标准残差相加
            return self.final_relu(self.layer(x) + self.shortcut(x))

# --- 3. 统一架构系列 (Series) ---
# --- 统一深度管理：确保所有模型都是 10 层激活 ---

def make_reservoir(in_features, width, a_level, depth=6):
    """
    后端蓄水池管理
    确保无论 a_level 如何，总 ReLU 激活层数 = depth
    """
    layers = []
    curr_in = in_features
    
    # 计算实际需要放置的块数
    # A=0,1: 每一层1个激活; A=2: 每一块2个激活
    if a_level < 2:
        num_blocks = depth - 1 # 减1是为了留出一层给最后的 Linear(w, 10)
        blocks_to_add = num_blocks
    else:
        num_blocks = (depth - 1) // 2 # 残差块自带两层
        blocks_to_add = num_blocks

    for i in range(blocks_to_add):
        layers.append(UnifiedBlock(curr_in, width, a_level, b_level=0, mode='linear'))
        curr_in = width
    
    # 如果是 A=2 且 depth 是奇数，补一层普通的层对齐深度
    if a_level == 2 and (depth - 1) % 2 != 0:
        layers.append(UnifiedBlock(curr_in, width, a_level=1, b_level=0, mode='linear'))
        curr_in = width

    layers.append(nn.Linear(curr_in, 10))
    return nn.Sequential(*layers)

# ================= 3x3 矩阵最终定义 =================

# B0: MLP (总深度 10)
def make_b0_series(width, a_level):
    # 10 层全连接
    return nn.Sequential(Flatten(), make_reservoir(3072, int(width), a_level, depth=10))

# B1: CNN (总深度 10)
def make_b1_series(width, a_level):
    c_w = 64
    # 前端 4 层卷积 (对齐深度)
    conv_part = nn.Sequential(
        UnifiedBlock(3, c_w, a_level, b_level=1, mode='conv'), # L1
        nn.MaxPool2d(2),
        UnifiedBlock(c_w, c_w*2, a_level, b_level=1, mode='conv'), # L2
        nn.MaxPool2d(2),
        UnifiedBlock(c_w*2, c_w*2, a_level, b_level=1, mode='conv'), # L3
        nn.MaxPool2d(2),
        UnifiedBlock(c_w*2, c_w*4, a_level, b_level=1, mode='conv'), # L4
    )
    # 4x4 * 256 = 4096
    # 后端 6 层蓄水池
    return nn.Sequential(conv_part, Flatten(), make_reservoir(4096, int(width), a_level, depth=6))

# B2: Multi-Scale (总深度 10)
def make_b2_series(width, a_level):
    c_w = 64
    # 前端 4 层多尺度
    conv_part = nn.Sequential(
        UnifiedBlock(3, c_w, a_level, b_level=2, mode='conv'), # L1
        nn.MaxPool2d(2),
        UnifiedBlock(c_w, c_w*2, a_level, b_level=2, mode='conv'), # L2
        nn.MaxPool2d(2),
        UnifiedBlock(c_w*2, c_w*2, a_level, b_level=2, mode='conv'), # L3
        nn.MaxPool2d(2),
        UnifiedBlock(c_w*2, c_w*4, a_level, b_level=2, mode='conv'), # L4
    )
    return nn.Sequential(conv_part, Flatten(), make_reservoir(4096, int(width), a_level, depth=6))

# 封装供 auto_scale 调用
def node_00(w): return make_b0_series(w, 0)
def node_10(w): return make_b0_series(w, 1)
def node_20(w): return make_b0_series(w, 2)
def node_01(w): return make_b1_series(w, 0)
def node_11(w): return make_b1_series(w, 1)
def node_21(w): return make_b1_series(w, 2)
def node_02(w): return make_b2_series(w, 0)
def node_12(w): return make_b2_series(w, 1)
def node_22(w): return make_b2_series(w, 2)

def auto_scale(factory):
    # ... (保持之前的二分查找逻辑不变) ...
    low, high = 1, 4096
    
    best_m = None; best_err = float('inf')
    
    for _ in range(20):
        mid = (low + high) // 2
        if mid < 1: mid = 1
        try:
            # 1. 在 CPU 上构建模型
            m = factory(mid) 
            
            # 2. 计算参数 (CPU)
            p = count_params(m)
            
            err = abs(p - TARGET_PARAMS) / TARGET_PARAMS
            if err < best_err: 
                best_err = err; best_m = m
            
            if err < TOLERANCE: break
            
            if p < TARGET_PARAMS: low = mid + 1
            else: high = mid - 1
            
            # 3. 释放内存
            del m
            
        except Exception as e: 
            # 捕捉 OOM 或构建错误
            high = mid - 1
            
    print(f"   Built {factory.__name__}: {count_params(best_m)/1e6:.2f}M Params")
    return best_m # 返回的是 CPU 模型
    # (在此处插入 Model Factory 代码)

# --- 3. 数据加载 (Train 带噪, Val 干净) ---
def get_loaders():
    # 增加问题复杂度 (数据增强 + 归一化)
    tf = transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2), # 增加颜色干扰
        transforms.RandomGrayscale(p=0.1),         # 增加模态干扰
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
    sub = int(len(full)*DATA_SUBSET); idx = np.arange(sub)
    data = full.data[idx]; tgt = np.array(full.targets)[idx]
    
    if NOISE_RATIO>0:
        rng = np.random.default_rng(42); n_n = int(sub*NOISE_RATIO)
        idx_n = rng.choice(idx, n_n, replace=False)
        tgt[idx_n] = rng.integers(0,10,n_n)
        
    val_idx = np.arange(sub, sub+1000)
    
    class DS(torch.utils.data.Dataset):
        def __init__(self, x, y, tf): self.x, self.y, self.tf = x, y, tf
        def __len__(self): return len(self.x)
        def __getitem__(self, i): 
            img = Image.fromarray(self.x[i])
            if self.tf: img = self.tf(img)
            return img, int(self.y[i])
            
    train_dl = torch.utils.data.DataLoader(DS(data, tgt, tf), BATCH_SIZE, True)
    val_dl = torch.utils.data.DataLoader(DS(full.data[val_idx], np.array(full.targets)[val_idx], tf), 256, False)
    return train_dl, val_dl

# --- 4. 实验主程序 (修正版: Window Velocity + Best Loss Snapshot) ---
def run_zigzag():
    print(f"🚀 Experiment 2: Zig-Zag Inertia (Loss < {TARGET_VAL_LOSS})")
    
    # 获取 Loader
    train_dl, val_dl = get_loaders()
    fixed_val = next(iter(val_dl))
    
    # 1. 定标 D (保持不变)
    print("\n[Phase 1] Calibrating Universe on MLP...")
    # 1. 获取 CPU 模型
    base_cpu = auto_scale(node_00)
    # 2. [关键修正] 放入 GPU
    base = base_cpu.to(DEVICE)
    
    opt = optim.SGD(base.parameters(), lr=LR, momentum=MOMENTUM)
    # Cosine Annealing: T_max 设为预热轮数+几轮缓冲，保证定标时 LR 平滑
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=WARMUP_EPOCHS+5)
    crit = nn.CrossEntropyLoss()
    
    calib_tracker = OrthogonalPhysicsTracker(manual_D=None)
    
    for epoch in range(WARMUP_EPOCHS + 1): 
        base.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(base(x), y).backward(); gt=get_grad_vector(base)
            
            base.eval(); opt.zero_grad()
            vx, vy = fixed_val; vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            vl = crit(base(vx), vy); vl.backward(); gv=get_grad_vector(base)
            val_l_val = vl.item()
            
            base.train(); opt.zero_grad()
            crit(base(x), y).backward() 
            nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0) # Clipping
            
            opt.step()
            calib_tracker.step_batch(gt, gv, opt.param_groups[0]['lr'], val_l_val)
        
        sched.step() 
            
        if epoch == WARMUP_EPOCHS - 1:
            calib_tracker.start_tracking()
            
    GLOBAL_D = calib_tracker.D
    print(f"   🔒 Global Constant Locked: ||D|| = {GLOBAL_D:.4f}")
    
    # 2. 运行网格
    nodes = [
        # A 轴基础方向: 纵向看 R
        (0, 0, node_00, "MLP-A0"),
        (1, 0, node_10, "MLP-A1"),
        (2, 0, node_20, "MLP-A2"),
        
        # B 轴改进方向: 横向看 S_ext
        (0, 1, node_01, "CNN-A0"),
        (0, 2, node_02, "MCNN-A0"),

        # 鞍部节点
        (1, 1, node_11, "CNN-A1"),
        (2, 1, node_21, "CNN-A2"),
        (1, 2, node_12, "MCNN-A1"),
        (2, 2, node_22, "MCNN-A2")
    ]
    
    results = [] 
    
    # 窗口大小
    WINDOW_SIZE = 5 
    
    for a, b, factory, name in nodes:
        print(f"\n>>> Testing {name} (A={a}, B={b})...")
        # 1. 获取 CPU 模型
        model_cpu = auto_scale(factory)
        
        # 2. 只有最终确定的模型才放入 GPU
        model = model_cpu.to(DEVICE)

        # === [核心修改] 计算单次 Forward 的 FLOPs ===
        # 使用一个 dummy input
        dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        # FLOPs ≈ 2 * MACs
        flops_per_forward = macs * 2
        # Backward FLOPs ≈ 2 * Forward FLOPs
        flops_per_step = flops_per_forward * 3
        
        print(f"    Model FLOPs per step: {flops_per_step / 1e9:.2f} GFLOPs")

        
        # 使用 Cosine Annealing + 较大的初始 LR
        initial_lr = LR 
        opt = optim.SGD(model.parameters(), lr=initial_lr, momentum=MOMENTUM)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_LIMIT)
        crit = nn.CrossEntropyLoss()
        
        tracker = OrthogonalPhysicsTracker(manual_D=GLOBAL_D)
        tracker.is_warmed_up = True 
        
        reached = False
        
        # 最佳状态记录
        best_val_loss = float('inf')
        epoch_at_best = 1
        dR_at_best = 0.0
        dS_ext_at_best = 0.0
        v_window_at_best = 0.0
        # 记录累积的 FLOPs
        total_flops = 0.0
        flops_at_best = 0.0
        
        # 窗口状态: (epoch, total_dR, total_dS_ext)
        window_states = []
        
        for epoch in range(MAX_EPOCHS_LIMIT):
            model.train()
            current_lr = opt.param_groups[0]['lr'] 
            
            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # 更新 FLOPs
                # * BATCH_SIZE
                total_flops += flops_per_step * x.size(0)

                # Train Backward
                opt.zero_grad(); crit(model(x), y).backward(); gt=get_grad_vector(model)
                
                # Val Backward
                model.eval(); opt.zero_grad()
                vx, vy = fixed_val; vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                vl = crit(model(vx), vy); vl.backward(); gv=get_grad_vector(model)
                val_l_val = vl.item()
                
                # Update
                model.train(); opt.zero_grad(); crit(model(x), y).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) # Clipping
                opt.step()
                tracker.step_batch(gt, gv, current_lr, val_l_val)
            
            # Step Epoch
            tracker.step_epoch()
            sched.step()
            
            # Check Target
            model.eval()
            loss_sum = 0
            with torch.no_grad():
                for vx, vy in val_dl:
                    loss_sum += crit(model(vx.to(DEVICE)), vy.to(DEVICE)).item()
            val_loss = loss_sum / len(val_dl)
            
            # === Window Velocity Calculation ===
            # 记录当前累积状态
            window_states.append((epoch, tracker.total_dR, tracker.total_dS_ext))
            if len(window_states) > WINDOW_SIZE:
                window_states.pop(0) 
            
            # 计算窗口速度
            v_window = 0.0
            if len(window_states) > 1:
                # Delta = End - Start
                win_dR = window_states[-1][1] - window_states[0][1]
                win_dS_ext = window_states[-1][2] - window_states[0][2]
                denom = win_dR + win_dS_ext
                if denom > 0: v_window = win_dR / denom
            else:
                # Fallback to global velocity if window not full
                win_dR = tracker.total_dR
                win_dS_ext = tracker.total_dS_ext
                denom_g = tracker.total_dR + tracker.total_dS_ext
                v_window = tracker.total_dR / (denom_g + 1e-9)
                
            # === Snapshot Best State ===
            if val_loss <= best_val_loss - 1e-3:
                best_val_loss = val_loss
                dR_at_best = win_dR
                epoch_at_best = epoch
                dS_ext_at_best = win_dS_ext
                flops_at_best = total_flops
                v_window_at_best = v_window # 记录最佳时刻的窗口速度

            if epoch % 5 == 0:
                print(f"    Ep {epoch}: Loss {val_loss:.3f} (Best {best_val_loss:.3f})| FLOPs {total_flops/1e9:.2f} (Best {flops_at_best/1e9:.2f})| dR {win_dR:.1f} | v_win {v_window:.3f}")
                
            if val_loss <= TARGET_VAL_LOSS:
                print(f"    ✅ Reached Target in {epoch} epochs!")
                reached = True
                break

            if epoch - epoch_at_best > PATIENCE:
                print(f"    ⏸️  Early stopping at epoch {epoch} (best at {epoch_at_best})")
                break
        
        # === 结果结算 (使用 Best Snapshot) ===
        # 无论是 Reached 还是 Failed，都使用 best_val_loss 时刻的数据
        
        start_loss = 3
        target_gap = max(1e-4, start_loss - TARGET_VAL_LOSS)
        actual_drop = max(0, start_loss - best_val_loss)
        progress = max(0.01, min(1.0, actual_drop / target_gap))
        
        if reached: progress = 1.0
        
        # 惯性 = 达到目标所需的总 cost (Projected)
        projected_cost = flops_at_best / progress
        
        # 速度 = 最佳时刻的窗口速度
        velocity = v_window_at_best
        
        if not reached:
            print(f"    ❌ Failed. Prog: {progress*100:.1f}% | Proj Cost (Inertia): {projected_cost/1e9:.1f}")
        else:
            print(f"    ✅ Success. Inertia: {projected_cost:.1f}")
            
        entry = {
            'name': name, 'A': a, 'B': b,
            'success': reached, 'progress': progress,
            'flops': flops_at_best,
            'inertia': projected_cost, # Y轴：惯性
            'velocity': velocity,    # 辅助数据
            'final_loss': best_val_loss,
            'actual_dR': dR_at_best,
            'actual_dS': dS_ext_at_best
        }
        results.append(entry)
        
    with open('exp02_zigzag_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n✅ Detailed data saved to exp02_zigzag_data.json")


# 不再包含 plot 函数，数据已导出
if __name__ == "__main__":
    run_zigzag()