import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import copy
import json
import time
from PIL import Image

import torch
import numpy as np

# ==============================================================
# R-S 统一智能理论实验配置 (Experiment 3: Engineering Application)
# ==============================================================

# 1. 硬件与设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# 2. 架构与数据规模 (基于 5M 参数量级压测)
TARGET_PARAMS = 1e6      # 目标参数量 5M
TOLERANCE = 0.05           # 参数量允许误差 5%
DATA_SUBSET = 0.4          # 使用 40% 的 CIFAR-10 数据 (约 20,000 张)
BATCH_SIZE = 256           # 4070 建议值 (考虑双向梯度计算，不建议设为 256)

# 3. R-S 调节器核心参数 (The Regulator)
LR_BASE = 0.05             # 调节器允许的最大搜索步长 (基准动能)
ANCHOR_WINDOW = 20         # 调节器寻找 T_best (最优锚点) 的记忆长度
DECAY_WINDOW = 40          # 调节器记忆池的衰减窗口
TAU_RELAX = 500             # 逻辑弛豫时间 (从冻结到恢复 63.2% 动能所需的 Epoch 数)
BETA_THAW = 1.0 / TAU_RELAX # 自动计算的耗散系数 (1/Tau)

# 4. 物理观测参数 (The Monitor)
VELOCITY_WINDOW = 20        # 测量瞬时速度 v_win 的滑动窗口 (对齐实验 1 & 2)
WARMUP_EPOCHS = 1          # 预热轮数：用于定标 ||D||，跳过初始混沌期

# 5. 训练动力学参数
MOMENTUM = 0.9             # SGD 动量系数
WEIGHT_DECAY = 1e-3        # 权重衰减 (正则化防止过早死记硬背)
# MAX_EPOCHS_LIMIT = 40     # 单个任务的最大迭代轮数
MAX_EPOCHS_LIMIT = 40     # 单个任务的最大迭代轮数，测定极限时使用更大的值

TARGET_VAL_LOSS = 0.1      # 理想收敛目标
PATIENCE = 50              # 容忍 50 轮 Loss 不下降则强制触发解冻或停止

# 6. 环境压力设置 (Shock & Continual)
NOISE_RATIO = 0.0          # 基础环境噪声 (实验三中会在特定 Epoch 动态注入)
LOCKED_D = None            # 全局常数 ||D|| (留 None 则等待自动定标)

# 滤波设置
MUTUAL_PROJECTION_TTHRESHOLD = 10/BATCH_SIZE  # 互投影掩码阈值 (Causal Coherence Mask)
FIXED_E2B_LOSS_P_BASE = 1.0  # Epoch->Batch 损失基准 (用于调节一致性计算)
FIXED_E2B_LOSS_L_BASE = 1.0  # Epoch->Batch 损失基准 (用于调节一致性计算)

# ==============================================================
# --- 1. 物理追踪器：增加元素级路径统计 ---
class OrthogonalPhysicsTracker:
    def __init__(self, velocity_window=VELOCITY_WINDOW, manual_D=None, fixed_E2B_LOSS_P_BASE=None, fixed_E2B_LOSS_L_BASE=None):
        self.D = manual_D
        self.E2B_LOSS_P_BASE = fixed_E2B_LOSS_P_BASE
        self.E2B_LOSS_L_BASE = fixed_E2B_LOSS_L_BASE
        self.is_warmed_up = False
        self.vel_window_size = velocity_window
        self.v_window_pool = 0.5
        self.v_window_p = None
        self.v_window_l = None
        self.t_window = 0.0
        self.lr_window = LR_BASE
 
        
        # 寄存器：仅存储本 Epoch 的矢量求和
        self.epoch_g_t_sum = None # Sum(lr * g_train)
        self.epoch_g_s_sum = None # Sum(g_val * lr / Loss)

        self.train_batch_count = 0
        self.val_batch_count = 0
        
        # epoch to batch 损失
        self.epoch_g_t_sum_batch_abs_acc = None # Sum abs (lr * g_train)
        self.epoch_g_s_sum_batch_abs_acc = None # Sum abs (g_val * lr / Loss)

        self.epoch2batch_loss_g_t = 1.0 # 当在 batch 层次计算时，运用 epoch 层次数据所带来的损失
        self.epoch2batch_loss_g_s = 1.0 # 当在 batch 层次计算时，运用 epoch 层次数据所带来的损失
        
        # 预热定标累加器
        self.warmup_dR_vec_acc = None
        self.warmup_dR_vec_abs_acc = None
        self.warmup_dS_raw_vec_acc = None

        # [新增] 记录本 Epoch 所有的 Batch LR
        self.epoch_lrs = []

        self.obs_history = [] 

    def step_batch_rule(self, g_t, lr):
        """
        [训练阶段调用] 仅累加规则空间位移
        """
        self.epoch_lrs.append(lr)
        v_r = g_t.detach() * lr

        self.last_batch_v_r = v_r.clone()

        if self.epoch_g_t_sum is None:
            self.epoch_g_t_sum = v_r.clone()
            self.epoch_g_t_sum_batch_abs_acc = torch.abs(v_r).clone()

        else:
            self.epoch_g_t_sum += v_r
            self.epoch_g_t_sum_batch_abs_acc += torch.abs(v_r)
        self.train_batch_count += 1
        
        # 
        # if self.is_warmed_up == False:
        #     if self.warmup_dR_vec_abs_acc is None:
        #         self.warmup_dR_vec_abs_acc = torch.abs(v_r).clone()
        #     else:
        #         self.warmup_dR_vec_abs_acc += torch.abs(v_r)
    
    def mutual_projection_mask(self, v_anc, v_bat, threshold=MUTUAL_PROJECTION_TTHRESHOLD):
        """
        互投影掩码函数 (Causal Coherence Mask)
        v_anc: 锚点矢量 (Anchor Vector)
        v_bat: 当前批次矢量 (Batch Vector)
        threshold: 相似度判定阈值 (epsilon_ratio)
        """
        # 1. 提取绝对值 (为了计算量级比例)
        abs_anc = torch.abs(v_anc)
        abs_bat = torch.abs(v_bat)

        # 2. 计算元素级最小值和最大值
        min_vals = torch.min(abs_anc, abs_bat)
        max_vals = torch.max(abs_anc, abs_bat) + 1e-9 # 防止除零

        # 3. 核心判定规则：
        # 规则 1: 量级比例必须大于阈值 (排除极端异常点)
        # 规则 2: 方向必须一致 (v_anc * v_bat > 0)
        # 使用位运算加速
        mask = (min_vals / max_vals > threshold) & (v_anc * v_bat > 0)

        # 4. 执行 Mask 操作
        # 不满足条件的位点全部归零
        v_anc_masked = v_anc * mask
        v_bat_masked = v_bat * mask

        return v_anc_masked, v_bat_masked
    

    def step_batch_statext(self, g_v, lr, vl):
        """
        [评估阶段调用] 计算受双重相干性过滤后的状态位移
        v_anchor_l: 传入调节器 Pool 中计算出的环境力虚拟锚点
        """
        # 原始环境增益矢量
        v_s_raw = g_v.detach() * lr
        
        # --- 1. 时间自相干 (Temporal Coherence) ---
        # 逻辑：dS_ext 必须是 dR 动作产生的相干反馈
        if self.last_batch_v_r is not None:
            norm_r = torch.norm(self.last_batch_v_r)
            norm_s = torch.norm(v_s_raw)
            cos_temp = torch.dot(v_s_raw, self.last_batch_v_r) / (norm_r * norm_s + 1e-7)
            zeta_temp = torch.clamp(cos_temp, 0.0, 1.0) # 仅保留正向继承部分
        else:
            zeta_temp = 1.0

        # --- 2. 空间自相干 (Spatial Coherence) ---
        # 逻辑：当前增益必须符合窗口内的共识逻辑轴
        if (self.v_window_l is not None) and (len(self.obs_history) >= self.vel_window_size):
            # if self.val_batch_count%50 == 0:
            #     print(f"       [Tracker] Applying Spatial Coherence Mask at Batch ")
            # v_anc_l = self.v_window_l.to(v_s_raw.device)
            # v_anc_l_p, v_s_raw_p = self.mutual_projection_mask(v_anc_l, v_s_raw, threshold=min(2 * MUTUAL_PROJECTION_TTHRESHOLD,0.9)) # 大致方向对就行，微观batch的数据与宏观epoch，特别是window之间差异实在过大，连续性太低
            # cos_spat = torch.dot(v_s_raw_p, v_anc_l_p) / (torch.norm(v_anc_l_p) * torch.norm(v_s_raw_p) + 1e-7)
            # zeta_spat = torch.clamp(cos_spat, 0.0, 1.0)
            zeta_spat = 1.0
        else:
            zeta_spat = 1.0

        # --- 3. 复合过滤与势能归一化 ---
        # 应用相干性衰减：v_s = v_s_raw * (时间继承率) * (空间共鸣率)
        # v_s_coherent = v_s_raw * zeta_temp * zeta_spat
        v_s_coherent = v_s_raw * zeta_temp * zeta_spat
        
        # 引入环境信息密度 (1/Loss)
        # 注意：此处 dS_ext 的模长将因为相干性差而显著减小，导致 v -> 1
        energy_correction = 1.0 / (vl + 1e-7)
        v_s_final = v_s_coherent * energy_correction

        if self.epoch_g_s_sum is None:
            self.epoch_g_s_sum = v_s_final.clone()
            self.epoch_g_s_sum_batch_abs_acc = torch.abs(v_s_final).clone()
        else:
            self.epoch_g_s_sum += v_s_final
            self.epoch_g_s_sum_batch_abs_acc += torch.abs(v_s_final)

        self.val_batch_count += 1
        if self.val_batch_count%50 == 0:
            print(f"       [Tracker] Batch {self.val_batch_count} | zeta_temp:{zeta_temp:.4f} | zeta_spat:{zeta_spat:.4f} | energy_corr:{energy_correction:.4f} | win_size:{self.vel_window_size} | history_len:{len(self.obs_history)}")

    def step_epoch(self, current_val_loss):
        # 1. 提取本轮总位移矢量 (Displacement Sum)
        dR_epoch_vec = self.epoch_g_t_sum / self.train_batch_count
        dS_raw_epoch_vec = self.epoch_g_s_sum / self.val_batch_count

        # 计算元素级绝对值矢量 (Path Sum)
        dR_epoch_vec_batch_abs = self.epoch_g_t_sum_batch_abs_acc / self.train_batch_count
        dS_raw_epoch_vec_batch_abs = self.epoch_g_s_sum_batch_abs_acc / self.val_batch_count

        self.epoch2batch_loss_g_t = torch.norm(dR_epoch_vec_batch_abs).item() / (torch.norm(dR_epoch_vec).item() + 1e-9)
        self.epoch2batch_loss_g_s = torch.norm(dS_raw_epoch_vec_batch_abs).item() / (torch.norm(dS_raw_epoch_vec).item() + 1e-9)

        # 2. 正式期：将本轮矢量存入历史数据库，供池化层计算混乱度
        self.obs_history.append({
            'dR_vec': dR_epoch_vec.cpu(), 
            'dS_raw_vec': dS_raw_epoch_vec.cpu(),
            'loss': current_val_loss,
            'epoch2batch_loss_g_t': self.epoch2batch_loss_g_t,
            'epoch2batch_loss_g_s': self.epoch2batch_loss_g_s,
            'lr': self.epoch_lrs[-1] if self.epoch_lrs else LR_BASE,
            't': len(self.obs_history)
        })

        if not self.is_warmed_up:
            # 预热期：累加各 Epoch 的总矢量，直接退出后续逻辑
            if self.warmup_dR_vec_acc is None:
                self.warmup_dR_vec_acc = dR_epoch_vec.clone()
                self.warmup_dS_raw_vec_acc = dS_raw_epoch_vec.clone()
            else:
                self.warmup_dR_vec_acc += dR_epoch_vec
                self.warmup_dS_raw_vec_acc += dS_raw_epoch_vec
            self._reset_buffers()
            return 0.5, 0.5

            
        # 3. 计算v_window
        self.vel_window_size = len(self.obs_history) // 2
        window_pool = self.obs_history[-self.vel_window_size:]
        vec_dR_pool = torch.stack([e['dR_vec'] for e in window_pool])
        vec_dS_pool = torch.stack([e['dS_raw_vec'] for e in window_pool])
        
        vec_lr_pool = torch.tensor([e['lr'] for e in window_pool])
        
        
        # 矢量直接相加 (位移)
        disp_R = vec_dR_pool.sum(dim=0)
        disp_S_raw = vec_dS_pool.sum(dim=0)
        
        # 元素绝对值之和 (路径)
        path_R = torch.abs(vec_dR_pool).sum(dim=0)
        path_S_raw = torch.abs(vec_dS_pool).sum(dim=0)
        
        # --- 计算混乱度 L (只在池化层发生) ---
        L_R = torch.norm(path_R).item() / (torch.norm(disp_R).item() + 1e-9)
        L_s = torch.norm(path_S_raw).item() / (torch.norm(disp_S_raw).item() + 1e-9)

        # --- 更新窗口锚点参考系 ---
        self.v_window_p = disp_R.to(DEVICE)
        self.v_window_l = disp_S_raw.to(DEVICE)
        
        lr_sum = vec_lr_pool.sum().item()
        self.lr_window = lr_sum / self.vel_window_size if self.vel_window_size > 0 else LR_BASE

        
        # 4. 加权虚拟时间 (你的平滑机制)
        losses = np.array([e['loss'] for e in window_pool])
        epoch2batch_loss_dR_pool = np.array([e['epoch2batch_loss_g_t'] for e in window_pool])
        epoch2batch_loss_dS_pool = np.array([e['epoch2batch_loss_g_s'] for e in window_pool])

        weights = (1.0 / (losses + 1e-6))**2
        self.t_window = np.sum(weights * np.array([e['t'] for e in window_pool])) / np.sum(weights)
        epoch2batch_anchor_p_loss = np.sum(weights * epoch2batch_loss_dR_pool) / np.sum(weights)
        epoch2batch_anchor_l_loss = np.sum(weights * epoch2batch_loss_dS_pool) / np.sum(weights)


        # --- 计算池内有效速度 v_pool ---
        avg_loss = np.mean([e['loss'] for e in window_pool])
        R_eff = torch.norm(disp_R).item() * L_R * epoch2batch_anchor_p_loss / self.E2B_LOSS_P_BASE
        S_eff = (torch.norm(disp_S_raw).item() / self.D) / (avg_loss * L_s * epoch2batch_anchor_l_loss / self.E2B_LOSS_L_BASE + 1e-9)
        self.v_window_pool = R_eff / (R_eff + S_eff + 1e-9)        
        self._reset_buffers()
        print(f"       [Tracker] t_window:{self.t_window:.4f} | v_window_pool:{self.v_window_pool:.4f}")
        return self.v_window_pool, self.v_window_pool, self.v_window_pool # 这里的返回仅用于日志
    
    def _reset_buffers(self):
        self.epoch_g_t_sum = None
        self.epoch_g_s_sum = None
        self.train_batch_count = 0
        self.val_batch_count = 0
        self.epoch_g_t_sum_batch_abs_acc = None
        self.epoch_g_s_sum_batch_abs_acc = None

    def start_tracking(self):
        # [核心定标修正] D = ||Sum(dS_vec)|| / ||Sum(dR_vec)||
        # 这确保了在定标周期内，D 反映的是宏观位移的比例
        if self.D is None:
            self.D = torch.norm(self.warmup_dS_raw_vec_acc).item() / (torch.norm(self.warmup_dR_vec_acc).item() + 1e-9)
            
            e2b_loss_p = 0.0
            e2b_loss_l = 0.0
            for epoch_data in self.obs_history:
                e2b_loss_p += epoch_data['epoch2batch_loss_g_t']
                e2b_loss_l += epoch_data['epoch2batch_loss_g_s']
            self.E2B_LOSS_P_BASE = e2b_loss_p / len(self.obs_history) if self.obs_history else 1.0
            self.E2B_LOSS_L_BASE = e2b_loss_l / len(self.obs_history) if self.obs_history else 1.0

            print(f"       [Calibration] warmup_dS_raw_acc:{torch.norm(self.warmup_dS_raw_vec_acc).item():.4f} | warmup_dR_acc:{torch.norm(self.warmup_dR_vec_acc).item():.4f} |D = {self.D:.4f} (v_base=0.5) | E2B_LOSS_P_BASE={self.E2B_LOSS_P_BASE:.4f} | E2B_LOSS_L_BASE={self.E2B_LOSS_L_BASE:.4f}")
        self.is_warmed_up = True



class RSInertiaRegulator:
    def __init__(self, eta_base, elite_size=10, tau=TAU_RELAX, fixed_D=1.0, fixed_E2B_LOSS_P_BASE=FIXED_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=FIXED_E2B_LOSS_L_BASE):
        self.eta_base = eta_base
        self.gamma_factor = 1.0 # 相对论修正因子
        self.modulated_base_lr = eta_base
        self.elite_size = elite_size
        self.beta = 1.0 / tau
        self.D = fixed_D
        self.fixed_e2b_loss_p_base = fixed_E2B_LOSS_P_BASE
        self.fixed_e2b_loss_l_base = fixed_E2B_LOSS_L_BASE
        self.elite_pool = [] 
        self.v_anchor_p = None # 虚拟锚点矢量
        self.v_anchor_l = None
        self.epoch2batch_anchor_p_loss = 1.0
        self.epoch2batch_anchor_l_loss = 1.0
        self.t_best_anchor = 0
        self.lr_window = LR_BASE

        # 用于控制 get_batch_lr 的输出频率
        self.batch_print_counter = 0

    def record_epoch_state(self, tracker:"OrthogonalPhysicsTracker", current_epoch:int):
        # 1. 拿到本轮的矢量数据
        data = tracker.obs_history[-1]
        self.lr_window = tracker.lr_window

        new_entry = {
            't': current_epoch,
            'dR_vec': data['dR_vec'],      # Epoch 级的位移矢量
            'dS_raw_vec': data['dS_raw_vec'],
            'loss': data['loss'],
            'epoch2batch_loss_g_t': data['epoch2batch_loss_g_t'],
            'epoch2batch_loss_g_s': data['epoch2batch_loss_g_s'],
            'lr': self.modulated_base_lr 
        }

        # 2. 更新精英池 (Top-K)
        if len(self.elite_pool) < self.elite_size:
            self.elite_pool.append(new_entry)
        else:
            idx_max = np.argmax([e['loss'] for e in self.elite_pool])
            if new_entry['loss'] < self.elite_pool[idx_max]['loss']:
                self.elite_pool[idx_max] = new_entry

        # 3. [理论核心] 计算池内有效物理量
        # 我们对池内的 10 个 Epoch 矢量进行：
        # - 矢量求和 (Displacement Sum)
        # - 元素级绝对值求和 (Path Sum)
        
        vec_dR_pool = torch.stack([e['dR_vec'] for e in self.elite_pool])
        vec_dS_pool = torch.stack([e['dS_raw_vec'] for e in self.elite_pool])
        
        
        # 矢量直接相加 (位移)
        disp_R = vec_dR_pool.sum(dim=0)
        disp_S_raw = vec_dS_pool.sum(dim=0)
        
        # 元素绝对值之和 (路径)
        path_R = torch.abs(vec_dR_pool).sum(dim=0)
        path_S_raw = torch.abs(vec_dS_pool).sum(dim=0)
        
        # --- 计算混乱度 L (只在池化层发生) ---
        L_R = torch.norm(path_R).item() / (torch.norm(disp_R).item() + 1e-9)
        L_s = torch.norm(path_S_raw).item() / (torch.norm(disp_S_raw).item() + 1e-9)
        
        
        # --- 更新锚点参考系 ---
        self.v_anchor_p = disp_R.to(DEVICE)
        self.v_anchor_l = disp_S_raw.to(DEVICE)

        
        # 4. 加权虚拟时间 (你的平滑机制)
        losses = np.array([e['loss'] for e in self.elite_pool])
        epoch2batch_loss_dR_pool = np.array([e['epoch2batch_loss_g_t'] for e in self.elite_pool])
        epoch2batch_loss_dS_pool = np.array([e['epoch2batch_loss_g_s'] for e in self.elite_pool])

        weights = (1.0 / (losses + 1e-6))**2
        self.t_best_anchor = np.sum(weights * np.array([e['t'] for e in self.elite_pool])) / np.sum(weights)
        self.epoch2batch_anchor_p_loss = np.sum(weights * epoch2batch_loss_dR_pool) / np.sum(weights)
        self.epoch2batch_anchor_l_loss = np.sum(weights * epoch2batch_loss_dS_pool) / np.sum(weights)


        # --- 计算池内有效速度 v_pool ---
        avg_loss = np.mean([e['loss'] for e in self.elite_pool])
        R_eff = torch.norm(disp_R).item() * L_R * self.epoch2batch_anchor_p_loss / self.fixed_e2b_loss_p_base
        S_eff = (torch.norm(disp_S_raw).item() / self.D) / (avg_loss * L_s * self.epoch2batch_anchor_l_loss / self.fixed_e2b_loss_l_base + 1e-9)
        v_pool = R_eff / (R_eff + S_eff + 1e-9)

        # 4. 计算池内速度与特征相角
        # phi_best 代表了稳态下的 R-S 能量分配比例
        self.v_anchor_phi = S_eff / (R_eff + 1e-9)
        
        # --- 相对论 LR 修正 ---
        self.gamma_factor = np.sqrt(1 - tracker.v_window_pool**2) / 0.866
        self.modulated_base_lr = self.eta_base * self.gamma_factor


        # --- 调试输出 ---
        print(f"\n   [Pool Stats] Ep {current_epoch} | Size: {len(self.elite_pool)} | MinL: {min(losses):.4f}")
        print(f"   >>> Disorder: L_R={L_R:.2f}, L_s={L_s:.2f} | R_eff={R_eff:.4f} | S_eff={S_eff:.4f} | D: {self.D:.4f} | v_pool={v_pool:.4f} | v_window={tracker.v_window_pool:.4f}")
        print(f"   >>> Physical: phi_best={self.v_anchor_phi:.3f} | t_anchor={self.t_best_anchor:.2f}")
        print(f"   >>> Control:  Modulated_Base_LR={self.modulated_base_lr:.8f}\n")


        return self.modulated_base_lr
    
    def _calc_cosine(self, v1, v2):
        return torch.clamp(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-9), 0.0, 1.0)
    
    def mutual_projection_mask(self, v_anc, v_bat, threshold=MUTUAL_PROJECTION_TTHRESHOLD):
        """
        互投影掩码函数 (Causal Coherence Mask)
        v_anc: 锚点矢量 (Anchor Vector)
        v_bat: 当前批次矢量 (Batch Vector)
        threshold: 相似度判定阈值 (epsilon_ratio)
        """
        # 1. 提取绝对值 (为了计算量级比例)
        abs_anc = torch.abs(v_anc)
        abs_bat = torch.abs(v_bat)

        # 2. 计算元素级最小值和最大值
        min_vals = torch.min(abs_anc, abs_bat)
        max_vals = torch.max(abs_anc, abs_bat) + 1e-9 # 防止除零

        # 3. 核心判定规则：
        # 规则 1: 量级比例必须大于阈值 (排除极端异常点)
        # 规则 2: 方向必须一致 (v_anc * v_bat > 0)
        # 使用位运算加速
        mask = (min_vals / max_vals > threshold) & (v_anc * v_bat > 0)

        # 4. 执行 Mask 操作
        # 不满足条件的位点全部归零
        v_anc_masked = v_anc * mask
        v_bat_masked = v_bat * mask

        return v_anc_masked, v_bat_masked

    def get_batch_lr(self, g_train_batch, g_val_batch, val_loss_batch, current_epoch, scheduled_lr=None, w_inertia = 1.0, max_lr_limit=1.0):
        """
        [Batch Level] 使用凝聚态虚拟锚点进行实时保护，不修改历史
        正常来说，g_val_batch 需要在 g_train_batch 上投影，但实际调用中，其为同一个tensor，所以无需此操作
        """
        if self.v_anchor_p is None: return self.eta_base, 1.0, 0, 0.0
        
        # Batch 级只做简单的投影计算 S_t，不涉及混乱度
        # 物理意义：查看当前脉冲是否符合池化层确立的宏观方向
        p_batch = g_train_batch * self.eta_base # 探测用基准步长
        l_batch = g_val_batch * (self.D / (val_loss_batch + 1e-7))
        
        # 1. 计算当前相对于"池化凝聚态锚点"的三个一致性分量
        p_anc = self.v_anchor_p.to(DEVICE)
        l_anc = self.v_anchor_l.to(DEVICE)

        # 3. [核心步骤] 执行互投影过滤
        # 设定阈值为 0.1 (即：能量级差距超过 10 倍的分量被视为噪声)
        p_anc_m, p_bat_m = self.mutual_projection_mask(p_anc, p_batch, threshold=MUTUAL_PROJECTION_TTHRESHOLD)
        l_anc_m, l_bat_m = self.mutual_projection_mask(l_anc, l_batch, threshold=MUTUAL_PROJECTION_TTHRESHOLD)
        
        e2b_loss_p_ratio = self.epoch2batch_anchor_p_loss / self.fixed_e2b_loss_p_base
        e2b_loss_l_ratio = self.epoch2batch_anchor_l_loss / self.fixed_e2b_loss_l_base

        c_p_raw = self._calc_cosine(p_bat_m, p_anc_m)
        c_p = torch.clamp(c_p_raw / (e2b_loss_p_ratio + 1e-9), 0.0, 1.0)
        c_l_raw = self._calc_cosine(l_bat_m, l_anc_m)
        c_l = torch.clamp(c_l_raw / (e2b_loss_l_ratio + 1e-9), 0.0, 1.0)
        
        # (c) 比例一致性 C_phi
        # 比例计算建议使用 Mask 后的模长，反映有效信号的能量配比
        norm_p_bat = torch.norm(p_bat_m)
        norm_l_bat = torch.norm(l_bat_m)
        norm_p_anc = torch.norm(p_anc_m)
        norm_l_anc = torch.norm(l_anc_m)

        ratio_curr = norm_l_bat / (norm_p_bat + 1e-9)
        ratio_best = norm_l_anc / (norm_p_anc + 1e-9)
        
        k_ratio_raw = torch.min(ratio_curr, ratio_best) / (torch.max(ratio_curr, ratio_best) + 1e-9)
        k_ratio = torch.clamp(k_ratio_raw /(e2b_loss_p_ratio*e2b_loss_l_ratio + 1e-9), 0.0, 1.0)
        c_phi_raw = torch.clamp(torch.sin(np.pi / 2 * k_ratio_raw), 0.0, 1.0)
        c_phi = torch.clamp(torch.sin(np.pi / 2 * k_ratio), 0.0, 1.0)
        
        # 三角度乘积法计算同态度 S (Relativistic Consistency)
        # S_batch 越小，说明当前步越属于"无法被系统吸收的高能自旋/噪声"
        S_batch = torch.clamp(c_p * c_l * c_phi, 0.0, 1.0).item()
        
        # 规模修正 (考虑池化层实际大小的影响)
        S_batch = S_batch * len(self.elite_pool) / self.elite_size + (1-len(self.elite_pool)/self.elite_size)
        # S_batch = torch.clamp(c_p_raw * c_l_raw * c_phi_raw, 0.0, 1.0).item()

        # 2. 当采用外部scheduler提供基础学习率时，重新调制学习率
        # if scheduled_lr is not None:
        #     self.eta_base = scheduled_lr
        #     self.modulated_base_lr = self.eta_base * self.gamma_factor
        # else:
        #     scheduled_lr = self.eta_base

        eta_rel = self.modulated_base_lr * S_batch
        
        # 3. 耗散解冻 (基于与池内最优时刻的 Epoch 距离)
        delta_t = current_epoch - self.t_best_anchor
        omega = 1.0 - np.exp(-self.beta * delta_t)
        eta_thaw = self.eta_base * omega
        


        # 最终步长：取防御值与保底值的最大者 与 scheduler提供lr的几何平均
        final_lr = min(max(eta_rel, eta_thaw), self.eta_base, 1.5 * eta_rel) # 防止无限发散
        
        if scheduled_lr is not None:
            # if final_lr > scheduled_lr:
            #     # 强制向scheduled_lr靠拢以收敛到0
            #     w_inertia = w_inertia / (final_lr / scheduled_lr+1e-9) **0.5
            final_lr = ((final_lr)**w_inertia) * ((scheduled_lr) **(1.0 - w_inertia)) 
        else:
            scheduled_lr = final_lr
        # 这里采用调和平均
        if self.lr_window > max_lr_limit:
            # final_lr = (1/final_lr**2 + 1/max_lr_limit**2)**(-0.5) # 调和平均防止过大值的影响过于剧烈 
            final_lr = final_lr * max_lr_limit / (self.lr_window + 1e-9) # 直接缩放到限制范围内 

        # --- 定时调试输出 (每 10 个 Batch) ---
        self.batch_print_counter += 1
        if self.batch_print_counter % 50 == 0:
            print(f"      (B-Reg) S_b:{S_batch:.4f} [c_p:{c_p:.2f} c_p_raw:{c_p_raw:.2f} e2b_loss_p_ratio: {self.epoch2batch_anchor_p_loss/self.fixed_e2b_loss_p_base:.2f} | c_l:{c_l:.2f} c_l_raw:{c_l_raw:.2f} e2b_loss_l_ratio: {self.epoch2batch_anchor_l_loss/self.fixed_e2b_loss_l_base:.2f} | c_phi:{c_phi:.2f} c_phi_raw:{c_phi_raw:.2f}] "
                  f"| LR_rel:{eta_rel:.7f} | LR_thaw:{eta_thaw:.7f} | dt:{delta_t:.2f} | scheduled_lr:{scheduled_lr:.7f} | lr_window:{self.lr_window:.7f} | max_lr_limit:{max_lr_limit:.7f} | final_lr:{final_lr:.7f}")
            
        
        return final_lr, S_batch, delta_t, omega
    

# --- 辅助功能 (复用实验1/2) ---
def get_grad_vector(model):
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(grads) if grads else torch.tensor([]).to(DEVICE)



# def get_grad_vector(model):
#     grads = []
#     for p in model.parameters():
#         if p.grad is not None:
#             grads.append(p.grad.view(-1))
#     if not grads: return torch.tensor([]).to(DEVICE)
#     return torch.cat(grads)





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
    return nn.Sequential(Flatten(), make_reservoir(3072, int(width), a_level, depth=8))

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
    return nn.Sequential(conv_part, Flatten(), make_reservoir(4096, int(width), a_level, depth=4))

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
    return nn.Sequential(conv_part, Flatten(), make_reservoir(4096, int(width), a_level, depth=4))


# 封装供 auto_scale 调用
def node_00(w): return make_b0_series(w, 0)
def node_10(w): return make_b0_series(w, 1)
def node_20(w): return make_b0_series(w, 2)
def node_01(w): return make_b1_series(w, 0)
def node_11(w): return make_b1_series(w, 1)
def node_21(w): return make_b1_series(w, 2)
def node_02(w): return make_b2_series(w, 0)
def node_12(w): return make_b2_series(w, 1)
def node_22(w): return make_b0_series(w, 2)

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

def get_branch_loaders(branch_type="shock", task_id=0):
    """
    针对实验三设计的特殊 Loader
    branch_type: 
      - "shock": 0  干净, 1 epoch 注入 50% 标签噪声 (考察防撞能力)
      - "continual": Task 0 (0-4类) -> Task 1 (5-9类) (考察知识保护)
    """
    tf = transforms.Compose([
        transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
    full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
    
    if branch_type == "shock":
        # Shock 实验使用全量数据，噪声在训练循环中动态注入
        return torch.utils.data.DataLoader(full_train, BATCH_SIZE, True), \
               torch.utils.data.DataLoader(full_test, 256, False)
               
    elif branch_type == "continual":
        # 类别拆分逻辑
        targets_train = np.array(full_train.targets)
        targets_test = np.array(full_test.targets)
        
        # 确定当前任务需要的标签范围
        if task_id == 0:
            mask_train = (targets_train < 5)
            mask_test = (targets_test < 5)   # 测试集也只取对应类别，防止认知外干扰
        else:
            mask_train = (targets_train >= 5)
            mask_test = (targets_test >= 5)
            
        train_idx = np.where(mask_train)[0]
        test_idx = np.where(mask_test)[0]
        
        # 构建子集
        sub_train = torch.utils.data.Subset(full_train, train_idx)
        sub_test = torch.utils.data.Subset(full_test, test_idx)
        
        # 返回对应任务的训练集和【该任务对应的专用测试集】
        return torch.utils.data.DataLoader(sub_train, BATCH_SIZE, shuffle=True), \
               torch.utils.data.DataLoader(sub_test, 256, shuffle=False)

# --- 1. 突变预警辅助函数 ---
def detect_and_alert_shock(current_s, last_s, epoch, mode):
    """
    监测逻辑一致性崩溃
    """
    if last_s is not None and current_s < (last_s * 0.5):
        print(f"\n⚠️  [LOGIC SHOCK DETECTED] Epoch {epoch} | Mode: {mode}")
        print(f"   >>> Consistency $S_t$ collapsed from {last_s:.4f} to {current_s:.4f}")
        print(f"   >>> R-S Regulator: Relativistic Braking Engaged (Freezing Parameters).\n")
        return True
    return False


# --- 3. 手动更新辅助函数 (Manual SGD with Momentum) ---
def manual_sgd_step(model, grad_list, momentum_buffer, lr, momentum, weight_decay):
    """
    grad_list: 预先保存的梯度列表，防止被环境探测覆盖
    """
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            if grad_list[i] is None: continue
            
            d_p = grad_list[i].data # 使用传入的训练梯度
            
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)
            
            if momentum != 0:
                if name not in momentum_buffer:
                    buf = momentum_buffer[name] = torch.clone(d_p).detach()
                else:
                    buf = momentum_buffer[name]
                    buf.mul_(momentum).add_(d_p)
                d_p = buf
            
            p.data.add_(d_p, alpha=-lr)

# --- 4. 逻辑冲击分支实验 (修正版：实时因果控制 + 双循环结构) ---
def run_branch_shock(factory, name, fixed_D, fixed_E2B_LOSS_P_BASE=None, fixed_E2B_LOSS_L_BASE=None):
    print(f"\n{'='*30}\n🚀 SHOCK TEST: {name}\n{'='*30}")
    
    train_loader, val_loader = get_branch_loaders(branch_type="shock")
    model = auto_scale(factory).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # 物理系统
    tracker = OrthogonalPhysicsTracker(velocity_window=VELOCITY_WINDOW, manual_D=fixed_D, fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    regulator = RSInertiaRegulator(LR_BASE, elite_size=ANCHOR_WINDOW, tau=TAU_RELAX, fixed_D=fixed_D, fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    tracker.is_warmed_up = True
    
    momentum_buffer = {}

    dummy_optimizer = optim.SGD(model.parameters(), lr=LR_BASE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    gamma = (0.01)**(1/MAX_EPOCHS_LIMIT)
    scheduler = optim.lr_scheduler.ExponentialLR(dummy_optimizer, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=MAX_EPOCHS_LIMIT)

    history = []
    
    for epoch in range(MAX_EPOCHS_LIMIT):
        is_noisy = (epoch >= MAX_EPOCHS_LIMIT // 2) and (epoch % 2 == 0) # 从中点开始，每隔一个 epoch 注入一次噪声
        if epoch == MAX_EPOCHS_LIMIT // 2: print(f"\n📢 [SHOCK] Injecting 100% Noise...")

        if 'dummy_optimizer' in locals():
            current_lr_epoch = dummy_optimizer.param_groups[0]['lr']
        else:
            current_lr_epoch = LR_BASE
        # ------------------------------------------------------
        # 步骤 A: 训练与因果探测 (同一 Batch 内完成)
        # ------------------------------------------------------
        model.train()
        last_S_t, last_lr = 1.0, LR_BASE
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            if is_noisy:
                # y = torch.where(torch.rand_like(y.float()) < 0.5, torch.randint(0, 10, y.shape).to(DEVICE), y.to(DEVICE))
                y = torch.randint(0, 10, y.shape).to(DEVICE)
            x, y = x.to(DEVICE), y.to(DEVICE).long()

            # 1. 探测“前状态”梯度
            model.zero_grad()
            out_pre = model(x); loss_pre = criterion(out_pre, y); loss_pre.backward()
            g_train_vec = get_grad_vector(model)
            g_train_list = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            
            # 2. 调节器裁定当前 LR
            # 注意：此处使用的 g_val 逻辑需要保持一致，为计算方便用 g_train_vec 探测
            curr_lr, S_t, dt, omega = regulator.get_batch_lr(g_train_vec, g_train_vec, loss_pre.item(), epoch, max_lr_limit=current_lr_epoch)
            
            # 3. [物理记录] 记录因果链的起点：规则位移 dR
            tracker.step_batch_rule(g_train_vec, curr_lr)
            
            # 4. [手动更新] 执行更新
            manual_sgd_step(model, g_train_list, momentum_buffer, curr_lr, MOMENTUM, WEIGHT_DECAY)
            
            # 5. [因果探测] 探测“后状态”增益 (在相同 x, y 上)
            model.eval()
            with torch.set_grad_enabled(True): # 强制开启梯度用于探测 dS_ext
                model.zero_grad()
                out_post = model(x); loss_post = criterion(out_post, y); loss_post.backward()
                g_ripple_vec = get_grad_vector(model) # 这就是因果相连的相邻状态
                
                # [物理记录] 记录因果链的终点：状态位移 dS_ext
                tracker.step_batch_statext(g_ripple_vec, curr_lr, loss_post.item())
            
            model.train()
            last_S_t, last_lr = S_t, curr_lr

        # ------------------------------------------------------
        # 步骤 B: 宏观环境评估 (背景势能)
        # ------------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE).long()
                total_val_loss += criterion(model(vx), vy).item()
        avg_v_loss = total_val_loss / len(val_loader)

        scheduler.step()

        # ------------------------------------------------------
        # 步骤 C: 轮次结算
        # ------------------------------------------------------
        v_anc, v_ins, lr_snr = tracker.step_epoch(avg_v_loss)
        regulator.record_epoch_state(tracker, epoch)
        
        status = "🔴 SHOCK" if v_ins > 0.9 else "🟢 STABLE"
        print(f"   Ep {epoch:02d} | L:{avg_v_loss:.3f} | v_anc:{v_anc:.3f} | v_ins:{v_ins:.3f} | S_t:{last_S_t:.4f} | LR:{last_lr:.10f} | SNR:{lr_snr:.2f} | {status}")
        history.append({'epoch': epoch, 'v_anc': v_anc, 'v_ins': v_ins, 'S_t': last_S_t, 'LR': last_lr, 'loss': avg_v_loss, 'snr': lr_snr})
        
    return history




def evaluate_old_task(model, val_loader_old):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader_old:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            outputs = model(x)
            loss_sum += criterion(outputs, y).item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return loss_sum / len(val_loader_old), 100.0 * correct / total


_, global_val_loader_full = get_branch_loaders()
def evaluate_full_task(model):
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in global_val_loader_full:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            outputs = model(x)
            loss_sum += criterion(outputs, y).item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return loss_sum / len(global_val_loader_full), 100.0 * correct / total


def run_branch_continual(factory, name, fixed_D, fixed_E2B_LOSS_P_BASE=None, fixed_E2B_LOSS_L_BASE=None):
    print(f"\n{'='*30}\n🚀 REGULATED CONTINUAL TEST: {name}\n{'='*30}")
    
    tasks = [0, 1]
    epochs_per_task = MAX_EPOCHS_LIMIT // 2
    model = auto_scale(factory).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # 初始化物理系统
    tracker = OrthogonalPhysicsTracker(velocity_window=VELOCITY_WINDOW, manual_D=fixed_D,fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE,fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    regulator = RSInertiaRegulator(LR_BASE, elite_size=ANCHOR_WINDOW, tau=TAU_RELAX, fixed_D=fixed_D, fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    tracker.is_warmed_up = True
    
    dummy_optimizer = optim.SGD(model.parameters(), lr=LR_BASE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    gamma = (0.01)**(1/MAX_EPOCHS_LIMIT)
    scheduler = optim.lr_scheduler.ExponentialLR(dummy_optimizer, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=MAX_EPOCHS_LIMIT)

    momentum_buffer = {}
    history = []

    # 预先加载旧任务 Loader 用于记忆评估
    _, val_loader_old = get_branch_loaders(branch_type="continual", task_id=0)

    for task_id in tasks:
        print(f"\n📢 [TASK {task_id}] Switching Reference Frame...")
        train_dl, val_dl = get_branch_loaders(branch_type="continual", task_id=task_id)

        for epoch in range(epochs_per_task):

            if 'dummy_optimizer' in locals():
                current_lr_epoch = dummy_optimizer.param_groups[0]['lr']
            else:
                current_lr_epoch = LR_BASE
            abs_epoch = task_id * epochs_per_task + epoch
            model.train()
            
            last_S_t, last_lr = 1.0, LR_BASE

            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                
                # --- 1. dR 测量 ---
                model.zero_grad()
                criterion(model(x), y).backward()
                g_train_vec = get_grad_vector(model)
                g_train_list = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                
                # --- 2. 调节决策 (使用当前感知) ---
                curr_lr, S_t, dt, omega = regulator.get_batch_lr(g_train_vec, g_train_vec, 2.3, abs_epoch, max_lr_limit=current_lr_epoch) # 基准势能
                
                tracker.step_batch_rule(g_train_vec, curr_lr)
                manual_sgd_step(model, g_train_list, momentum_buffer, curr_lr, MOMENTUM, WEIGHT_DECAY)
                
                # --- 3. dS_ext 测量 (因果映射) ---
                model.eval()
                with torch.set_grad_enabled(True):
                    model.zero_grad()
                    v_out = model(x); v_loss = criterion(v_out, y); v_loss.backward()
                    g_ripple_vec = get_grad_vector(model)
                    tracker.step_batch_statext(g_ripple_vec, curr_lr, v_loss.item())
                
                model.train()
                last_S_t, last_lr = S_t, curr_lr

            # --- Epoch 结算 ---
            model.eval()
            total_v_loss = 0.0
            with torch.no_grad():
                for vx, vy in val_dl:
                    total_v_loss += criterion(model(vx.to(DEVICE)), vy.to(DEVICE).long()).item()
            avg_val_loss = total_v_loss / len(val_dl)

            v_anc, v_ins, lr_snr = tracker.step_epoch(avg_val_loss)
            regulator.record_epoch_state(tracker, abs_epoch)

            scheduler.step()
            
            # --- 7. 跨任务记忆评估 (重点：检测是否遗忘了 Task 0) ---
            old_loss, old_acc = evaluate_old_task(model, val_loader_old)
            # 8. 全任务评估
            full_loss, full_acc = evaluate_full_task(model)
            
            # 状态判定
            status = "❄️ PROTECT" if last_S_t < 0.1 else "🔥 ADAPT"
            if v_ins > 0.9: status = "🔴 SHOCK"

            print(f"   Ep {abs_epoch:02d} | T:{task_id} | L_curr:{avg_val_loss:.3f} | L_old:{old_loss:.3f} Acc_old:{old_acc:.2f}% | L_full:{full_loss:.3f} Acc_full:{full_acc:.2f}% | "
                  f"v_ins:{v_ins:.3f} | S_t:{last_S_t:.4f} | SNR:{lr_snr:.2f} | LR:{last_lr:.10f} | {status}")
            
            history.append({
                'epoch': abs_epoch, 
                'task': task_id, 
                'loss_curr': avg_val_loss, 
                'loss_old': old_loss,
                'acc_old': old_acc, 
                'loss_full': full_loss,
                'acc_full': full_acc,
                'v_anc': v_anc, 
                'v_ins': v_ins, 
                'S_t': last_S_t, 
                'lr': last_lr, 
                'lr_snr': lr_snr,
                'omega': omega
            })
            
    return history
def run_branch_baseline(factory, name, test_type="continual", fixed_D = None, fixed_E2B_LOSS_P_BASE=None, fixed_E2B_LOSS_L_BASE=None):
    """
    对照组核心逻辑：执行标准优化算法 (Cosine SGD)
    虽然不进行调节，但同步记录物理速度 (v_anc, v_ins) 以便进行量化对比。
    """
    print(f"\n{'='*30}\n💀 BASELINE {test_type.upper()}: {name} (Control Group)\n{'='*30}")
    
    # 1. 架构与优化器初始化 (使用与实验组完全一致的参数)
    model = auto_scale(factory).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR_BASE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    momentum_buffer = {}
    history = []

    # 2. 物理追踪器 (仅作为观察者观测 Baseline 产生的混乱速度)
    tracker = OrthogonalPhysicsTracker(velocity_window=VELOCITY_WINDOW, manual_D=fixed_D,fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE,fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    tracker.is_warmed_up = True 

    # --- 分支 A: 逻辑冲击 (Shock Test) ---
    if test_type == "shock":
        train_loader, val_loader = get_branch_loaders(branch_type="shock")
        # 对照组使用标准的余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_LIMIT)
        
        for epoch in range(MAX_EPOCHS_LIMIT):
            is_noisy = (epoch >= MAX_EPOCHS_LIMIT // 2) and (epoch % 2 == 0)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 步骤 1: 训练循环 (执行标准更新)
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                if is_noisy:
                    # y = torch.where(torch.rand_like(y.float()) < 0.5, torch.randint(0, 10, y.shape).to(DEVICE), y)
                    y = torch.randint(0, 10, y.shape).to(DEVICE)
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                
                # 获取前状态梯度 (dR)
                model.zero_grad()
                out_pre = model(x); loss_pre = criterion(out_pre, y); loss_pre.backward()
                g_train_vec = get_grad_vector(model)
                g_train_list = [p.grad.clone() for p in model.parameters()]
                
                # [记录] 内部做功 dR
                tracker.step_batch_rule(g_train_vec, current_lr)
                
                # [手动执行更新] 模拟标准 SGD，没有任何阻尼
                manual_sgd_step(model, g_train_list, momentum_buffer, current_lr, MOMENTUM, WEIGHT_DECAY)
                
                # [探测] 记录应用更新后的因果反馈 (dS_ext)
                model.eval()
                with torch.set_grad_enabled(True):
                    model.zero_grad()
                    out_post = model(x); loss_post = criterion(out_post, y); loss_post.backward()
                    tracker.step_batch_statext(get_grad_vector(model), current_lr, loss_post.item())
                model.train()

            # 步骤 2: 宏观背景探测 (Epoch 结算)
            model.eval()
            total_v_loss = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    total_v_loss += criterion(model(vx.to(DEVICE)), vy.to(DEVICE).long()).item()
            avg_v_loss = total_v_loss / len(val_loader)
            
            v_anc, v_ins, _ = tracker.step_epoch(avg_v_loss)
            scheduler.step() # 推进标准调度

            print(f"   [Base-Shock] Ep {epoch:02d} | L:{avg_v_loss:.3f} | v_ins:{v_ins:.3f} | LR:{current_lr:.6f}")
            history.append({'epoch': epoch, 'loss': avg_v_loss, 'v_ins': v_ins, 'lr': current_lr})

    # --- 分支 B: 持续学习 (Continual Test) ---
    elif test_type == "continual":
        # 预加载旧任务(Task 0)加载器，用于全程观测遗忘曲线
        _, val_loader_old = get_branch_loaders(branch_type="continual", task_id=0)
        
        tasks = [0, 1]
        epochs_per_task = MAX_EPOCHS_LIMIT // 2
        # 基准组通常不重置调度器，模拟持续训练
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS_LIMIT)
        
        for task_id in tasks:
            print(f"\n📢 [Baseline] Switching to Task {task_id}...")
            train_loader, val_loader = get_branch_loaders(branch_type="continual", task_id=task_id)
            
            for epoch in range(epochs_per_task):
                abs_epoch = task_id * epochs_per_task + epoch
                current_lr = optimizer.param_groups[0]['lr']
                
                model.train()
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE).long()
                    
                    # 1. 训练梯度记录 (dR)
                    model.zero_grad(); criterion(model(x), y).backward()
                    gt_vec = get_grad_vector(model)
                    gt_list = [p.grad.clone() for p in model.parameters()]
                    tracker.step_batch_rule(gt_vec, current_lr)
                    
                    # 2. 执行盲目更新 (不加保护)
                    manual_sgd_step(model, gt_list, momentum_buffer, current_lr, MOMENTUM, WEIGHT_DECAY)
                    
                    # 3. 因果反馈记录 (dS_ext)
                    model.eval()
                    with torch.set_grad_enabled(True):
                        model.zero_grad(); v_out = model(x); v_l = criterion(v_out, y); v_l.backward()
                        tracker.step_batch_statext(get_grad_vector(model), current_lr, v_l.item())
                    model.train()

                # 结算本 Epoch 物理状态
                model.eval()
                total_v_l = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader: total_v_l += criterion(model(vx.to(DEVICE)), vy.to(DEVICE).long()).item()
                avg_l = total_v_l / len(val_loader)
                
                v_anc, v_ins, _ = tracker.step_epoch(avg_l)
                
                # --- 核心评估：双重指标监测 ---
                # 评估旧任务 0 的保持程度 (遗忘观测)
                old_l, old_acc = evaluate_old_task(model, val_loader_old)
                # 评估全量 10 类的表现 (综合能力观测)
                full_l, full_acc = evaluate_full_task(model)
                
                print(f"   [Base-Cont] Ep {abs_epoch:02d} | T:{task_id} | v_ins:{v_ins:.3f} | OldL:{old_l:.3f} | OldAcc:{old_acc:.1f}% | FullAcc:{full_acc:.1f}% | LR:{current_lr:.6f}")
                
                history.append({
                    'epoch': abs_epoch, 'loss_old': old_l, 'acc_old': old_acc, 
                    'loss_full': full_l, 'acc_full': full_acc, 'v_ins': v_ins, 'lr': current_lr
                })
                scheduler.step()
                
    return history

def run_limit_test(factory, name, fixed_D, controller_type="inertia", scheduler_name=None, fixed_E2B_LOSS_P_BASE=None, fixed_E2B_LOSS_L_BASE=None,composed=False,w_inertia=0.33):
    """
    计算不同调控器下的误差降低极限 (L_min)
    factory: 模型工厂
    fixed_D: 定标后的物理常数
    controller_type: "inertia" (UIT调节器) 或 "standard" (标准调度器)
    """
    print(f"\n{'='*40}")
    print(f"🌡️  LIMIT TEST: {name} | Controller: {controller_type.upper()} ({scheduler_name if scheduler_name else ''})")
    print(f"{'='*40}")
    
    # 使用纯净数据 (branch_type="shock" 且无噪声)
    train_loader, val_loader = get_branch_loaders(branch_type="shock") 
    model = auto_scale(factory).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # 1. 物理追踪系统 (观测所有调控器的物理表现)
    tracker = OrthogonalPhysicsTracker(velocity_window=VELOCITY_WINDOW, manual_D=fixed_D,fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE,fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)
    tracker.is_warmed_up = True
    
    # 2. 调控器初始化
    regulator = None
    scheduler = None
    momentum_buffer = {}
    
    # 虚拟优化器，仅用于辅助 PyTorch Scheduler 管理 LR 状态
    dummy_optimizer = optim.SGD(model.parameters(), lr=LR_BASE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    regulator = RSInertiaRegulator(LR_BASE, elite_size=ANCHOR_WINDOW, tau=TAU_RELAX, fixed_D=fixed_D, fixed_E2B_LOSS_P_BASE=fixed_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=fixed_E2B_LOSS_L_BASE)

    if controller_type == "standard" or composed == True:
        # 对照组：8 种业界常用调度器，涵盖了不同的收敛哲学
        if scheduler_name == 'cosine':
            # [平滑收敛标杆] 始终不重启，最常用的强力基准
            scheduler = optim.lr_scheduler.CosineAnnealingLR(dummy_optimizer, T_max=MAX_EPOCHS_LIMIT)
            
        elif scheduler_name == 'cosine_restart':
            # [主动探索标杆] 模拟周期性跳出局部最优
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(dummy_optimizer, T_0=MAX_EPOCHS_LIMIT//2)
            
        elif scheduler_name == 'one_cycle':
            # [快速收敛标杆] 模仿快速升温再退火，学术界公认的“超收敛”策略
            scheduler = optim.lr_scheduler.OneCycleLR(dummy_optimizer, max_lr=LR_BASE*1.5, 
                                                       total_steps=MAX_EPOCHS_LIMIT, steps_per_epoch=1)
            
        elif scheduler_name == 'plateau':
            # [结果驱动标杆] 唯一不依赖时间，只依赖 Loss 反馈的经典方法
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(dummy_optimizer, mode='min', factor=0.5, patience=5)
            
        elif scheduler_name == 'multistep':
            # [阶梯衰减标杆] 工业界最传统的、分阶段收敛的方法
            scheduler = optim.lr_scheduler.MultiStepLR(dummy_optimizer, 
                                                       milestones=[MAX_EPOCHS_LIMIT//3, 2*MAX_EPOCHS_LIMIT//3], 
                                                       gamma=0.1)

        elif scheduler_name == 'cyclic':
            # [震荡搜索标杆] 在两个能级之间不断震荡。
            # 这与我们的"解冻"逻辑形成了有趣的对比：它是盲目震荡，我们是按需解冻。
            scheduler = optim.lr_scheduler.CyclicLR(dummy_optimizer, base_lr=LR_BASE/10, max_lr=LR_BASE, 
                                                     step_size_up=MAX_EPOCHS_LIMIT//4, mode='triangular2')

        elif scheduler_name == 'exponential':
            # [恒定衰减标杆] 物理上的纯阻尼模拟，不考虑任何地形变化。
            # 自动计算 gamma 使得在最后一步正好衰减到初始的 1%
            gamma = (0.01)**(1/MAX_EPOCHS_LIMIT)
            scheduler = optim.lr_scheduler.ExponentialLR(dummy_optimizer, gamma=gamma)

        elif scheduler_name == 'polynomial':
            # [凸性衰减标杆] 常用于语义分割等复杂任务。
            # power=2.0 使得衰减曲线比线性更陡，比余弦更晚进入深水区。
            scheduler = optim.lr_scheduler.PolynomialLR(dummy_optimizer, total_iters=MAX_EPOCHS_LIMIT, power=2.0)            

    history = []
    best_loss_overall = float('inf') 
    best_epoch = -1

    # ------------------------------------------------------
    # 核心训练循环 (因果探测结构)
    # ------------------------------------------------------
    for epoch in range(MAX_EPOCHS_LIMIT):
        model.train()
        
        # === [修复点 1] 显式初始化本轮基准学习率 ===
        # 无论哪种模式，先从 dummy_optimizer 获取当前被调度器锁定的学习率
        if 'dummy_optimizer' in locals():
            current_lr_epoch = dummy_optimizer.param_groups[0]['lr']
        else:
            current_lr_epoch = LR_BASE

        epoch_S_t_sum = 0.0
        batch_count = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            
            # --- A. 获取前状态梯度 (dR 趋势) ---
            model.train()
            model.zero_grad()
            out_pre = model(x); loss_pre = criterion(out_pre, y); loss_pre.backward()
            
            g_train_vec = get_grad_vector(model)
            g_train_list = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            
            # --- B. 调控器实时裁定 ---
            if controller_type == "inertia" or composed == True:
                # 惯性组：在当前调度器步长的基础上，进行相对论过滤
                # [修复点 2] 将当前的 current_lr_epoch 作为 base 传入调节器
                curr_lr, S_t, dt, omega = regulator.get_batch_lr(
                    g_train_vec, 
                    g_train_vec, 
                    loss_pre.item(), 
                    epoch, 
                    scheduled_lr=current_lr_epoch if composed else None, # 传递当前调度器建议的步长
                    w_inertia=w_inertia
                )
                epoch_S_t_sum += S_t
            
            if controller_type == "standard" and not composed:
                # 标准组：直接沿用 Epoch 级的调度器步长，不进行任何物理干预
                curr_lr = current_lr_epoch
                S_t = 1.0
                epoch_S_t_sum += 1.0
            
            # --- C. 物理记录 (因果起点) ---
            tracker.step_batch_rule(g_train_vec, curr_lr)
            
            # --- D. 手动更新权重 (即时制动) ---
            manual_sgd_step(model, g_train_list, momentum_buffer, curr_lr, MOMENTUM, WEIGHT_DECAY)
            
            # --- E. 因果探测 (测量产生的 dS_ext) ---
            # 必须在同一 batch 数据上测量更新后的反应
            model.eval()
            with torch.set_grad_enabled(True):
                model.zero_grad()
                out_post = model(x); loss_post = criterion(out_post, y); loss_post.backward()
                g_ripple_vec = get_grad_vector(model)
                
                # [核心修正] 记录因果反馈位移
                tracker.step_batch_statext(g_ripple_vec, curr_lr, loss_post.item())
            
            batch_count += 1

        # ------------------------------------------------------
        # Epoch 结算 (宏观评估 L_min)
        # ------------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE).long()
                total_val_loss += criterion(model(vx), vy).item()
                val_steps += 1
        avg_v_loss = total_val_loss / val_steps

        # 记录热力学极限
        if avg_v_loss < best_loss_overall:
            best_loss_overall = avg_v_loss
            best_epoch = epoch

        # 1. 物理观测结算 (由 Tracker 统一处理矢量坍缩)
        v_anc, v_ins, lr_snr = tracker.step_epoch(avg_v_loss)
        
        # 2. 推进记忆/调度器
        if controller_type == "standard":
            if scheduler_name == 'plateau':
                scheduler.step(avg_v_loss)
            else:
                scheduler.step()

        if controller_type == "inertia" or composed == True:
            regulator.record_epoch_state(tracker, epoch)
        

        # 3. 日志输出
        avg_S_t = epoch_S_t_sum / batch_count
        print(f"   Ep {epoch:02d} | Loss:{avg_v_loss:.4f} | MinL:{best_loss_overall:.4f} | bestEp:{best_epoch} | v_ins:{v_ins:.3f} | "
              f"v_anc:{v_anc:.3f} | S_t_avg:{avg_S_t:.3f} | SNR:{lr_snr:.2f} | LR:{curr_lr:.8f}")
        
        history.append({
            'epoch': epoch, 
            'loss': avg_v_loss, 
            'min_loss': best_loss_overall,
            'best_epoch': best_epoch,
            'velocity': v_ins, 
            'lr': curr_lr,
            'lr_snr': lr_snr,
            'S_t': avg_S_t
        })

    return history


def run_experiment_3():
    # 架构配置 (建议使用 ResNet-GAP node_22)
    target_factory = node_22
    target_name = "ResNet-GAP"
    
    # ==============================================================
    # [Phase 1] 自动定标 (Calibration)
    # 使用 node_22 (ResNet-GAP) 作为基准，确定 ||D|| 常数
    # ==============================================================
    print(f"\n{'='*30}")
    print(f"🚀 Phase 1: Calibrating Universe Constant ||D||")
    print(f"{'='*30}")

    # 1. 准备定标加载器
    # 使用 shock 模式获取干净的 CIFAR-10 全量数据
    train_dl_calib, val_dl_calib = get_branch_loaders(branch_type="shock")

    # 2. 构建定标模型与环境
    base_cpu = auto_scale(node_22) # 使用 ResNet-GAP 进行定标
    base = base_cpu.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 3. 初始化手动更新所需的动量缓冲区
    momentum_buffer = {}
    calib_tracker = OrthogonalPhysicsTracker(manual_D=None)

    for epoch in range(WARMUP_EPOCHS):
        # ------------------------------------------------------
        # 步骤 A: 训练阶段 (测量规则位移 dR)
        # ------------------------------------------------------
        base.train()
        print(f"   Epoch {epoch:02d} | [Training Path] Accumulating dR...")
        
        for epoch in range(WARMUP_EPOCHS):
            base.train()
            for x, y in train_dl_calib:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                # dR
                base.zero_grad(); criterion(base(x), y).backward(); gt_vec = get_grad_vector(base)
                g_list = [p.grad.clone() for p in base.parameters()]
                calib_tracker.step_batch_rule(gt_vec, LR_BASE)
                manual_sgd_step(base, g_list, momentum_buffer, LR_BASE, MOMENTUM, WEIGHT_DECAY)
                # dS_ext Ripple
                base.eval()
                with torch.set_grad_enabled(True):
                    base.zero_grad(); v_out = base(x); v_loss = criterion(v_out, y); v_loss.backward()
                    calib_tracker.step_batch_statext(get_grad_vector(base),LR_BASE, v_loss.item())
            
            # Eval for Epoch
            base.eval()
            v_l_sum = 0
            with torch.no_grad():
                for vx, vy in val_dl_calib: v_l_sum += criterion(base(vx.to(DEVICE)), vy.to(DEVICE).long()).item()
            avg_val_loss = v_l_sum / len(val_dl_calib)
            # 结算本 Epoch 的有效模长，用于最终计算 D
            calib_tracker.step_epoch(avg_val_loss)
            
            print(f"   Epoch {epoch:02d} | Avg Val Loss: {avg_val_loss:.4f}")

    # 4. 锁定全局常数 ||D||, GLOBAL_E2B_LOSS_P_BASE, GLOBAL_E2B_LOSS_L_BASE
    calib_tracker.start_tracking()
    GLOBAL_D = calib_tracker.D
    GLOBAL_E2B_LOSS_P_BASE = calib_tracker.E2B_LOSS_P_BASE
    GLOBAL_E2B_LOSS_L_BASE = calib_tracker.E2B_LOSS_L_BASE
    print(f"\n✅ [Calibration Done] Global Constant Locked: ||D|| = {GLOBAL_D:.6f} | E2B_LOSS_P_BASE = {GLOBAL_E2B_LOSS_P_BASE:.6f} | E2B_LOSS_L_BASE = {GLOBAL_E2B_LOSS_L_BASE:.6f}\n")

    # 彻底释放显存
    del base, gt_vec
    torch.cuda.empty_cache()

    # Phase 2: 分支实验
    final_results = {
        'metadata': {'D': GLOBAL_D, 'LR_BASE': LR_BASE, 'E2B_LOSS_P_BASE': GLOBAL_E2B_LOSS_P_BASE, 'E2B_LOSS_L_BASE': GLOBAL_E2B_LOSS_L_BASE},
        'shock': {
            # 'regulated': run_branch_shock(target_factory, target_name, fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
            # 'baseline': run_branch_baseline(target_factory, target_name, GLOBAL_D, test_type="shock"),
        },
        'continual': {
            # 'baseline': run_branch_baseline(target_factory, target_name, test_type="continual"),
            # 'regulated': run_branch_continual(target_factory, target_name, fixed_D=GLOBAL_D),
        },

        # 'limit_results' : {
        #     'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        # },

   
    }

    # 数据导出
    # with open('exp03_final_comparison_limit_results.json', 'w') as f:
    #     json.dump(final_results, f, indent=4)
    
    final_results = {
        'metadata': {'D': GLOBAL_D, 'LR_BASE': LR_BASE, 'E2B_LOSS_P_BASE': GLOBAL_E2B_LOSS_P_BASE, 'E2B_LOSS_L_BASE': GLOBAL_E2B_LOSS_L_BASE},
        'shock': {
            'regulated': run_branch_shock(target_factory, target_name, fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
            'baseline': run_branch_baseline(target_factory, target_name, test_type="shock", fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        },
        # 'continual': {
        #     'regulated': run_branch_continual(target_factory, target_name, fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'baseline': run_branch_baseline(target_factory, target_name, test_type="continual"),
        # },

        # 'limit_results' : {
        #     'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),

            
        # },

        # 'composed_limit_results20' : {
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },

        # 'composed_limit_results20_A' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },

        # 'composed_limit_results20_B' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
        # 'composed_limit_results20_C' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
        # 'composed_limit_results20_D' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },


        # 'composed_limit_results25' : {
        #     # 'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.25),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.25),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.25),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.25),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.25),
        # },
        # 'composed_limit_results33' : {
        #     # 'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.33),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.33),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.33),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.33),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.33),
        # }, 
        # 'composed_limit_results50' : {
        #     # 'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        # },
        # 'composed_limit_results_20-75' : {
        #     'cosine_baseline20': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cosine_baseline30': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.30),
        #     'cosine_baseline40': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.40),
        #     'cosine_baseline50': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.50),
        #     'cosine_baseline60': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.70),
            
        # },


        # 'composed_limit_results20_A' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
        # 'composed_limit_results20_B' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
        # 'composed_limit_results20_C' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
        # 'composed_limit_results20_D' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=True, w_inertia=0.20),
        # },
           
    }

    with open('exp03_final_comparison_composed_limit_results1_pool6.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    


    final_results = {
        'metadata': {'D': GLOBAL_D, 'LR_BASE': LR_BASE, 'E2B_LOSS_P_BASE': GLOBAL_E2B_LOSS_P_BASE, 'E2B_LOSS_L_BASE': GLOBAL_E2B_LOSS_L_BASE},
        # 'shock': {
        #     # 'regulated': run_branch_shock(target_factory, target_name, fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     # 'baseline': run_branch_baseline(target_factory, target_name, GLOBAL_D, test_type="shock"),
        # },
        # 'continual': {
        #     # 'baseline': run_branch_baseline(target_factory, target_name, test_type="continual"),
        #     # 'regulated': run_branch_continual(target_factory, target_name, fixed_D=GLOBAL_D),
        # },

        'continual': {
            'regulated': run_branch_continual(target_factory, target_name, fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
            'baseline': run_branch_baseline(target_factory, target_name, test_type="continual", fixed_D=GLOBAL_D, fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        },

        # 'limit_results' : {
        #     'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE),

            
        # },

        # 'composed_limit_results20' : {
        #     'cosine_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     # 'inertia_regulator': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="inertia", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False),
        #     'cosine_restart_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cosine_restart", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        # },


        # 'composed_limit_results20_A' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        # },

        # 'composed_limit_results20_B' : {
        #     'one_cycle_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="one_cycle", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'plateau_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="plateau", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        # },

        # 'composed_limit_results20_A' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        # },

        # 'composed_limit_results20_B' : {
        #     'multistep_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="multistep", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'cyclic_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="cyclic", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'exponential_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="exponential", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        #     'polynomial_baseline': run_limit_test(target_factory, target_name, GLOBAL_D, controller_type="standard", scheduler_name="polynomial", fixed_E2B_LOSS_P_BASE=GLOBAL_E2B_LOSS_P_BASE, fixed_E2B_LOSS_L_BASE=GLOBAL_E2B_LOSS_L_BASE, composed=False, w_inertia=0.20),
        # },
  
    }

    with open('exp03_final_comparison_composed_limit_results2_pool6.json', 'w') as f:
        json.dump(final_results, f, indent=4)


    print("\n🎉 Experiment 3 Finished. Data exported to exp03_final_comparison.json")


if __name__ == "__main__":
    # 执行完整实验流程
    run_experiment_3()

