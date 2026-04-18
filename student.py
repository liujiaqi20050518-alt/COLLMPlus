import os

_omp_threads = os.environ.get("OMP_NUM_THREADS")
if _omp_threads is None or not _omp_threads.isdigit() or int(_omp_threads) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

from minigpt4.models.rec_base_models import MatrixFactorization, LightGCN 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 
import time
import scipy.sparse as sp # 新增依赖
# =========== 新增：构建 KNN 图滤波器 ===========
def build_graph_filter(features, num_items, k=10, alpha=0.45):
    """根据大模型特征构建 KNN 图，并返回稀疏图滤波器 H = I - alpha * L_tilde"""
    print(f"🌐 正在构建 KNN 图滤波器 (k={k}, alpha={alpha})...")
    norm_feat = F.normalize(features, p=2, dim=-1)
    sim_matrix = torch.matmul(norm_feat, norm_feat.T)
    
    _, topk_indices = torch.topk(sim_matrix, k=k+1, dim=-1)
    
    row_indices = torch.arange(num_items).view(-1, 1).expand(-1, k+1).flatten().cuda()
    col_indices = topk_indices.flatten().cuda()
    values = torch.ones_like(row_indices, dtype=torch.float32).cuda()
    
    indices = torch.stack([row_indices, col_indices])
    A = torch.sparse_coo_tensor(indices, values, (num_items, num_items)).coalesce()
    
    degree = torch.sparse.sum(A, dim=1).to_dense()
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    A_dense = A.to_dense()
    D_inv_sqrt_mat = torch.diag(d_inv_sqrt)
    L_sym = torch.eye(num_items).cuda() - torch.mm(torch.mm(D_inv_sqrt_mat, A_dense), D_inv_sqrt_mat)
    
    H = torch.eye(num_items).cuda() - alpha * L_sym
    H_sparse = H.to_sparse()
    print("✅ 滤波器构建完成！")
    return H_sparse

# !!!!!!!!版本1！！！！！！！！！！！！！！！！！！！
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
# class DynamicKDLoss(nn.Module):
#     def __init__(self, gamma=0.01):
#         super().__init__()
#         # 调小 gamma 到 0.01，让更多中低频（长尾）物品能享受到蒸馏的红利
#         # 比如出现 50 次的物品，alpha 也能保留 0.6 左右的蒸馏强度
#         self.gamma = gamma

#     def forward(self, student_logits, teacher_logits, true_labels, item_freqs):
#         # 1. 基础长尾动态权重 Alpha
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         # 2. Hard Loss (学生 vs 真实世界)
#         hard_loss = F.binary_cross_entropy_with_logits(
#             student_logits, true_labels.float(), reduction='none'
#         )
        
#         # 3. 转化为概率
#         student_probs = torch.sigmoid(student_logits)
#         teacher_probs = torch.sigmoid(teacher_logits)
        
#         # 4. 🌟 新创新：平滑置信度 (Soft Confidence) 代替一刀切！
#         # 老师的预测离真实标签越近，confidence 越大 (最高为1)；越离谱，confidence 越小 (最低趋近于0)
#         teacher_error = torch.abs(true_labels.float() - teacher_probs)
#         confidence = 1.0 - teacher_error 
        
#         # 5. 🌟 推荐专属 Soft Loss：均方误差 (MSE)
#         # 强迫学生的概率分布在二维空间上向老师靠拢，不搞温度软化那套虚的
#         soft_loss = F.mse_loss(student_probs, teacher_probs, reduction='none')
        
#         # 6. 究极融合公式
#         # MSE 的值天然比 BCE 小很多，所以我们给蒸馏 Loss 乘一个放大系数 (比如 5.0)
#         # 让长尾物品 (alpha高) 且老师有把握 (confidence高) 的样本，爆发出强大的蒸馏引导力！
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + kd_weight * soft_loss
        
#         return total_loss.mean()


# !!!!!!!!版本2！！！！！！！！！！！！！！！！！！！
# class DynamicKDLoss(nn.Module):
#     def __init__(self, gamma=0.01):
#         super().__init__()
#         self.gamma = gamma

#     def forward(self, student_logits, teacher_logits, true_labels, item_freqs):
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         hard_loss = F.binary_cross_entropy_with_logits(
#             student_logits, true_labels.float(), reduction='none'
#         )
        
#         # 🌟 开启 Hinton 蒸馏温度魔法
#         T = 6.0  
        
#         student_probs = torch.sigmoid(student_logits)
#         teacher_probs = torch.sigmoid(teacher_logits)
        
#         teacher_error = torch.abs(true_labels.float() - teacher_probs)
#         confidence = 1.0 - teacher_error 
        
#         # 🌟 使用温度软化概率分布，挖掘大模型的“暗知识”
#         student_probs_soft = torch.sigmoid(student_logits / T)
#         teacher_probs_soft = torch.sigmoid(teacher_logits / T)
        
#         # 🌟 必须乘以 T^2 来补偿梯度缩放！
#         # soft_loss = F.mse_loss(student_probs_soft, teacher_probs_soft, reduction='none') * (T * T)
#         # 用交叉熵去对齐两个分布，梯度的指向会比距离绝对值（MSE）更加敏锐和精准！
#         # 同样别忘了乘以 (T * T) 来补偿梯度缩放
#         soft_loss = F.binary_cross_entropy(student_probs_soft, teacher_probs_soft, reduction='none') * (T * T)
#         # 保持 8.0 的大模型辅导强度！
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + kd_weight * soft_loss
        
#         return total_loss.mean()

# # 版本3，加入跨模态表征对齐（修复版）
# class DualLevelKDLoss(nn.Module):
#     def __init__(self, gamma=0.01, feat_weight=0.8): 
#         super().__init__()
#         self.gamma = gamma
#         self.feat_weight = feat_weight

#     def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         # 1. 硬标签
#         hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
        
#         # 2. 软标签 (保持你的进度不变)
#         T = 5.0  
#         teacher_error = torch.abs(true_labels.float() - torch.sigmoid(teacher_logits))
#         confidence = 1.0 - teacher_error 
        
#         stu_probs_soft = torch.sigmoid(student_logits / T)
#         tea_probs_soft = torch.sigmoid(teacher_logits / T)
#         soft_loss = F.binary_cross_entropy(stu_probs_soft, tea_probs_soft, reduction='none') * (T * T)
#         kd_weight = alpha * confidence * 8.0
        
#         # 🌟 3. 跨模态表征对齐 (修复版：直接计算余弦相似度！)
#         # cosine_similarity 输出范围是 [-1, 1]。1代表完全同向。
#         # 我们用 1.0 减去它，让 Loss 的范围变成 [0, 2]，数值极其稳定！
#         cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1).unsqueeze(1)
#         feat_loss = 1.0 - cos_sim 
        
#         # 终极融合！
#         total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * feat_loss)
#         return total_loss.mean()
#===========================================================================================================
# def dct1d(x):
#     """标准的 1D DCT 变换 (保持不变)"""
#     N = x.shape[-1]
#     x_even = x[:, ::2]
#     x_odd = x[:, 1::2].flip([1])
#     v = torch.cat([x_even, x_odd], dim=1)
#     V = torch.fft.fft(v, dim=1)
#     k = -torch.arange(N, device=x.device).float() * np.pi / (2 * N)
#     W_r = torch.cos(k)
#     W_i = torch.sin(k)
#     return 2 * (V.real * W_r - V.imag * W_i)



def dct1d(x):
    """
    对输入张量的最后一个维度进行 1D 离散余弦变换 (DCT-II)
    x: (batch, dim)
    """
    N = x.shape[-1]
    # 重新排列序列以利用 FFT 计算 DCT
    x_even = x[:, ::2]
    x_odd = x[:, 1::2].flip([1])
    v = torch.cat([x_even, x_odd], dim=1)
    
    # 快速傅里叶变换
    V = torch.fft.fft(v, dim=1)
    
    # 根据 DCT-II 定义进行相位补偿
    k = -torch.arange(N, device=x.device).float() * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    
    # 提取实部作为 DCT 结果
    dct_output = 2 * (V.real * W_r - V.imag * W_i)
    return dct_output

class SpectralKDLoss(nn.Module):
    def __init__(self, gamma=0.01, feat_dim=4096, feat_weight=0.01): 
        super().__init__()
        self.gamma = gamma
        self.feat_weight = feat_weight
        self.low_bound = feat_dim // 4   
        self.mid_bound = feat_dim // 2   

    def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
        alpha = torch.exp(-self.gamma * item_freqs)
        teacher_probs = torch.sigmoid(teacher_logits)
        confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
        T = 5.0
        soft_loss = F.binary_cross_entropy(
            torch.sigmoid(student_logits/T), 
            torch.sigmoid(teacher_logits/T), 
            reduction='none'
        ) * (T * T)
        
        # ==========================================
        # 🌟 核心修复：特征归一化 (L2 Normalization)
        # ==========================================
        # 这一步极其关键！把向量的长度缩放到 1。
        # 这样能保证后续计算的平方差 (MSE) 被死死限制在安全范围内，彻底消除梯度爆炸！
        student_feat_norm = F.normalize(student_feat, p=2, dim=-1)
        teacher_feat_norm = F.normalize(teacher_feat, p=2, dim=-1)
        
        # 3. 对归一化后的安全特征进行真正的频率域拆解
        s_spectrum = dct1d(student_feat_norm)
        t_spectrum = dct1d(teacher_feat_norm)
        
        spec_diff = (s_spectrum - t_spectrum)**2
        
        loss_low = spec_diff[:, :self.low_bound].mean(dim=-1, keepdim=True)
        loss_mid = spec_diff[:, self.low_bound:self.mid_bound].mean(dim=-1, keepdim=True)
        loss_high = spec_diff[:, self.mid_bound:].mean(dim=-1, keepdim=True)
        
        # 放大低频 (2.5)，削弱高频 (0.1)
        weighted_feat_loss = 2.5 * loss_low + 1.0 * loss_mid + 0.1 * loss_high
        
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
        kd_weight = alpha * confidence * 8.0
        
        # 融合计算
        total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * weighted_feat_loss)
        return total_loss.mean()

# class SpectralKDLoss(nn.Module):
#     def __init__(self, gamma=0.01, feat_dim=4096, feat_weight=0.01): 
#         super().__init__()
#         self.gamma = gamma
#         self.feat_weight = feat_weight
        
#         # 定义频段切分点
#         self.low_bound = feat_dim // 4   
#         self.mid_bound = feat_dim // 2   

#     def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
#         # 1. 计算长尾权重 Alpha (越冷门越接近 1，越热门越接近 0)
#         # item_freqs 的形状通常是 [Batch_Size, 1]
#         alpha = torch.exp(-self.gamma * item_freqs)
        
#         teacher_probs = torch.sigmoid(teacher_logits)
#         confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
#         # 2. 预测层软标签蒸馏
#         T = 5.0
#         soft_loss = F.binary_cross_entropy(
#             torch.sigmoid(student_logits/T), 
#             torch.sigmoid(teacher_logits/T), 
#             reduction='none'
#         ) * (T * T)
        
#         # 3. 真正的频率域特征蒸馏 (包含 L2 归一化安全阀)
#         student_feat_norm = F.normalize(student_feat, p=2, dim=-1)
#         teacher_feat_norm = F.normalize(teacher_feat, p=2, dim=-1)
        
#         s_spectrum = dct1d(student_feat_norm)
#         t_spectrum = dct1d(teacher_feat_norm)
        
#         spec_diff = (s_spectrum - t_spectrum)**2
        
#         # 提取各个频段的差异能量，形状均为 [Batch_Size, 1]
#         loss_low = spec_diff[:, :self.low_bound].mean(dim=-1, keepdim=True)
#         loss_mid = spec_diff[:, self.low_bound:self.mid_bound].mean(dim=-1, keepdim=True)
#         loss_high = spec_diff[:, self.mid_bound:].mean(dim=-1, keepdim=True)
        
#         # ==========================================
#         # 🌟 核心创新：流行度感知的动态低频加权
#         # ==========================================
#         # 基础骨架权重保底为 2.0，动态增益部分由 alpha 控制（最高加 1.0）。
#         # 因为 alpha 和 loss_low 的形状都是 [Batch_Size, 1]，这里会自动实现逐样本加权！
#         dynamic_low_weight = 2.0 + (1.0 * alpha)
        
#         # 融合分段特征损失
#         weighted_feat_loss = (dynamic_low_weight * loss_low) + (1.0 * loss_mid) + (0.1 * loss_high)
        
#         # 4. 计算 Hard Loss 并汇总
#         hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
#         kd_weight = alpha * confidence * 8.0
        
#         # 使用外部控制的 self.feat_weight (建议保持 0.01 并配合预热)
#         total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * weighted_feat_loss)
        
#         return total_loss.mean()

# class SpectralKDLoss(nn.Module):
#     def __init__(self, gamma=0.01, feat_dim=4096, feat_weight=0.05): # 🌟 因为用了余弦，权重可以安心回到 0.05
#         super().__init__()
#         self.gamma = gamma
#         self.feat_weight = feat_weight
#         self.low_bound = feat_dim // 4   
#         self.mid_bound = feat_dim // 2   

#     def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
#         # 1. 基础长尾权重
#         alpha = torch.exp(-self.gamma * item_freqs)
#         teacher_probs = torch.sigmoid(teacher_logits)
#         confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
#         # 2. 预测层蒸馏
#         T = 5.0
#         soft_loss = F.binary_cross_entropy(
#             torch.sigmoid(student_logits/T), 
#             torch.sigmoid(teacher_logits/T), 
#             reduction='none'
#         ) * (T * T)
        
#         # ==========================================
#         # 🌟 核心替换部分：频域分段余弦相似度
#         # ==========================================
#         # 不需要 F.normalize，直接对原始特征做 DCT 变换
#         s_spectrum = dct1d(student_feat)
#         t_spectrum = dct1d(teacher_feat)
        
#         # 对三个频段分别计算 余弦相似度 (输出范围 [-1, 1])
#         cos_low = F.cosine_similarity(s_spectrum[:, :self.low_bound], t_spectrum[:, :self.low_bound], dim=-1).unsqueeze(1)
#         cos_mid = F.cosine_similarity(s_spectrum[:, self.low_bound:self.mid_bound], t_spectrum[:, self.low_bound:self.mid_bound], dim=-1).unsqueeze(1)
#         cos_high = F.cosine_similarity(s_spectrum[:, self.mid_bound:], t_spectrum[:, self.mid_bound:], dim=-1).unsqueeze(1)
        
#         # 转换为 Loss 形式 (范围 [0, 2])
#         loss_low = 1.0 - cos_low
#         loss_mid = 1.0 - cos_mid
#         loss_high = 1.0 - cos_high
        
#         # 应用你成功的动态流行度加权 (依然是极冷门=3.0, 极热门=2.0)
#         dynamic_low_weight = 2.0 + (1.0 * alpha)
        
#         # 融合分段特征损失
#         weighted_feat_loss = (dynamic_low_weight * loss_low) + (1.0 * loss_mid) + (0.1 * loss_high)
#         # ==========================================
        
#         # 4. 汇总 (这就是你说的那行，保持不变！)
#         hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * weighted_feat_loss)
        
#         return total_loss.mean()

# class UniformKDLoss(nn.Module):
#     def __init__(self, gamma=0.01, feat_weight=0.05, uniform_weight=0.02): 
#         super().__init__()
#         self.gamma = gamma
#         self.feat_weight = feat_weight
#         self.uniform_weight = uniform_weight

#     def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
#         # 1. 基础长尾权重
#         alpha = torch.exp(-self.gamma * item_freqs)
#         teacher_probs = torch.sigmoid(teacher_logits)
#         confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
#         T = 5.0
#         soft_loss = F.binary_cross_entropy(
#             torch.sigmoid(student_logits/T), 
#             torch.sigmoid(teacher_logits/T), 
#             reduction='none'
#         ) * (T * T)
        
#         # 🌟 引擎 1：纯净的全局余弦
#         cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1).unsqueeze(1)
#         feat_loss = 1.0 - cos_sim
        
#         # ==========================================
#         # 🌟 引擎 2：特征均匀性惩罚 (内存安全版 🚀)
#         # ==========================================
#         student_feat_norm = F.normalize(student_feat, p=2, dim=-1)
        
#         # a. 矩阵乘法算相似度矩阵，大小只有 [2048, 2048]，绝不爆显存！
#         sim_matrix = torch.matmul(student_feat_norm, student_feat_norm.T)
        
#         # b. 生成对角线掩码（去除自己和自己的比较）
#         mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        
#         # c. 数学转换：惩罚 exp(-2 * sq_dist) 等价于惩罚 exp(4 * sim)
#         # 我们希望不同物品的 sim_matrix 越小越好
#         uniform_loss = torch.log(torch.mean(torch.exp(4.0 * sim_matrix[~mask])))
#         # ==========================================
        
#         # 4. 汇总总损失
#         hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
#         kd_weight = alpha * confidence * 8.0
        
#         total_loss = hard_loss + \
#                      (kd_weight * soft_loss) + \
#                      (self.feat_weight * feat_loss) + \
#                      (self.uniform_weight * uniform_loss)
                     
#         return total_loss.mean()


# class DualVictoryKDLoss(nn.Module):
#     def __init__(self, gamma=0.01, feat_dim=4096, feat_weight=1.0): 
#         super().__init__()
#         self.gamma = gamma
#         self.feat_weight = feat_weight # 接收外部传入的预热比例 (取消预热时默认为 1.0)
        
#         # ==============================================
#         # ⚙️ 参数一：动态力度 (决定了向大模型靠拢的绝对力量)
#         # ==============================================
#         self.max_weight = 0.05  # 极冷门物品的最高拉扯力 (你的神丹配置)
#         self.min_weight = 0.01  # 极热门物品的保底拉扯力
        
#         # ==============================================
#         # ⚙️ 参数二：动态及格线 (Margin，方案一的核心创新)
#         # ==============================================
#         self.strict_margin = 0.99  # 冷门物品及格线：必须极度相似 (听老师的)
#         self.loose_margin = 0.30   # 热门物品及格线：大方向没错就行 (放过它)

#     def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, true_labels, item_freqs):
#         # 1. 计算长尾衰减系数 alpha (越冷门越接近 1.0，越热门越接近 0.0)
#             alpha = torch.exp(-self.gamma * item_freqs)
        
#         # 2. 软标签置信度 
#             teacher_probs = torch.sigmoid(teacher_logits)
#             confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
#         # 3. 软标签交叉熵 Loss 
#             T = 5.0
#             soft_loss = F.binary_cross_entropy(
#                 torch.sigmoid(student_logits/T), 
#                 torch.sigmoid(teacher_logits/T), 
#                 reduction='none'
#             ) * (T * T)
        
#         # ==========================================================
#         # 🌟 绝杀逻辑：频率自适应宽容度 (Frequency-Adaptive Margin)
#         # ==========================================================
#         # 计算每个样本真实的余弦相似度, shape: [Batch_Size, 1], 值域 [-1, 1]
#             cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1).unsqueeze(1)
        
#         # 为每个样本动态计算专属的“及格线”
#         # - 纯冷门 (alpha≈1) -> 及格线逼近 0.99
#         # - 超热门 (alpha≈0) -> 及格线逼近 0.30
#             target_margin = self.loose_margin + (self.strict_margin - self.loose_margin) * alpha
        
#         # Margin Loss：核心中的核心！
#         # 如果 cos_sim >= target_margin，相减为负数，clamp 后直接变成 0 (不产生梯度)
#         # 如果 cos_sim < target_margin，才会产生正向 Loss，逼迫模型学习
#             sample_cos_loss = torch.clamp(target_margin - cos_sim, min=0.0)
        
#         # 继续施加动态权重 (力度的缩放)
#             dynamic_feat_weight = self.min_weight + (self.max_weight - self.min_weight) * alpha
        
#         # 计算最终的特征蒸馏 Loss
#             weighted_feat_loss = (dynamic_feat_weight * sample_cos_loss).mean() * self.feat_weight
#         # ==========================================================
        
#         # 4. 汇总总损失
#             hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
#             kd_weight = alpha * confidence * 8.0
        
#         # 总 Loss = 图协同过滤 + (软标签蒸馏) + (带截断的特征对齐)
#             total_loss = hard_loss + (kd_weight * soft_loss) + weighted_feat_loss
        
#             return total_loss.mean()
        

# =========== 新增：正宗的 FreqD 特征蒸馏 Loss ===========
class FreqDKDLoss(nn.Module):
    def __init__(self, gamma=0.01, feat_weight=0.05): 
        super().__init__()
        self.gamma = gamma
        self.feat_weight = feat_weight

    def forward(self, student_logits, teacher_logits, 
                s_feat_filtered_batch, t_feat_filtered_batch, 
                true_labels, item_freqs):
        
        # 1. 基础长尾衰减权重 (保留你现有的逻辑)
        alpha = torch.exp(-self.gamma * item_freqs)
        teacher_probs = torch.sigmoid(teacher_logits)
        confidence = 1.0 - torch.abs(true_labels.float() - teacher_probs)
        
        # 2. 预测层软标签蒸馏
        T = 5.0
        soft_loss = F.binary_cross_entropy(
            torch.sigmoid(student_logits/T), 
            torch.sigmoid(teacher_logits/T), 
            reduction='none'
        ) * (T * T)
        
        # 3. FreqD 特征蒸馏：直接计算图滤波后的均方误差
        feat_loss = F.mse_loss(s_feat_filtered_batch, t_feat_filtered_batch, reduction='none').mean(dim=-1, keepdim=True)
        
        # 4. 融合计算
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels.float(), reduction='none')
        kd_weight = alpha * confidence * 8.0
        
        total_loss = hard_loss + (kd_weight * soft_loss) + (self.feat_weight * feat_loss)
        return total_loss.mean()
# ========================================================
def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 64
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/"
        self.dataset = 'ml-100k' #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        self.layer_size='[64,64]'
        self.verbose = 1
        self.Ks='[10]'
        self.data_type='retraining'

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 100
        self.A_split = False
        self.dropout = False
        self.pretrain=0
        self.init_emb=1e-4
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)
    
    def hyper_para_info(self):
        print(self.__dict__)


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = model_hyparameters()
    args.reset(train_config)
    args.hyper_para_info()

    # load dataset
    # data_dir = "/home/zyang/LLM/MiniGPT-4/dataset/ml-100k/"
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-100k/"
    data_dir = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/"
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values



    # === 🌟 核心数据融合开始 ===
    print("🚀 正在加载大模型脑电波数据...")
    # 请填入你真实生成的 npy 文件名！
    teacher_logits = np.load("/root/autodl-tmp/CoLLM/result/20260330_214720/distill_soft_labels_20260330_214720.npy") 
    if len(teacher_logits.shape) == 1:
        teacher_logits = teacher_logits.reshape(-1, 1)
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    print("🧠 正在加载大模型 4096 维文本语义特征空间...")
    llm_features_path = "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/item_llm_features.npy"
    teacher_features_all = torch.from_numpy(np.load(llm_features_path)).float().cuda()
    train_data = train_data[:teacher_logits.shape[0]]
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    items = train_data[:, 1].astype(int)
    item_counts = np.bincount(items)
    item_freqs = item_counts[items].reshape(-1, 1)
    
    # 将 [uid, iid, label] 扩充为 [uid, iid, label, teacher_logit, item_freq]
    train_data = np.concatenate([train_data, teacher_logits, item_freqs], axis=1)
    print(f"✅ 数据融合完毕！新的训练集形状: {train_data.shape}")
    # === 🌟 核心数据融合结束 ===

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    # train_config={
    #     "lr": 1e-2,
    #     "wd": 1e-4,
    #     "epoch": 5000,
    #     "eval_epoch":1,
    #     "patience":50,
    #     "batch_size":1024
    # }

    user_num = train_data[:,0].max() + 1
    item_num = train_data[:,1].max() + 1

    # =========== 新增：预构建滤波器并处理大模型特征 ===========
    # H_filter = build_graph_filter(teacher_features_all, int(item_num), k=10, alpha=0.01) 
    # print("⚡ 预计算大模型的全局图滤波特征 (仅需计算一次)...")
    # filtered_teacher_feat_all = torch.sparse.mm(H_filter, teacher_features_all)
    # ==========================================================

    lgcn_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size']),
        "embed_size": int(train_config['embedding_size']),
        "data_path": "/root/autodl-tmp/CoLLM/collm-datasets/ml-1m/",
        "dataset": 'ml-1m', #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        "layer_size": '[64,64]',

        # lightgcn hyper-parameters
        "gcn_layers": train_config['gcn_layer'],
        "keep_prob" : 0.6,
        "A_n_fold": 100,
        "A_split": False,
        "dropout": False,
        "pretrain": 0,
        "init_emb": 1e-1,
        }
    lgcn_config = omegaconf.OmegaConf.create(lgcn_config)
    gnndata = GnnDataset(lgcn_config, data_dir)
    lgcn_config['user_num'] = int(gnndata.m_users)
    lgcn_config['item_num'] = int(gnndata.n_items)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    # model = MatrixFactorization(mf_config).cuda()
    # model = LightGCN(lgcn_config).cuda()
    # model._set_graph(gnndata.Graph)
    model = LightGCN(lgcn_config, llm_features=teacher_features_all).cuda()
    model._set_graph(gnndata.Graph)
    
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    # criterion = nn.BCEWithLogitsLoss() ！！！！！！！！！！！！！修改
    # criterion = SpectralKDLoss(gamma=0.01, feat_dim=4096)
    # =========== 修改：实例化新 Loss 并准备全局 ID ===========
    criterion = FreqDKDLoss(gamma=0.01, feat_weight=1)
    all_iids = torch.arange(int(item_num)).long().cuda()
    # =========================================================

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc,_,_ = uAUC_me(users, pre, label)

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return 
    

    # ！！！！！！！！！！！！！！！！！版本12使用的！！！！！！！！！！！！！！！
    # for epoch in range(train_config['epoch']):
    #     model.train()
    #     for bacth_id, batch_data in enumerate(train_data_loader):
    #         batch_data = batch_data.cuda()
    #         # ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
    #         # loss = criterion(ui_matching,batch_data[:,-1].float())
    #         # opt.zero_grad()
    #         # loss.backward()
    #         # opt.step()！！！！！！！！！！！！！！！！！修改过了
    #         uids = batch_data[:, 0].long()
    #         iids = batch_data[:, 1].long()
            
    #         # 因为数据被拼接了，真正的 label 现在固定在第 3 列 (索引为2)
    #         # unsqueeze(1) 是为了把形状从 [Batch_Size] 变成 [Batch_Size, 1]，对齐公式计算
    #         true_labels = batch_data[:, 2].float().unsqueeze(1)    
    #         teacher_logits = batch_data[:, 3].float().unsqueeze(1) 
    #         item_freqs = batch_data[:, 4].float().unsqueeze(1)     
            
    #         # 1. 小模型前向传播，算出自己的预测
    #         student_logits = model(uids, iids).unsqueeze(1)
            
    #         # 2. 扔进你的专属超级引擎：DynamicKDLoss
    #         loss = criterion(student_logits, teacher_logits, true_labels, item_freqs)
    #         # ==========================================
    #         # 🌟 核心修改结束
    #         # ==========================================

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()


    for epoch in range(train_config['epoch']):
        model.train()
        target_weight = 0.01
        current_feat_weight = target_weight
            
        
        criterion.feat_weight = current_feat_weight
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            
            uids = batch_data[:, 0].long()
            iids = batch_data[:, 1].long()
            
            true_labels = batch_data[:, 2].float().unsqueeze(1)    
            teacher_logits = batch_data[:, 3].float().unsqueeze(1) 
            item_freqs = batch_data[:, 4].float().unsqueeze(1)     
            
            # 1. 小模型前向传播，算出自己的预测
            student_logits = model(uids, iids).unsqueeze(1)
            
            # 🌟 2. 获取用于对齐的特征：
            # 注意：因为 model 内部现在已经把语义和 ID 融合了
            # 所以 get_projected_item_emb 获取到的是【融合后】的物品再放大到 4096 维的特征
            student_feat = model.get_projected_item_emb(iids)
            teacher_feat = teacher_features_all[iids]
            
            # 🌟 3. 计算总 Loss：
            # 这里如果你用的依然是 FreqDKDLoss，你需要确保 Loss 函数接收这五个参数
            # （直接把特征喂给 Loss 即可，不需要在外面做额外的稀疏滤波了，因为底层特征已经很稳）
            student_feat_norm = F.normalize(student_feat, p=2, dim=-1)
            teacher_feat_norm = F.normalize(teacher_feat, p=2, dim=-1)
            
            # Cosine Loss: 1.0 - 余弦相似度 (当方向完全一致时，Loss为0)
            align_loss = 1.0 - (student_feat_norm * teacher_feat_norm).sum(dim=-1).mean()
            
            # 4. 计算原来的推荐 BCE Loss
            bce_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels)
            
            # 5. 总 Loss (feat_weight 我们给一个非常强烈的信号：2.0)
            loss = bce_loss + 2.0 * align_loss
            opt.zero_grad()
            loss.backward()
            opt.step()


        if epoch% train_config['eval_epoch']==0:
            
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
                users.extend(batch_data[:,0].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc,_,_ = uAUC_me(users, pre, label)
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
                users.extend(batch_data[:,0].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc,_,_ = uAUC_me(users, pre, label)
            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc': valid_uauc,'test_auc':test_auc, 'test_uauc': test_uauc,'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, early_count:{}".format(epoch, valid_auc, valid_uauc, test_auc, test_uauc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.52")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)



# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4]
#     dw_ = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64, 128, 256]
#     gcn_layers = [1, 2, 3]
#     try:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'rw+')
#         # f = open("ood-ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#     except:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'w+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 for gcn_layer in gcn_layers:
#                     train_config={
#                         'lr': lr,
#                         'wd': wd,
#                         'embedding_size': embedding_size,
#                         "epoch": 5000,
#                         "eval_epoch":1,
#                         "patience":100,
#                         "batch_size":2048,
#                         "gcn_layer": gcn_layer
#                     }
#                     print(train_config)
#                     run_a_trail(train_config=train_config, log_file=f, save_mode=False)
#     f.close()



#train_config: {'lr': 0.01, 'wd': 0.0001, 'embedding_size': 64, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}

# {'lr': 0.0001, 'wd': 1e-07, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 
# 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4] #1e-2
#     dw_ = [1e-5]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64]
#     save_path = "/root/autodl-tmp/CoLLM/PretrainedModels/lgcn/"
#     # save_path = "/home/sist/zyang/LLM/PretrainedModels/mf/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048,
#                     "gcn_layer": 2
#                 }
#                 print(train_config)
#                 save_path += "0918-OODv2_lgcn_ml1m_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()


# # with prtrain version:


# !!!!!!!!!!!!!!train训练！！！！！！！！！！！！！！！！！！！！！！
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-3] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [64]
    save_path = "/root/autodl-tmp/CoLLM/lgcnresult/"
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":100,
                    "batch_size":2048,
                    "gcn_layer": 2
                }
                
                # print(train_config)
                # if os.path.exists(save_path + "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"


                # save_path += "0918-OODv2_lgcn_ml1m_best_model_d64lr-0.01wd0.0001.pth" 生成lgcn图参数使用
                save_file_name = save_path + "student_dynamic_kd_best.pth"
                run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=save_file_name, need_train=True, warm_or_cold=None)
    if f is not None:
        f.close()
        
if __name__=='__main__':
    # 🌟 1. 严格锁定你创造历史的黄金配置（不要做任何修改）
    train_config={
        'lr': 1e-4, 
        'wd': 1e-4,
        'embedding_size': 64,
        "epoch": 5000,
        "eval_epoch":1,
        "patience":50,
        "batch_size":2048,
        "gcn_layer": 2
    }
    
    # 🌟 2. 指向你那颗 0.6356 的极品神丹的绝对路径
    # （请核对一下这个路径是不是你最新跑出最好成绩的那个 pth 文件）
    save_file_name = "/root/autodl-tmp/CoLLM/lgcnresult/student_dynamic_kd_best.pth"
    
    print("\n" + "🔥"*10 + " 正在评估 Warm (热门/活跃) 试卷 " + "🔥"*10)
    # 🌟 3. need_train=False 极其关键！这意味着不训练，直接加载权重去考试
    run_a_trail(train_config=train_config, 
                save_file=save_file_name, 
                need_train=False,  
                warm_or_cold='warm') 
    
    print("\n" + "❄️"*10 + " 正在评估 Cold (冷门/长尾/OOD) 试卷 " + "❄️"*10)
    # 🌟 4. 测试 Cold 数据
    run_a_trail(train_config=train_config, 
                save_file=save_file_name, 
                need_train=False,  
                warm_or_cold='cold')





        
            







