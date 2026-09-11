# -*- coding: utf-8 -*-
"""
common.py —— 所有网络共享的配置、数据集、损失、指标
====================================================
被 train_mae / train_aux / train_teacher / train_direct / train_student / evaluate 共同 import。
任何路径、超参数只在这里改，五个训练脚本不要各自硬编码。
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ======================================================================
# 0. 全局配置
# ======================================================================
class CFG:
    # ---- 路径（改成你的实际路径）----
    DATA_ROOT   = Path("data")
    MANIFEST    = DATA_ROOT / "manifest.csv"          # ROI 总表
    PATCH_DIR   = DATA_ROOT / "patches"               # DINOv2 [CLS] 特征 .npy
    VOLUME_DIR  = DATA_ROOT / "volumes"               # OCT 体数据 .npy (3,64,64,64)
    NORM_JSON   = DATA_ROOT / "norm_constants.json"   # 归一化常数（A/B 计算后冻结）
    CKPT_DIR    = Path("checkpoints")                 # 所有模型权重输出目录
    LOG_DIR     = Path("logs")                        # 训练日志 CSV（供 evaluate.py 画图）

    # ---- 晶圆划分（2-1-2 协议）----
    TRAIN_WAFERS = ["A", "B"]     # 训练片（内部再 8:2 切 train/val）
    VAL_WAFER    = "C"            # 验证片：早停、模型选择
    TEST_WAFERS  = ["D", "E"]     # 测试片：全程封存，最后一次开封

    # ---- 维度 ----
    DINO_DIM   = 384      # DINOv2 ViT-S/14 [CLS]
    META_DIM   = 5        # P = [x, y, w, h, a]
    FUSE_DIM   = 773      # 384 + 384 + 5
    FEAT_DIM   = 256      # 融合特征维度

    # ---- Teacher 损失权重（文档 5.5 节）----
    W_T_Q, W_T_MORPH, W_T_U = 1.0, 0.5, 2.0
    # ---- Student 蒸馏权重（文档 7.4 节）----
    W_FEAT, W_KD, W_KDU, W_RANK = 1.0, 0.5, 1.0, 1.0
    KD_TAU = 3.0          # 蒸馏温度
    HUBER_DELTA = 1.0     # 回归目标均在 [0,1]，残差>1 属极端

    # ---- 训练超参 ----
    BATCH        = 32
    SUB_OVERSAMPLE = 0.30  # 每 batch subsurface 占比下限（对抗类别失衡）
    SEED         = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
# 1. 归一化常数：只用训练片 A/B 计算，冻结后应用于所有晶圆
# ======================================================================
def compute_or_load_norm(df_train: pd.DataFrame) -> dict:
    """
    输入：Wafer A/B 训练部分的 manifest 子表
    输出/读取：norm_constants.json
      P 五维的均值/标准差、m* 四分量 min/max、r 的 min/max（辅助分类器阶段回填）
    原则：一切统计量只碰训练片，防止信息泄漏（文档 1.3 节）。
    """
    if CFG.NORM_JSON.exists():
        return json.loads(CFG.NORM_JSON.read_text())

    norm = {}
    for k in ["x", "y", "w", "h", "a"]:
        norm[f"{k}_mean"] = float(df_train[k].mean())
        norm[f"{k}_std"]  = float(df_train[k].std() + 1e-8)
    # m* 四分量 min-max：只在 subsurface（z*=2）样本上统计（文档 3.5 节）
    sub = df_train[df_train.z_star == 2]
    for k in ["d", "v", "o", "c"]:
        norm[f"{k}_min"] = float(sub[k].min())
        norm[f"{k}_max"] = float(sub[k].max())
    CFG.NORM_JSON.write_text(json.dumps(norm, indent=2, ensure_ascii=False))
    return norm


# ======================================================================
# 2. 数据集
# ======================================================================
class ROIDataset(Dataset):
    """
    每个样本 = 一个候选 ROI。
    输入（网络用）：
      feat_local (384,)  M_local 的 DINOv2 [CLS]（离线预提取，backbone 不进训练循环）
      feat_ctx   (384,)  M_ctx  的 [CLS]
      meta       (5,)    P，已按冻结常数 z-score
      volume     (3,64,64,64) OCT 体数据（仅 Teacher/MAE 用；Direct/Student 不加载）
    监督（真值）：
      z_star (long)      类别 {0,1,2}
      q_star (float)     可观测性 [0,1]
      m_star (4,)        形态真值，非 subsurface 为 NaN → 用 morph_mask 掩掉
      morph_mask (bool)  z_star==2
      U_star (float)     效用真值 [0,1]
    """
    def __init__(self, df: pd.DataFrame, norm: dict, load_volume: bool):
        self.df = df.reset_index(drop=True)
        self.norm = norm
        self.load_volume = load_volume

    def __len__(self):
        return len(self.df)

    def _meta(self, row) -> torch.Tensor:
        v = [(row[k] - self.norm[f"{k}_mean"]) / self.norm[f"{k}_std"]
             for k in ["x", "y", "w", "h", "a"]]
        return torch.tensor(v, dtype=torch.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = {
            "roi_id":  row.roi_id,
            "feat_local": torch.from_numpy(np.load(CFG.PATCH_DIR / f"{row.roi_id}_local.npy")).float(),
            "feat_ctx":   torch.from_numpy(np.load(CFG.PATCH_DIR / f"{row.roi_id}_ctx.npy")).float(),
            "meta":       self._meta(row),
            "z_star":     torch.tensor(int(row.z_star), dtype=torch.long),
            "q_star":     torch.tensor(float(row.q_star), dtype=torch.float32),
            "U_star":     torch.tensor(float(row.U_star), dtype=torch.float32),
            "morph_mask": torch.tensor(bool(row.z_star == 2)),
        }
        # m*：subsurface 才有值；NaN 位置由 morph_mask 挡住，不进入损失（文档 3.5 节）
        m = [row.d, row.v, row.o, row.c]
        item["m_star"] = torch.tensor(
            [np.nan if pd.isna(x) else float(x) for x in m], dtype=torch.float32)
        if self.load_volume:
            item["volume"] = torch.from_numpy(
                np.load(CFG.VOLUME_DIR / f"{row.roi_id}.npy")).float()  # (3,64,64,64)
        return item


class SubsurfaceOversampler(torch.utils.data.Sampler):
    """
    每 batch 内 subsurface 占比 >= CFG.SUB_OVERSAMPLE（文档 7.5 节）。
    原理：subsurface 样本索引重复采样，normal/surface 不重复。
    """
    def __init__(self, df):
        self.sub_idx  = df.index[df.z_star == 2].tolist()
        self.rest_idx = df.index[df.z_star != 2].tolist()
        n_sub = max(1, int(CFG.SUB_OVERSAMPLE * len(df)))
        self.epoch_idx = self.rest_idx + random.choices(self.sub_idx, k=n_sub)

    def __iter__(self):
        random.shuffle(self.epoch_idx)
        return iter(self.epoch_idx)

    def __len__(self):
        return len(self.epoch_idx)


def load_split(norm=None):
    """返回 (df_train, df_val_internal, df_val_waferC, df_test)"""
    df = pd.read_csv(CFG.MANIFEST)
    df_tr  = df[(df.wafer.isin(CFG.TRAIN_WAFERS)) & (df.split == "train")]
    df_vi  = df[(df.wafer.isin(CFG.TRAIN_WAFERS)) & (df.split == "val")]
    df_vc  = df[df.wafer == CFG.VAL_WAFER]
    df_te  = df[df.wafer.isin(CFG.TEST_WAFERS)]
    if norm is None:
        norm = compute_or_load_norm(df_tr)
    return df_tr, df_vi, df_vc, df_te, norm


# ======================================================================
# 3. 损失函数（文档第 11 章附录的逐条实现）
# ======================================================================
def weighted_ce(logits, target, class_counts):
    """加权交叉熵：w_c = N / (C * N_c)，补偿 subsurface 稀少（11.1 节）"""
    w = 1.0 / (class_counts.float() + 1e-8)
    w = w / w.sum() * len(class_counts)          # 归一化到均值≈1
    return F.cross_entropy(logits, target, weight=w.to(logits.device))


def huber(pred, target, delta=CFG.HUBER_DELTA):
    return F.huber_loss(pred, target, delta=delta)


def morph_huber(pred4, m_star, mask):
    """
    形态损失：只在 subsurface 掩码内计算（文档 3.5/5.5 节）。
    必须带掩码——NaN 参与会让整个 loss 变 NaN 崩掉训练；填 0 则是假监督。
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred4.device)
    return F.huber_loss(pred4[mask], m_star[mask], delta=CFG.HUBER_DELTA)


def feat_align_loss(f_s, f_t, proj):
    """L_feat = MSE(φ(F_S), F_T) + (1 - cos)：数值对齐 + 方向对齐（11.5 节）"""
    g = proj(f_s)
    mse = F.mse_loss(g, f_t)
    cos = 1 - F.cosine_similarity(g, f_t, dim=-1).mean()
    return mse + cos


def kd_loss(logits_s, logits_t, tau=CFG.KD_TAU):
    """温度蒸馏 + τ² 梯度补偿（11.3 节）"""
    p = F.log_softmax(logits_s / tau, dim=-1)
    q = F.softmax(logits_t / tau, dim=-1)
    return tau * tau * F.kl_div(p, q, reduction="batchmean")


def rank_loss(U_s, U_star):
    """
    成对排序损失：batch 内所有 U*_i > U*_j 的样本对（文档 7.4 式(7)）。
    应用目标是排序，逐样本回归之外必须直接优化排序一致性。
    """
    diff_pred = U_s.unsqueeze(1) - U_s.unsqueeze(0)      # (B,B)
    diff_true = U_star.unsqueeze(1) - U_star.unsqueeze(0)
    pair_mask = (diff_true > 0).float()
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=U_s.device)
    loss = -F.logsigmoid(diff_pred) * pair_mask
    return loss.sum() / pair_mask.sum()


# ======================================================================
# 4. 评估指标（evaluate.py 与各训练脚本的早停共用）
# ======================================================================
def spearman(a, b):
    """Spearman 秩相关：先各自取秩再算 Pearson（11.6 节）"""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def per_class_metrics(y_true, y_pred, n_cls=3):
    """
    逐类 precision / recall / F1 + 混淆矩阵。
    ★ 证明"Direct 预测不了深度缺陷"的核心证据就在这里：
      - recall[2]（subsurface 召回率）
      - conf[2,1] / conf[2,:].sum()（subsurface 被误判成 surface 的比例 = 深度歧义）
    """
    names = ["normal", "surface", "subsurface"]
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    out = {}
    for c in range(n_cls):
        tp = conf[c, c]
        prec = tp / max(conf[:, c].sum(), 1)
        rec  = tp / max(conf[c, :].sum(), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        out[names[c]] = {"precision": prec, "recall": rec, "f1": f1}
    return out, conf


def topk_hit_rate(U_pred, z_star, rho=0.2):
    """
    预算命中率：按预测效用降序取前 ρ 比例，其中真 subsurface 的占比。
    端到端应用指标——部署时这就是"扫出来的区域有多少真藏了深度缺陷"。
    """
    k = max(1, int(len(U_pred) * rho))
    top = np.argsort(-np.asarray(U_pred))[:k]
    return float((np.asarray(z_star)[top] == 2).mean())


# ======================================================================
# 5. 训练日志：逐 epoch 写 CSV，evaluate.py 负责画图
# ======================================================================
class CSVLogger:
    def __init__(self, name):
        CFG.LOG_DIR.mkdir(exist_ok=True)
        self.path = CFG.LOG_DIR / f"{name}.csv"
        self.rows = []

    def log(self, **kw):
        self.rows.append(kw)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
