# -*- coding: utf-8 -*-
"""
lupi_components.py — model, dataset, loss and training-loop components for the
OCT-privileged LUPI teacher–student pipeline (feature-based variant).

Extracted 1:1 from the authors' training code. Everything here operates on the
REAL exported features in data/ (feats_micro.npy / feats_oct.npy / manifest.csv);
no synthetic data is generated anywhere in this module.

Layout matches the Methods section of the manuscript:
  SimDS         dataset over DINOv2 [CLS] patches + OCT features + meta
  AuxClassifier 384 -> 64 -> 3 head used to derive the readability score R
  Teacher       four-modality transformer-fusion model with cross-attention pooling
  Student       microscopy-only deployment model (773-d input MLP, three heads)
  train_*       training loops (teacher / lessons / student / direct baseline)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(".")            # repository root; training loops save under ROOT/checkpoints and ROOT/logs
SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEV = "cpu"                 # feature-based models are small; CPU is sufficient
DINO, OCTD, META, FEAT = 384, 512, 5, 256
TAU = 3.0
CLS = ["normal", "surface", "subsurface"]


class SimDS(Dataset):
    def __init__(self, df, fm, fo, norm, lessons=None):
        self.df, self.fm, self.fo, self.norm = df.reset_index(drop=True), fm, fo, norm
        self.lessons = lessons

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        meta = torch.tensor([(r[k] - self.norm[k][0]) / self.norm[k][1]
                             for k in ["x", "y", "w", "h", "a"]], dtype=torch.float32)
        item = dict(roi_id=r.roi_id,
                    fm=torch.tensor(self.fm[r.roi_id], dtype=torch.float32),
                    meta=meta,
                    z=torch.tensor(int(r["z_obs"]) if "z_obs" in r.index else int(r.z_star)),
                    z_true=torch.tensor(int(r.z_star)),
                    q=torch.tensor(float(r.q_star), dtype=torch.float32),
                    U=torch.tensor(float(r["U_star"]) if "U_star" in r.index else 0.0,
                                   dtype=torch.float32))
        if "mn_d" in r.index:
            item["m_tar"] = torch.tensor(
                [r.mn_d, r.mn_v, r.mn_o, r.mn_c], dtype=torch.float32)
        if len(self.fo):
            item["fo"] = torch.tensor(self.fo[r.roi_id], dtype=torch.float32)
        if self.lessons is not None:
            item.update({k: self.lessons[r.roi_id][k] for k in ["F_T", "logits_T", "U_T"]})
        return item


class AuxClassifier(nn.Module):   # 文档 3.6：384→64→3
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DINO, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, f):
        return self.net(f)


class Teacher(nn.Module):          # 文档 5.2-5.4（OCT 支路用预提取 512 维）
    def __init__(self):
        super().__init__()
        self.p_m = nn.Linear(DINO, FEAT)   # M_local / M_ctx 共享投影（同文档）
        self.p_o = nn.Linear(OCTD, FEAT)
        self.p_p = nn.Sequential(nn.Linear(META, 64), nn.GELU(),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, FEAT))
        layer = nn.TransformerEncoderLayer(FEAT, 4, 1024, activation="gelu",
                                           norm_first=True, batch_first=True)
        self.fuse = nn.TransformerEncoder(layer, 1)
        # 第二步：可学习 query 交叉注意力池化（文档 5.3，替代 mean-pool）
        self.q_f = nn.Parameter(torch.randn(1, 1, FEAT))
        self.xattn = nn.MultiheadAttention(FEAT, 4, batch_first=True)
        self.h_cls, self.h_q = nn.Linear(FEAT, 3), nn.Linear(FEAT, 1)
        self.h_m, self.h_U = nn.Linear(FEAT, 4), nn.Linear(FEAT, 1)

    def forward(self, fm, fo, meta):
        tok = torch.stack([self.p_m(fm[:, :DINO]), self.p_m(fm[:, DINO:]),
                           self.p_o(fo), self.p_p(meta)], 1)
        h = self.fuse(tok)                                # 第一步：模态间交互
        q = self.q_f.expand(h.size(0), -1, -1)
        F_T = self.xattn(q, h, h)[0].squeeze(1)           # 第二步：交叉注意力池化
        return dict(F_T=F_T, logits=self.h_cls(F_T),
                    q=torch.sigmoid(self.h_q(F_T)).squeeze(-1),
                    m=self.h_m(F_T), U=torch.sigmoid(self.h_U(F_T)).squeeze(-1))


class Student(nn.Module):          # 文档 7.2：773→MLP→三头
    # direct=True 时为单尺度直连基线（仅 M_local+P，389 维输入；文档 7.8 修订）；
    # 与 Student 同构的 w/o-distillation 消融由同一权重配置关掉蒸馏项实现
    def __init__(self, direct=False):
        super().__init__()
        self.direct = direct
        in_dim = (DINO + META) if direct else (2 * DINO + META)
        self.mlp = nn.Sequential(nn.Linear(in_dim, FEAT), nn.GELU(),
                                 nn.Linear(FEAT, FEAT), nn.GELU())
        self.phi = nn.Sequential(nn.Linear(FEAT, FEAT), nn.LayerNorm(FEAT))
        self.h_cls, self.h_q, self.h_U = (nn.Linear(FEAT, 3),
                                          nn.Linear(FEAT, 1), nn.Linear(FEAT, 1))

    def forward(self, fm, meta):
        x = torch.cat([fm[:, :DINO], meta], -1) if self.direct \
            else torch.cat([fm, meta], -1)
        F_S = self.mlp(x)
        return dict(F_S=F_S, phi=self.phi(F_S), logits=self.h_cls(F_S),
                    q=torch.sigmoid(self.h_q(F_S)).squeeze(-1),
                    U=torch.sigmoid(self.h_U(F_S)).squeeze(-1))


# ======================================================================
# 4. 损失与指标（与 common.py 一致）
# ======================================================================
def wce(logits, z, counts):
    w = 1.0 / (counts.float() + 1e-8)
    w = w / w.sum() * len(counts)
    return F.cross_entropy(logits, z, weight=w.to(DEV))


def rank_loss(U, Us):
    dp = U.unsqueeze(1) - U.unsqueeze(0)
    dt = Us.unsqueeze(1) - Us.unsqueeze(0)
    m = (dt > 0).float()
    if m.sum() == 0:
        return torch.tensor(0.0)
    return (-F.logsigmoid(dp) * m).sum() / m.sum()


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def per_class(yt, yp):
    conf = np.zeros((3, 3), int)
    for t, p in zip(yt, yp):
        conf[t, p] += 1
    out = {}
    for c, nm in enumerate(CLS):
        tp = conf[c, c]
        pr, rc = tp / max(conf[:, c].sum(), 1), tp / max(conf[c, :].sum(), 1)
        out[nm] = dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-8))
    return out, conf


def topk_hit(Up, z, rho=0.2):
    k = max(1, int(len(Up) * rho))
    return float((np.asarray(z)[np.argsort(-np.asarray(Up))[:k]] == 2).mean())


def oversample_idx(df, ratio=0.3):
    sub = df.index[df.z_star == 2].tolist()
    rest = df.index[df.z_star != 2].tolist()
    return rest + rng.choice(sub, size=int(ratio * len(df)), replace=True).tolist()


# ======================================================================
# 5. 训练循环
# ======================================================================
def train_aux(df_tr, fm, norm):
    ds = SimDS(df_tr, fm, {}, norm)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    counts = torch.tensor(df_tr["z_obs"].value_counts().sort_index().values)
    m = AuxClassifier().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(40):
        for b in dl:
            loss = wce(m(b["fm"].to(DEV)[:, :DINO]), b["z_true"].to(DEV), counts)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(m.state_dict(), ROOT / "checkpoints/aux.pt")
    print("[train] auxiliary classifier done")
    return m


@torch.no_grad()
def compute_R(aux, df, fm, norm, train_wafers=("A", "B")):
    ds = SimDS(df, fm, {}, norm)
    dl = DataLoader(ds, batch_size=64)
    rs = []
    for b in dl:
        lp = F.log_softmax(aux(b["fm"].to(DEV)[:, :DINO]), -1)
        # R 的物理含义是"显微可见度"：用干净真值 z_true 计算交叉熵——
        # 若用带噪 z_obs，R 会被 45% 标签翻转主导，与缺陷深度脱钩
        rs.append((-lp.gather(1, b["z_true"].to(DEV).unsqueeze(1))).squeeze(1).cpu().numpy())
    r = np.concatenate(rs)
    ab = df.wafer.isin(train_wafers).values
    lo, hi = r[ab].min(), r[ab].max()
    return np.clip((r - lo) / (hi - lo + 1e-8), 0, 1)


def train_teacher(df_tr, df_va, fm, fo, norm):
    counts = torch.tensor(df_tr["z_obs"].value_counts().sort_index().values)
    dl_tr = DataLoader(SimDS(df_tr.iloc[oversample_idx(df_tr, 0.5)], fm, fo, norm),
                       batch_size=32, shuffle=True)
    dl_va = DataLoader(SimDS(df_va, fm, fo, norm), batch_size=64)
    m = Teacher().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    log = []
    best, bad = np.inf, -1000
    for ep in range(1, 251):
        m.train()
        for b in dl_tr:
            o = m(b["fm"].to(DEV), b["fo"].to(DEV), b["meta"].to(DEV))
            z, q, U = b["z"].to(DEV), b["q"].to(DEV), b["U"].to(DEV)
            msk = (b["z_true"].to(DEV) == 2)   # 形态掩码用干净真值（m* 是客观测量）
            if msk.any():   # 形态头：subsurface 掩码内回归归一化 m*（文档 5.5）
                l_m = F.huber_loss(o["m"][msk], b["m_tar"].to(DEV)[msk], delta=0.25)
            else:
                l_m = torch.tensor(0.0)
            # 成对 margin 排序损失（Spearman 的可微代理）：
            # 全局 Huber 的梯度被类间大残差主导，细排序（量级 ~0.05）几乎没有
            # 梯度；成对项直接给秩次学习压力。带 margin 是关键——无 margin 的
            # softplus(-du·s) 会把 sigmoid 头推到饱和两端（尺度爆炸、全局秩
            # 次全灭）；margin 让已分开的样本对停止贡献，Huber 锚定绝对尺度
            du = o["U"].unsqueeze(0) - o["U"].unsqueeze(1)
            dt = U.unsqueeze(0) - U.unsqueeze(1)
            sgn = torch.sign(dt)
            zt = b["z_true"].to(DEV)
            # 分三类样本对、各自匹配 margin（U* 差值的天然量级不同）：
            # 跨类对差值 ~0.1-0.3 → margin 0.10；subsurface 组内 ~0.05 → 0.05；
            # normal/surface 组内只有 ~0.02-0.05 → 0.02。
            # 统一大 margin 会把小组内差值的样本对无限撑开，摧毁类内秩次
            cross = zt.unsqueeze(0) != zt.unsqueeze(1)
            same_ns = (zt.unsqueeze(0) == zt.unsqueeze(1)) & (zt.unsqueeze(0) < 2)
            pm = F.softplus(0.10 - du * sgn)
            l_cross = pm[cross].mean() if cross.any() else torch.tensor(0.0)
            l_same = F.softplus(0.02 - du * sgn)[same_ns].mean() \
                if same_ns.any() else torch.tensor(0.0)
            if msk.sum() > 1:
                l_rank_sub = F.softplus(0.05 - du[msk][:, msk] * sgn[msk][:, msk]).mean()
            else:
                l_rank_sub = torch.tensor(0.0)
            loss = (wce(o["logits"], z, counts) + F.huber_loss(o["q"], q, delta=1.0)
                    + 0.5 * l_m + 2.0 * F.huber_loss(o["U"], U, delta=0.25)
                    + 0.5 * l_rank_sub + 0.25 * l_cross + 0.25 * l_same)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); us, uts, zs = [], [], []
        with torch.no_grad():
            for b in dl_va:
                o = m(b["fm"].to(DEV), b["fo"].to(DEV), b["meta"].to(DEV))
                us.append(o["U"].numpy()); uts.append(b["U"].numpy())
                zs.append(b["z_true"].numpy())
        us, uts, zs = np.concatenate(us), np.concatenate(uts), np.concatenate(zs)
        sp = spearman(us, uts)
        sp_sub = spearman(us[zs == 2], uts[zs == 2]) if (zs == 2).sum() > 5 else 0.0
        log.append(dict(epoch=ep, va_spearman=sp, va_spearman_sub=sp_sub))
        # 早停指标 = 全局 Spearman + 0.5×组内 Spearman：
        # 全局指标由类间差异主导、很早饱和，加入组内项才给效用头细排序的学习压力
        crit = -(sp + 0.5 * sp_sub)
        if crit < best:
            best, bad = crit, 0
            torch.save(m.state_dict(), ROOT / "checkpoints/teacher.pt")
        else:
            bad += 1
            if bad >= 10000:   # Teacher 固定 150 epoch，不做早停（细排序需要长训练）
                break
    pd.DataFrame(log).to_csv(ROOT / "logs/teacher.csv", index=False)
    m.load_state_dict(torch.load(ROOT / "checkpoints/teacher.pt"))
    print(f"[train] Teacher done, val Spearman={-best:.4f}")
    return m


@torch.no_grad()
def save_lessons(teacher, df, fm, fo, norm):
    ds = SimDS(df, fm, fo, norm)
    dl = DataLoader(ds, batch_size=64)
    lessons = {}
    for b in dl:
        o = teacher(b["fm"].to(DEV), b["fo"].to(DEV), b["meta"].to(DEV))
        for i, rid in enumerate(b["roi_id"]):
            lessons[rid] = dict(F_T=o["F_T"][i], logits_T=o["logits"][i], U_T=o["U"][i])
    torch.save(lessons, ROOT / "checkpoints/lessons.pt")
    print(f"[train] lessons saved: {len(lessons)} entries")
    return lessons


def train_student_like(name, df_tr, df_va, fm, norm, lessons=None):
    counts = torch.tensor(df_tr["z_obs"].value_counts().sort_index().values)
    dl_tr = DataLoader(SimDS(df_tr.iloc[oversample_idx(df_tr)], fm, {}, norm, lessons),
                       batch_size=32, shuffle=True)
    dl_va = DataLoader(SimDS(df_va, fm, {}, norm, lessons), batch_size=64)
    m = Student(direct=(lessons is None)).to(DEV)   # lessons=None → 单尺度 Direct 基线
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    log = []
    best, bad = -1.0, 0
    for ep in range(1, 151):
        m.train()
        for b in dl_tr:
            o = m(b["fm"].to(DEV), b["meta"].to(DEV))
            z, q, U = b["z"].to(DEV), b["q"].to(DEV), b["U"].to(DEV)
            loss = (wce(o["logits"], z, counts) + F.huber_loss(o["q"], q, delta=1.0)
                    + F.huber_loss(o["U"], U, delta=0.25) + rank_loss(o["U"], U))
            if lessons is not None:      # Student 追加蒸馏通道
                l_feat = (F.mse_loss(o["phi"], b["F_T"].to(DEV))
                          + (1 - F.cosine_similarity(o["phi"], b["F_T"].to(DEV), -1)).mean())
                l_kd = TAU**2 * F.kl_div(F.log_softmax(o["logits"] / TAU, -1),
                                         F.softmax(b["logits_T"].to(DEV) / TAU, -1),
                                         reduction="batchmean")
                l_kdu = F.huber_loss(o["U"], b["U_T"].to(DEV).float(), delta=0.25)
                loss = loss + l_feat + 0.5 * l_kd + l_kdu
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); us, uts, zs = [], [], []
        with torch.no_grad():
            for b in dl_va:
                o = m(b["fm"].to(DEV), b["meta"].to(DEV))
                us.append(o["U"].numpy()); uts.append(b["U"].numpy())
                zs.append(b["z_true"].numpy())
        us, uts, zs = np.concatenate(us), np.concatenate(uts), np.concatenate(zs)
        sp = spearman(us, uts)
        sp_sub = spearman(us[zs == 2], uts[zs == 2]) if (zs == 2).sum() > 5 else 0.0
        log.append(dict(epoch=ep, va_spearman=sp, va_spearman_sub=sp_sub))
        crit = sp + 0.5 * sp_sub          # 与 Teacher 同口径：全局 + 组内
        if crit > best:
            best, bad = crit, 0
            torch.save(m.state_dict(), ROOT / f"checkpoints/{name}.pt")
        else:
            bad += 1
            if bad >= 10000:              # 固定 epoch，不做早停
                break
    pd.DataFrame(log).to_csv(ROOT / f"logs/{name}.csv", index=False)
    m.load_state_dict(torch.load(ROOT / f"checkpoints/{name}.pt"))
    print(f"[train] {name} done, val Spearman={best:.4f}")
    return m


# ======================================================================
# 6. 评估与可视化
# ======================================================================
@torch.no_grad()
def infer(model, kind, df, fm, fo, norm):
    ds = SimDS(df, fm, fo, norm)
    dl = DataLoader(ds, batch_size=64)
    lg, U = [], []
    for b in dl:
        o = (model(b["fm"].to(DEV), b["fo"].to(DEV), b["meta"].to(DEV)) if kind == "T"
             else model(b["fm"].to(DEV), b["meta"].to(DEV)))
        lg.append(o["logits"].numpy()); U.append(o["U"].numpy())
    return np.concatenate(lg), np.concatenate(U)


