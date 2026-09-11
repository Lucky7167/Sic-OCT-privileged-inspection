# -*- coding: utf-8 -*-
"""
train_aux_classifier.py —— 辅助显微分类器与回看置信度 R（文档 3.6 节）
======================================================================
【输入】M_local 的 DINOv2 [CLS] 特征 (384,) + 标签 z*（全体训练样本）
【输出】① checkpoints/aux_classifier.pt
        ② manifest.csv 回填两列：r_i（逐样本交叉熵）和 R（min-max 归一化）
           —— R 是 U* 的合成原料，必须在 Teacher 训练前完成
【结构】Linear(384→64) + GELU + Linear(64→3)（小 MLP， deliberately 简单）
【评估】三分类 macro-F1——它只是"显微镜视角的自评考官"，够用即可，不追求 SOTA

它的角色：只给 R 用，不参与任何下游网络结构。
r_i = -log p_aux(z*_i | M_i)：只看显微图，分类器对正确答案有多"意外"——
越隐蔽的缺陷 r 越大 → R 越大 → 越值得 OCT 补看（U* 的 R 项）。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (CFG, ROIDataset, CSVLogger, load_split, set_seed,
                    weighted_ce, per_class_metrics)


class AuxClassifier(nn.Module):
    """结构见文档 3.6 节：384 → 64 → 3"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CFG.DINO_DIM, 64), nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, feat_local):
        return self.net(feat_local)


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()

    # 用"全体训练样本"（A/B 的 train+val）训练——文档 3.6 节
    df_all = pd.concat([df_tr, df_vi]).reset_index(drop=True)
    ds = ROIDataset(df_all, norm, load_volume=False)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=True)
    class_counts = df_all.z_star.value_counts().sort_index().values

    model = AuxClassifier().to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    logger = CSVLogger("aux")

    for epoch in range(1, 51):
        model.train(); tot, n = 0.0, 0
        for b in dl:
            logits = model(b["feat_local"].to(CFG.DEVICE))
            loss = weighted_ce(logits, b["z_star"].to(CFG.DEVICE), class_counts)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        logger.log(epoch=epoch, ce=tot / n)
        if epoch % 10 == 0:
            print(f"[AUX] epoch {epoch:02d}  ce={tot/n:.4f}")

    CFG.CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CFG.CKPT_DIR / "aux_classifier.pt")

    # ---------------- 训练集上自评（ sanity check ）----------------
    model.eval()
    with torch.no_grad():
        logits = model(ds[:]["feat_local"].to(CFG.DEVICE)) if len(ds) < 4096 else None
    # 大数据量时分批（这里样本 ~500，直接全量）
    feats = torch.stack([ds[i]["feat_local"] for i in range(len(ds))]).to(CFG.DEVICE)
    with torch.no_grad():
        logits = model(feats)
    pred = logits.argmax(-1).cpu().numpy()
    metrics, conf = per_class_metrics(df_all.z_star.values, pred)
    print("[AUX] 训练集逐类指标：", {k: {m: round(v, 3) for m, v in d.items()}
                                   for k, d in metrics.items()})
    print("[AUX] 混淆矩阵（行=真值）：\n", conf)

    # ---------------- 回看：逐样本交叉熵 r_i，回填 manifest ----------------
    df_full = pd.read_csv(CFG.MANIFEST)
    ds_full = ROIDataset(df_full, norm, load_volume=False)
    feats = torch.stack([ds_full[i]["feat_local"] for i in range(len(ds_full))]).to(CFG.DEVICE)
    z = torch.tensor(df_full.z_star.values).to(CFG.DEVICE)
    with torch.no_grad():
        logp = torch.log_softmax(model(feats), dim=-1)
    r = (-logp.gather(1, z.unsqueeze(1))).squeeze(1).cpu().numpy()  # (N,)

    # min-max 归一化只用 Wafer A/B（常数冻结，写入 norm_constants.json）
    ab = df_full.wafer.isin(CFG.TRAIN_WAFERS).values
    r_min, r_max = r[ab].min(), r[ab].max()
    R = np.clip((r - r_min) / (r_max - r_min + 1e-8), 0, 1)

    df_full["r_i"] = r
    df_full["R"] = R
    df_full.to_csv(CFG.MANIFEST, index=False)

    norm["r_min"], norm["r_max"] = float(r_min), float(r_max)
    CFG.NORM_JSON.write_text(__import__("json").dumps(norm, indent=2, ensure_ascii=False))

    # ---------------- 回填 U*（若还没合成）----------------
    # U* = q* · [αD + (1−α)R]，D = 𝕀(z*=2)(0.5+0.5·m̄)，α = 0.7（文档 3.7 节）
    sub = df_full.z_star == 2
    mbar = np.zeros(len(df_full))
    for k in ["d", "v", "o", "c"]:
        col = df_full[k].values.astype(float)
        col_n = (col - norm[f"{k}_min"]) / (norm[f"{k}_max"] - norm[f"{k}_min"] + 1e-8)
        mbar += np.where(sub & ~np.isnan(col_n), col_n, 0.0)
    mbar = mbar / 4.0
    D = np.where(sub, 0.5 + 0.5 * mbar, 0.0)
    df_full["m_bar"] = mbar
    df_full["U_star"] = df_full.q_star.values * (0.7 * D + 0.3 * R)
    df_full.to_csv(CFG.MANIFEST, index=False)
    print("[AUX] 已回填 r_i / R / m_bar / U_star 到 manifest.csv")


if __name__ == "__main__":
    main()
