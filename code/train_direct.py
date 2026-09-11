# -*- coding: utf-8 -*-
"""
train_direct.py —— Direct 基线：仅显微镜 + 真值监督（文档 7.8 节）
====================================================================
【输入】与 Student 完全相同：feat_local, feat_ctx, meta（无 OCT、无教案）
【输出】checkpoints/direct.pt + logs/direct.csv
【结构】Student 同构（student_model.Student）
【损失】L_direct = L_cls^GT + L_q^GT + L_U^GT + L_rank —— 关闭全部蒸馏项

它的使命（论文叙事第一幕）：
  证明"不看 OCT，仅凭显微图，无法可靠预测深度缺陷"——
  预期不对称失败：浅表大缺陷能猜，subsurface 分类混淆、排序崩盘。
  评估时重点看：subsurface 召回率、混淆矩阵、subsurface 组内 Spearman。
【公平性】数据划分、过采样、早停指标与 train_student.py 完全一致——
  Direct 与 Student 的唯一差别是"有没有 Teacher 教案"，
  这样二者之差才能干净地归因于蒸馏。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (CFG, ROIDataset, SubsurfaceOversampler, CSVLogger,
                    load_split, set_seed, weighted_ce, huber, rank_loss, spearman)
from student_model import Student


def run_epoch(model, dl, opt, class_counts, train=True):
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    U_all, U_star_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            out = model(b["feat_local"].to(CFG.DEVICE), b["feat_ctx"].to(CFG.DEVICE),
                        b["meta"].to(CFG.DEVICE))
            z  = b["z_star"].to(CFG.DEVICE)
            qs = b["q_star"].to(CFG.DEVICE)
            Us = b["U_star"].to(CFG.DEVICE)
            loss = (weighted_ce(out["logits_S"], z, class_counts)
                    + huber(out["q_S"], qs)
                    + huber(out["U_S"], Us)
                    + rank_loss(out["U_S"], Us))       # 排序损失保留：与 Student 可比
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
            U_all.append(out["U_S"].detach().cpu().numpy())
            U_star_all.append(Us.cpu().numpy())
    return tot / n, spearman(np.concatenate(U_all), np.concatenate(U_star_all))


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    ds_tr = ROIDataset(df_tr, norm, load_volume=False)  # ★ 不加载 OCT
    ds_vc = ROIDataset(df_vc, norm, load_volume=False)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Student().to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # 同 7.5 节
    logger = CSVLogger("direct")

    best, patience, bad = -1.0, 15, 0
    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 101):                        # 上限 100 epoch
        tr_loss, sp_tr = run_epoch(model, dl_tr, opt, class_counts, train=True)
        va_loss, sp_va = run_epoch(model, dl_vc, None, class_counts, train=False)
        logger.log(epoch=epoch, tr_loss=tr_loss, va_loss=va_loss, va_spearman=sp_va)
        print(f"[Direct] ep{epoch:03d}  va_loss={va_loss:.4f}  va_spearman={sp_va:.4f}")
        if sp_va > best:                               # 早停监控验证集 Spearman（同 7.5）
            best, bad = sp_va, 0
            torch.save(model.state_dict(), CFG.CKPT_DIR / "direct.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[Direct] 早停于 epoch {epoch}")
                break
    print(f"[Direct] 完成，最优验证 Spearman = {best:.4f}")


if __name__ == "__main__":
    main()
