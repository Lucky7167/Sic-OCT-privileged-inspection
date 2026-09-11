# -*- coding: utf-8 -*-
"""
train_student.py —— 阶段 D：Student 网络训练（文档第 7 章）
==============================================================
【输入】feat_local, feat_ctx, meta（与 Direct 完全相同）+ lessons.pt（教案）
【输出】checkpoints/student.pt —— 最终部署模型
【损失】七项双通道（文档 7.4）：
  通道1（真值监督）：L_cls^GT + L_q^GT + L_U^GT
  通道2（蒸馏）：    1.0·L_feat + 0.5·L_KD(τ=3) + 1.0·L_KD-U + 1.0·L_rank

训练要点（容易踩的坑）：
1. 全部随机初始化，与 Teacher 无参数共享（文档 6.2）——传递的是教案不是权重；
2. 教案只从 lessons.pt 读，训练循环里不出现 Teacher 网络和 OCT 数据；
3. subsurface 过采样 ≥30%/batch 必须开——否则 L_morph 类损失梯度稀疏，
   且排序损失中学不到"subsurface 排最前"的关键样本对；
4. 早停只看验证集 Spearman(U_S, U*)——排序保真度是应用目标，别看 loss；
5. 验收（7.6）：与 Teacher 的 Spearman 差距 ≤0.05、准确率差距 ≤3%、CPU <10ms，
   全部满足才保存部署——evaluate.py 里有现成检查。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (CFG, ROIDataset, SubsurfaceOversampler, CSVLogger,
                    load_split, set_seed, weighted_ce, huber, rank_loss,
                    feat_align_loss, kd_loss, spearman)
from student_model import Student


class StudentDataset(Dataset):
    """包装 ROIDataset：给每个样本挂上教案"""
    def __init__(self, base: ROIDataset, lessons: dict):
        self.base, self.lessons = base, lessons

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        les = self.lessons[item["roi_id"]]
        item["F_T"] = les["F_T"]
        item["logits_T"] = les["logits_T"]
        item["U_T"] = les["U_T"]
        return item


def run_epoch(model, dl, opt, class_counts, train=True):
    model.train() if train else model.eval()
    tot = {"cls": 0, "q": 0, "U": 0, "feat": 0, "kd": 0, "kdu": 0, "rank": 0}
    U_all, U_star_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            dev = CFG.DEVICE
            out = model(b["feat_local"].to(dev), b["feat_ctx"].to(dev), b["meta"].to(dev))
            z, qs, Us = b["z_star"].to(dev), b["q_star"].to(dev), b["U_star"].to(dev)
            # ---- 通道 1：真值监督（权重均 1.0）----
            l_cls = weighted_ce(out["logits_S"], z, class_counts)
            l_q   = huber(out["q_S"], qs)
            l_U   = huber(out["U_S"], Us)
            # ---- 通道 2：蒸馏（教案）----
            l_feat = feat_align_loss(out["F_S"], b["F_T"].to(dev), model.phi)
            l_kd   = kd_loss(out["logits_S"], b["logits_T"].to(dev))
            l_kdu  = huber(out["U_S"], b["U_T"].to(dev).float())
            l_rank = rank_loss(out["U_S"], Us)          # 比较对来自 U*（真值）
            loss = (l_cls + l_q + l_U
                    + CFG.W_FEAT * l_feat + CFG.W_KD * l_kd
                    + CFG.W_KDU * l_kdu + CFG.W_RANK * l_rank)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            for k, v in zip(tot, [l_cls, l_q, l_U, l_feat, l_kd, l_kdu, l_rank]):
                tot[k] += v.item()
            U_all.append(out["U_S"].detach().cpu().numpy())
            U_star_all.append(Us.cpu().numpy())
    n = len(dl)
    return {k: v / n for k, v in tot.items()}, spearman(np.concatenate(U_all),
                                                        np.concatenate(U_star_all))


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    lessons = torch.load(CFG.CKPT_DIR / "lessons.pt")   # ★ 教案（先跑 save_lessons.py）

    ds_tr = StudentDataset(ROIDataset(df_tr, norm, load_volume=False), lessons)
    ds_vc = StudentDataset(ROIDataset(df_vc, norm, load_volume=False), lessons)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Student().to(CFG.DEVICE)                    # ★ 随机初始化
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    logger = CSVLogger("student")

    best, patience, bad = -1.0, 15, 0
    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 101):
        tr, sp_tr = run_epoch(model, dl_tr, opt, class_counts, train=True)
        va, sp_va = run_epoch(model, dl_vc, None, class_counts, train=False)
        logger.log(epoch=epoch, **{f"tr_{k}": v for k, v in tr.items()},
                   **{f"va_{k}": v for k, v in va.items()}, va_spearman=sp_va)
        print(f"[Student] ep{epoch:03d}  va_spearman={sp_va:.4f}")
        if sp_va > best:
            best, bad = sp_va, 0
            torch.save(model.state_dict(), CFG.CKPT_DIR / "student.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[Student] 早停于 epoch {epoch}")
                break
    print(f"[Student] 完成，最优验证 Spearman = {best:.4f}")
    print("下一步：python evaluate.py（三方对比 Direct / Student / Teacher + 验收检查）")


if __name__ == "__main__":
    main()
