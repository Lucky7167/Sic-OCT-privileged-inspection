# -*- coding: utf-8 -*-
"""
save_lessons.py —— 阶段 C：教案保存（文档第 6 章）
====================================================
【输入】teacher.pt（已训练达标并永久冻结）+ 训练集 + 验证集全部样本
【输出】checkpoints/lessons.pt —— {roi_id: {F_T, logits_T, q_T, U_T}}

为什么"保存"而不是在线联训（文档第 6 章）：
- Teacher 前向要跑 OCT 支路，在线联训会让 Student 管线依赖 OCT 数据流，徒增 IO/显存；
- 缓存后 Student 训练是纯监督任务：稳定、可复现、可反复重训。
"""

import torch
from torch.utils.data import DataLoader

from common import CFG, ROIDataset, load_split, set_seed
from train_teacher import Teacher


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    # 教案覆盖 训练集 + Wafer C 验证集（Student 的训练与早停都需要教案）
    import pandas as pd
    df_all = pd.concat([df_tr, df_vi, df_vc]).reset_index(drop=True)

    model = Teacher().to(CFG.DEVICE)
    model.load_state_dict(torch.load(CFG.CKPT_DIR / "teacher.pt"))
    model.eval()                                     # ★ 永久冻结
    for p in model.parameters():
        p.requires_grad = False

    ds = ROIDataset(df_all, norm, load_volume=True)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=False, num_workers=4)

    lessons = {}
    with torch.no_grad():
        for b in dl:
            out = model(b["feat_local"].to(CFG.DEVICE), b["feat_ctx"].to(CFG.DEVICE),
                        b["volume"].to(CFG.DEVICE), b["meta"].to(CFG.DEVICE))
            for i, rid in enumerate(b["roi_id"]):
                lessons[rid] = {
                    "F_T":      out["F_T"][i].cpu(),        # 256维，L_feat 对齐目标
                    "logits_T": out["logits_T"][i].cpu(),   # 3维，L_KD 教师分布
                    "q_T":      out["q_T"][i].cpu(),        # 标量，一致性检查
                    "U_T":      out["U_T"][i].cpu(),        # 标量，L_KD-U 蒸馏锚点
                }
    torch.save(lessons, CFG.CKPT_DIR / "lessons.pt")
    print(f"[Lessons] 已保存 {len(lessons)} 条教案 → checkpoints/lessons.pt")


if __name__ == "__main__":
    main()
