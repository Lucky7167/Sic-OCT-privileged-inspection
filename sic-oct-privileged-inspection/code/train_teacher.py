# -*- coding: utf-8 -*-
"""
train_teacher.py —— 阶段 B：Teacher 网络训练（文档第 5 章）
=============================================================
【输入】四模态：M_local/M_ctx 的 [CLS] 特征 (384×2)、OCT 体数据 V (3,64,64,64)、P (5,)
【输出】checkpoints/teacher.pt + 验证集指标日志 logs/teacher.csv
【结构】四路投影 → 4 token (4×256) → 1层 Transformer Encoder（模态间交互）→ 可学习query交叉注意力池化 → F_T → 四个头
【评估】验证集 L_U + Spearman(U_T, U*)（早停指标，文档 5.6 节）

训练要点（容易踩的坑）：
1. DINOv2 冻结——训练时根本不加载它，直接用离线 [CLS] 特征，显存需求骤降；
2. r3d_18 只解冻 layer4（~800万参数），前三层冻结——400 个体数据撑不起全量微调；
3. L_morph 必须在 subsurface 掩码内计算，掩码外样本对形态头零监督（设计意图）；
4. MAE 权重先加载再训练，不要从 Kinetics 权重直接开始。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from common import (CFG, ROIDataset, SubsurfaceOversampler, CSVLogger,
                    load_split, set_seed, weighted_ce, huber, morph_huber, spearman)


# ----------------------------------------------------------------------
class OCTEncoder(nn.Module):
    """
    r3d_18 加载 MAE 域适应权重；全局池化得 512 维。
    冻结策略：stem/layer1-3 冻结，仅 layer4 可训练（文档 5.2/5.6 节）。
    """
    def __init__(self, mae_ckpt=CFG.CKPT_DIR / "mae_r3d18.pt"):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # 去 fc
        if mae_ckpt.exists():
            self.features.load_state_dict(torch.load(mae_ckpt))          # ★ MAE 初始化
            print("[Teacher] 已加载 MAE 域适应权重")
        else:
            print("[Teacher] 警告：未找到 MAE 权重，从 Kinetics 直接开始（请先跑 train_mae.py）")
        # 冻结除 layer4 外全部
        for name, p in self.features.named_parameters():
            p.requires_grad = name.startswith("7")   # children 序号为 7 的是 layer4

    def forward(self, v):                     # v: (B,3,64,64,64)
        f = self.features(v)                  # (B,512,1,4,4,4)
        return f.mean(dim=[2, 3, 4])          # 全局池化 → (B,512)


class Teacher(nn.Module):
    """
    文档 5.2–5.4：
      M_local [CLS] 384 → Linear(384→256)      ┐
      M_ctx   [CLS] 384 → Linear(384→256)      ├ 4 token (4×256)
      V → r3d_18 512   → Linear(512→256)      ┤ → Transformer(1层,4头,pre-LN)
      P 5维 → MLP(5→64→32→256) → GELU → 256   ┘ → 可学习query交叉注意力池化
                                                   → F_T(256) → 四个头
      融合分两步（文档 5.3）：① self-attention 模态间交互；② 可学习 query q_f
      对交互后的 4 个 token 做交叉注意力池化（替代固定等权的 mean-pool），
      逐样本自适应模态权重 w_i 作为副产品返回（仅供分析，不蒸馏）。
    """
    def __init__(self):
        super().__init__()
        self.oct_enc = OCTEncoder()
        self.proj_local = nn.Linear(CFG.DINO_DIM, CFG.FEAT_DIM)
        self.proj_ctx   = nn.Linear(CFG.DINO_DIM, CFG.FEAT_DIM)
        self.proj_vol   = nn.Linear(512, CFG.FEAT_DIM)
        self.meta_mlp   = nn.Sequential(nn.Linear(5, 64), nn.GELU(),
                                        nn.Linear(64, 32), nn.GELU(),
                                        nn.Linear(32, CFG.FEAT_DIM))
        layer = nn.TransformerEncoderLayer(
            d_model=CFG.FEAT_DIM, nhead=4, dim_feedforward=1024,
            activation="gelu", norm_first=True, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=1)
        # 第二步：可学习 query 的交叉注意力池化（文档 5.3，替代 mean-pool）
        self.q_f   = nn.Parameter(torch.randn(1, 1, CFG.FEAT_DIM))
        self.xattn = nn.MultiheadAttention(CFG.FEAT_DIM, 4, batch_first=True)
        # 四个输出头（文档 5.4）
        self.head_cls   = nn.Linear(CFG.FEAT_DIM, 3)   # 分类头 p_T（保 logits）
        self.head_q     = nn.Linear(CFG.FEAT_DIM, 1)   # 可观测性 q_T
        self.head_morph = nn.Linear(CFG.FEAT_DIM, 4)   # 形态头 m_T
        self.head_U     = nn.Linear(CFG.FEAT_DIM, 1)   # 效用头 U_T

    def forward(self, feat_local, feat_ctx, volume, meta):
        tokens = torch.stack([
            self.proj_local(feat_local),
            self.proj_ctx(feat_ctx),
            self.proj_vol(self.oct_enc(volume)),
            self.meta_mlp(meta),
        ], dim=1)                                     # (B,4,256)
        h = self.fusion(tokens)                       # 第一步：模态间交互 (B,4,256)
        q = self.q_f.expand(h.size(0), -1, -1)        # (B,1,256)
        F_T, attn_w = self.xattn(q, h, h)             # 第二步：交叉注意力池化
        F_T = F_T.squeeze(1)                          # (B,256)
        return {
            "F_T": F_T,
            "w_T": attn_w.squeeze(1),                 # (B,4) 逐样本模态权重，仅供分析
            "logits_T": self.head_cls(F_T),
            "q_T": torch.sigmoid(self.head_q(F_T)).squeeze(-1),
            "m_T": self.head_morph(F_T),
            "U_T": torch.sigmoid(self.head_U(F_T)).squeeze(-1),
        }


# ----------------------------------------------------------------------
def run_epoch(model, dl, opt, class_counts, train=True):
    model.train() if train else model.eval()
    tot = {"cls": 0, "q": 0, "morph": 0, "U": 0}
    U_all, U_star_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            out = model(b["feat_local"].to(CFG.DEVICE), b["feat_ctx"].to(CFG.DEVICE),
                        b["volume"].to(CFG.DEVICE), b["meta"].to(CFG.DEVICE))
            z   = b["z_star"].to(CFG.DEVICE)
            qs  = b["q_star"].to(CFG.DEVICE)
            ms  = b["m_star"].to(CFG.DEVICE)
            msk = b["morph_mask"].to(CFG.DEVICE)
            Us  = b["U_star"].to(CFG.DEVICE)
            # 文档 5.5：L_T = L_cls + 1.0·L_q + 0.5·L_morph + 2.0·L_U
            l_cls   = weighted_ce(out["logits_T"], z, class_counts)
            l_q     = huber(out["q_T"], qs)
            l_morph = morph_huber(out["m_T"], ms, msk)   # ★ 掩码内计算
            l_U     = huber(out["U_T"], Us)
            loss = l_cls + CFG.W_T_Q * l_q + CFG.W_T_MORPH * l_morph + CFG.W_T_U * l_U
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            for k, v in zip(tot, [l_cls, l_q, l_morph, l_U]):
                tot[k] += v.item()
            U_all.append(out["U_T"].detach().cpu().numpy())
            U_star_all.append(Us.cpu().numpy())
    n = len(dl)
    spear = spearman(np.concatenate(U_all), np.concatenate(U_star_all))
    return {k: v / n for k, v in tot.items()}, spear


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    ds_tr = ROIDataset(df_tr, norm, load_volume=True)
    ds_vc = ROIDataset(df_vc, norm, load_volume=True)   # Wafer C 独立验证
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False, num_workers=4)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Teacher().to(CFG.DEVICE)
    # 只把 requires_grad=True 的参数交给优化器（自动遵守冻结策略）
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    logger = CSVLogger("teacher")

    best, patience, bad = float("inf"), 20, 0
    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 151):                       # 上限 150 epoch（文档 5.6）
        tr, sp_tr = run_epoch(model, dl_tr, opt, class_counts, train=True)
        va, sp_va = run_epoch(model, dl_vc, None, class_counts, train=False)
        # 早停指标：验证集 L_U + Spearman（文档 5.6）——L_U 要小、Spearman 要大
        criterion = va["U"] - sp_va
        logger.log(epoch=epoch, **{f"tr_{k}": v for k, v in tr.items()},
                   **{f"va_{k}": v for k, v in va.items()}, va_spearman=sp_va)
        print(f"[Teacher] ep{epoch:03d}  va_U={va['U']:.4f}  va_spearman={sp_va:.4f}")
        if criterion < best:
            best, bad = criterion, 0
            torch.save(model.state_dict(), CFG.CKPT_DIR / "teacher.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[Teacher] 早停于 epoch {epoch}")
                break
    print(f"[Teacher] 完成，最优验证指标 {best:.4f}，权重已存 checkpoints/teacher.pt")


if __name__ == "__main__":
    main()
