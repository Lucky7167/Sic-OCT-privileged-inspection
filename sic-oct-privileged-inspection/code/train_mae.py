# -*- coding: utf-8 -*-
"""
train_mae.py —— 阶段 A：OCT 编码器 MAE 域适应预训练（文档第 4 章）
==================================================================
【输入】训练片 A/B 的全部 OCT 体数据 V (3,64,64,64)，无标签
【输出】checkpoints/mae_r3d18.pt —— 域适应后的 r3d_18 编码器权重（移交 Teacher）
【结构】r3d_18（Kinetics-400 初始化）+ 轻量 3D 转置卷积解码器
【评估】掩码位置重建 MSE（只看下降趋势；真正检验是下游 Teacher 性能）

为什么需要它：r3d_18 官方权重在自然视频（Kinetics-400）上预训练，
自然视频与 OCT 在纹理、噪声、灰度分布上差异巨大；训练样本仅 ~400 个，
不足以从头训练。MAE 自监督先把编码器"搬进" OCT 域。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from common import CFG, ROIDataset, CSVLogger, load_split, set_seed


# ----------------------------------------------------------------------
# MAE 模型：编码器复用 r3d_18 主干，解码器只在预训练期存在，训完即弃
# ----------------------------------------------------------------------
class TubeMAE(nn.Module):
    def __init__(self, mask_ratio=0.85):
        super().__init__()
        self.mask_ratio = mask_ratio
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        # 编码器 = 去掉 fc 的 r3d_18；输出特征图 (512, 4, 4, 4)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # 轻量解码器：把 (512,4,4,4) 上采样回 (1,64,64,64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=4),          # 4→16
            nn.GELU(),
            nn.ConvTranspose3d(256, 64, 4, stride=2, padding=1), # 16→32
            nn.GELU(),
            nn.ConvTranspose3d(64, 1, 4, stride=2, padding=1),   # 32→64
        )

    def tube_mask(self, x):
        """
        tube masking：沿"帧"方向整列掩掉（同一空间位置在所有帧上同时掩），
        掩码比例 75–90%（文档 4.2 节）。掩码处置 0。
        x: (B, 3, 64, 64, 64)，把通道维并入深度处理
        返回 masked_x 和 mask（True = 被掩位置，损失只算这些位置）
        """
        B = x.shape[0]
        # 空间网格 8×8（每格 8×8 像素），逐格决定掩/留
        g = 8
        grid = torch.rand(B, g, g, device=x.device) < self.mask_ratio   # (B,8,8)
        mask = grid.repeat_interleave(64 // g, 1).repeat_interleave(64 // g, 2)
        mask = mask.unsqueeze(1).unsqueeze(1)                            # (B,1,1,64,64)
        mask = mask.expand(-1, 3, 64, -1, -1)                            # 全帧共享 → tube
        return x * (~mask), mask

    def forward(self, x):
        xm, mask = self.tube_mask(x)
        feat = self.encoder(xm)            # (B,512,1,4,4,4)→压平
        feat = feat.squeeze(2)             # (B,512,4,4,4)
        recon = self.decoder(feat)         # (B,1,64,64,64)
        # 灰度复制 3 路的输入，重建单通道后与第 1 路比较
        target = x[:, :1]
        loss = ((recon - target) ** 2)[mask[:, :1]].mean()   # 只算掩码位置
        return loss


def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    # MAE 用训练片全部体数据（train+val 内部切分都用，无标签、无泄漏）
    import pandas as pd
    df_all = pd.concat([df_tr, df_vi])
    ds = ROIDataset(df_all, norm, load_volume=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    model = TubeMAE().to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    logger = CSVLogger("mae")

    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 41):                      # 文档 4.2：40 epoch
        model.train()
        tot, n = 0.0, 0
        for batch in dl:
            loss = model(batch["volume"].to(CFG.DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        logger.log(epoch=epoch, recon_mse=tot / n)
        print(f"[MAE] epoch {epoch:02d}/40  recon_mse={tot/n:.5f}")

    # 只保存编码器（解码器是训练脚手架，产出后弃用——文档 4.3 节）
    torch.save(model.encoder.state_dict(), CFG.CKPT_DIR / "mae_r3d18.pt")
    print("已保存 checkpoints/mae_r3d18.pt（编码器权重，移交 Teacher）")


if __name__ == "__main__":
    main()
