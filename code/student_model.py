# -*- coding: utf-8 -*-
"""
student_model.py —— Student 网络结构定义（文档 7.2 节）
==========================================================
Direct 基线与 Student 共用此结构，仅损失不同（文档 7.8 节）。

【输入】（部署时全部可得，无 OCT）：
  feat_local (384,)   M_local 的 DINOv2 [CLS]
  feat_ctx   (384,)   M_ctx  的 [CLS]
  meta       (5,)     P，z-score 标准化（常数冻结）
【输出】三个头：p_S (3), q_S (1), U_S (1)；训练期另有 φ(F_S) 用于 L_feat

注意：Student 无 Transformer——773 维已是"对齐后拼接"，
跨模态交互由 Teacher 的教案间接传递（蒸馏），Student 自身只是 MLP。
"""

import torch
import torch.nn as nn

from common import CFG


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        # 融合 MLP：约 26 万参数（文档 7.5 节"可训练参数"）
        self.mlp = nn.Sequential(
            nn.Linear(CFG.FUSE_DIM, CFG.FEAT_DIM), nn.GELU(),   # 773 → 256
            nn.Linear(CFG.FEAT_DIM, CFG.FEAT_DIM), nn.GELU(),   # 256 → 256
        )
        # φ 投影层：仅训练期存在，把 F_S 投影到教师特征空间做 L_feat
        self.phi = nn.Sequential(nn.Linear(CFG.FEAT_DIM, CFG.FEAT_DIM),
                                 nn.LayerNorm(CFG.FEAT_DIM))
        # 三个输出头（文档 7.3）
        self.head_cls = nn.Linear(CFG.FEAT_DIM, 3)   # p_S：展示，写报告
        self.head_q   = nn.Linear(CFG.FEAT_DIM, 1)   # q_S：预警，误检归因
        self.head_U   = nn.Linear(CFG.FEAT_DIM, 1)   # U_S：决策，排序选区唯一依据

    def forward(self, feat_local, feat_ctx, meta):
        x = torch.cat([feat_local, feat_ctx, meta], dim=-1)    # (B, 773)
        F_S = self.mlp(x)                                       # (B, 256)
        return {
            "F_S": F_S,
            "phi_F_S": self.phi(F_S),                           # 训练专用
            "logits_S": self.head_cls(F_S),
            "q_S": torch.sigmoid(self.head_q(F_S)).squeeze(-1),
            "U_S": torch.sigmoid(self.head_U(F_S)).squeeze(-1),
        }
