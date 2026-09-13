import torch
import torch.nn as nn
from common import CFG

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(CFG.FUSE_DIM, CFG.FEAT_DIM), nn.GELU(), 
            nn.Linear(CFG.FEAT_DIM, CFG.FEAT_DIM), nn.GELU(),  
        )

        self.phi = nn.Sequential(nn.Linear(CFG.FEAT_DIM, CFG.FEAT_DIM),
                                 nn.LayerNorm(CFG.FEAT_DIM))

        self.head_cls = nn.Linear(CFG.FEAT_DIM, 3)  
        self.head_q   = nn.Linear(CFG.FEAT_DIM, 1)   
        self.head_U   = nn.Linear(CFG.FEAT_DIM, 1)   

    def forward(self, feat_local, feat_ctx, meta):
        x = torch.cat([feat_local, feat_ctx, meta], dim=-1)   
        F_S = self.mlp(x)                                   
        return {
            "F_S": F_
            "phi_F_S": self.phi(F_S),                          
            "logits_S": self.head_cls(F_S),
            "q_S": torch.sigmoid(self.head_q(F_S)).squeeze(-1),
            "U_S": torch.sigmoid(self.head_U(F_S)).squeeze(-1),
        }
