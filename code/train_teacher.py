import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from common import (CFG, ROIDataset, SubsurfaceOversampler, CSVLogger,
                    load_split, set_seed, weighted_ce, huber, morph_huber, spearman)

class OCTEncoder(nn.Module):
    def __init__(self, mae_ckpt=CFG.CKPT_DIR / "mae_r3d18.pt"):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  
        if mae_ckpt.exists():
            self.features.load_state_dict(torch.load(mae_ckpt))         
            print("[Teacher] MAE ")
        else:
            print("[Teacher] warning")

        for name, p in self.features.named_parameters():
            p.requires_grad = name.startswith("7")  

    def forward(self, v):                    
        f = self.features(v)                
        return f.mean(dim=[2, 3, 4])      


class Teacher(nn.Module):
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
  
        self.q_f   = nn.Parameter(torch.randn(1, 1, CFG.FEAT_DIM))
        self.xattn = nn.MultiheadAttention(CFG.FEAT_DIM, 4, batch_first=True)
 
        self.head_cls   = nn.Linear(CFG.FEAT_DIM, 3)
        self.head_q     = nn.Linear(CFG.FEAT_DIM, 1) 
        self.head_morph = nn.Linear(CFG.FEAT_DIM, 4) 
        self.head_U     = nn.Linear(CFG.FEAT_DIM, 1)  

    def forward(self, feat_local, feat_ctx, volume, meta):
        tokens = torch.stack([
            self.proj_local(feat_local),
            self.proj_ctx(feat_ctx),
            self.proj_vol(self.oct_enc(volume)),
            self.meta_mlp(meta),
        ], dim=1)                                  
        h = self.fusion(tokens)                    
        q = self.q_f.expand(h.size(0), -1, -1)      
        F_T, attn_w = self.xattn(q, h, h)           
        F_T = F_T.squeeze(1)                       
        return {
            "F_T": F_T,
            "w_T": attn_w.squeeze(1),             
            "logits_T": self.head_cls(F_T),
            "q_T": torch.sigmoid(self.head_q(F_T)).squeeze(-1),
            "m_T": self.head_morph(F_T),
            "U_T": torch.sigmoid(self.head_U(F_T)).squeeze(-1),
        }


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
            l_morph = morph_huber(out["m_T"], ms, msk) 
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
    ds_vc = ROIDataset(df_vc, norm, load_volume=True)   
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False, num_workers=4)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Teacher().to(CFG.DEVICE)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    logger = CSVLogger("teacher")

    best, patience, bad = float("inf"), 20, 0
    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 151):                     
        tr, sp_tr = run_epoch(model, dl_tr, opt, class_counts, train=True)
        va, sp_va = run_epoch(model, dl_vc, None, class_counts, train=False)
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
                print(f"[Teacher] epoch {epoch}")
                break

if __name__ == "__main__":
    main()
