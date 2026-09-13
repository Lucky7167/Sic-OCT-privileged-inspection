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
                    + rank_loss(out["U_S"], Us))  
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
            U_all.append(out["U_S"].detach().cpu().numpy())
            U_star_all.append(Us.cpu().numpy())
    return tot / n, spearman(np.concatenate(U_all), np.concatenate(U_star_all))

def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    ds_tr = ROIDataset(df_tr, norm, load_volume=False) 
    ds_vc = ROIDataset(df_vc, norm, load_volume=False)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Student().to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  
    logger = CSVLogger("direct")

    best, patience, bad = -1.0, 15, 0
    CFG.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, 101):                      
        tr_loss, sp_tr = run_epoch(model, dl_tr, opt, class_counts, train=True)
        va_loss, sp_va = run_epoch(model, dl_vc, None, class_counts, train=False)
        logger.log(epoch=epoch, tr_loss=tr_loss, va_loss=va_loss, va_spearman=sp_va)
        print(f"[Direct] ep{epoch:03d}  va_loss={va_loss:.4f}  va_spearman={sp_va:.4f}")
        if sp_va > best:                           
            best, bad = sp_va, 0
            torch.save(model.state_dict(), CFG.CKPT_DIR / "direct.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[Direct] epoch {epoch}")
                break

if __name__ == "__main__":
    main()
