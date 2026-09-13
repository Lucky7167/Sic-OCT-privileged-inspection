import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (CFG, ROIDataset, SubsurfaceOversampler, CSVLogger,
                    load_split, set_seed, weighted_ce, huber, rank_loss,
                    feat_align_loss, kd_loss, spearman)
from student_model import Student

class StudentDataset(Dataset):
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
            l_cls = weighted_ce(out["logits_S"], z, class_counts)
            l_q   = huber(out["q_S"], qs)
            l_U   = huber(out["U_S"], Us)
            l_feat = feat_align_loss(out["F_S"], b["F_T"].to(dev), model.phi)
            l_kd   = kd_loss(out["logits_S"], b["logits_T"].to(dev))
            l_kdu  = huber(out["U_S"], b["U_T"].to(dev).float())
            l_rank = rank_loss(out["U_S"], Us)        
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
    lessons = torch.load(CFG.CKPT_DIR / "lessons.pt") 

    ds_tr = StudentDataset(ROIDataset(df_tr, norm, load_volume=False), lessons)
    ds_vc = StudentDataset(ROIDataset(df_vc, norm, load_volume=False), lessons)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH,
                       sampler=SubsurfaceOversampler(df_tr), num_workers=4)
    dl_vc = DataLoader(ds_vc, batch_size=CFG.BATCH, shuffle=False)
    class_counts = df_tr.z_star.value_counts().sort_index().values

    model = Student().to(CFG.DEVICE)             
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
                print(f"[Student] epoch {epoch}")
                break

if __name__ == "__main__":
    main()
