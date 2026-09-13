import torch
from torch.utils.data import DataLoader
from common import CFG, ROIDataset, load_split, set_seed
from train_teacher import Teacher

def main():
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    import pandas as pd
    df_all = pd.concat([df_tr, df_vi, df_vc]).reset_index(drop=True)

    model = Teacher().to(CFG.DEVICE)
    model.load_state_dict(torch.load(CFG.CKPT_DIR / "teacher.pt"))
    model.eval()                              
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
                    "F_T":      out["F_T"][i].cpu(),      
                    "logits_T": out["logits_T"][i].cpu(), 
                    "q_T":      out["q_T"][i].cpu(),      
                    "U_T":      out["U_T"][i].cpu(),       
                }
    torch.save(lessons, CFG.CKPT_DIR / "lessons.pt")

if __name__ == "__main__":
    main()
