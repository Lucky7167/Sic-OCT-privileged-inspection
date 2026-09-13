import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CFG:
    DATA_ROOT   = Path("data")
    MANIFEST    = DATA_ROOT / "manifest.csv"          
    PATCH_DIR   = DATA_ROOT / "patches"             
    VOLUME_DIR  = DATA_ROOT / "volumes"              
    NORM_JSON   = DATA_ROOT / "norm_constants.json"   
    CKPT_DIR    = Path("checkpoints")                
    LOG_DIR     = Path("logs")                       
    TRAIN_WAFERS = ["A", "B"]     
    VAL_WAFER    = "C"            
    TEST_WAFERS  = ["D", "E"]     
    DINO_DIM   = 384      
    META_DIM   = 5     
    FUSE_DIM   = 773   
    FEAT_DIM   = 256    
    W_T_Q, W_T_MORPH, W_T_U = 1.0, 0.5, 2.0
    W_FEAT, W_KD, W_KDU, W_RANK = 1.0, 0.5, 1.0, 1.0
    KD_TAU = 3.0         
    HUBER_DELTA = 1.0     
    BATCH        = 32
    SUB_OVERSAMPLE = 0.30  
    SEED         = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def compute_or_load_norm(df_train: pd.DataFrame) -> dict:
   
    if CFG.NORM_JSON.exists():
        return json.loads(CFG.NORM_JSON.read_text())

    norm = {}
    for k in ["x", "y", "w", "h", "a"]:
        norm[f"{k}_mean"] = float(df_train[k].mean())
        norm[f"{k}_std"]  = float(df_train[k].std() + 1e-8)
    sub = df_train[df_train.z_star == 2]
    for k in ["d", "v", "o", "c"]:
        norm[f"{k}_min"] = float(sub[k].min())
        norm[f"{k}_max"] = float(sub[k].max())
    CFG.NORM_JSON.write_text(json.dumps(norm, indent=2, ensure_ascii=False))
    return norm

class ROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, norm: dict, load_volume: bool):
        self.df = df.reset_index(drop=True)
        self.norm = norm
        self.load_volume = load_volume

    def __len__(self):
        return len(self.df)

    def _meta(self, row) -> torch.Tensor:
        v = [(row[k] - self.norm[f"{k}_mean"]) / self.norm[f"{k}_std"]
             for k in ["x", "y", "w", "h", "a"]]
        return torch.tensor(v, dtype=torch.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = {
            "roi_id":  row.roi_id,
            "feat_local": torch.from_numpy(np.load(CFG.PATCH_DIR / f"{row.roi_id}_local.npy")).float(),
            "feat_ctx":   torch.from_numpy(np.load(CFG.PATCH_DIR / f"{row.roi_id}_ctx.npy")).float(),
            "meta":       self._meta(row),
            "z_star":     torch.tensor(int(row.z_star), dtype=torch.long),
            "q_star":     torch.tensor(float(row.q_star), dtype=torch.float32),
            "U_star":     torch.tensor(float(row.U_star), dtype=torch.float32),
            "morph_mask": torch.tensor(bool(row.z_star == 2)),
        }
        m = [row.d, row.v, row.o, row.c]
        item["m_star"] = torch.tensor(
            [np.nan if pd.isna(x) else float(x) for x in m], dtype=torch.float32)
        if self.load_volume:
            item["volume"] = torch.from_numpy(
                np.load(CFG.VOLUME_DIR / f"{row.roi_id}.npy")).float()  # (3,64,64,64)
        return item

class SubsurfaceOversampler(torch.utils.data.Sampler):
    def __init__(self, df):
        self.sub_idx  = df.index[df.z_star == 2].tolist()
        self.rest_idx = df.index[df.z_star != 2].tolist()
        n_sub = max(1, int(CFG.SUB_OVERSAMPLE * len(df)))
        self.epoch_idx = self.rest_idx + random.choices(self.sub_idx, k=n_sub)

    def __iter__(self):
        random.shuffle(self.epoch_idx)
        return iter(self.epoch_idx)

    def __len__(self):
        return len(self.epoch_idx)

def load_split(norm=None):
    df = pd.read_csv(CFG.MANIFEST)
    df_tr  = df[(df.wafer.isin(CFG.TRAIN_WAFERS)) & (df.split == "train")]
    df_vi  = df[(df.wafer.isin(CFG.TRAIN_WAFERS)) & (df.split == "val")]
    df_vc  = df[df.wafer == CFG.VAL_WAFER]
    df_te  = df[df.wafer.isin(CFG.TEST_WAFERS)]
    if norm is None:
        norm = compute_or_load_norm(df_tr)
    return df_tr, df_vi, df_vc, df_te, norm

def weighted_ce(logits, target, class_counts):
    w = 1.0 / (class_counts.float() + 1e-8)
    w = w / w.sum() * len(class_counts)       
    return F.cross_entropy(logits, target, weight=w.to(logits.device))
    
def huber(pred, target, delta=CFG.HUBER_DELTA):
    return F.huber_loss(pred, target, delta=delta)

def morph_huber(pred4, m_star, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred4.device)
    return F.huber_loss(pred4[mask], m_star[mask], delta=CFG.HUBER_DELTA)

def feat_align_loss(f_s, f_t, proj):
    g = proj(f_s)
    mse = F.mse_loss(g, f_t)
    cos = 1 - F.cosine_similarity(g, f_t, dim=-1).mean()
    return mse + cos

def kd_loss(logits_s, logits_t, tau=CFG.KD_TAU):
    p = F.log_softmax(logits_s / tau, dim=-1)
    q = F.softmax(logits_t / tau, dim=-1)
    return tau * tau * F.kl_div(p, q, reduction="batchmean")

def rank_loss(U_s, U_star):
    diff_pred = U_s.unsqueeze(1) - U_s.unsqueeze(0)      # (B,B)
    diff_true = U_star.unsqueeze(1) - U_star.unsqueeze(0)
    pair_mask = (diff_true > 0).float()
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=U_s.device)
    loss = -F.logsigmoid(diff_pred) * pair_mask
    return loss.sum() / pair_mask.sum()

def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])

def per_class_metrics(y_true, y_pred, n_cls=3):
    names = ["normal", "surface", "subsurface"]
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    out = {}
    for c in range(n_cls):
        tp = conf[c, c]
        prec = tp / max(conf[:, c].sum(), 1)
        rec  = tp / max(conf[c, :].sum(), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        out[names[c]] = {"precision": prec, "recall": rec, "f1": f1}
    return out, conf

def topk_hit_rate(U_pred, z_star, rho=0.2):
    k = max(1, int(len(U_pred) * rho))
    top = np.argsort(-np.asarray(U_pred))[:k]
    return float((np.asarray(z_star)[top] == 2).mean())

class CSVLogger:
    def __init__(self, name):
        CFG.LOG_DIR.mkdir(exist_ok=True)
        self.path = CFG.LOG_DIR / f"{name}.csv"
        self.rows = []

    def log(self, **kw):
        self.rows.append(kw)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
