import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lupi_components as sp

ROOT = Path(__file__).resolve().parents[1]   
mani = pd.read_csv(ROOT / "data/manifest.csv")
norm = json.loads((ROOT / "data/norm.json").read_text())
feats_m = np.load(ROOT / "data/feats_micro.npy", allow_pickle=True).item()
feats_o = np.load(ROOT / "data/feats_oct.npy", allow_pickle=True).item()
DEV = "cpu"
TAU = sp.TAU

def wce(logits, z, counts):
    w = 1.0 / counts.float(); w = w / w.sum() * len(counts)
    return F.cross_entropy(logits, z, weight=w.to(logits.device))

def multiotsu3(vals, nbins=256):
    v = np.asarray(vals, float); lo, hi = v.min(), v.max()
    if hi <= lo: return lo, hi
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    p = hist.astype(float); p /= p.sum()
    c = (edges[:-1] + edges[1:]) / 2
    om = np.cumsum(p); mu = np.cumsum(p * c); mt = mu[-1]
    best, bt = -1, (c[0], c[-1])
    for i in range(1, nbins - 2):
        for j in range(i + 1, nbins - 1):
            w0, w1, w2 = om[i], om[j] - om[i], 1 - om[j]
            if w0 <= 0 or w1 <= 0 or w2 <= 0: continue
            m0, m1, m2 = mu[i]/w0, (mu[j]-mu[i])/w1, (mt-mu[j])/w2
            b = w0*(m0-mt)**2 + w1*(m1-mt)**2 + w2*(m2-mt)**2
            if b > best: best, bt = b, (c[i], c[j])
    return bt

def budget_full_recall(s, z):
    t_lo_o, t_hi = multiotsu3(s)
    t_lo = min(t_lo_o, s[z == 2].min() - 1e-9)
    band = (s > t_lo) & (s < t_hi)
    return band.mean()

@torch.no_grad()
def make_lessons(teacher, df):
    dl = DataLoader(sp.SimDS(df, feats_m, feats_o, norm), batch_size=64)
    L = {}
    for b in dl:
        o = teacher(b["fm"], b["fo"], b["meta"])
        for i, rid in enumerate(b["roi_id"]):
            L[rid] = dict(F_T=o["F_T"][i].clone(), logits_T=o["logits"][i].clone(), U_T=o["U"][i].clone())
    return L

def train_variant(name, df_tr, df_va, lessons, flags, epochs=150, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    counts = torch.tensor(df_tr["z_obs"].value_counts().sort_index().values)
    dl_tr = DataLoader(sp.SimDS(df_tr.iloc[sp.oversample_idx(df_tr)], feats_m, {}, norm, lessons),
                       batch_size=32, shuffle=True)
    dl_va = DataLoader(sp.SimDS(df_va, feats_m, {}, norm, lessons), batch_size=64)
    m = sp.Student(direct=flags.get("direct_arch", False))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1e9, None
    for ep in range(epochs):
        m.train()
        for b in dl_tr:
            o = m(b["fm"], b["meta"])
            z, q, U = b["z"], b["q"], b["U"]
            loss = 0.0
            if flags.get("ce", True):  loss = loss + wce(o["logits"], z, counts)
            if flags.get("q"):         loss = loss + F.huber_loss(o["q"], q, delta=1.0)
            if flags.get("ustar"):     loss = loss + F.huber_loss(o["U"], U, delta=0.25)
            if flags.get("rank"):      loss = loss + sp.rank_loss(o["U"], U)
            if lessons is not None:
                if flags.get("feat"):
                    loss = loss + (F.mse_loss(o["phi"], b["F_T"])
                                   + (1 - F.cosine_similarity(o["phi"], b["F_T"], -1)).mean())
                if flags.get("kd"):
                    loss = loss + 0.5 * TAU**2 * F.kl_div(
                        F.log_softmax(o["logits"] / TAU, -1),
                        F.softmax(b["logits_T"] / TAU, -1), reduction="batchmean")
                if flags.get("kdu"):
                    loss = loss + F.huber_loss(o["U"], b["U_T"].float(), delta=0.25)
            opt.zero_grad(); loss.backward(); opt.step()

        m.eval(); us, uts, zs, ps = [], [], [], []
        with torch.no_grad():
            for b in dl_va:
                o = m(b["fm"], b["meta"])
                us.append(o["U"].numpy()); uts.append(b["U"].numpy())
                zs.append(b["z_true"].numpy())
                ps.append(o["logits"].argmax(-1).numpy())
        us, uts, zs, ps = np.concatenate(us), np.concatenate(uts), np.concatenate(zs), np.concatenate(ps)
        if flags.get("ustar") or flags.get("kdu"):
            crit = sp.spearman(us, uts)
        else:
            crit = f1_score(zs, ps, average="macro")
        if crit > best:
            best, best_state = crit, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m

@torch.no_grad()
def predict(m, df, use_U):
    dl = DataLoader(sp.SimDS(df, feats_m, {}, norm, None), batch_size=64)
    rows = []
    for b in dl:
        o = m(b["fm"], b["meta"])
        prob = F.softmax(o["logits"], -1)
        score = o["U"].numpy() if use_U else prob[:, 2].numpy()
        for i, rid in enumerate(b["roi_id"]):
            rows.append(dict(roi_id=rid, score=float(score[i]), p_sub=float(prob[i, 2]),
                             pred=int(o["logits"][i].argmax())))
    return pd.DataFrame(rows)

VARIANTS = {
    "V0 Direct (single-scale)":      dict(direct_arch=True, ce=True, q=True, ustar=True, rank=True),
    "V1 Student CE-only":            dict(ce=True),
    "V2 +logit KD":                  dict(ce=True, kd=True),
    "V3 +utility KD":                dict(ce=True, kdu=True),
    "V4 full Student":               dict(ce=True, q=True, ustar=True, rank=True, feat=True, kd=True, kdu=True),
}

all_preds, summary = [], []
t_start = time.time()
for fold, held in enumerate(["A", "B", "C", "D", "E"]):
    df_pool = mani[mani.wafer != held].reset_index(drop=True)
    df_te   = mani[mani.wafer == held].reset_index(drop=True)

    idx_va = (df_pool.groupby("z_star", group_keys=False)
              .apply(lambda g: g.sample(frac=0.10, random_state=fold)).index)
    df_va = df_pool.loc[idx_va].reset_index(drop=True)
    df_tr = df_pool.drop(index=idx_va).reset_index(drop=True)

    teacher = sp.train_teacher(df_tr, df_va, feats_m, feats_o, norm)
    lessons = make_lessons(teacher, pd.concat([df_tr, df_va]))
 
    dl_te = DataLoader(sp.SimDS(df_te, feats_m, feats_o, norm), batch_size=64)
    trows = []
    with torch.no_grad():
        for b in dl_te:
            o = teacher(b["fm"], b["fo"], b["meta"])
            for i, rid in enumerate(b["roi_id"]):
                trows.append(dict(roi_id=rid, score=float(o["U"][i])))
    tp = pd.DataFrame(trows); tp["variant"] = "Teacher (privileged)"; tp["fold"] = held
    all_preds.append(tp)

    for vn, flags in VARIANTS.items():
        lessons_arg = None if flags.get("direct_arch") else lessons
        m = train_variant(vn, df_tr, df_va, lessons_arg, flags, seed=100 + fold)
        use_U = flags.get("ustar") or flags.get("kdu")
        p = predict(m, df_te, use_U); p["variant"] = vn; p["fold"] = held
        all_preds.append(p)
        el = time.time() - t_start
        print(f"[fold {held}] {vn} done  ({el/60:.1f} min elapsed)", flush=True)

P = pd.concat(all_preds)
P = P.merge(mani[["roi_id", "wafer", "z_star"]], on="roi_id")
P.to_csv(ROOT / "results/ablation_student_loss_predictions.csv", index=False)

print("\n================ ABLATION SUMMARY (pooled LOWO×5) ================")
rows = []
for vn in list(VARIANTS) + ["Teacher (privileged)"]:
    g = P[P.variant == vn]
    z = (g.z_star == 2).astype(int).values
    auc = roc_auc_score(z, g.score.values)
    budgets = [budget_full_recall(g[g.fold == w].score.values,
                                  (g[g.fold == w].z_star == 2).values) for w in "ABCDE"]
    rows.append(dict(variant=vn, AUROC=round(auc, 3),
                     budget100=round(100 * np.mean(budgets), 1),
                     per_wafer=" ".join(f"{100*b:.0f}" for b in budgets)))
    print(f"{vn:28s} AUROC={auc:.3f}  budget@100%={100*np.mean(budgets):.1f}%  per-wafer(%): "
          + " ".join(f"{100*b:.0f}" for b in budgets))
pd.DataFrame(rows).to_csv(ROOT / "results/ablation_student_loss_summary.csv", index=False)
print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
print("Done. Per-fold predictions and summary written to results/.")
