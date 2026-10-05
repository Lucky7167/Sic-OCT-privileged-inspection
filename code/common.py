
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class Config:
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")
    dino_dim: int = 384
    metadata_dim: int = 5
    feature_dim: int = 256
    batch_size: int = 32
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
META_COLUMNS = ("x", "y", "w", "h", "a")
CLASS_NAMES = ("normal", "surface", "subsurface")


def set_seed(seed: int = CFG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or CFG.data_root / "manifest.csv")
    required = {"roi_id", "wafer", "z_star", *META_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if frame.roi_id.duplicated().any():
        raise ValueError("roi_id must be unique")
    return frame


def make_outer_fold(frame: pd.DataFrame, test_wafer: str, validation_wafer: str):
    wafers = set(frame.wafer.astype(str).unique())
    if test_wafer == validation_wafer:
        raise ValueError("test and validation wafers must differ")
    if {test_wafer, validation_wafer} - wafers:
        raise ValueError("unknown test or validation wafer")
    train_wafers = wafers - {test_wafer, validation_wafer}
    train = frame[frame.wafer.astype(str).isin(train_wafers)].copy()
    valid = frame[frame.wafer.astype(str) == validation_wafer].copy()
    test = frame[frame.wafer.astype(str) == test_wafer].copy()
    assert test_wafer not in set(train.wafer.astype(str))
    assert test_wafer not in set(valid.wafer.astype(str))
    return tuple(x.reset_index(drop=True) for x in (train, valid, test))


def validation_rotation(wafers: Iterable[str]) -> dict[str, str]:
    ordered = sorted(map(str, wafers))
    if len(ordered) < 3:
        raise ValueError("at least three wafers are required")
    return {w: ordered[(i + 1) % len(ordered)] for i, w in enumerate(ordered)}


def fit_normalization(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for column in META_COLUMNS:
        std = float(train[column].std(ddof=1))
        result[column] = {
            "mean": float(train[column].mean()),
            "std": std if std > 0 else 1.0,
        }
    return result


def save_fold_metadata(path: Path, train, valid, test, normalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_wafers": sorted(map(str, train.wafer.unique())),
        "validation_wafers": sorted(map(str, valid.wafer.unique())),
        "test_wafers": sorted(map(str, test.wafer.unique())),
        "normalization": normalization,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_q_star(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "deff", "dmax", "snr", "snr_ref", "rho_shadow", "rho_sat",
        "c_cont", "has_depth_structure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cannot compute q_star; missing {sorted(missing)}")
    present = frame.has_depth_structure.astype(bool).to_numpy()
    depth = np.divide(
        frame.deff, frame.dmax, out=np.zeros(len(frame)), where=frame.dmax != 0
    )
    snr = np.divide(
        frame.snr, frame.snr_ref, out=np.zeros(len(frame)), where=frame.snr_ref != 0
    )
    depth = np.where(present, depth, 0.0)
    snr = np.where(present, np.clip(snr, 0.0, 1.0), 0.0)
    continuity = np.where(present, frame.c_cont.to_numpy(float), 0.0)
    q = (
        depth + snr + (1.0 - frame.rho_shadow.to_numpy(float))
        + (1.0 - frame.rho_sat.to_numpy(float)) + continuity
    ) / 5.0
    return np.clip(q, 0.0, 1.0)


def compute_u_star(q_star, risk, alpha: float = 0.7) -> np.ndarray:
    q_star, risk = np.asarray(q_star, float), np.asarray(risk, float)
    if q_star.shape != risk.shape:
        raise ValueError("q_star and risk must have the same shape")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return q_star * (alpha + (1.0 - alpha) * risk)


class ROIDataset(Dataset):
    def __init__(self, frame, normalization, load_volume=False, lessons=None):
        self.frame = frame.reset_index(drop=True)
        self.normalization = normalization
        self.load_volume = load_volume
        self.lessons = lessons

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        root = CFG.data_root
        metadata = [
            (float(row[c]) - self.normalization[c]["mean"])
            / self.normalization[c]["std"] for c in META_COLUMNS
        ]
        item = {
            "roi_id": str(row.roi_id),
            "feat_local": torch.as_tensor(
                np.load(root / "patches" / f"{row.roi_id}_local.npy"), dtype=torch.float32
            ),
            "feat_ctx": torch.as_tensor(
                np.load(root / "patches" / f"{row.roi_id}_ctx.npy"), dtype=torch.float32
            ),
            "metadata": torch.tensor(metadata, dtype=torch.float32),
            "z_star": torch.tensor(int(row.z_star), dtype=torch.long),
            "q_star": torch.tensor(float(row.get("q_star", np.nan)), dtype=torch.float32),
            "U_star": torch.tensor(float(row.get("U_star", np.nan)), dtype=torch.float32),
        }
        if self.load_volume:
            item["volume"] = torch.as_tensor(
                np.load(root / "volumes" / f"{row.roi_id}.npy"), dtype=torch.float32
            )
        if self.lessons is not None:
            item.update(self.lessons[str(row.roi_id)])
        return item


class SubsurfaceSampler(Sampler[int]):
    def __init__(self, frame: pd.DataFrame, fraction: float = 0.30):
        self.positive = np.flatnonzero(frame.z_star.to_numpy() == 2)
        self.other = np.flatnonzero(frame.z_star.to_numpy() != 2)
        if len(self.positive) == 0:
            raise ValueError("training fold has no subsurface examples")
        self.fraction = fraction
        self.length = max(len(frame), int(np.ceil(len(self.other) / (1 - fraction))))

    def __iter__(self):
        count = max(int(np.ceil(self.fraction * self.length)), self.length - len(self.other))
        rng = np.random.default_rng(torch.initial_seed() % (2**32))
        indices = np.concatenate([self.other, rng.choice(self.positive, count, replace=True)])
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.length


def weighted_cross_entropy(logits, target, counts):
    weights = 1.0 / counts.float().clamp_min(1.0)
    weights = weights / weights.sum() * len(counts)
    return F.cross_entropy(logits, target, weight=weights.to(logits.device))


def pairwise_rank_loss(scores, targets):
    predicted = scores[:, None] - scores[None, :]
    expected = targets[:, None] - targets[None, :]
    mask = expected > 0
    return -F.logsigmoid(predicted[mask]).mean() if mask.any() else scores.sum() * 0.0


def feature_alignment_loss(projected_student, teacher_feature):
    return 1.0 - F.cosine_similarity(projected_student, teacher_feature, dim=-1).mean()


def logit_distillation_loss(student, teacher, temperature=3.0):
    log_p = F.log_softmax(student / temperature, dim=-1)
    p = F.softmax(teacher / temperature, dim=-1)
    return temperature**2 * F.kl_div(log_p, p, reduction="batchmean")
