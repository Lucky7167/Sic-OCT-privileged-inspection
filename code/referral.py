from __future__ import annotations
import numpy as np
import pandas as pd


def multi_otsu_three_regions(scores, bins=256):
    values = np.asarray(scores, float)
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = histogram / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(probability); moment = np.cumsum(probability * centers)
    total = moment[-1]; best, thresholds = -np.inf, None
    for first in range(0, bins - 2):
        for second in range(first + 1, bins - 1):
            weights = (omega[first], omega[second] - omega[first], 1 - omega[second])
            if min(weights) <= 0:
                continue
            means = (
                moment[first] / weights[0],
                (moment[second] - moment[first]) / weights[1],
                (total - moment[second]) / weights[2],
            )
            objective = sum(w * (m - total) ** 2 for w, m in zip(weights, means))
            if objective > best:
                best, thresholds = objective, (centers[first], centers[second])
    if thresholds is None:
        raise ValueError("score distribution cannot be split into three non-empty regions")
    return thresholds


def deploy(scores, microscopy_predictions, thresholds=None):
    scores = np.asarray(scores, float); predictions = np.asarray(microscopy_predictions, int)
    low, high = thresholds or multi_otsu_three_regions(scores)
    referred = (scores >= low) & (scores < high)
    final = predictions.copy(); final[scores >= high] = 2
    return {"low": low, "high": high, "referred": referred, "direct_prediction": final}


def footprint_cost(frame: pd.DataFrame, selected) -> int:
    required = {"x0_aline", "x1_aline", "y0_aline", "y1_aline"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"footprint cost requires {sorted(missing)}")
    rectangles = frame.loc[np.asarray(selected, bool), list(required)]
    cells = set()
    for row in rectangles.itertuples(index=False):
        for x in range(int(row.x0_aline), int(row.x1_aline)):
            cells.update((x, y) for y in range(int(row.y0_aline), int(row.y1_aline)))
    return len(cells)


def normalized_budget(frame, selected, reference):
    denominator = footprint_cost(frame, reference)
    return 100.0 * footprint_cost(frame, selected) / denominator


def oracle_recall_budget_curve(frame, order, positive_label=2):
    labels = frame.z_star.to_numpy(); selected = np.zeros(len(frame), dtype=bool)
    total = (labels == positive_label).sum(); rows = []
    reference = np.ones(len(frame), dtype=bool)
    for index in order:
        selected[index] = True
        recall = ((labels == positive_label) & selected).sum() / max(total, 1)
        rows.append((normalized_budget(frame, selected, reference), recall))
    return pd.DataFrame(rows, columns=["budget_percent", "recall"])
