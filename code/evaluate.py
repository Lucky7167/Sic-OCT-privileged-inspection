import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from referral import deploy, normalized_budget


def evaluate(predictions, manifest):
    merged = predictions.merge(manifest, on=["roi_id", "wafer", "z_star"], validate="many_to_one")
    rows = []
    for (model, wafer), group in merged.groupby(["model", "wafer"]):
        augmented = group
        operational = group[group.z_star > 0].copy()
        labels = (operational.z_star == 2).astype(int)
        row = {
            "model": model,
            "wafer": wafer,
            "auroc": roc_auc_score(labels, operational.score),
            "auprc": average_precision_score(labels, operational.score),
        }
        if model == "student":
            decision = deploy(operational.score, operational.prediction)
            referred = decision["referred"]
            final = decision["direct_prediction"]
            final[referred] = operational.z_star.to_numpy()[referred]
            row.update({
                "threshold_low": decision["low"],
                "threshold_high": decision["high"],
                "referral_fraction": referred.mean(),
                "subsurface_recall": (final[operational.z_star.to_numpy() == 2] == 2).mean(),
                "budget_candidate_guided_percent": normalized_budget(
                    operational, referred, np.ones(len(operational), dtype=bool)
                ),
            })

            final_augmented = augmented.prediction.to_numpy().copy()
            positions = augmented.index.get_indexer(operational.index)
            final_augmented[positions] = final
            row["macro_f1"] = f1_score(augmented.z_star, final_augmented, average="macro")
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("outputs/lowo/predictions_all_folds.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lowo/metrics_per_wafer.csv"))
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.predictions), pd.read_csv(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
