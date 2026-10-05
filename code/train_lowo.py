import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import CFG, ROIDataset, SubsurfaceSampler, load_manifest, make_outer_fold, save_fold_metadata, set_seed, validation_rotation
from losses import direct_loss, direct_reg_loss, student_loss, teacher_loss
from models import Student, Teacher
from targets import build_fold_targets
from train_mae import train as train_mae


def _counts(frame):
    values = frame.z_star.value_counts().reindex(range(3), fill_value=0).to_numpy()
    return torch.tensor(values, dtype=torch.float32)


def _move(batch):
    return {k: v.to(CFG.device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _teacher_forward(model, batch):
    return model(batch["feat_local"], batch["feat_ctx"], batch["volume"], batch["metadata"])


def _student_forward(model, batch):
    return model(batch["feat_local"], batch["feat_ctx"], batch["metadata"])


def _fit(model, train_loader, valid_loader, loss_function, class_counts, forward, learning_rate, epochs, checkpoint):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _move(batch); output = forward(model, batch)
            loss = loss_function(output, batch, class_counts)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval(); values = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = _move(batch); output = forward(model, batch)
                values.append(loss_function(output, batch, class_counts).item())
        value = float(np.mean(values))
        if value < best:
            best, best_state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 15:
                break
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    return model


@torch.no_grad()
def _cache_lessons(model, frame, normalization):
    model.eval(); lessons = {}
    loader = DataLoader(ROIDataset(frame, normalization, load_volume=True), batch_size=32)
    for batch in loader:
        moved = _move(batch); output = _teacher_forward(model, moved)
        for i, roi_id in enumerate(batch["roi_id"]):
            lessons[roi_id] = {
                "teacher_feature": output["feature"][i].cpu(),
                "teacher_logits": output["logits"][i].cpu(),
                "teacher_score": output["score"][i].cpu(),
            }
    return lessons


@torch.no_grad()
def _predict(model, frame, normalization, fold, name):
    model.eval(); rows = []
    for batch in DataLoader(ROIDataset(frame, normalization), batch_size=64):
        output = _student_forward(model, _move(batch))
        probabilities = F.softmax(output["logits"], -1).cpu().numpy()
        scores = output["score"].cpu().numpy()
        for i, roi_id in enumerate(batch["roi_id"]):
            rows.append({
                "roi_id": roi_id, "fold": fold, "model": name,
                "score": float(scores[i]), "prediction": int(probabilities[i].argmax()),
                "p_normal": float(probabilities[i, 0]),
                "p_surface": float(probabilities[i, 1]),
                "p_subsurface": float(probabilities[i, 2]),
            })
    return pd.DataFrame(rows)


def run_fold(manifest, test_wafer, validation_wafer, output_root):
    set_seed(CFG.seed + sorted(map(str, manifest.wafer.unique())).index(test_wafer))
    train, valid, test = make_outer_fold(manifest, test_wafer, validation_wafer)
    train, valid, test, normalization = build_fold_targets(train, valid, test)
    fold_dir = output_root / f"fold_{test_wafer}"
    save_fold_metadata(fold_dir / "fold.json", train, valid, test, normalization)
    pd.concat([train.assign(role="train"), valid.assign(role="validation"), test.assign(role="test")]).to_csv(fold_dir / "targets.csv", index=False)
    mae_path = fold_dir / "mae_encoder.pt"
    train_mae(train, mae_path)
    counts = _counts(train)

    teacher = Teacher(mae_path).to(CFG.device)
    teacher = _fit(
        teacher,
        DataLoader(ROIDataset(train, normalization, True), batch_size=32, sampler=SubsurfaceSampler(train)),
        DataLoader(ROIDataset(valid, normalization, True), batch_size=64),
        teacher_loss, counts, _teacher_forward, 1e-4, 150, fold_dir / "teacher.pt",
    )
    lessons = _cache_lessons(teacher, pd.concat([train, valid]), normalization)

    predictions = []
    variants = {
        "direct": direct_loss,
        "direct_reg": direct_reg_loss,
        "student": student_loss,
    }
    for name, loss_function in variants.items():
        use_lessons = lessons if name == "student" else None
        train_ds = ROIDataset(train, normalization, lessons=use_lessons)
        valid_ds = ROIDataset(valid, normalization, lessons=use_lessons)
        model = Student().to(CFG.device)
        model = _fit(
            model,
            DataLoader(train_ds, batch_size=32, sampler=SubsurfaceSampler(train)),
            DataLoader(valid_ds, batch_size=64),
            loss_function, counts, _student_forward, 3e-4, 100, fold_dir / f"{name}.pt",
        )
        predictions.append(_predict(model, test, normalization, test_wafer, name))
    result = pd.concat(predictions, ignore_index=True).merge(
        test[["roi_id", "wafer", "z_star"]], on="roi_id", validate="many_to_one"
    )
    result.to_csv(fold_dir / "predictions.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/lowo"))
    args = parser.parse_args()
    manifest = load_manifest(); wafers = sorted(map(str, manifest.wafer.unique()))
    rotation = validation_rotation(wafers)
    all_predictions = [run_fold(manifest, test, rotation[test], args.output) for test in wafers]
    pd.concat(all_predictions, ignore_index=True).to_csv(args.output / "predictions_all_folds.csv", index=False)
