# data/ — manifest, features, and prediction exports

Everything needed to reproduce the figures and the Table 1 ablation without
touching the raw imaging data.

## Files

| File | Description |
|---|---|
| `manifest.csv` | One row per ROI (986 total: wafer A–E). Columns: `roi_id, wafer, split, x, y, w, h, a, z_star, q_star, d, v, o, c, U_star, …`. Splits are by wafer (never by ROI) to prevent group leakage. |
| `patches/{roi_id}_local.npy`, `patches/{roi_id}_ctx.npy` | DINOv2 [CLS] features (384-d) of the local and context microscopy crops; extracted once, offline. |
| `feats_micro.npy` | Dict `{roi_id: 768-d vector}` = concat of local and context [CLS] features. |
| `feats_oct.npy` | Dict `{roi_id: 512-d vector}` = pre-extracted OCT volume features (privileged modality). |
| `norm_constants.json` | Flat `{key_mean/key_std/key_min/key_max}` normalization constants, computed on the A/B training split and frozen. Read by `code/common.py`. |
| `norm.json` | Nested `{key: [mean, std]}` variant of the same constants. Read by `analysis/`. |

## Not included (to be added in a later release)

- `volumes/{roi_id}.npy` — raw OCT volumes, `(3, 64, 64, 64)` float32. Required
  only by the volume-based pipeline in `code/` (MAE pretraining and teacher
  training from raw volumes). Every figure/analysis script uses the exported
  features above instead.
- Raw microscopy images.

## Cohorts

- **Augmented set (986):** 250 normal + 431 surface + 305 subsurface ROIs.
- **Operational set (736):** the 431 + 305 defect ROIs (`z_star > 0`); the
  referral strategy and all budget numbers are evaluated on this cohort.
