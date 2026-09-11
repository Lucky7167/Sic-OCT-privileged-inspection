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
| `fig2_direct_probs_5fold.csv` | Per-ROI direct-model outputs pooled over the five LOWO folds: `z_true, direct_label, p_normal, p_surface, p_subsurface`. Basis of Figs. 2–5. |
| `adjusted_predictions_5fold.csv` | Per-ROI direct/student/teacher predictions and utility scores for the five folds (`Direct_*/Student_*/Teacher_*` columns). Basis of Figs. 4, 5 and SI Figs. S1–S5. |
| `adjusted_predictions_test_DE.csv` | Same schema, test wafers D/E only (Fig. 1 wafer maps). |
| `heuristic_baseline_scores_5fold.csv` | Heuristic (non-learned) baseline ranking scores used for comparison in Fig. 4. |
| `fig1_wafer_photo.png` | Wafer photo underlying the Fig. 1 map. |
| `figS7_curves_source.pdf` | Vector source from which `figures/si/draw_figS7_arial.py` recovers the SI Fig. S7 curves. |

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
