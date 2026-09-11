# Learning when to scan: OCT-privileged selective inspection for efficient defect localization in silicon carbide wafers

Code, data exports, trained checkpoints, and figure-reproduction scripts for the
manuscript submitted to *Light: Advanced Manufacturing*.

Wide-field bright-field microscopy localizes candidate defects on a 6-inch SiC
wafer, but cannot resolve whether a defect lies on the surface or buried below
it; spectral-domain optical coherence tomography (OCT) can, yet is too slow to
scan every candidate. This project trains a microscopy-only **student** model
with **learning using privileged information (LUPI)** — an OCT-informed
**teacher** distills subsurface knowledge into the student — so that only the
candidates the student cannot confidently resolve are referred to OCT. On five
6-inch wafers (986 annotated ROIs; 736 operational surface/subsurface defects),
the selective strategy preserves all 305 subsurface defects while spending only
8.2–11.2 % of the exhaustive OCT scan time (9.8 % pooled), a 67 % reduction in
per-wafer inspection time.

## Repository layout

```
sic-oct-privileged-inspection/
├── code/                 # Full LUPI training pipeline (volume-based, Methods-faithful)
│   ├── common.py         #   config / datasets / losses / metrics
│   ├── train_mae.py      #   stage A: masked-autoencoder pretraining on OCT volumes
│   ├── train_aux_classifier.py
│   ├── train_teacher.py  #   stage B: four-modality privileged teacher
│   ├── save_lessons.py   #   stage C: freeze teacher, export lessons
│   ├── train_direct.py   #   direct baseline (no privileged information)
│   ├── train_student.py  #   stage D: microscopy-only deployment student
│   └── evaluate.py
├── analysis/             # Feature-based training components + Table 1 ablation
│   ├── lupi_components.py        # models / dataset / losses / training loops
│   └── run_ablation_student_loss.py   # LOWO×5 student-loss ablation
├── figures/              # One folder per figure; see figures/README.md
├── data/                 # Manifest, features, prediction CSVs; see data/README.md
├── checkpoints/          # Trained weights: teacher / student / direct / lessons
├── results/              # Ablation and strategy-matrix result tables (CSV)
└── outputs/              # All scripts write their figures here
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

All figure and analysis scripts are run **from the repository root** and read
from `data/`, writing to `outputs/`:

```bash
python figures/fig3/draw_fig3d_736.py     # Fig. 3d error-capture curve
python figures/fig4/draw_fig4_v2.py       # Fig. 4
python figures/fig5/draw_fig5a_v2.py      # Fig. 5 panels a–f (one script each)
python analysis/run_ablation_student_loss.py   # Table 1 ablation (LOWO×5, CPU)
```

The volume-based training pipeline in `code/` expects raw OCT volumes under
`data/volumes/{roi_id}.npy` (`(3, 64, 64, 64)` float32), which are not part of
this release (see `data/README.md`); the DINOv2 patch features, OCT features,
manifest, normalization constants, and trained checkpoints needed by every
other script are included.

## Cohorts and headline numbers

| Cohort | Composition | Use |
|---|---|---|
| Augmented (986 ROIs) | 250 normal + 431 surface + 305 subsurface | confidence analysis (Fig. 3c) |
| Operational (736 ROIs) | 431 surface + 305 subsurface | referral strategy (Figs. 3d, 4, 5) |

Selective inspection: 4.1–5.1 min/wafer vs. 11.7–15.5 min for candidate-guided
OCT and 35.4 min for exhaustive OCT (Table 2).

## Notes

- Prediction CSVs and checkpoints are the current five-fold
  leave-one-wafer-out pipeline export. After any retraining, regenerate them
  with `code/` + `analysis/` and replace the files in place (same names).
- Raw microscopy images and OCT volumes will be added in a later release.

## License

Code is released under the MIT License (see `LICENSE`). Data exports are
provided for reproducibility of the manuscript; please cite the paper if you
use them.
