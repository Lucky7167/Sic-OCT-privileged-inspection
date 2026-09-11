# figures/ — reproduction scripts

Run every script **from the repository root**. Inputs come from `data/`,
outputs are written to `outputs/`.

| Figure | Script | Inputs | Output |
|---|---|---|---|
| Fig. 1 (wafer maps) | `fig1/draw_fig1_full.py` | `data/fig1_wafer_photo.png`, `data/adjusted_predictions_test_DE.csv` | `outputs/Fig1_full.png` |
| Fig. 3 (composite) | `fig3/draw_fig3_5fold_v2.py` | `data/adjusted_predictions_5fold.csv`, `data/fig2_direct_probs_5fold.csv` | `outputs/Fig3_5fold_v2.png` |
| Fig. 3d (error capture, operational cohort) | `fig3/draw_fig3d_736.py` | `data/fig2_direct_probs_5fold.csv` | `outputs/Fig3d_736.png` |
| Fig. 4 | `fig4/draw_fig4_v2.py` | prediction CSVs + `data/heuristic_baseline_scores_5fold.csv` + `data/feats_micro.npy` | `outputs/Fig4_updated.png` |
| Fig. 5a | `fig5/draw_fig5a_v2.py` | prediction CSVs + `data/feats_micro.npy` | `outputs/Fig5a_*.png` |
| Fig. 5b–f | `fig5/draw_fig5b.py` … `draw_fig5f.py` | same | `outputs/Fig5{b..f}_*.png` |
| Architecture schematic | `schematic/draw_architecture.py` | — (pure drawing) | `outputs/Fig3_architecture.png` |
| SI Figs. S1–S5 + SI tables | `si/draw_supplementary_current.py` | prediction CSVs + `data/feats_micro.npy` | `outputs/supplementary/` |
| SI Fig. S7 (Arial redraw) | `si/draw_figS7_arial.py` | `data/figS7_curves_source.pdf` (vector curve recovery) | `outputs/figS7_arial.png/.pdf` |

Notes

- `fig3/draw_fig3d_736.py` documents the exact Fig. 3d definition: rank the 736
  operational ROIs by ascending max-softmax confidence, refer the least
  confident first, and plot cumulative error capture against referral fraction;
  the tau = 0.8, 95 %, and 100 % operating points are computed from the data.
- The SI script writes its intermediate plot data
  (`补充图S1_绘图数据.csv`) into `outputs/supplementary/` as well.
