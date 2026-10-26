---
type: log
---

# Project Memory Log

## 2026-10-18
- Initialized project OKF memory bundle.
- Documented Kaggle FUSE I/O and DataLoader deadlock resolution in `concepts/kaggle_fuse_dataloader_deadlock.md`.
- Documented StyleGAN2 frequency residual modeling, print-scan domain orthogonality, and calibrated decision fusion in `concepts/stylegan2_frequency_residual_orthogonality.md`.
- Consolidated master progress record, weekly presentation guide, and complete Phase 3/4 blueprints into `team_internal/master_progress_and_future_plan.md`.
- Streamlined repository workspace: moved raw zip archives into `data/` and preserved lean, self-contained standalone execution scripts.
- Added `team_internal/team_debrief_and_walkthrough.md` with first-principles technical walkthroughs and Q&A defense scripts.

## 2026-10-26
- Converted mid-semester evaluation report into an IEEE conference/transactions LaTeX manuscript (`ieee_report/main.tex`, `references.bib`, `figures/`, `ieee_report.zip`).
- Corrected mathematical formulations: patch maximum operator ($\mathbf{z}_{\max}$), LoRA rank-16 projections, Haar HH subband, ArcFace loss, and bona fide Z-score calibration.
- Integrated missing empirical results from `results/`: Phase 1 4-epoch SMDD validation record, Phase 2 8-epoch SMDD convergence record, complete 15-slice FRLL-Morphs benchmark with BPCER@5% and loss, 7-variant architectural ablation, and official ROC curves.
- Formally defined ISO/IEC 30107-3 biometric metrics (APCER, BPCER, EER, BPCER@1%, BPCER@5%, ROC-AUC) and single-image latency profile (17.3 ms on dual T4).
- Overhauled system architecture graphic: migrated from overflowing TikZ to a 300 DPI vector graphic (`figures/system_architecture.png`) featuring 3D isometric convolutional feature maps and clear feature token bars.
- Streamlined author block for B.Tech mid-semester evaluation: consolidated four separate repeating department addresses into a single shared affiliation with Himank Gupta first-listed.
- Converted Table IV and Table V to auto-wrapping full-width environments (`tabularx` with `X` column and `tabular*`) to resolve right-margin overflow.
- Resolved all table margin overflows across Tables I, II, III, IV, and V by wrapping in `\resizebox{\textwidth}{!}{...}`, guaranteeing zero margin clipping.
- Added 3-panel empirical confusion matrix figure (`figures/confusion_matrices.png`, Figure 4) illustrating the 51.0% generative collapse on StyleGAN2 and the dual-stream restoration.
- Re-packaged complete Overleaf-ready project into `ieee_report.zip`.
