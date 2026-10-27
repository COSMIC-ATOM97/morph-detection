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
- Enforced authentic Seaborn/Kaggle notebook styling on confusion matrices (`figures/confusion_matrices.png`, Figure 4) with standard `Blues` gradient, integer counts, and clean percentage labels.
- Added qualitative forensic analysis figure (`figures/qualitative_samples.png`, Figure 5) displaying genuine, landmark morph, generative latent morph, and print-scan portraits alongside high-pass Laplacian frequency residuals $R = |I * K|$.
- Added Table VI providing exemplar quantitative prediction breakdowns ($S_s, S_f, Z_s, Z_f, Z_{\text{final}}$) for validation test portraits.
- Resolved Overleaf free-tier compile timeout ("Your project needs more compile time") by replacing runtime TikZ/PGFPlots bar rendering with a pre-rendered 300 DPI graphic (`figures/quantitative_eval_charts.png`), removing `pgfplots`/`tikz` packages, and optimizing image memory footprint down to 2.47 MB.
- Balanced bibliography columns with `\IEEEtriggeratref{6}` and float spacing to pull citation [11] onto Page 11, eliminating the blank 12th page.
- Repackaged final Overleaf-ready draft into `ieee_report.zip`.
## 2026-10-27
- Converted manuscript into a clean single-column B.Tech Major Project Progress Report (CSPC401, NIT Kurukshetra) with 1-inch margins.
- Redesigned architecture diagram with bold, readable typography (348 KB PNG) and generated an editable Draw.io XML source (`figures/uid_mad_architecture.drawio`).
- Streamlined contents to remove research-paper filler while preserving all empirical findings, 6 full-width tables, Kaggle Seaborn confusion matrices, and qualitative forensic visualizations.
- Repackaged into `ieee_report.zip` (2.26 MB) for instant one-click Overleaf compilation.
