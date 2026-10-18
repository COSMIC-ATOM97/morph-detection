# Experimental Results & Quantitative Artifacts

This directory archives the official evaluation metrics, convergence histories, and ROC curves generated during model benchmarking on Kaggle (Dual NVIDIA T4 GPUs).

---

## Directory Overview

### [`phase1/`](phase1/) -- Single-Stream Spatial Baseline
- **Model:** Frozen and fine-tuned DINOv2 ViT-S/14 with Multi-Scale Patch Token Aggregator.
- **Evaluation Standards:** ISO/IEC 30107-3 compliant zero-shot out-of-distribution evaluation on FRLL-Morphs across 15 splits (5 generator algorithms x 3 environmental conditions).
- **Artifacts:**
  - `training_history.csv`: 4-epoch SMDD training convergence log (0.01% validation EER, 0.999999 validation ROC-AUC).
  - `roc_morph_opencv.png`: Near-perfect landmark detection (0.9993 AUC, 0.94% EER).
  - `roc_morph_facemorpher.png`: Near-perfect landmark detection (0.9958 AUC, 2.45% EER).
  - `roc_morph_webmorph.png`: Strong landmark detection (0.9886 AUC, 4.91% EER).
  - `roc_morph_stylegan.png`: Generative latent-space collapse (0.4988 AUC, 51.0% EER).

### [`phase2/`](phase2/) -- Dual-Stream UID-MAD (Spatial + Frequency Residual)
- **Model:** Dual-Stream architecture combining DINOv2 ViT-S/14 (Stream A) with LoRA rank-16 adapters and a 4-stage Laplacian/Wavelet ConvNet (Stream B, 128-d), coupled with Decoupled Multi-Task ArcFace heads and Z-score Calibrated Decision Fusion.
- **Evaluation Standards:** ISO/IEC 30107-3 compliant zero-shot out-of-distribution evaluation across all 15 FRLL-Morphs splits (5 generator algorithms x 3 environmental conditions).
- **Artifacts:**
  - `training_history.csv`: 8-epoch two-stage SMDD training log (99.12% accuracy, 0.0000 validation EER).
  - `frll_results.csv`: Per-slice performance metrics across all 15 evaluation slices.
  - `frll_results_comparison.csv`: Comparative breakdown across all 5 inference pathways (`fused_head`, `calibrated`, `spatial`, `frequency`, `ens_margin`).
  - `roc_morph_opencv.png`: Clean 0.9997 AUC, JPEG 0.9996 AUC, Print-Scan 0.9973 AUC.
  - `roc_morph_facemorpher.png`: Clean 0.9999 AUC, JPEG 0.9999 AUC, Print-Scan 0.9994 AUC.
  - `roc_morph_webmorph.png`: Clean 0.9968 AUC, JPEG 0.9967 AUC, Print-Scan 0.9893 AUC.
  - `roc_morph_stylegan.png`: Clean 0.9399 AUC, JPEG 0.9384 AUC, Print-Scan 0.9854 AUC (StyleGAN2 breakthrough).
  - `roc_morph_amsl.png`: Clean 0.9938 AUC, JPEG 0.9935 AUC, Print-Scan 0.9809 AUC.

