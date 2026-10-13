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
