# Experimental Results & Quantitative Artifacts

This directory archives the official evaluation metrics, convergence histories, and ROC curves generated during model benchmarking on Kaggle (Dual NVIDIA T4 GPUs).

---

## 📂 Structure

### 1. [`phase1/`](phase1/) -- Single-Stream Spatial Baseline
- **Model:** Frozen DINOv2 ViT-S/14 + Multi-Scale Patch Token Aggregator.
- **Artifacts:**
  - `training_history.csv`: 4-epoch SMDD training convergence.
  - `roc_morph_opencv.png`: Near-perfect detection (0.9993 AUC, 0.94% EER).
  - `roc_morph_facemorpher.png`: Near-perfect detection (0.9958 AUC, 2.45% EER).
  - `roc_morph_webmorph.png`: Strong detection (0.9886 AUC, 4.91% EER).
  - `roc_morph_stylegan.png`: Domain collapse (0.4988 AUC, 51.0% EER).

### 2. [`phase2/`](phase2/) -- Dual-Stream UID-MAD (Spatial + Frequency Residual)
- **Model:** Dual-Stream ViT + 4-Stage Laplacian ConvNet + Multi-Branch ArcFace Heads.
- **Artifacts:**
  - `frll_results.csv`: Complete ISO/IEC 30107-3 evaluation across all 15 splits (5 generators x 3 degradation conditions).
  - `training_history.csv`: 6-epoch in-domain SMDD convergence (99.30% accuracy, 0.01% val EER).
  - `roc_morph_*.png`: Clean condition ROC curves across all 5 generator families.
