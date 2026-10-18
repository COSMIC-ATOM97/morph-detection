---
type: memory
---

# Morph Attack Detection (UID-MAD) Active Architecture & Invariants

- **Active Model:** Dual-Stream UID-MAD (`vit_small_patch14_dinov2.lvd142m` with LoRA rank 16 on blocks 6-11 + 4-stage Laplacian/Wavelet ConvNet 128-d).
- **Classification:** Decoupled multi-task ArcFace heads (`s=32.0, m=0.25`) with Z-score calibrated decision fusion.
- **Hardware/Kaggle Rule:** Extract archives to `/tmp/`, run with `NUM_WORKERS=0` and `PIN_MEMORY=False`.
- **Empirical Benchmarks:** StyleGAN2 clean 0.9399 AUC (14.66% EER), printscan 0.9854 AUC (3.92% EER); printscan frequency stream >0.997 AUC across all generators; calibrated fusion >= 0.9809 AUC across all 15 FRLL splits (mean 0.992).
