---
type: concept
title: StyleGAN2 Frequency Residual Modeling and Domain Orthogonality
date: 2026-10-18
---

# StyleGAN2 Frequency Residual Modeling and Domain Orthogonality

## Context
Face morph attack detection (S-MAD) models must generalize out-of-distribution across both landmark-warped faces (OpenCV, FaceMorpher, WebMorph, AMSL) and latent-space generative morphs (StyleGAN2) under physical print-scan degradation (ISO/IEC 30107-3).

## Key Learnings & Empirical Discoveries

1. **Why Single-Stream Spatial ViTs Fail on StyleGAN2:**
   Landmark morphs create localized spatial blending seams, cut-out masks, and edge ghosting captured effectively by DINOv2 patch token variance statistics (>0.99 AUC). Generative latent morphs synthesize photorealistic continuous RGB surfaces with zero seams, causing spatial ViT models to collapse to random chance (0.4988 ROC-AUC, 51.0% EER).

2. **Stream B (High-Pass Frequency Residuals):**
   Fixed Laplacian high-pass filtering strips low-frequency facial identity, isolating high-frequency generative noise:
   - Captures transposed-convolution Nyquist checkerboard peaks and unnatural phase distributions from StyleGAN2 upsampling layers.
   - Surges clean StyleGAN2 detection from 0.4988 to 0.9399 ROC-AUC (14.66% EER).

3. **Print-Scan Orthogonal Complementarity:**
   Physical CMYK printing and optical flatbed scanning attenuate spatial edge seams (lowering spatial ViT AUC on StyleGAN2 to 0.3155 and AMSL to 0.3743). Under identical print-scan degradation, the frequency residual stream surges to **>0.997 ROC-AUC across all 5 generator families**.

4. **Inversion of Early Metric Concatenation vs Late Calibrated Decision Fusion:**
   Concatenating spatial and frequency embeddings into an early linear projection before ArcFace (`fused_head`) overfits synthetic training data, inverting on out-of-distribution real cameras (AUC drops to 0.03 - 0.39).
   Standardizing margin predictions against bona fide baselines via Z-score calibrated decision fusion:
   $$Z_{\text{calibrated}} = 0.5 \cdot \left(\frac{S_{\text{spatial}} - \mu_{\text{bf}, s}}{\sigma_{\text{bf}, s}}\right) + 0.5 \cdot \left(\frac{S_{\text{freq}} - \mu_{\text{bf}, f}}{\sigma_{\text{bf}, f}}\right)$$
   eliminates PRNU sensor noise false alarms, delivering $\ge 0.9809$ ROC-AUC across all 15 operational slices (mean 0.992).

5. **Trigonometric Margin Stability:**
   ArcFace margin application in FP16 mixed precision must use angle addition identities ($\cos(\theta + m) = \cos\theta \cos m - \sin\theta \sin m$) with Taylor fallback for $\theta + m > \pi$ to prevent NaN gradient overflows.
