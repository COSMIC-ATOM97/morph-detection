# Phase 1 Evaluation Report: Single-Stream DINOv2 Spatial Baseline

**Date:** October 11, 2026  
**Model:** Stream A Baseline (DINOv2 ViT-S/14 with Multi-Scale Patch Token Aggregator + ArcFace Head)  
**Training Dataset:** SMDD (`SMDD_release_train`, 32,000 synthetic face samples)  
**Evaluation Dataset:** FRLL-Morphs (7,274 samples across 5 generator families under Clean, JPEG q=30, and Simulated Print-Scan)  

---

## 1. Executive Summary

Phase 1 established the single-stream spatial foundation baseline for the unified morph attack detection architecture. The model was trained exclusively on the synthetic SMDD dataset and tested strictly out-of-distribution (cross-dataset, cross-generator, and cross-degradation) on FRLL-Morphs without any fine-tuning on the target domain.

### Key Empirical Findings
1. **Landmark-Based Morphs Solved in Spatial Domain:**
   - On digital landmark-warped morphs, DINOv2 dense patch statistics achieved near-perfect discrimination:
     - **OpenCV:** 0.94% EER, **0.9993 ROC-AUC**
     - **FaceMorpher:** 2.45% EER, **0.9958 ROC-AUC**
     - **WebMorph:** 4.91% EER, **0.9886 ROC-AUC**
     - **AMSL:** 13.20% EER, **0.9461 ROC-AUC**
2. **Extreme Robustness to JPEG Compression:**
   - Under heavy JPEG compression ($q=30$), detection performance remained virtually identical (OpenCV AUC dropped by only $0.0001$, from $0.9993$ to $0.9992$). Anti-shortcut augmentations during training prevented compression artifacts from misleading the classifier.
3. **Catastrophic Domain Collapse on StyleGAN2 Morphs:**
   - On StyleGAN2 morphs, detection collapsed to random chance:
     - **StyleGAN2 Clean:** 51.00% EER, **0.4988 ROC-AUC**
     - **StyleGAN2 JPEG:** 49.98% EER, **0.4973 ROC-AUC**
     - **StyleGAN2 Print-Scan:** 45.77% EER, **0.5780 ROC-AUC**

---

## 2. In-Domain Training Convergence (SMDD)

Training executed across 4 warm-up epochs with a frozen DINOv2 ViT-S/14 backbone and trainable multi-scale projection + ArcFace head ($s=32.0, m=0.25$):

| Epoch | Train Loss | Train Acc | Val Loss | Val EER | Val ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.5422 | 92.06% | 0.0072 | 0.0013 (0.13%) | 0.99998 |
| 2 | 0.1544 | 95.54% | 0.0036 | 0.0006 (0.06%) | 0.99999 |
| 3 | 0.1226 | 96.17% | 0.0061 | 0.0004 (0.04%) | 0.99999 |
| 4 | 0.1223 | 96.15% | 0.0071 | **0.0001 (0.01%)** | **0.999999** |

Best model weights saved to `best_model.pt` (91.9 MB).

---

## 3. Cross-Dataset Evaluation on FRLL-Morphs (ISO/IEC 30107-3)

Evaluated across all 5 morph generator algorithms and 3 degradation condition slices (15 evaluation passes):

| Morph Generator | Condition | EER | BPCER @ APCER=1% | BPCER @ APCER=5% | ROC-AUC | Val Loss |
|---|---|---|---|---|---|---|
| **morph_opencv** | Clean | **0.0094** | 0.0000 | 0.0000 | **0.9993** | 5.4053 |
| **morph_opencv** | JPEG (q=30) | **0.0094** | 0.0098 | 0.0000 | **0.9992** | 5.3539 |
| **morph_opencv** | Print-Scan | **0.0916** | 0.2990 | 0.1569 | **0.9786** | 2.7859 |
| **morph_facemorpher** | Clean | **0.0245** | 0.1912 | 0.0000 | **0.9958** | 5.9695 |
| **morph_facemorpher** | JPEG (q=30) | **0.0249** | 0.1618 | 0.0000 | **0.9956** | 5.9273 |
| **morph_facemorpher** | Print-Scan | **0.0585** | 0.2647 | 0.0735 | **0.9859** | 2.6364 |
| **morph_webmorph** | Clean | **0.0491** | 0.2745 | 0.0490 | **0.9886** | 10.2887 |
| **morph_webmorph** | JPEG (q=30) | **0.0585** | 0.3039 | 0.0735 | **0.9875** | 10.2548 |
| **morph_webmorph** | Print-Scan | **0.2539** | 0.7892 | 0.5833 | **0.8320** | 8.7468 |
| **morph_amsl** | Clean | **0.1320** | 0.7598 | 0.3382 | **0.9461** | 12.8552 |
| **morph_amsl** | JPEG (q=30) | **0.1367** | 0.7647 | 0.3235 | **0.9434** | 12.8603 |
| **morph_amsl** | Print-Scan | **0.3978** | 0.9216 | 0.8039 | **0.6491** | 12.3142 |
| **morph_stylegan** | Clean | **0.5100** | 0.9951 | 0.9902 | **0.4988** | 12.9096 |
| **morph_stylegan** | JPEG (q=30) | **0.4998** | 0.9951 | 0.9853 | **0.4973** | 12.9027 |
| **morph_stylegan** | Print-Scan | **0.4577** | 0.9608 | 0.8824 | **0.5780** | 12.1117 |

---

## 4. Deep-Dive Diagnosis: Why StyleGAN2 Collapsed (The Scientific Rationale for Stream B)

The failure on StyleGAN2 (AUC = 0.4988) is not a tuning issue; it is a fundamental limitation of spatial-domain patch analysis when faced with latent-space generative morphing.

### 4.1 Digital Landmark Morphing vs. Generative Latent Morphing
- **Landmark-Based Morphs (OpenCV, FaceMorpher, WebMorph, AMSL):**
  - Two input faces are triangulated using 68 Delaunay landmark meshes.
  - Affine warping maps the donor geometry to the midpoint, followed by pixel-level alpha blending:
    $$I_{\text{morph}} = \alpha \cdot \text{Warp}(I_A) + (1 - \alpha) \cdot \text{Warp}(I_B)$$
  - **Artifacts:** Ghosting along the iris/pupil borders, double edges in high-frequency regions (eyebrows, eyelashes, nostrils), and alpha boundary blending seams where the cropped facial mask blends into the background.
  - **Why Stream A Succeeds:** DINOv2 self-supervised patch tokens are trained on dense spatial correspondence. Aggregating patch statistics (`patch_std`, `patch_max`) directly flags these local boundary variance spikes.

- **Generative Morphs (StyleGAN2):**
  - Real input images $I_A$ and $I_B$ are projected into the extended latent space $W^+$ of StyleGAN2 via GAN inversion:
    $$w_A = \text{Encoder}(I_A), \quad w_B = \text{Encoder}(I_B)$$
  - Morphing is performed by linear interpolation of latent vectors:
    $$w_{\text{morph}} = \alpha w_A + (1 - \alpha) w_B$$
  - The morphed face is synthesized from scratch by the StyleGAN2 generator network:
    $$I_{\text{morph}} = G(w_{\text{morph}})$$
  - **Artifacts in Spatial Domain:** There are **zero pixel-level alpha-blending seams**, zero ghosting contours, and no landmark boundary cuts. The output image is photorealistically smooth and coherent in RGB space.
  - **Why Stream A Fails:** In RGB spatial patch space, a StyleGAN2 morphed face looks indistinguishable from a bona fide face to a model looking for blending seams.

### 4.2 The Frequency Domain Footprint of StyleGAN2
While StyleGAN2 leaves no seams in the spatial domain, it leaves prominent fingerprints in the **frequency domain**:
1. **Upconvolution Spectral Peaks:** The progressive upsampling layers in StyleGAN2 introduce periodic grid artifacts (Nyquist checkerboard patterns). When analyzed via 2D Discrete Cosine Transform (DCT) or FFT power spectrum, GAN-generated faces show pronounced high-frequency spikes absent in camera optics.
2. **Phase Anomalies:** Natural optical sensors (CMOS/CCD) produce camera sensor pattern noise (PRNU). Synthetic generator upsampling lacks natural sensor noise and instead produces synthetic phase coherence across frequency bands.
3. **High-Pass Residual Visibility:** Subtracting low-frequency facial structure ($R = I - \text{GaussianFilter}(I)$) strips semantic facial identity and isolates pure generator noise.

---

## 5. Architectural Mandate for Phase 2: Dual-Stream UID-MAD

Phase 2 will implement **Stream B (Frequency Residual Stream)** to fuse with Stream A:

```
                      INPUT IMAGE (224x224 RGB)
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
    STREAM A (SPATIAL)                  STREAM B (FREQUENCY RESIDUAL)
    DINOv2 ViT-S/14                     High-Pass Residual: R = I - Blur(I)
    Dense Patch Aggregator              2D Discrete Cosine Transform (DCT)
    (CLS, Mean, Std, Max -> 256-d)      Lightweight 3-Layer ConvNet (128-d)
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                        FUSION HEAD
                  Concat(256 + 128) -> 256-d
                               ▼
                        ARCFACE HEAD
```

### Specific Objectives for Phase 2:
1. Lift StyleGAN2 AUC from **0.4988 (random)** to **>0.85**.
2. Preserve >0.99 AUC on landmark morphs (OpenCV, FaceMorpher).
3. Improve print-scan resilience on WebMorph and AMSL.
4. Optimize DataLoader pipeline to achieve 1.5–2.5 it/s training speed.
