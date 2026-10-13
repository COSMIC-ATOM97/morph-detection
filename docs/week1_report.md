# Week 1 Research Report: Single-Stream Spatial DINOv2 Morph Attack Detection Pipeline

**Project:** Unified Interpretable Face Morph Attack Detection (UID-MAD)  
**Reporting Period:** Week 1  
**Architecture:** Single-Stream Spatial Baseline (DINOv2 ViT-S/14 + Multi-Scale Patch Aggregator + ArcFace Head)  
**Standard:** ISO/IEC 30107-3 (Biometric Presentation Attack Detection)  
**Primary Artifacts:** [`scripts/phase1_dino_spatial.py`](file:///home/crimsond/project/scripts/phase1_dino_spatial.py), [`scripts/phase1_dino_spatial.ipynb`](file:///home/crimsond/project/scripts/phase1_dino_spatial.ipynb), [`results/phase1/`](file:///home/crimsond/project/results/phase1)

---

## 1. Executive Summary & Week 1 Objectives

During Week 1, the research team established the foundational single-stream spatial baseline for Single-Image Morphing Attack Detection (S-MAD). Morph attacks pose a major security vulnerability to automated border control (ABC) e-Gates and electronic passport issuance because a single morphed image can verify against two distinct human identities.

```
       Subject A (Accomplice) ──┐
                                ├─► [Morph Generation] ─► Matches Both Passports
       Subject B (Criminal)   ──┘
```

### Primary Objectives of Week 1
1. **Foundation Model Feasibility:** Evaluate whether self-supervised Vision Transformers (DINOv2 ViT-S/14) learn spatial representations capable of detecting subtle facial morphing artifacts without requiring dataset-specific landmark engineering.
2. **Angular Margin Metric Learning:** Implement Additive Angular Margin Loss (ArcFace) to enforce hyperspherical metric clustering and reduce intra-class feature dispersion.
3. **Rigorous Out-of-Distribution Protocol:** Train exclusively on synthetic data (SMDD) and evaluate strictly zero-shot out-of-distribution across 5 morphing algorithms and 3 degradation conditions on FRLL-Morphs under ISO/IEC 30107-3 standards.

### Key Empirical Findings
* **Landmark Morphs Solved in Spatial Domain:** Achieved **0.9993 ROC-AUC (0.94% EER)** on OpenCV and **0.9958 ROC-AUC (2.45% EER)** on FaceMorpher.
* **JPEG Compression Robustness:** Maintained **0.9992 ROC-AUC** under heavy JPEG compression ($q=30$) via anti-shortcut data augmentations.
* **The Catastrophic StyleGAN2 Blind Spot:** Collapsed to random guessing on generative morphs (**0.4988 ROC-AUC, 51.00% EER** on StyleGAN2), proving that spatial RGB patch features alone cannot discriminate latent-space generative morphs.

---

## 2. Week 1 Pipeline Architecture

The Week 1 pipeline processes single RGB facial images through a spatial Vision Transformer and projects dense patch statistics into an angular margin metric space.

```
                      INPUT IMAGE: I ∈ R^(224×224×3)
                                    │
                                    ▼
                     DINOv2 ViT-S/14 BACKBONE (14×14 Patches)
             Outputs: CLS Token z_cls ∈ R^384, Patch Tokens z_1...z_256 ∈ R^384
                                    │
                                    ▼
                     MULTI-SCALE PATCH TOKEN AGGREGATOR
         f = [ z_cls || Mean(z_patches) || Std(z_patches) || Max(z_patches) ]
                              f ∈ R^1536
                                    │
                                    ▼
                       PROJECTION BOTTLENECK (MLP)
                     Linear(1536 -> 512) -> BatchNorm -> GELU
                     Linear(512 -> 256)  -> BatchNorm
                              e ∈ R^256
                                    │
                                    ▼
                         ARCFACE HEAD (s=30.0, m=0.35)
                L2 Normalization of Feature e and Weight Matrix W
                   Angular Margin Penalty on Ground-Truth Class
                                    │
                                    ▼
                       CROSS-ENTROPY LOSS / INFERENCE
```

### 2.1 Spatial Feature Extraction (DINOv2 ViT-S/14)
* **Backbone:** `vit_small_patch14_dinov2.lvd142m` (21.6 million parameters, 384 embedding dimension).
* **Self-Supervised Representation:** Unlike ImageNet-supervised models that optimize for high-level object labels (discarding fine-grained pixel details), DINOv2 is pretrained via joint self-distillation (DINO) and patch-level masked image modeling (iBOT). This forces patch tokens to preserve dense geometric and texture correspondence.
* **Multi-Scale Patch Token Pooling:** Facial morph blending seams are spatially localized. Rather than utilizing only the classification token $\mathbf{z}_{\text{cls}}$, we pool across all 256 patch tokens:
  $$\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}$$
  The patch standard deviation ($\boldsymbol{\sigma}_{\text{patch}}$) explicitly captures the variance across spatial patches, directly registering localized blending discontinuities.

### 2.2 Additive Angular Margin Head (ArcFace)
To enforce compact intra-class clustering, features $\mathbf{e}$ and class weights $\mathbf{w}_j$ are $L_2$-normalized:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}, \quad \hat{\mathbf{w}}_j = \frac{\mathbf{w}_j}{\|\mathbf{w}_j\|_2} \implies \cos(\theta_j) = \hat{\mathbf{w}}_j^T \hat{\mathbf{e}}$$

An additive angular penalty $m = 0.35$ radians is added to the target class angle, scaled by radius $s = 30.0$:

$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\Big(s \cdot \cos(\theta_{y_i} + m)\Big)}{\exp\Big(s \cdot \cos(\theta_{y_i} + m)\Big) + \sum_{j \neq y_i} \exp\Big(s \cdot \cos(\theta_j)\Big)}$$

This guarantees a geodesic separation of at least $m$ radians on the hypersphere between bona fide and morph attack representations.

---

## 3. Training Protocol & Experimental Setup

### 3.1 Datasets
* **Training & In-Domain Validation:** **SMDD (`SMDD_release_train`)**
  * Total images: 32,000 synthetic faces.
  * Bona Fide: 20,030 samples ($62.6\%$).
  * Morphs: 11,970 samples ($37.4\%$).
  * Handled $5:3$ class imbalance using a `WeightedRandomSampler` ($w_c = 1 / N_c$).
* **Out-of-Distribution Evaluation:** **FRLL-Morphs**
  * Total images: 7,274 samples across 5 morphing generator families (OpenCV, FaceMorpher, WebMorph, AMSL, StyleGAN2) under 3 degradation conditions (Clean, JPEG $q=30$, Simulated Print-Scan).

### 3.2 Two-Stage Optimization Schedule
1. **Stage 1: Head Warmup (Epochs 1-3):** DINOv2 backbone is frozen. Only the projector and ArcFace head are trained using AdamW ($\eta = 10^{-3}$, weight decay $= 10^{-4}$).
2. **Stage 2: Differential Fine-Tuning (Epochs 4-10):** The final 2 transformer blocks (blocks 10 & 11) and LayerNorm are unfrozen. A differential learning rate is applied ($\eta_{\text{backbone}} = 10^{-5}, \eta_{\text{head}} = 10^{-4}$) scheduled via Cosine Annealing.

### 3.3 Anti-Shortcut Data Augmentation
To prevent the model from learning superficial dataset shortcuts (such as JPEG compression differences or resizing filters), Albumentations was used to apply:
* Randomized JPEG compression: $q \in [20, 70]$ ($p = 0.50$).
* Gaussian & motion blur: blur limit 3-7 ($p = 0.30$).
* Simulated print-scan: downscale to 40-60%, JPEG $q \in [30, 50]$, Gaussian noise $\sigma^2 \in [5, 25]$, and upscale ($p = 0.20$).
* Photometric jitter: Random Gamma ($p=0.30$), Brightness/Contrast ($\pm 20\%$), Color Jitter ($p=0.30$).

---

## 4. In-Domain Training Convergence (SMDD)

Training executed on NVIDIA GPU hardware using PyTorch Mixed Precision (`torch.cuda.amp`):

| Epoch | Optimization Stage | Train Loss | Train Acc | Val Loss | Val EER | Val ROC-AUC |
|---|---|---|---|---|---|---|
| **1** | Stage 1 (Warmup) | 0.5422 | 92.06% | 0.0072 | 0.0013 (0.13%) | 0.99998 |
| **2** | Stage 1 (Warmup) | 0.1544 | 95.54% | 0.0036 | 0.0006 (0.06%) | 0.99999 |
| **3** | Stage 1 (Warmup) | 0.1226 | 96.17% | 0.0061 | 0.0004 (0.04%) | 0.99999 |
| **4** | Stage 2 (Fine-Tune) | 0.1223 | 96.15% | 0.0071 | **0.0001 (0.01%)** | **0.999999** |

*The model achieved near-perfect in-domain convergence by Epoch 4, locking checkpoint `best_model.pt` (91.9 MB).*

---

## 5. Cross-Dataset Evaluation on FRLL-Morphs (15 Slices)

The locked Week 1 model was evaluated zero-shot on FRLL-Morphs under ISO/IEC 30107-3 metrics:

| Morph Generator | Condition | Equal Error Rate (EER) | BPCER @ APCER = 1% | BPCER @ APCER = 5% | ROC-AUC |
|---|---|---|---|---|---|
| **morph_opencv** | Clean | **0.0094 (0.94%)** | **0.0000 (0.0%)** | 0.0000 (0.0%) | **0.9993** |
| **morph_opencv** | JPEG (q=30) | **0.0094 (0.94%)** | 0.0098 (0.98%) | 0.0000 (0.0%) | **0.9992** |
| **morph_opencv** | Print-Scan | 0.0916 (9.16%) | 0.2990 (29.9%) | 0.1569 (15.7%) | 0.9786 |
| **morph_facemorpher** | Clean | **0.0245 (2.45%)** | 0.1912 (19.1%) | 0.0000 (0.0%) | **0.9958** |
| **morph_facemorpher** | JPEG (q=30) | 0.0249 (2.49%) | 0.1618 (16.2%) | 0.0000 (0.0%) | 0.9956 |
| **morph_facemorpher** | Print-Scan | 0.0585 (5.85%) | 0.2647 (26.5%) | 0.0735 (7.4%) | 0.9859 |
| **morph_webmorph** | Clean | **0.0491 (4.91%)** | 0.2745 (27.5%) | 0.0490 (4.9%) | **0.9886** |
| **morph_webmorph** | JPEG (q=30) | 0.0585 (5.85%) | 0.3039 (30.4%) | 0.0735 (7.4%) | 0.9875 |
| **morph_webmorph** | Print-Scan | 0.2539 (25.4%) | 0.7892 (78.9%) | 0.5833 (58.3%) | 0.8320 |
| **morph_amsl** | Clean | 0.1320 (13.2%) | 0.7598 (76.0%) | 0.3382 (33.8%) | 0.9461 |
| **morph_amsl** | JPEG (q=30) | 0.1367 (13.7%) | 0.7647 (76.5%) | 0.3235 (32.4%) | 0.9434 |
| **morph_amsl** | Print-Scan | 0.3978 (39.8%) | 0.9216 (92.2%) | 0.8039 (80.4%) | 0.6491 |
| **morph_stylegan** | Clean | **0.5100 (51.0%)** | **0.9951 (99.5%)** | 0.9902 (99.0%) | **0.4988** |
| **morph_stylegan** | JPEG (q=30) | 0.4998 (50.0%) | 0.9951 (99.5%) | 0.9853 (98.5%) | 0.4973 |
| **morph_stylegan** | Print-Scan | 0.4577 (45.8%) | 0.9608 (96.1%) | 0.8824 (88.2%) | 0.5780 |

---

## 6. Critical Shortcomings & Failure Analysis

### 6.1 The Mathematical Cause of the StyleGAN2 Collapse
The collapse on StyleGAN2 (AUC = 0.4988, EER = 51.00%) represents a fundamental theoretical limitation of spatial-domain analysis:

* **Landmark-Based Morphing:**
  $$I_{\text{landmark}}(\mathbf{x}) = \alpha I_A\Big(\mathcal{W}_A(\mathbf{x})\Big) + (1-\alpha) I_B\Big(\mathcal{W}_B(\mathbf{x})\Big)$$
  Operates via affine warping of 68 Delaunay triangles followed by pixel alpha blending. This operation leaves localized spatial artifacts: double edges along high-gradient features (pupil borders, iris texture, nostrils) and boundary blending seams around the facial mask perimeter. These physical discontinuities are readily flagged by ViT patch token variance statistics ($\boldsymbol{\sigma}_{\text{patch}}, \mathbf{z}_{\text{max}}$).
* **Generative Latent Morphing (StyleGAN2):**
  Images are inverted into StyleGAN2's extended latent space $\mathcal{W}^+$ and linearly interpolated:
  $$\mathbf{w}_{\text{morph}} = \alpha \mathbf{w}_A + (1-\alpha) \mathbf{w}_B \implies I_{\text{morph}} = \mathcal{G}(\mathbf{w}_{\text{morph}})$$
  The morphed face is synthesized from scratch through a generative network. There are **zero pixel-level alpha blending seams, zero cut-out masks, and zero ghosting double edges**. The pixel gradients are continuous and photorealistic.
* **Conclusion:** In the spatial RGB domain, a StyleGAN2 morphed face looks indistinguishable from a genuine human face. A spatial-only detector cannot discriminate generative morphs.

### 6.2 High Operational False Rejection Rates (BPCER @ APCER = 1%)
While clean EER on AMSL was 13.20%, BPCER at APCER=1% surged to **75.98%**. In an automated border control e-Gate, enforcing that no more than 1 in 100 morph attacks pass through ($\text{APCER} \le 1\%$) would cause **76% of legitimate travelers to be falsely rejected**. AMSL's edge feathering softens spatial seams, causing score distributions to overlap with genuine presentations.

### 6.3 Susceptibility to Print-Scan Low-Pass Filtering
Under simulated print-scan degradation, EER on WebMorph rose from 4.91% to 25.39%, and AMSL EER rose from 13.20% to 39.78%. Physical printing and scanning introduce an analog Modulation Transfer Function (MTF) that acts as a low-pass filter, smoothing out high-frequency spatial edge anomalies while adding printer halftoning grain that masks localized patch variance.

---

## 7. Week 1 Deliverables & Transition to Week 2

### Completed Deliverables in Week 1
1. **Standalone Script & Notebook:** [`scripts/phase1_dino_spatial.py`](file:///home/crimsond/project/scripts/phase1_dino_spatial.py) and [`scripts/phase1_dino_spatial.ipynb`](file:///home/crimsond/project/scripts/phase1_dino_spatial.ipynb).
2. **Empirical Artifacts:** [`results/phase1/training_history.csv`](file:///home/crimsond/project/results/phase1/training_history.csv) and ROC curve plots across generator splits.
3. **Comprehensive Defense Briefing:** [`docs/phase1_comprehensive_briefing.md`](file:///home/crimsond/project/docs/phase1_comprehensive_briefing.md).

### The Scientific Mandate for Week 2
While StyleGAN2 leaves no seams in the spatial RGB domain, it leaves prominent periodic upsampling fingerprints (transposed convolution Nyquist peaks) in the **frequency domain**. Week 2 will introduce **Stream B (Frequency Residual Stream)** using high-pass Laplacian filtering and a dedicated convolutional encoder to fuse with Stream A.
