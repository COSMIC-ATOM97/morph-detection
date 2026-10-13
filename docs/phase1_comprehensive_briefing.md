# Phase 1 Comprehensive Research Briefing: Single-Stream Spatial MAD Architecture, Empirical Benchmark Metrics, Failure Analysis, and Defense Guide

**Author / Researcher:** Morph Attack Detection Research Initiative  
**Target Audience:** Academic Advisor / Thesis Committee / Technical Reviewers  
**Topic:** Single-Image Face Morphing Attack Detection (S-MAD) via Self-Supervised Foundation ViT & Angular Margin Metric Learning  
**Biometric Evaluation Standard:** ISO/IEC 30107-3 (Information Technology -- Biometric Presentation Attack Detection)  
**Primary Artifacts:** `scripts/phase1_dino_spatial.py`, `scripts/phase1_dino_spatial.ipynb`, `results/phase1/`

---

## 1. Executive Summary & Research Narrative

Face morphing attacks constitute an existential vulnerability for automated border control (ABC) e-Gates and electronic Identity Document (eMRTD / ICAO Doc 9303) issuance systems. In an operational morph attack, a malicious actor synthetically blends facial images of an accomplice and an applicant, producing a hybrid photograph that matches the biometric template of both individuals in commercial Automated Facial Recognition Systems (AFRS).

```
    Subject A (Accomplice) ──┐
                             ├─► [Morph Generation] ─► Mappable to Both Passports
    Subject B (Criminal)   ──┘
```

Phase 1 established the **Single-Stream Spatial Baseline** for our Morph Attack Detection (MAD) research pipeline. The primary scientific objectives of Phase 1 were:
1. **Benchmark Foundation Vision Transformers:** Investigate whether high-capacity self-supervised Vision Transformers (DINOv2 ViT-S/14) learn spatial representations capable of detecting subtle facial morphing artifacts without task-specific engineering.
2. **Angular Margin Optimization:** Replace standard binary cross-entropy with Additive Angular Margin Loss (ArcFace) to enforce hyperspherical metric clustering and suppress intra-class feature dispersion.
3. **Rigorous Out-of-Distribution Protocol:** Train exclusively on synthetic data (SMDD) and benchmark zero-shot generalization against out-of-domain datasets (FRLL-Morphs) across 5 morph generator algorithms and 3 physical/compression degradation conditions under **ISO/IEC 30107-3** metrics.

### Key Empirical Outcomes
* **Near-Perfect Detection on Landmark Morphs:** Achieved **0.9993 ROC-AUC (0.94% EER)** on OpenCV and **0.9958 ROC-AUC (2.45% EER)** on FaceMorpher.
* **Invariance to Digital Compression:** Preserved **0.9992 ROC-AUC** under aggressive JPEG compression ($q=30$) through anti-shortcut data augmentations.
* **The StyleGAN2 Blind Spot (Core Theoretical Shortcoming):** Detection collapsed completely to random chance on generative latent morphs (**0.4988 ROC-AUC, 51.00% EER** on StyleGAN2), proving empirically that purely spatial RGB patch features cannot discriminate latent-space generative morphs. This finding served as the direct scientific justification for Phase 2's dual-stream architecture.

---

## 2. System Architecture & Mathematical Formulation

The Phase 1 architecture combines a frozen self-supervised Vision Transformer backbone, multi-scale patch token pooling, a non-linear projection bottleneck, and an Additive Angular Margin classification head.

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

### 2.1 The Theoretical Justification for DINOv2
Standard convolutional neural networks (e.g., ResNet-50) and supervised ViTs trained on ImageNet-1k optimize for broad semantic categories (e.g., "dog", "car", "person"). In doing so, supervised backbones intentionally discard fine-grained high-frequency pixel anomalies to achieve translation and texture invariance.

In contrast, **DINOv2 (Vision Transformer Small with 14x14 patch size, 21.6M parameters)** is trained via a self-supervised multi-task objective:
1. **DINO Self-Distillation:** Matches the global distribution of the CLS token across student and teacher networks under multi-crop transformations.
2. **iBOT Masked Image Modeling:** Reconstructs patch tokens from masked input views at the feature level.

This self-supervised pretraining forces DINOv2's 256 patch tokens ($16 \times 16$ grid for a $224 \times 224$ image) to retain **dense semantic correspondence and local patch variance**, making them sensitive to localized pixel blending, ghosting contours, and interpolation seams.

### 2.2 Multi-Scale Patch Token Pooling
In standard classification tasks, only the classification token $\mathbf{z}_{\text{cls}}$ is utilized. However, in facial morphing attacks, blending artifacts often occupy localized spatial regions (such as the iris boundary, nasolabial folds, or the perimeter of the face mask).

To capture both holistic identity semantics and local anomaly variance, we construct a 1536-dimensional feature vector $\mathbf{f}$:

$$\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}$$

Where $N = 256$ is the number of spatial patches, and:

$$\boldsymbol{\mu}_{\text{patch}} = \frac{1}{N}\sum_{i=1}^N \mathbf{z}_i$$

$$\boldsymbol{\sigma}_{\text{patch}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{z}_i - \boldsymbol{\mu}_{\text{patch}})^2 + \epsilon}$$

$$\mathbf{z}_{\text{max}} = \max_{i=1,\dots,N} \mathbf{z}_i \quad \text{(element-wise maximum)}$$

* $\boldsymbol{\mu}_{\text{patch}}$ provides average background structural context.
* $\boldsymbol{\sigma}_{\text{patch}}$ explicitly quantifies the variance of representation across spatial patches, directly flagging localized discontinuities.
* $\mathbf{z}_{\text{max}}$ captures peak localized anomaly signals that would otherwise be smoothed out by average pooling.

The concatenated vector is compressed through a two-layer bottleneck with Batch Normalization:

$$\mathbf{h} = \text{GELU}\Big(\text{BatchNorm1d}(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1)\Big) \in \mathbb{R}^{512}$$

$$\mathbf{e} = \text{BatchNorm1d}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{256}$$

### 2.3 Additive Angular Margin Loss (ArcFace) Formulation
Conventional Binary Cross-Entropy (BCE) with a sigmoid output evaluates classification via a hyperplane in Euclidean space:

$$\mathcal{L}_{\text{BCE}} = -y \log \sigma(\mathbf{w}^T \mathbf{e} + b) - (1-y)\log(1 - \sigma(\mathbf{w}^T \mathbf{e} + b))$$

BCE optimizes separability, but does not constrain intra-class compactness. In open-set biometric evaluation, unseen morphing attacks scatter widely across Euclidean space, leading to false acceptances.

We utilize **Additive Angular Margin Loss (ArcFace)**. The embeddings $\mathbf{e}$ and class weight vectors $\mathbf{w}_j$ are $L_2$-normalized:

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}, \quad \hat{\mathbf{w}}_j = \frac{\mathbf{w}_j}{\|\mathbf{w}_j\|_2} \implies \cos(\theta_j) = \hat{\mathbf{w}}_j^T \hat{\mathbf{e}}$$

An additive angular penalty $m$ is enforced on the ground-truth angle $\theta_{y_i}$, scaled by hypersphere radius $s$:

$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\Big(s \cdot \cos(\theta_{y_i} + m)\Big)}{\exp\Big(s \cdot \cos(\theta_{y_i} + m)\Big) + \sum_{j \neq y_i} \exp\Big(s \cdot \cos(\theta_j)\Big)}$$

**Hyperparameters:** $s = 30.0$, $m = 0.35$, $C = 2$ (Class 0: Bona Fide, Class 1: Morph Attack).

**Geometric Interpretation:** ArcFace enforces that for an input to be classified as class 1 (Morph), its cosine angle must satisfy:

$$\cos(\theta_1 + m) > \cos(\theta_0) \implies \theta_1 < \theta_0 - m$$

This forces an angular margin of at least $m$ radians between the bona fide and morph attack manifolds on the unit hypersphere, suppressing intra-class variance and improving out-of-distribution separation.

---

## 3. Training Protocol & Optimization Strategy

### 3.1 Two-Stage Differential Optimization
Training a 21.6M parameter Vision Transformer on a binary task from scratch can result in catastrophic forgetting of self-supervised representations. We implemented a two-stage training schedule:

```
                  STAGE 1: HEAD WARMUP (Epochs 1-3)
           Backbone: FROZEN (requires_grad = False)
           Projector & ArcFace Head: TRAINABLE (lr = 1e-3, AdamW)
           Objective: Align projection manifold with ArcFace hypersphere
                                   │
                                   ▼
              STAGE 2: DIFFERENTIAL FINE-TUNING (Epochs 4-10)
           Backbone: UNFREEZE LAST 2 BLOCKS (Blocks 10 & 11) + Norm
           Backbone lr: 1e-5 (Preserves self-supervised features)
           Head lr:     1e-4 (Cosine Annealing scheduler)
           Weight Decay: 1e-4
```

1. **Stage 1 (Head Warmup, Epochs 1-3):** The entire DINOv2 backbone is frozen. Only the projector ($\mathbf{W}_1, \mathbf{W}_2$) and ArcFace weight matrix $\mathbf{W}_{\text{arc}}$ are updated with $\eta = 10^{-3}$ and AdamW. This prevents large random initialization gradients from corrupting the pretrained ViT weights.
2. **Stage 2 (Differential Fine-Tuning, Epochs 4-10):** The final 2 transformer blocks (Block 10 and Block 11) and the final LayerNorm of the ViT are unfrozen. A differential learning rate is applied:
   * Backbone parameters: $\eta_{\text{backbone}} = 10^{-5}$
   * Projector and Head parameters: $\eta_{\text{head}} = 10^{-4}$
   * Learning Rate Decay: Cosine Annealing scheduler over the remaining 7 epochs.

### 3.2 Dataset Imbalance & Weighted Random Sampling
The training corpus **SMDD (`SMDD_release_train`)** contains 32,000 synthetic face samples:
* Bona Fide: 20,030 samples ($62.6\%$)
* Morph Attacks: 11,970 samples ($37.4\%$)

The dataset presents a natural $5:3$ class imbalance. Without correction, gradient descent biases the decision boundary toward the majority class (bona fide). We implement a `WeightedRandomSampler` where sample probability is weighted inversely by class frequency:

$$w_c = \frac{1}{N_c}, \quad P(\text{sample}_i) = \frac{w_{y_i}}{\sum_{k} w_{y_k}}$$

This guarantees an expected $1:1$ balanced class distribution across every training batch without data discarding.

### 3.3 Anti-Shortcut Data Augmentation Pipeline
In deep learning for biometric forensics, models frequently learn trivial "shortcuts" (e.g., subtle differences in JPEG compression quality, resizing bicubic filters, or color tone shifts between generator outputs and source images). To eliminate these shortcuts, we deployed an augmentation pipeline using Albumentations:

* **Simulated JPEG Compression:** $q \in [20, 70]$ ($p = 0.50$).
* **Gaussian & Motion Blur:** Blur limit 3 to 7, $\sigma \in [0.1, 2.0]$ ($p = 0.30$).
* **Simulated Print-Scan Attack:** Downscale to $40\%-60\%$, apply JPEG ($q \in [30, 50]$), inject additive Gaussian noise ($\sigma^2 \in [5, 25]$), and upscale back ($p = 0.20$).
* **Photometric Perturbations:** Random Gamma ($\gamma \in [0.7, 1.3]$), Random Brightness/Contrast ($\pm 20\%$), Color Jitter ($p = 0.30$).
* **Geometric Regularization:** Random Horizontal Flip ($p = 0.50$).

---

## 4. In-Domain Training Convergence (SMDD)

Training executed on an NVIDIA GPU cluster (Dual T4 / P100) using PyTorch Mixed Precision (`torch.cuda.amp.autocast`). The model converged rapidly:

| Epoch | Optimization Stage | Train Loss | Train Acc | Val Loss | Val EER | Val ROC-AUC |
|---|---|---|---|---|---|---|
| **1** | Stage 1 (Warmup) | 0.5422 | 92.06% | 0.0072 | 0.0013 (0.13%) | 0.99998 |
| **2** | Stage 1 (Warmup) | 0.1544 | 95.54% | 0.0036 | 0.0006 (0.06%) | 0.99999 |
| **3** | Stage 1 (Warmup) | 0.1226 | 96.17% | 0.0061 | 0.0004 (0.04%) | 0.99999 |
| **4** | Stage 2 (Fine-Tune) | 0.1223 | 96.15% | 0.0071 | **0.0001 (0.01%)** | **0.999999** |

*By Epoch 4, in-domain validation EER reached 0.01% ($1 \times 10^{-4}$), and validation ROC-AUC reached 0.999999. The best model checkpoint (`best_model.pt`, 91.9 MB) was locked for cross-dataset evaluation.*

---

## 5. Cross-Dataset Evaluation on FRLL-Morphs (ISO/IEC 30107-3)

### 5.1 Biometric Evaluation Metrics Explained
In compliance with international standard **ISO/IEC 30107-3**:
1. **APCER (Attack Presentation Classification Error Rate):** The proportion of morph attacks misclassified as bona fide (False Acceptance Rate on attacks / False Negatives).
   $$\text{APCER}(\tau) = \frac{1}{N_{\text{attack}}} \sum_{i=1}^{N_{\text{attack}}} \mathbb{I}(s_i < \tau)$$
2. **BPCER (Bona Fide Presentation Classification Error Rate):** The proportion of genuine presentations misclassified as morph attacks (False Rejection Rate on genuine users / False Positives).
   $$\text{BPCER}(\tau) = \frac{1}{N_{\text{bona}}} \sum_{j=1}^{N_{\text{bona}}} \mathbb{I}(s_j \ge \tau)$$
3. **EER (Equal Error Rate):** The operating point where $\text{APCER}(\tau) = \text{BPCER}(\tau)$.
4. **BPCER @ APCER = 1.0% ($0.01$):** Mandated by automated border control (Frontex / ICAO). In high-security passport e-Gates, no more than $1\%$ of morph attacks may be accepted ($\text{APCER} \le 1\%$). BPCER at this threshold indicates what fraction of bona fide travelers will be falsely flagged for secondary inspection.
5. **BPCER @ APCER = 5.0% ($0.05$):** Secondary operating benchmark for commercial/civil verification systems.

### 5.2 Full Empirical Benchmark: All 15 Evaluation Slices
The locked model was evaluated **zero-shot** on FRLL-Morphs across all 5 morphing algorithms and 3 degradation conditions (7,274 samples):

| Generator Split | Condition | Equal Error Rate (EER) | BPCER @ APCER = 1% | BPCER @ APCER = 5% | ROC-AUC | Val Loss |
|---|---|---|---|---|---|---|
| **morph_opencv** | Clean | **0.0094 (0.94%)** | **0.0000 (0.0%)** | 0.0000 (0.0%) | **0.9993** | 5.4053 |
| **morph_opencv** | JPEG (q=30) | **0.0094 (0.94%)** | 0.0098 (0.98%) | 0.0000 (0.0%) | **0.9992** | 5.3539 |
| **morph_opencv** | Print-Scan | 0.0916 (9.16%) | 0.2990 (29.9%) | 0.1569 (15.7%) | 0.9786 | 2.7859 |
| **morph_facemorpher** | Clean | **0.0245 (2.45%)** | 0.1912 (19.1%) | 0.0000 (0.0%) | **0.9958** | 5.9695 |
| **morph_facemorpher** | JPEG (q=30) | 0.0249 (2.49%) | 0.1618 (16.2%) | 0.0000 (0.0%) | 0.9956 | 5.9273 |
| **morph_facemorpher** | Print-Scan | 0.0585 (5.85%) | 0.2647 (26.5%) | 0.0735 (7.4%) | 0.9859 | 2.6364 |
| **morph_webmorph** | Clean | **0.0491 (4.91%)** | 0.2745 (27.5%) | 0.0490 (4.9%) | **0.9886** | 10.2887 |
| **morph_webmorph** | JPEG (q=30) | 0.0585 (5.85%) | 0.3039 (30.4%) | 0.0735 (7.4%) | 0.9875 | 10.2548 |
| **morph_webmorph** | Print-Scan | 0.2539 (25.4%) | 0.7892 (78.9%) | 0.5833 (58.3%) | 0.8320 | 8.7468 |
| **morph_amsl** | Clean | 0.1320 (13.2%) | 0.7598 (76.0%) | 0.3382 (33.8%) | 0.9461 | 12.8552 |
| **morph_amsl** | JPEG (q=30) | 0.1367 (13.7%) | 0.7647 (76.5%) | 0.3235 (32.4%) | 0.9434 | 12.8603 |
| **morph_amsl** | Print-Scan | 0.3978 (39.8%) | 0.9216 (92.2%) | 0.8039 (80.4%) | 0.6491 | 12.3142 |
| **morph_stylegan** | Clean | **0.5100 (51.0%)** | **0.9951 (99.5%)** | 0.9902 (99.0%) | **0.4988** | 12.9096 |
| **morph_stylegan** | JPEG (q=30) | 0.4998 (50.0%) | 0.9951 (99.5%) | 0.9853 (98.5%) | 0.4973 | 12.9027 |
| **morph_stylegan** | Print-Scan | 0.4577 (45.8%) | 0.9608 (96.1%) | 0.8824 (88.2%) | 0.5780 | 12.1117 |

---

## 6. Comprehensive Shortcomings & Scientific Failure Analysis

The Phase 1 experiments revealed three critical shortcomings. Rather than viewing these as project deficiencies, each failure provides clear theoretical insights that directly led to the Phase 2 architecture.

```
                              PHASE 1 SHORTCOMINGS
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 [STYLEGAN2 BLIND SPOT]     [OPERATIONAL BPCER SPIKES]     [PRINT-SCAN DEGRADATION]
 Latent space interpolation   At APCER=1% threshold,        Analog LP filter washes
 leaves zero RGB spatial      bona fide false rejection     out boundary seams;
 seams; AUC drops to 0.4988   reaches 76% on AMSL           injects halftone noise
```

---

### Shortcoming 1: The Catastrophic Generative Collapse (StyleGAN2 Failure)

#### The Empirical Finding
On StyleGAN2 morphs, Phase 1 achieved **ROC-AUC = 0.4988** and **EER = 51.00%**. An AUC of 0.4988 is statistically indistinguishable from a random coin toss ($0.50$). The model was completely blind to StyleGAN2 morphs.

#### The Mathematical & Generative Cause
The collapse is rooted in the mathematical formulation of how the morph is synthesized:

##### 1. Landmark-Based Digital Warping (OpenCV, FaceMorpher, WebMorph, AMSL)
Two donor face photographs $I_A$ and $I_B$ are annotated with 68 facial landmarks. Delaunay triangulation partitions the facial mesh into triangles $T_k$. Affine transformation matrices $\mathcal{W}_A^{(k)}$ and $\mathcal{W}_B^{(k)}$ warp the source pixels to an intermediate average geometry. The final image is constructed via pixel-wise alpha blending:

$$I_{\text{landmark}}(\mathbf{x}) = \alpha \cdot I_A\Big(\mathcal{W}_A(\mathbf{x})\Big) + (1 - \alpha) \cdot I_B\Big(\mathcal{W}_B(\mathbf{x})\Big)$$

* **Artifact Footprint:** This operation creates physical discontinuities in the spatial RGB domain:
  * Double contours and ghosting along high-gradient features (pupil borders, iris textures, eyelashes).
  * Boundary cut lines where the morphed facial triangle mask is blended back into the donor's original hair and torso background.
* **Why DINOv2 Succeeds:** DINOv2 patch tokens compute localized self-attention. The patch statistics ($\boldsymbol{\sigma}_{\text{patch}}, \mathbf{z}_{\text{max}}$) register elevated variance spikes exactly at these blending seams, achieving $>0.99$ AUC.

##### 2. Generative Latent-Space Morphing (StyleGAN2)
Donor images $I_A$ and $I_B$ are projected into the extended latent space $\mathcal{W}^+$ of a pretrained StyleGAN2 generator via GAN inversion:

$$\mathbf{w}_A = \mathcal{E}(I_A), \quad \mathbf{w}_B = \mathcal{E}(I_B) \quad \in \mathbb{R}^{18 \times 512}$$

Morphing is performed via linear vector interpolation in latent space:

$$\mathbf{w}_{\text{morph}} = \alpha \mathbf{w}_A + (1 - \alpha) \mathbf{w}_B$$

The morphed facial image is synthesized from scratch through the StyleGAN2 generator network $\mathcal{G}$:

$$I_{\text{generative}} = \mathcal{G}(\mathbf{w}_{\text{morph}})$$

* **Artifact Footprint:** StyleGAN2 synthesizes every pixel coherently from a deep neural network. There are **zero pixel-level alpha blending seams, zero cut-out masks, and zero ghosting double edges**.
* **Spatial Appearance:** In the spatial RGB domain, a StyleGAN2 face has smooth, continuous gradients and natural lighting transitions.
* **Why DINOv2 Fails:** To a spatial Vision Transformer seeking localized warping discontinuities, a StyleGAN2 morphed face looks identical to a genuine human photograph. The spatial patch variance signature is completely absent.

#### The Scientific Solution (The Bridge to Phase 2)
While StyleGAN2 leaves no seams in the spatial RGB domain, it leaves distinct structural artifacts in the **frequency domain**:
1. **Periodic Upsampling Spikes (Nyquist Checkerboard Artifacts):** StyleGAN2 relies on transposed convolutions and bilinear upsampling layers. In the 2D frequency spectrum (DCT/FFT), this induces periodic spectral peaks that do not occur in camera lenses.
2. **Missing Sensor PRNU:** Camera sensors introduce Photo-Response Non-Uniformity (PRNU) noise. Generative synthesis lacks natural optical sensor noise.
3. High-pass residual filtering ($R = I - \text{GaussianFilter}(I)$) removes low-frequency facial identity and exposes the generator's synthetic noise footprint. This observation formed the foundation of Phase 2's Stream B.

---

### Shortcoming 2: High False Rejection Rates at Operational Thresholds (BPCER @ APCER = 1%)

#### The Empirical Finding
While the Equal Error Rate on AMSL appears moderate ($13.20\%$), the operational metric **BPCER @ APCER = 1% surges to 75.98%**.

#### Operational Significance
In an Automated Border Control (ABC) deployment:
* **APCER = 1% Requirement:** Border security regulations mandate that no more than 1 out of 100 morph attacks may bypass the e-Gate undetected.
* **Resulting BPCER = 75.98%:** To force the decision threshold high enough to catch 99% of morph attacks, the system must set a threshold where **76 out of 100 legitimate, bona fide travelers are falsely rejected** and flagged for manual secondary inspection.
* **Practical Consequence:** The e-Gate infrastructure becomes practically unusable under real passenger traffic loads.

#### Root Cause
AMSL applies advanced landmark blending and post-processing feathering, which softens spatial seams. The score distribution for AMSL morphs overlaps heavily with the upper tail of the bona fide distribution. Because the ArcFace loss was trained with fixed margin $m=0.35$ on synthetic data, it did not learn sufficiently separated margins for softened, post-processed landmark morphs.

---

### Shortcoming 3: Vulnerability to Physical Medium Transfer (Print-Scan Degradation)

#### The Empirical Finding
Under simulated print-scan degradation:
* OpenCV EER rose from **0.94% to 9.16%** (AUC dropped from 0.9993 to 0.9786).
* WebMorph EER surged from **4.91% to 25.39%** (AUC dropped from 0.9886 to 0.8320).
* AMSL EER surged from **13.20% to 39.78%** (AUC dropped from 0.9461 to 0.6491).

#### Root Cause
Physical printing (CMYK halftoning) followed by flatbed scanning acts as an **analog low-pass modulation transfer filter (MTF)**:
1. Low-pass smoothing attenuates the subtle high-frequency spatial edge anomalies created by Delaunay warping.
2. The printing process introduces halftone dot patterns and paper grain texture across the entire image.
3. The spatial Vision Transformer interprets halftone grain as global image noise, which masks the localized variance spikes ($\boldsymbol{\sigma}_{\text{patch}}$) that previously signaled a morph.

---

## 7. Oral Defense Guide: Questions & Technical Answers for Your Professor

Use these structured talking points to defend Phase 1 during your presentation.

---

### Q1: "Why did you use DINOv2 self-supervised ViT instead of an existing face recognition model like ArcFace / InsightFace trained on Glint360k?"

> **Strong Technical Answer:**  
> "A facial recognition backbone like InsightFace or ArcFace is trained to optimize identity verification. Its training objective explicitly learns to discard intra-person variations such as lighting, minor expression shifts, and local skin imperfections in order to extract an identity-invariant feature embedding. Morphing attacks, however, are specifically designed to retain identity consistency while introducing subtle, localized physical anomalies (such as blending seams and landmark warping). Using an identity-optimized model causes the network to overlook the very artifacts we need to detect.  
> In contrast, DINOv2 is trained with a self-supervised objective combining patch-level masked image modeling (iBOT) and global distillation (DINO). Its patch tokens retain rich geometric and dense pixel correspondence without identity projection, making it far more sensitive to structural and edge discontinuities."

---

### Q2: "Why did you use ArcFace loss instead of standard Binary Cross-Entropy?"

> **Strong Technical Answer:**  
> "Binary Cross-Entropy optimizes a separating hyperplane in Euclidean space. While BCE can achieve high training accuracy, it does not constrain intra-class dispersion. In open-set biometric evaluation, morph attacks from unseen generator algorithms scatter across Euclidean space and can cross the decision boundary, leading to poor generalization.  
> ArcFace projects feature vectors onto a unit hypersphere and enforces an additive angular margin penalty ($m = 0.35$ radians) on the target class. This forces the network to minimize intra-class angular variance while maximizing the geodesic distance between bona fide and attack representations. This compact clustering is essential for out-of-distribution biometric generalization."

---

### Q3: "Your validation EER on SMDD was 0.01%, but on AMSL it was 13.20% and on StyleGAN it was 51.00%. Did the model overfit to SMDD?"

> **Strong Technical Answer:**  
> "No, this is not classical statistical overfitting; it represents a cross-dataset, cross-generator domain shift. SMDD is an entirely synthetic dataset where morphs were generated using specific splicing and blending routines.  
> On landmark morphs with similar geometric foundations (OpenCV and FaceMorpher), the model achieved 0.94% and 2.45% EER zero-shot, proving that the learned spatial representations generalized across datasets.  
> The drop on AMSL occurs because AMSL applies heavy seam-smoothing and post-processing, which reduces spatial gradient variance.  
> The collapse on StyleGAN2 (51.00% EER) is a fundamental representation limit: StyleGAN2 morphs are generated via latent vector interpolation in $\mathcal{W}^+$ space and synthesized de novo. They have zero spatial blending seams. A purely spatial RGB model has no physical mechanism to distinguish a StyleGAN face from a genuine face."

---

### Q4: "Why do you report BPCER @ APCER = 1% instead of simply reporting Accuracy, Precision, or F1-Score?"

> **Strong Technical Answer:**  
> "In biometric presentation attack detection, overall accuracy and F1-score are misleading because they depend heavily on the evaluation dataset's class ratio and implicitly assume an operational threshold of $\tau = 0.5$.  
> Under international standard ISO/IEC 30107-3, biometric systems must be evaluated across operational risk operating points. In high-security automated border control (ICAO 9303 / Frontex), border authorities cannot tolerate security breaches; they require APCER $\le 1\%$ (fewer than 1 in 100 morph attacks accepted). Reporting BPCER @ APCER = 1% directly measures the real-world operational cost: the percentage of genuine travelers who would be falsely rejected and delayed at the border to maintain that security posture."

---

### Q5: "Why did JPEG compression at quality 30 barely degrade performance on OpenCV (AUC dropped only from 0.9993 to 0.9992)?"

> **Strong Technical Answer:**  
> "This robustness is a direct result of our anti-shortcut data augmentation pipeline. During training on SMDD, we applied randomized JPEG compression with quality factors ranging from 20 to 70 with a 50% probability.  
> This prevented the Vision Transformer from relying on high-frequency DCT block artifacts as a shortcut for classification. The network was forced to learn structural and geometric warping features (such as iris ghosting and contour distortions) that survive 8x8 DCT quantization."

---

### Q6: "Why did print-scan degradation hurt performance so significantly across all landmark morphs?"

> **Strong Technical Answer:**  
> "Physical printing and scanning introduce an analog Modulation Transfer Function (MTF) that behaves as a low-pass spatial filter. Printing halftones the image into CMYK ink dots, and optical scanning blurs high-frequency pixel transitions.  
> Because landmark morph detection in Stream A relies on high-frequency spatial edge discontinuities along Delaunay triangle borders, the analog low-pass filtering attenuates these seams, while the printed halftone pattern injects paper-grain noise across the face. This washes out the localized patch variance signals that DINOv2 uses to identify morphs."

---

### Q7: "What specific modifications did these Phase 1 shortcomings motivate for Phase 2?"

> **Strong Technical Answer:**  
> "Phase 1 confirmed that spatial Vision Transformers are effective for landmark-warped morphs, but fundamentally blind to generative morphs like StyleGAN2.  
> To address this, Phase 2 developed a **Dual-Stream Architecture (UID-MAD)**:
> 1. **Stream A (Spatial):** Retains the DINOv2 patch aggregator to handle landmark warping and boundary seams.
> 2. **Stream B (Frequency Residual):** Introduces a 3x3 high-pass Laplacian filter followed by a 4-stage residual convolutional network. This stream isolates the high-frequency residual $R = I - \text{GaussianFilter}(I)$, exposing the periodic checkerboard upsampling artifacts and spectral peaks characteristic of GAN generators.
> 3. **Multi-Branch Fusion:** Fuses both representations into a unified feature vector to provide simultaneous coverage across both landmark and generative morph attacks."

---

## 8. Summary Comparison: Landmark vs. Generative Morphing

To assist in oral presentation, this summary table contrasts the two morphing paradigms evaluated in Phase 1:

| Dimension | Landmark-Based Morphing | Generative Latent Morphing |
|---|---|---|
| **Representative Generators** | OpenCV, FaceMorpher, WebMorph, AMSL | StyleGAN2, StyleGAN3 |
| **Mathematical Mechanism** | Delaunay triangulation + Affine Warping + Alpha Blending: $I = \alpha \mathcal{W}(I_A) + (1-\alpha)\mathcal{W}(I_B)$ | GAN Inversion + Latent Interpolation: $\mathbf{w} = \alpha \mathbf{w}_A + (1-\alpha)\mathbf{w}_B$; $I = \mathcal{G}(\mathbf{w})$ |
| **Spatial RGB Footprint** | Iris double contours, blending mask seams, boundary cut lines | Coherent, photorealistic gradients; zero splicing seams |
| **Frequency Footprint** | Localized high-frequency edge phase shifts | Periodic Nyquist upsampling spikes, absent optical PRNU |
| **Phase 1 Stream A AUC** | **0.9993 (OpenCV), 0.9958 (FaceMorpher)** | **0.4988 (StyleGAN2 -- Random Chance)** |
| **Phase 1 Detection Mechanism** | ViT Patch token variance ($\boldsymbol{\sigma}_{\text{patch}}, \mathbf{z}_{\text{max}}$) | **Fails completely (Blind to latent synthesis)** |
| **Architectural Remedy** | Preserved in Stream A | **Stream B: High-Pass Laplacian Residual ConvNet** |

---

## 9. Conclusion & Transition to Phase 2

Phase 1 established that self-supervised Vision Transformers equipped with ArcFace heads provide an effective spatial baseline for detecting landmark-based morph attacks, maintaining strong robustness against digital JPEG compression.

However, the complete failure on StyleGAN2 (0.4988 AUC) demonstrated that single-stream spatial architectures cannot serve as a complete defense against modern generative morphing techniques. This finding validated our research hypothesis: **effective morph attack detection requires a multi-domain approach combining spatial structural analysis with frequency-domain residual analysis.**

This conclusion directly motivated the implementation of the **Dual-Stream UID-MAD architecture** in Phase 2.
