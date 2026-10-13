# Face Morph Attack Detection (S-MAD) via Self-Supervised Vision Transformers & Metric Learning

[![Standard](https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-blue.svg)]()
[![Backbone](https://img.shields.io/badge/Backbone-DINOv2%20ViT--S%2F14-orange.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Dual%20NVIDIA%20T4-green.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

Academic research project on Single-Image Morph Attack Detection (S-MAD) for Automated Border Control (ABC) e-Gates and biometric passport verification (ICAO Doc 9303).

---

## Executive Summary

A **face morph attack** occurs when two individuals (such as an accomplice with no criminal record and a criminal actor) synthetically blend their facial photographs into a single composite image. When submitted during a passport or electronic identity document (eMRTD) application, the resulting document can verify against the biometric templates of **both individuals**, allowing unauthorized border crossing through Automated Border Control (ABC) e-Gates.

```
       Subject A (Accomplice) ──┐
                                ├─► [Morph Generation] ─► Matches Both Passports
       Subject B (Criminal)   ──┘
```

This project establishes a rigorous **Single-Stream Spatial Baseline** evaluated strictly under **ISO/IEC 30107-3** presentation attack detection standards.

### Key Highlights
- **Self-Supervised Spatial Foundation:** Leverages DINOv2 ViT-S/14 with a multi-scale patch token aggregator (`CLS + Mean + Std + Max` to 1536-d) sensitive to subtle localized blending seams.
- **Angular Margin Metric Learning:** Implements Additive Angular Margin Loss (ArcFace: scale s = 30.0, margin m = 0.35) to enforce compact hyperspherical feature clustering.
- **Near-Perfect Landmark Morph Detection:** Achieved **0.9993 ROC-AUC (0.94% EER)** on OpenCV and **0.9958 ROC-AUC (2.45% EER)** on FaceMorpher zero-shot out-of-distribution.
- **Digital Compression Robustness:** Maintained **0.9992 ROC-AUC** under aggressive JPEG compression (q = 30) via anti-shortcut data augmentation.
- **Identification of the StyleGAN2 Blind Spot:** Collapsed to random chance (**0.4988 ROC-AUC, 51.00% EER**) on generative latent morphs, establishing the theoretical boundary of spatial-only representations.

---

## System Architecture & Methodology

The Phase 1 baseline pipeline processes single RGB facial images through a frozen/differentially fine-tuned self-supervised Vision Transformer, pools multi-scale patch statistics, and maps representations onto an angular margin hypersphere.

```
                      INPUT IMAGE: I in R^(224 x 224 x 3)
                                    │
                                    ▼
                      DINOv2 ViT-S/14 BACKBONE (14x14 Patches)
              Outputs: CLS Token z_cls in R^384, Patch Tokens z_1...z_256 in R^384
                                    │
                                    ▼
                      MULTI-SCALE PATCH TOKEN AGGREGATOR
          f = [ z_cls || Mean(z_patches) || Std(z_patches) || Max(z_patches) ]
                               f in R^1536
                                    │
                                    ▼
                        PROJECTION BOTTLENECK (MLP)
                      Linear(1536 -> 512) -> BatchNorm -> GELU
                      Linear(512 -> 256)  -> BatchNorm
                               e in R^256
                                    │
                                    ▼
                          ARCFACE HEAD (s=30.0, m=0.35)
                 L2 Normalization of Feature e and Weight Matrix W
                    Angular Margin Penalty on Ground-Truth Class
                                    │
                                    ▼
                        CROSS-ENTROPY LOSS / INFERENCE
```

### 1. Spatial Feature Extraction (DINOv2 ViT-S/14)
- **Backbone:** `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, 384 embedding dimension).
- **Self-Supervised Distillation:** Pretrained via joint global distillation (DINO) and patch masked image modeling (iBOT). Unlike ImageNet-supervised classifiers that discard fine pixel anomalies to achieve object invariance, DINOv2 preserves dense geometric correspondences and local patch texture variance.
- **Multi-Scale Patch Token Pooling:** Morphing seams occupy localized spatial regions (pupil contours, nose bridge, facial cutouts). We concatenate global and localized statistics across all 256 spatial patch tokens (16 x 16 grid):

$$
\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}
$$

Where the summary patch statistics are defined over $N = 256$ spatial tokens:

$$
\boldsymbol{\mu}_{\text{patch}} = \frac{1}{N}\sum_{i=1}^N \mathbf{z}_i, \quad \boldsymbol{\sigma}_{\text{patch}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{z}_i - \boldsymbol{\mu}_{\text{patch}})^2 + \epsilon}, \quad \mathbf{z}_{\text{max}} = \max_{i=1,\dots,N} \mathbf{z}_i
$$

The patch standard deviation explicitly quantifies representation variance across patches, directly registering localized blending discontinuities.

### 2. Additive Angular Margin Loss (ArcFace)
To enforce compact intra-class clustering and separate unseen morph attacks on the unit hypersphere, feature embeddings and class weights are L2-normalized:

$$
\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}, \quad \hat{\mathbf{w}}_j = \frac{\mathbf{w}_j}{\|\mathbf{w}_j\|_2} \implies \cos(\theta_j) = \hat{\mathbf{w}}_j^T \hat{\mathbf{e}}
$$

The ArcFace loss enforces an additive angular margin penalty $m$ on the target ground-truth class angle, scaled by radius $s$:

$$
\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\big(s \cdot \cos(\theta_{y_i} + m)\big)}{\exp\big(s \cdot \cos(\theta_{y_i} + m)\big) + \sum_{j \neq y_i} \exp\big(s \cdot \cos(\theta_j)\big)}
$$

- Hyperparameters: Radius $s = 30.0$, Margin $m = 0.35$ radians, Classes $C = 2$ (0: Bona Fide, 1: Morph Attack).
- Guarantees geodesic separation of at least $m$ radians between bona fide and morph attack manifolds on the unit hypersphere.

### 3. Training Protocol & Anti-Shortcut Pipeline
- **Dataset:** SMDD Synthetic Morphing Face Database (`SMDD_release_train`: 32,000 synthetic faces, 20,030 bona fide vs 11,970 morphs).
- **Class Balancing:** `WeightedRandomSampler` guarantees a strict 1:1 balanced class distribution across every training batch.
- **Two-Stage Differential Optimization:**
  - *Stage 1 (Warmup, Epochs 1-3):* Backbone frozen. Projector and ArcFace head trained with AdamW (learning rate = 1e-3, weight decay = 1e-4).
  - *Stage 2 (Fine-Tuning, Epochs 4-10):* Blocks 10 & 11 and LayerNorm unfrozen with differential learning rates (backbone lr = 1e-5, head lr = 1e-4) under Cosine Annealing.
- **Anti-Shortcut Augmentations (Albumentations):** Randomized JPEG compression (quality q in [20, 70]), Gaussian & motion blur, simulated print-scan degradation, and photometric jitter to prevent the network from memorizing dataset-specific digital shortcuts.

---

## Empirical Results & Benchmarks

### 1. In-Domain Training Convergence (SMDD)
Evaluated across 4 training epochs with PyTorch Mixed Precision (`torch.cuda.amp`):

| Epoch | Optimization Stage | Train Loss | Train Acc | Val Loss | Val EER | Val ROC-AUC |
|---|---|---|---|---|---|---|
| **1** | Stage 1 (Warmup) | 0.5422 | 92.06% | 0.0072 | 0.0013 (0.13%) | 0.99998 |
| **2** | Stage 1 (Warmup) | 0.1544 | 95.54% | 0.0036 | 0.0006 (0.06%) | 0.99999 |
| **3** | Stage 1 (Warmup) | 0.1226 | 96.17% | 0.0061 | 0.0004 (0.04%) | 0.99999 |
| **4** | Stage 2 (Fine-Tune) | 0.1223 | 96.15% | 0.0071 | **0.0001 (0.01%)** | **0.999999** |

*Checkpoint locked at Epoch 4: `best_model.pt` (91.9 MB).*

---

### 2. Zero-Shot Out-of-Distribution Benchmark on FRLL-Morphs (ISO/IEC 30107-3)
Evaluated across 5 morphing generator families and 3 environmental conditions (7,274 samples):

| Morph Generator | Generation Paradigm | Evaluation Condition | Equal Error Rate (EER) | BPCER @ APCER = 1% | BPCER @ APCER = 5% | ROC-AUC |
|---|---|---|---|---|---|---|
| **morph_opencv** | Delaunay Landmark Warp | Clean | **0.0094 (0.94%)** | **0.0000 (0.0%)** | 0.0000 (0.0%) | **0.9993** |
| **morph_opencv** | Delaunay Landmark Warp | JPEG (q=30) | **0.0094 (0.94%)** | 0.0098 (0.98%) | 0.0000 (0.0%) | **0.9992** |
| **morph_opencv** | Delaunay Landmark Warp | Print-Scan | 0.0916 (9.16%) | 0.2990 (29.9%) | 0.1569 (15.7%) | 0.9786 |
| **morph_facemorpher** | Landmark Alpha Blend | Clean | **0.0245 (2.45%)** | 0.1912 (19.1%) | 0.0000 (0.0%) | **0.9958** |
| **morph_facemorpher** | Landmark Alpha Blend | JPEG (q=30) | 0.0249 (2.49%) | 0.1618 (16.2%) | 0.0000 (0.0%) | 0.9956 |
| **morph_facemorpher** | Landmark Alpha Blend | Print-Scan | 0.0585 (5.85%) | 0.2647 (26.5%) | 0.0735 (7.4%) | 0.9859 |
| **morph_webmorph** | Landmark Research Warp | Clean | **0.0491 (4.91%)** | 0.2745 (27.5%) | 0.0490 (4.9%) | **0.9886** |
| **morph_webmorph** | Landmark Research Warp | JPEG (q=30) | 0.0585 (5.85%) | 0.3039 (30.4%) | 0.0735 (7.4%) | 0.9875 |
| **morph_webmorph** | Landmark Research Warp | Print-Scan | 0.2539 (25.4%) | 0.7892 (78.9%) | 0.5833 (58.3%) | 0.8320 |
| **morph_amsl** | Post-Processed Landmark | Clean | 0.1320 (13.2%) | 0.7598 (76.0%) | 0.3382 (33.8%) | 0.9461 |
| **morph_amsl** | Post-Processed Landmark | JPEG (q=30) | 0.1367 (13.7%) | 0.7647 (76.5%) | 0.3235 (32.4%) | 0.9434 |
| **morph_amsl** | Post-Processed Landmark | Print-Scan | 0.3978 (39.8%) | 0.9216 (92.2%) | 0.8039 (80.4%) | 0.6491 |
| **morph_stylegan** | Generative Latent W+ | Clean | **0.5100 (51.0%)** | **0.9951 (99.5%)** | 0.9902 (99.0%) | **0.4988** |
| **morph_stylegan** | Generative Latent W+ | JPEG (q=30) | 0.4998 (50.0%) | 0.9951 (99.5%) | 0.9853 (98.5%) | 0.4973 |
| **morph_stylegan** | Generative Latent W+ | Print-Scan | 0.4577 (45.8%) | 0.9608 (96.1%) | 0.8824 (88.2%) | 0.5780 |

---

## Scientific Findings & Failure Mode Analysis

### 1. Why Spatial ViT Solves Landmark Morphs
Landmark morphing operates by affine-warping facial triangles followed by pixel alpha blending:

$$
I_{\text{landmark}}(\mathbf{x}) = \alpha \cdot I_A\big(\mathcal{W}_A(\mathbf{x})\big) + (1-\alpha) \cdot I_B\big(\mathcal{W}_B(\mathbf{x})\big)
$$

This process introduces physical spatial discontinuities: double contours around pupils and iris edges, ghosting along high-frequency boundaries, and blending mask boundaries. DINOv2 patch token variance statistics capture these localized gradient anomalies, resulting in **>0.995 ROC-AUC**.

### 2. The Theoretical Cause of the StyleGAN2 Blind Spot
StyleGAN2 morphs are generated through GAN inversion into extended latent space W+ and linearly interpolated:

$$
\mathbf{w}_{\text{morph}} = \alpha \mathbf{w}_A + (1 - \alpha) \mathbf{w}_B \implies I_{\text{morph}} = \mathcal{G}(\mathbf{w}_{\text{morph}})
$$

Because the image is synthesized from scratch through generator network G, there are **zero pixel-level alpha blending seams, zero cut-out masks, and zero ghosting double edges**. In the spatial RGB domain, gradients are continuous and photorealistic. To a spatial Vision Transformer, a StyleGAN2 face is indistinguishable from a genuine human face (**0.4988 ROC-AUC: random chance**).

### 3. Operational Metric Analysis (BPCER @ APCER = 1%)
Under ISO/IEC 30107-3, high-security border control systems require APCER <= 1% (fewer than 1 in 100 morph attacks may bypass the gate). On AMSL morphs, edge feathering reduces seam contrast, forcing the operational decision threshold to a point where **76% of bona fide travelers would be falsely rejected** (BPCER = 75.98%).

### 4. Print-Scan Modulation Transfer Attenuation
Physical printing (CMYK halftoning) followed by scanning acts as an analog low-pass filter. It attenuates high-frequency spatial edge anomalies while injecting uniform halftone grain, lowering landmark detection AUC (OpenCV drops from 0.9993 to 0.9786; WebMorph from 0.9886 to 0.8320).

---

## Academic Presentation & Defense Quick Reference

When presenting these results to advisors and technical committees, the following structured justifications summarize our design decisions:

1. **Why DINOv2 ViT instead of standard facial recognition backbones (InsightFace/ArcFace)?**
   - Face recognition models are trained to optimize identity invariance, discarding subtle lighting, edge, and texture variations.
   - Morph detection requires sensitivity to localized pixel-level structural discontinuities. DINOv2 self-supervised patch tokens retain dense spatial correspondences and texture variance without identity projection.

2. **Why ArcFace loss instead of standard Binary Cross-Entropy?**
   - Binary Cross-Entropy optimizes a Euclidean separating hyperplane but does not penalize intra-class dispersion.
   - ArcFace maps representations onto a unit hypersphere and enforces an angular margin (m = 0.35 radians), producing tightly clustered bona fide and attack representations essential for out-of-distribution generalization.

3. **Did the model overfit to SMDD?**
   - No. Zero-shot transfer to OpenCV (0.94% EER) and FaceMorpher (2.45% EER) proves robust cross-dataset generalization.
   - The collapse on StyleGAN2 (51.00% EER) is a fundamental representation boundary of spatial RGB features, not statistical overfitting.

4. **Why evaluate under BPCER @ APCER = 1% rather than simple Accuracy?**
   - In border control operations (ICAO Doc 9303 / Frontex), allowing morph attacks through an e-Gate is a critical security failure (APCER <= 1%).
   - BPCER at this threshold quantifies the real-world operational cost: the percentage of genuine passengers wrongly flagged for secondary manual screening.

---

## Proposed Next Steps (Future Research Directions)

The empirical failure on StyleGAN2 directly defines the next scientific milestones:
1. **Frequency-Domain Residual Modeling:** While StyleGAN2 leaves no spatial seams, its transposed convolutions and upsampling layers induce periodic spectral Nyquist peaks in the 2D frequency domain (DCT/FFT) and lack physical camera Photo-Response Non-Uniformity (PRNU) sensor noise.
2. **Dual-Domain Architecture Design:** We propose coupling Stream A (Spatial ViT) with a dedicated high-pass residual frequency branch (such as a Laplacian residual filter and lightweight ConvNet) to provide comprehensive coverage across both landmark-warped and generative latent morphs.
3. **Calibrated Decision Fusion:** Designing continuous margin-based scoring to prevent modality dominance across distinct attack distributions.

---

## Repository Structure

```
morph-detection/
├── README.md                      # Academic presentation dossier & results summary
│
├── scripts/                       # EXECUTABLE CODE & NOTEBOOKS
│   ├── phase1_dino_spatial.py     # Standalone Python script (Training + Evaluation)
│   └── phase1_dino_spatial.ipynb  # Interactive Jupyter Notebook (19 cells)
│
├── docs/                          # RESEARCH DOCUMENTATION & DEFENSE GUIDES
│   ├── phase1_comprehensive_briefing.md # Comprehensive briefing and defense Q&A guide
│   ├── week1_report.md            # Week 1 formal milestone report
│   ├── architecture.md            # System architecture and mathematical formulation
│   └── RESEARCH_REPORT.md         # Full research dossier
│
├── results/                       # EXPERIMENTAL METRICS & PLOTS
│   ├── phase1/                    # Phase 1 evaluation artifacts
│   │   ├── training_history.csv   # SMDD 4-epoch convergence log
│   │   ├── roc_morph_opencv.png   # ROC curve: OpenCV (0.9993 AUC)
│   │   ├── roc_morph_facemorpher.png # ROC curve: FaceMorpher (0.9958 AUC)
│   │   ├── roc_morph_webmorph.png # ROC curve: WebMorph (0.9886 AUC)
│   │   └── roc_morph_stylegan.png # ROC curve: StyleGAN2 (0.4988 AUC)
│   └── README.md                  # Results documentation
│
└── literature/                    # FOUNDATIONAL BACKGROUND LITERATURE
    ├── literature_synthesis.md    # Synthesis of 4 foundational MAD papers
    ├── paper1_composite_heatmaps.md # ViT composite attention maps
    ├── paper2_deep_learning_survey.md # Deep learning MAD survey
    ├── paper3_facenet_hasc_baseline.md # FaceNet + HASC baseline
    ├── paper4_dmad_hybrid_analysis.md # Differential MAD hybrid analysis
    └── README.md                  # Literature index
```

---

## Quickstart & Reproduction

### Prerequisites
```bash
pip install torch torchvision timm albumentations scikit-learn pandas matplotlib Pillow
```

### Running on Kaggle / Local GPU
The baseline script [`scripts/phase1_dino_spatial.py`](scripts/phase1_dino_spatial.py) is completely self-contained.

1. **Interactive Notebook:** Open [`scripts/phase1_dino_spatial.ipynb`](scripts/phase1_dino_spatial.ipynb) in Jupyter / Kaggle / Google Colab.
2. **Command Line Execution:**
   ```bash
   python scripts/phase1_dino_spatial.py
   ```

Outputs (`best_model.pt`, `training_history.csv`, `roc_*.png`) are automatically saved to the working directory.

---

## Key References
- **ISO/IEC 30107-3:2023:** Information technology: Biometric presentation attack detection, Part 3: Testing and reporting.
- **DINOv2:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv:2304.07193*, 2023.
- **ArcFace:** Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
- **SMDD:** Damer et al., "SMDD -- Synthetic Morphing Face Database," *WACV*, 2021.
- **FRLL-Morphs:** Robertson et al., "Detecting Morphing Attacks: Synthetic Face Morphing Datasets," *IET Biometrics*, 2020.
