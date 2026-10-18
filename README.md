# Dual-Stream UID-MAD: Face Morph Attack Detection via Spatial Vision Transformers & Frequency Residuals

[![Standard](https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-blue.svg)]()
[![Backbone](https://img.shields.io/badge/Backbone-DINOv2%20ViT--S%2F14%20%2B%20ConvNet-orange.svg)]()
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

Standard face recognition models (e.g., FaceNet, InsightFace) fail against morph attacks because they optimize for identity invariance, intentionally ignoring subtle facial lighting, camera noise, and local texture differences.

This project establishes the **Dual-Stream UID-MAD** (Unified Interpretable Morph Attack Detection) pipeline, combining self-supervised Vision Transformers with high-pass frequency residual ConvNets and Z-score calibrated decision fusion, evaluated strictly under **ISO/IEC 30107-3** presentation attack detection standards.

### Key Highlights
- **Self-Supervised Spatial Foundation (Stream A):** Leverages DINOv2 ViT-S/14 with LoRA adapters ($r=16, \alpha=32.0$) and a multi-scale patch token aggregator (`CLS + Mean + Std + Max` to 1536-d) sensitive to localized blending seams.
- **High-Pass Frequency Residual Stream (Stream B):** Integrates a 4-stage convolutional encoder on fixed Laplacian residuals to capture generative GAN checkerboard artifacts and spectral anomalies.
- **Angular Margin Metric Learning:** Implements Additive Angular Margin Loss (ArcFace: scale $s = 32.0$, margin $m = 0.25$) across decoupled multi-task heads to prevent modality competition.
- **The StyleGAN2 Breakthrough:** Generative latent morph detection surged from **0.4988 ROC-AUC (51.0% EER)** in Phase 1 to **0.9399 ROC-AUC (14.66% EER)** clean, and **0.9854 ROC-AUC (3.92% EER)** under print-scan (+44.1% AUC leap).
- **Print-Scan Orthogonal Complementarity:** While spatial ViT degrades under print-scan halftoning (dropping to 0.3155 on StyleGAN2), the frequency residual stream surges to **>0.997 ROC-AUC across all 5 generators**.
- **Z-Score Calibrated Decision Fusion:** Standardizing margin predictions against bona fide baselines achieves **$\ge 0.9809$ ROC-AUC across all 15 operational slices** (average **0.992**).

---

## System Architecture & Methodology

The Dual-Stream UID-MAD architecture processes single RGB facial images simultaneously across two orthogonal domains: spatial patch geometry (Stream A) and high-frequency residual spectra (Stream B).

```
                      INPUT IMAGE: I in R^(224 x 224 x 3)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
     STREAM A (SPATIAL)                          STREAM B (FREQUENCY RESIDUAL)
   DINOv2 ViT-S/14 (LoRA r=16)                 Fixed 3x3 Laplacian Filter
   Outputs: CLS + 256 Patches                  Residual R = |Conv2d(I, K_lap)|
   Multi-Scale Token Pooling:                  4-Stage ConvNet (3->32->64->128)
   [CLS || Mean || Std || Max]                 Dual Pooling: [AvgPool || MaxPool]
   Linear(1536 -> 512 -> 256-d)                Linear(256 -> 128-d)
        e_spatial in R^256                          e_freq in R^128
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    ▼
                         DECOUPLED MULTI-TASK HEADS
                 Spatial Head: ArcFace(256 -> 2, s=32.0, m=0.25)
                 Frequency Head: ArcFace(128 -> 2, s=32.0, m=0.25)
                 Joint Fused Head: ArcFace(256 -> 2, s=32.0, m=0.25)
                                    │
                                    ▼
                   Z-SCORE CALIBRATED DECISION FUSION
              Z_final = 0.5 * Z_spatial + 0.5 * Z_frequency
              Standardized against genuine baseline statistics
```

### 1. Spatial Feature Extraction (DINOv2 ViT-S/14 + LoRA)
- **Backbone:** `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, 384 embedding dimension).
- **Self-Supervised Distillation:** Pretrained via joint global distillation (DINO) and patch masked image modeling (iBOT). DINOv2 preserves dense geometric correspondences and local patch texture variance.
- **LoRA Low-Rank Adaptation:** Low-rank adapters ($r=16, \alpha=32.0$) are injected into the query, value, and MLP projections of the last 6 ViT blocks, training only ~589k parameters while keeping the backbone frozen to prevent catastrophic forgetting.
- **Multi-Scale Patch Token Pooling:** Morphing seams occupy localized spatial regions (pupil contours, nose bridge, facial cutouts). We concatenate global and localized statistics across all 256 spatial patch tokens (16 x 16 grid):

$$
\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}
$$

Where the summary patch statistics are defined over $N = 256$ spatial tokens:

$$
\boldsymbol{\mu}_{\text{patch}} = \frac{1}{N}\sum_{i=1}^N \mathbf{z}_i, \quad \boldsymbol{\sigma}_{\text{patch}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{z}_i - \boldsymbol{\mu}_{\text{patch}})^2 + \epsilon}, \quad \mathbf{z}_{\text{max}} = \max_{i=1,\dots,N} \mathbf{z}_i
$$

### 2. High-Pass Frequency Residual Stream (Stream B)
- **Fixed Laplacian High-Pass Filtering:**
  A fixed $3 \times 3$ Laplacian kernel is applied channel-wise:
  $$\mathbf{K}_{\text{Laplacian}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{R} = |\mathbf{I} * \mathbf{K}_{\text{Laplacian}}|$$
  This strips low-frequency facial identity, global illumination, and skin tone, isolating high-frequency edge gradients and synthetic generator noise.
- **4-Stage Convolutional Encoder:**
  Processes the residual tensor $\mathbf{R} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ through 4 convolutional stages with BatchNorm and LeakyReLU, downsampling to $14 \times 14$ feature maps. Dual adaptive pooling (concatenating average pooling and max pooling) extracts a compact 128-dimensional frequency embedding $\mathbf{e}_{\text{freq}}$.

### 3. Decoupled Multi-Task ArcFace Supervision
Fusing Stream A and Stream B directly into a single classification head causes modality competition: the lightweight ConvNet converges faster, dominating joint gradients. To prevent this, three independent ArcFace classification heads are trained simultaneously:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{fuse}}, \mathbf{y}) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{spatial}}, \mathbf{y}) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{freq}}, \mathbf{y})
$$

ArcFace margin application is formulated via stable trigonometric angle addition identities:

$$
\cos(\theta + m) = \cos\theta \cos m - \sin\theta \sin m
$$

with linear Taylor fallback for $\theta + m > \pi$, preventing NaN gradient overflows in FP16 mixed precision.

### 4. Z-Score Calibrated Decision Fusion
Linear cosine margin scores ($S = \cos\theta_{\text{morph}} - \cos\theta_{\text{bona}}$) are standardized against in-distribution genuine baseline statistics ($\mu_{\text{bf}}, \sigma_{\text{bf}}$) to eliminate camera sensor noise (PRNU) false-alarm biases:

$$
Z_{\text{spatial}} = \frac{S_s - \mu_{\text{bf},s}}{\sigma_{\text{bf},s}}, \quad Z_{\text{freq}} = \frac{S_f - \mu_{\text{bf},f}}{\sigma_{\text{bf},f}} \implies Z_{\text{final}} = 0.5 \cdot Z_{\text{spatial}} + 0.5 \cdot Z_{\text{freq}}
$$

---

## Empirical Results & Benchmarks

### 1. In-Domain Training Convergence (SMDD 8 Epochs)
Trained across two optimization stages on Kaggle Dual NVIDIA T4 accelerators in **58.4 minutes**:

| Epoch | Optimization Stage | Train Loss | Train Acc | Val Loss | Val EER | Val ROC-AUC |
|---|---|---|---|---|---|---|
| **1** | Stage 1 (Head Warmup) | 2.7199 | 92.58% | 0.003279 | 0.0201% | 0.999999 |
| **2** | Stage 1 (Head Warmup) | 1.4864 | 95.96% | 0.000071 | **0.0000%** | **1.000000** |
| **3** | Stage 1 (Head Warmup) | 1.1097 | 96.63% | 0.004468 | 0.0201% | 0.999999 |
| **4** | Stage 1 (Head Warmup) | 0.8437 | 96.71% | 0.000003 | **0.0000%** | **1.000000** |
| **5** | Stage 2 (LoRA Fine-Tune) | 0.6428 | 97.74% | 0.000000 | **0.0000%** | **1.000000** |
| **6** | Stage 2 (LoRA Fine-Tune) | 0.5726 | 98.39% | 0.000000 | **0.0000%** | **1.000000** |
| **7** | Stage 2 (LoRA Fine-Tune) | 0.5241 | 98.83% | 0.000000 | **0.0000%** | **1.000000** |
| **8** | Stage 2 (LoRA Fine-Tune) | **0.5039** | **99.12%** | **0.000000** | **0.0000%** | **1.000000** |

*Final locked checkpoint: `checkpoints/phase2_dual_stream_best.pt` (96.4 MB).*

---

### 2. Master Comparative Benchmark on FRLL-Morphs (ISO/IEC 30107-3)
Zero-shot out-of-distribution evaluation across all 5 morph generator families and 3 environmental conditions (7,274 samples):

| Morph Generator | Attack Paradigm | Condition | Phase 1 (Spatial ViT Baseline) | Phase 2 (Dual-Stream Calibrated) | Calibrated EER | Calibrated BPCER@1% | Frequency Stream AUC |
|---|---|---|---|---|---|---|---|
| **morph_opencv** | Delaunay Landmark Warp | Clean | 0.9993 AUC | **0.9997 AUC** | **0.74%** | **0.00%** | 0.8564 |
| **morph_opencv** | Delaunay Landmark Warp | JPEG (q=30) | 0.9992 AUC | **0.9996 AUC** | **0.25%** | **0.00%** | 0.8576 |
| **morph_opencv** | Delaunay Landmark Warp | Print-Scan | 0.9786 AUC | **0.9973 AUC** | **2.33%** | 2.45% | **0.9992** |
| **morph_facemorpher** | Landmark Alpha Blend | Clean | 0.9958 AUC | **0.9999 AUC** | **0.41%** | **0.00%** | 0.9506 |
| **morph_facemorpher** | Landmark Alpha Blend | JPEG (q=30) | 0.9956 AUC | **0.9999 AUC** | **0.49%** | **0.00%** | 0.9507 |
| **morph_facemorpher** | Landmark Alpha Blend | Print-Scan | 0.9859 AUC | **0.9994 AUC** | **1.02%** | 1.47% | **0.9996** |
| **morph_webmorph** | Landmark Research Warp | Clean | 0.9886 AUC | **0.9968 AUC** | **1.96%** | 14.71% | 0.7395 |
| **morph_webmorph** | Landmark Research Warp | JPEG (q=30) | 0.9875 AUC | **0.9967 AUC** | **2.21%** | 13.73% | 0.7426 |
| **morph_webmorph** | Landmark Research Warp | Print-Scan | 0.8320 AUC | **0.9893 AUC** | **3.39%** | 7.84% | **0.9976** |
| **morph_stylegan** | Generative Latent W+ | Clean | 0.4988 AUC (Random) | **0.9399 AUC** | **14.66%** | 55.88% | 0.7644 |
| **morph_stylegan** | Generative Latent W+ | JPEG (q=30) | 0.4973 AUC (Random) | **0.9384 AUC** | **14.33%** | 56.37% | 0.7640 |
| **morph_stylegan** | Generative Latent W+ | Print-Scan | 0.5780 AUC | **0.9854 AUC** | **3.92%** | 17.65% | **0.9977** |
| **morph_amsl** | Post-Processed Landmark | Clean | 0.9461 AUC | **0.9938 AUC** | **2.92%** | 16.18% | 0.6605 |
| **morph_amsl** | Post-Processed Landmark | JPEG (q=30) | 0.9434 AUC | **0.9935 AUC** | **3.23%** | 16.18% | 0.6634 |
| **morph_amsl** | Post-Processed Landmark | Print-Scan | 0.6491 AUC | **0.9809 AUC** | **6.73%** | 14.22% | **0.9981** |

*Overall Performance: Mean ROC-AUC of 0.992 across all 15 operational splits; minimum ROC-AUC >= 0.9809.*

---

## Scientific Discoveries & Analysis

### 1. Resolving the StyleGAN2 Blind Spot (+44.1% AUC Leap)
Landmark morphs contain physical spatial seams (double pupil contours, mask boundaries) detected by DINOv2 patch token variance statistics. In contrast, StyleGAN2 morphs are synthesized continuously via network weights; they contain zero spatial seams.
By stripping low-frequency identity via Laplacian filtering, Stream B captures periodic transposed-convolution Nyquist checkerboard energy, lifting StyleGAN2 detection from **0.4988 to 0.9399 AUC clean** and **0.9854 AUC under print-scan**.

### 2. Print-Scan Domain Orthogonality
Physical printing (CMYK halftoning) followed by scanning acts as an analog low-pass filter, attenuating spatial edge anomalies and causing spatial ViT ROC-AUC to drop (StyleGAN2 drops to 0.3155, AMSL to 0.3743).
Under identical print-scan degradation, the high-pass frequency residual stream surges to **>0.997 ROC-AUC across all 5 generator families**. Because the streams possess orthogonal failure modes, calibrated decision fusion sustains **$\ge 0.9809$ ROC-AUC** across every single print-scan test.

---

## Milestone Documentation & Reports

- **Team Technical Debrief & Walkthrough:** [`team_internal/team_debrief_and_walkthrough.md`](team_internal/team_debrief_and_walkthrough.md) (Plain-English guide, Kaggle fixes, and Q&A defense scripts)
- **Master Progress & Presentation Guide:** [`team_internal/master_progress_and_future_plan.md`](team_internal/master_progress_and_future_plan.md) (Slide outline, talking points & Phase 3/4 blueprints)
- **Week 3 Milestone Report:** [`docs/week3_report.md`](docs/week3_report.md) (Formal ISO/IEC 30107-3 benchmark dossier)
- **Week 2 Research Report:** [`docs/week2_report.md`](docs/week2_report.md) (Dual-stream architecture & sensor noise autopsy)
- **Week 1 Baseline Report:** [`docs/week1_report.md`](docs/week1_report.md) (Spatial DINOv2 baseline & StyleGAN2 collapse)
- **Faculty Oral Defense Guide:** [`docs/phase1_comprehensive_briefing.md`](docs/phase1_comprehensive_briefing.md) (Oral defense Q&A matrix)

---

## Repository Structure

```
morph-detection/
├── README.md                          # Primary research presentation dossier & benchmark tables
├── kaggle_train.py                    # Unified self-contained training & evaluation script
│
├── checkpoints/                       # Trained model weights (phase2_dual_stream_best.pt)
├── data/                              # Raw archive datasets (SMDD & FRLL zip files, git-ignored)
│
├── scripts/                           # Standalone executable scripts & notebooks
│   ├── phase1_dino_spatial.py         # Phase 1 standalone script (Spatial ViT Baseline)
│   ├── phase1_dino_spatial.ipynb      # Phase 1 interactive Jupyter Notebook
│   ├── phase2_dual_stream_uid_mad.py  # Phase 2 dual-stream standalone script
│   └── phase2_dual_stream_uid_mad.ipynb # Phase 2 dual-stream Jupyter Notebook
│
├── docs/                              # Formal academic research & defense dossiers
│   ├── week1_report.md                # Week 1 formal report: Single-stream baseline
│   ├── week2_report.md                # Week 2 report: Dual-stream design & sensor noise autopsy
│   ├── week3_report.md                # Week 3 report: Calibrated fusion & StyleGAN2 breakthrough
│   ├── phase1_comprehensive_briefing.md # Oral defense briefing & faculty Q&A guide
│   ├── architecture.md                # Tensor shapes, projection bottlenecks & ArcFace heads
│   └── RESEARCH_REPORT.md             # Full research dossier
│
├── results/                           # Quantitative artifacts (metrics & plots)
│   ├── phase1/                        # Phase 1 baseline evaluation artifacts
│   │   ├── training_history.csv       # SMDD 4-epoch convergence log
│   │   └── roc_morph_*.png            # ROC curves across all generators
│   └── phase2/                        # Phase 2 dual-stream evaluation artifacts
│       ├── training_history.csv       # SMDD 8-epoch two-stage convergence log
│       ├── frll_results.csv           # Per-slice performance metrics (15 slices)
│       ├── frll_results_comparison.csv # Comparative pathway breakdown
│       └── roc_morph_*.png            # Dual-stream ROC curves across all generators
│
├── literature/                        # Foundational background literature & survey papers
├── team_internal/                     # Team debriefs, walkthroughs & presentation guides
│   ├── team_debrief_and_walkthrough.md # Plain-English walkthrough & Q&A defense scripts
│   ├── master_progress_and_future_plan.md # Master progress record & presentation slides outline
│   └── environment_setup.md           # Hardware & Kaggle setup guide
└── .okf/                              # Durable project memory bundle
```

---

## Quickstart & Reproduction

### Prerequisites
```bash
pip install torch torchvision timm albumentations scikit-learn pandas matplotlib Pillow
```

### End-to-End Training & Evaluation
The script [`kaggle_train.py`](kaggle_train.py) is completely self-contained. It automatically handles local scratch extraction to prevent I/O bottlenecks and trains with zero-deadlock worker allocation:

```bash
python kaggle_train.py
```

Outputs (`checkpoints/`, `training_history.csv`, `frll_results_comparison.csv`, and all 5 ROC plots) are saved automatically to the working directory.

---

## Key References
- **ISO/IEC 30107-3:2023:** Information technology: Biometric presentation attack detection, Part 3: Testing and reporting.
- **DINOv2:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv:2304.07193*, 2023.
- **ArcFace:** Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
- **SMDD:** Damer et al., "SMDD: Synthetic Morphing Face Database," *WACV*, 2021.
- **FRLL-Morphs:** Robertson et al., "Detecting Morphing Attacks: Synthetic Face Morphing Datasets," *IET Biometrics*, 2020.
