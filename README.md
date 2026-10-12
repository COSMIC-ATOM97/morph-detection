# UID-MAD: Unified Interpretable Dual-Stream Face Morph Attack Detection

[![ISO/IEC 30107-3 Compliant](https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Dual%20NVIDIA%20T4-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%2B%20DINOv2-orange.svg)]()

Research project on generalized Single-Image Morph Attack Detection (S-MAD) for Automated Border Control (ABC) e-Gates and biometric passport verification.

---

## 📂 Project Organization

The repository is structured into distinct, modular directories:

```
morph-detection/
├── README.md                      # Project landing page & reproduction instructions
├── kaggle_train.py                # Standalone script for Kaggle GPUs (Dual T4)
│
├── docs/                          # 🎓 PRESENTABLE: Formal academic deliverables for advisor & thesis
│   ├── RESEARCH_REPORT.md         # Master Research Dossier & Academic Paper
│   ├── architecture.md            # System Architecture & Mathematical Formulation
│   ├── phase1_report.md           # Phase 1 Evaluation Report (Spatial ViT Baseline)
│   └── phase2_report.md           # Phase 2 Evaluation Report (Dual-Stream Frequency Autopsy)
│
├── results/                       # 📊 RESULTS: Empirical artifacts, metrics & plots
│   ├── phase1/                    # Phase 1 ROC curves & SMDD convergence
│   └── phase2/                    # Phase 2 FRLL evaluation matrix & ROC plots across 15 splits
│
├── literature/                    # 📚 NORMAL FILES: Foundational background research
│   ├── compiled_result.md         # Multi-paper synthesis & gap analysis
│   └── paper1.md - paper4.md      # Literature summaries & baseline extractions
│
└── team_internal/                 # 🔒 TEAM DATA: Internal sprint planning & handoff (collaborators only)
    ├── README.md                  # Team navigation guide
    ├── WEEK1_TEAM_HANDOFF.md      # Plain-English Week 1 onboarding & task roadmap
    ├── plan.md                    # Living 4-week project plan & milestone tracker
    └── prep.md                    # Scratchpad & environment setup notes
```

---

## 📑 Core Documentation Map

1. **Academic Review:** Start with the **[Master Research Report](docs/RESEARCH_REPORT.md)** and **[System Architecture](docs/architecture.md)**.
2. **Experimental Data:** Check **[Phase 1 Results](results/phase1/)** and **[Phase 2 Results](results/phase2/)**.
3. **Literature Background:** See **[Literature Synthesis](literature/compiled_result.md)**.
4. **Internal Team Notes:** See **[Week 1 Team Handoff](team_internal/WEEK1_TEAM_HANDOFF.md)** and **[Multi-Week Plan](team_internal/plan.md)**.

---

## 🔬 Scientific Summary & Results

UID-MAD addresses the generalization collapse of Morph Attack Detection systems by decoupling facial representation into two orthogonal domains:
- **Stream A (Spatial):** DINOv2 ViT-S/14 dense patch token summary statistics (`CLS + Mean + Std + Max` -> 1536-d) targeting boundary alpha-blending seams and iris contours.
- **Stream B (Frequency):** Fixed 3x3 Laplacian residual filter + 4-stage convolutional encoder (128-d) targeting generator upsampling Nyquist peaks.
- **Multi-Branch Head:** Independent ArcFace angular margin heads ($s=32.0, m=0.25$) supervised jointly to prevent modality competition.

### Out-of-Distribution Performance on FRLL-Morphs (Clean Condition)

| Generator Algorithm | Generation Paradigm | Phase 1 (Spatial Only) | Phase 2 (Trial 1: Early Concat) | Phase 2 (Trial 2: Multi-Task) | Primary Detection Pathway |
|---|---|---|---|---|---|
| **OpenCV** | Delaunay Landmark Warp | **0.9993 AUC** (0.94% EER) | 0.2422 AUC (Inverted) | 0.7924 AUC (22.8% EER) | Spatial Patch Variance |
| **FaceMorpher** | Landmark Alpha Blend | **0.9958 AUC** (2.45% EER) | 0.4105 AUC | 0.8295 AUC (19.4% EER) | Spatial Patch Variance |
| **WebMorph** | Landmark Research Warp | **0.9886 AUC** (4.91% EER) | 0.3890 AUC | 0.5444 AUC (46.0% EER) | Spatial Patch Variance |
| **AMSL** | Landmark Baseline | **0.9461 AUC** (13.2% EER) | 0.3540 AUC | 0.5145 AUC (48.6% EER) | Spatial Patch Variance |
| **StyleGAN2** | Latent $W^+$ Synthesis | 0.4988 AUC (Random) | **0.8525 AUC** (24.0% EER) | 0.5363 AUC (46.4% EER) | Frequency Residual ConvNet |

---

## 🚀 Quick Start & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.0+, `timm`, `albumentations`, `scikit-learn`, `pandas`
- Dual NVIDIA T4 or A100 GPU

### Running on Kaggle

The training and evaluation script [kaggle_train.py](kaggle_train.py) is entirely self-contained.

1. **Evaluate Existing Checkpoint (2 minutes):**
   ```bash
   python kaggle_train.py --eval-only
   ```
   Outputs complete cross-dataset metrics across all 15 splits to `frll_results_comparison.csv` and generates comparative multi-stream ROC plots.

2. **Full Training from Scratch (~14 minutes):**
   ```bash
   python kaggle_train.py
   ```
   Executes Stage 1 head warm-up (4 epochs) and Stage 2 differential fine-tuning (2 epochs) on SMDD, followed by automatic zero-shot evaluation on FRLL-Morphs.
