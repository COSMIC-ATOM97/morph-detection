# UID-MAD: Unified Interpretable Dual-Stream Face Morph Attack Detection

[![ISO/IEC 30107-3 Compliant](https://img.shields.io/badge/Standard-ISO%2FIEC%2030107--3-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Dual%20NVIDIA%20T4-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%2B%20DINOv2-orange.svg)]()

Research project on generalized Single-Image Morph Attack Detection (S-MAD) for Automated Border Control (ABC) e-Gates and biometric passport verification.

---

## 📑 Core Documentation Dossier

For a complete academic and technical review, consult the formal documentation:

1. **[RESEARCH_REPORT.md](RESEARCH_REPORT.md):** Master academic research dossier containing problem formulation, mathematical models, empirical progression across all phases, scientific diagnoses, and operational recommendations for defense.
2. **[phase1_report.md](phase1_report.md):** Phase 1 report establishing the single-stream spatial baseline (DINOv2 patch aggregator) and the scientific autopsy of why spatial models collapse on StyleGAN2 morphs.
3. **[phase2_report.md](phase2_report.md):** Phase 2 report detailing the dual-stream frequency breakthrough (Nyquist checkerboard detection), modality competition, and sensor noise domain shift analysis.
4. **[architecture.md](architecture.md):** System architecture specification covering tensor shapes, Laplacian kernel formulation, and multi-branch ArcFace metric learning.
5. **[plan.md](plan.md):** Living engineering plan detailing milestones, hyperparameter specifications, and verification stages.
6. **[compiled_result.md](compiled_result.md):** Synthesis of foundational MAD literature across four key survey and benchmark papers.

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
