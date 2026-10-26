# UID-MAD: IEEE LaTeX Project Setup and Compilation Guide

This directory contains the complete IEEE-format manuscript and assets for:
**"UID-MAD: Unified Interpretable Dual-Stream Face Morph Attack Detection via Spatial Foundation Transformers and Spectral Residuals"**  
*B.Tech Major Project Mid-Semester Evaluation (CSPC401) — NIT Kurukshetra*

---

## 1. Directory Structure

```text
ieee_report/
├── main.tex            # Complete IEEE manuscript source code
├── references.bib      # Complete BibTeX bibliography (13 citations)
├── figures/            # High-resolution ROC and evaluation figures
│   ├── roc_stylegan_phase1.png   # Phase 1 collapse on StyleGAN2 (0.4988 AUC)
│   ├── roc_stylegan_phase2.png   # Phase 2 breakthrough on StyleGAN2 (0.940 AUC)
│   ├── roc_opencv_phase2.png     # Phase 2 OpenCV benchmark curve (1.000 AUC)
│   ├── roc_facemorpher_phase2.png
│   ├── roc_webmorph_phase2.png
│   └── roc_amsl_phase2.png
└── README.md           # This compilation and deployment guide
```

---

## 2. Compiling Online (Overleaf — Recommended)

To compile this project on [Overleaf](https://www.overleaf.com):

1. **Option A: Direct Upload as ZIP (Easiest)**
   - Zip this `ieee_report` folder:
     ```bash
     zip -r ieee_report.zip ieee_report/
     ```
   - On Overleaf, click **New Project** $\to$ **Upload Project**.
   - Select `ieee_report.zip`.
   - Ensure `main.tex` is set as the main document.
   - Click **Recompile** (Default engine: `pdfLaTeX` or `XeLaTeX`).

2. **Option B: Manual Upload**
   - Create a **New Project** $\to$ **Blank Project** named `UID-MAD-IEEE`.
   - Copy-paste the content of `main.tex` into Overleaf's `main.tex`.
   - Create a file named `references.bib` and paste its contents.
   - Create a folder named `figures/` and upload the 6 PNG files from `ieee_report/figures/`.
   - Click **Recompile**.

> **Note:** Overleaf has `IEEEtran.cls` and all standard packages (`pgfplots`, `tikz`, `booktabs`, `amsmath`, `subcaption`) pre-installed. The native TikZ architecture diagram and PGFPlots bar charts compile cleanly without any external dependencies!

---

## 3. Compiling Locally (Linux / macOS / Windows)

If you have a local TeX Live or MacTeX installation:

```bash
cd ieee_report

# 1. First pass with pdflatex
pdflatex -interaction=nonstopmode main.tex

# 2. Compile BibTeX bibliography
bibtex main

# 3. Second and third passes to resolve cross-references and citations
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Alternatively, use `latexmk` for single-command automated compilation:
```bash
latexmk -pdf main.tex
```

The resulting `main.pdf` is fully compliant with IEEE conference/transactions formatting requirements.

---

## 4. Key Highlights of this IEEE Conversion

1. **Corrected Mathematical Equations:**
   - Fixed equation formatting glitches (e.g., `\none` artifact removed from patch maximum equation).
   - Formally defined Multi-Scale Spatial Token Aggregation ($\boldsymbol{\mu}_{\text{patch}}, \boldsymbol{\sigma}_{\text{patch}}, \mathbf{z}_{\max}$).
   - Formally formulated LoRA parameter-efficient adaptation on blocks 6--11.
   - Formally defined 2D Haar Wavelet HH subband, Laplacian residual bank, and ArcFace angular margin loss.
   - Formally stated all ISO/IEC 30107-3 biometric metric formulas (APCER, BPCER, EER, BPCER@APCER=1\%/5\%, ROC-AUC).
   - Explicitly defined the bona fide Z-score calibration in linear cosine margin space.

2. **Integrated Complete Results Data from `results/`:**
   - Full 4-epoch Phase 1 training convergence record on SMDD.
   - Full 8-epoch Phase 2 training convergence record on SMDD with exact validation losses.
   - Complete 15-slice FRLL-Morphs benchmark table with all 9 columns (Spatial, Frequency, Fused head, Calibrated AUC, EER, BPCER@1\%, BPCER@5\%, and validation loss).
   - Comprehensive architectural and fusion strategy ablation table (including early concat inversion and softmax saturation).

3. **High-Grade Vector Graphics & Figures:**
   - **Figure 1:** System Architecture Block Diagram rendered natively in vector TikZ.
   - **Figure 2:** Phase 1 vs. Phase 2 Clean ROC-AUC Comparison bar chart rendered in native PGFPlots.
   - **Figure 3:** Print-Scan Stream Orthogonality bar chart rendered in native PGFPlots.
   - **Figure 4:** Calibrated EER across Degradation Conditions bar chart rendered in native PGFPlots.
   - **Figure 5:** Multi-panel subfigures featuring the official empirical ROC curves (StyleGAN2 Phase 1 vs Phase 2, and OpenCV benchmark).

4. **Preserved Content & Non-AI Voice:**
   - 100% of the project's factual findings, generator specifics (`morph_opencv`, `morph_facemorpher`, `morph_webmorph`, `morph_amsl`, `morph_stylegan`), hardware settings (dual NVIDIA T4, Kaggle, `NUM_WORKERS=0`), and author explanations are preserved without AI generic rewrites.
