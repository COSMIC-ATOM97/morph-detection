# Team Handoff & Week 1 Summary Report

**Project:** Face Morph Attack Detection (UID-MAD)  
**Date:** Week 1 Handoff (October 12, 2026)  
**Target Audience:** Teammates, Project Collaborators & Academic Advisor  
**Hardware:** Kaggle Dual NVIDIA T4 (GPU T4 x2)

---

## 1. Project TL;DR (What We Are Building & Why)

### The Security Problem
A **face morph attack** occurs when two individuals (e.g., an accomplice with no criminal record and a criminal actor) blend their facial photographs into a single composite image. If that image is accepted on a passport or ID card, **both individuals can pass through Automated Border Control (ABC) e-Gates using the same physical passport**.

### Why Existing Methods Fail
Current systems trained on simple landmark morphs (OpenCV, FaceMorpher) collapse when tested on modern generative morphs (StyleGAN2) or under realistic passport upload conditions (JPEG compression, print-scan). Our goal is to build **UID-MAD (Unified Interpretable Dual-Stream Morph Attack Detection)**, an architecture that catches both landmark-warped morphs and AI-generated morphs without failing on unseen data.

---

## 2. What We Accomplished in Week 1

Here is the exact step-by-step chronology of what was researched, implemented, and discovered this week:

```
[Week 1 Chronology]
Step 1: Literature Synthesis (4 Papers) -> Identified Spatial vs Frequency Gaps
Step 2: Phase 1 Single-Stream DINOv2 -> Solved Landmark Morphs (0.999 AUC)
Step 3: Discovery of StyleGAN Blindness -> 0.498 AUC (Random Guess)
Step 4: Phase 2 Dual-Stream Design -> Built Laplacian High-Pass Residual Stream
Step 5: Overnight Kaggle Training -> Breakthrough on StyleGAN (0.852 AUC)
Step 6: Empirical Autopsy -> Identified Modality Competition & Sensor Noise Shift
Step 7: Harness Engineering -> Built Cosine-Margin Scoring & Fast 2-Min Eval Mode
```

### Detailed Breakthroughs & Discoveries

#### Milestone 1: Literature Synthesis (`compiled_result.md`)
We reviewed and cross-analyzed four foundational papers to identify why previous models fail:
- Baseline models overfit tiny datasets (achieving fake 0.0% EER on ~38 images) and fail under ISO/IEC 30107-3 cross-dataset testing.
- Spatial-only networks miss generative artifacts; post-hoc occlusion heatmaps (Grad-CAM) are too slow for real-time border gates.
- **Our Decision:** A dual-stream architecture (Spatial Vision Transformer + Frequency Residual ConvNet) trained with ArcFace margin loss.

#### Milestone 2: Phase 1 Single-Stream DINOv2 Baseline (`phase1_report.md`)
- We implemented **Stream A**: Frozen DINOv2 ViT-S/14 backbone + a multi-scale patch token aggregator (`CLS + Mean + Std + Max` -> 1536-d) + ArcFace classification head ($s=32.0, m=0.25$).
- Trained on SMDD (32,000 synthetic faces), evaluated strictly out-of-distribution on FRLL-Morphs across 5 generators.
- **The Good News:** Landmark-blended morphs were virtually solved in the spatial domain:
  - **OpenCV:** **0.9993 ROC-AUC** (0.94% EER)
  - **FaceMorpher:** **0.9958 ROC-AUC** (2.45% EER)
  - **WebMorph:** **0.9886 ROC-AUC** (4.91% EER)
  - Extreme robustness to JPEG $q=30$ compression (OpenCV AUC only dropped from 0.9993 to 0.9992).
- **The Failure Mode:** Complete collapse on StyleGAN2 morphs (**0.4988 ROC-AUC**, 51.0% EER).
- **Why this happened:** Landmark morphs use Delaunay triangulation and alpha blending, leaving double edges around pupils and boundary cut seams. StyleGAN2 interpolates latent codes in $W^+$ space and synthesizes from scratch; it produces **zero pixel-level boundary seams**, making it invisible to spatial patch statistics alone.

#### Milestone 3: Phase 2 Dual-Stream Frequency Implementation (`phase2_report.md`)
- We designed **Stream B**: A depthwise non-trainable 3x3 Laplacian high-pass filter that strips facial identity and isolates high-frequency generator noise, followed by a 4-stage ConvNet (128-d).
- **Trial 1 (Early Fusion):** Concatenating Stream A + Stream B into a single ArcFace head proved our frequency hypothesis: **StyleGAN2 detection surged from 0.4988 to 0.8525 ROC-AUC!**
- **Trial 1 Complication (Modality Competition):** Because the ConvNet converged much faster than the frozen ViT, it dominated the fusion head. Real camera photos (FRLL OpenCV morphs) have no generative upsampling noise, so Stream B misclassified them as bona fide, inverting landmark predictions (OpenCV dropped to 0.2422 AUC).
- **Trial 2 (Current Overnight Run in `results(1).zip`):** We decoupled the architecture into three independent ArcFace heads (`spatial_head`, `freq_head`, `fused_head`) supervised jointly under multi-task loss ($\mathcal{L}_{\text{fuse}} + 0.5 \mathcal{L}_s + 0.5 \mathcal{L}_f$).
  - In-domain SMDD convergence was virtually perfect: **99.30% accuracy, 0.01% validation EER, 0.9999 validation AUC**.
  - Landmark scores recovered significantly: **OpenCV reached 0.7924 AUC**, **FaceMorpher reached 0.8295 AUC**.

#### Milestone 4: Autopsy of the Overnight Run (`results_run2/`)
When analyzing the overnight results, we diagnosed two crucial phenomena:
1. **The Sensor Noise Domain Mismatch:** SMDD training images are entirely synthetic (pristine digital noise floor). FRLL bona fides are real studio camera photographs with physical sensor pattern noise (Photo-Response Non-Uniformity, PRNU). Stream B, trained only on synthetic faces, interpreted physical camera sensor grain as an anomaly.
2. **The Uncalibrated `max()` Trap:** During test inference, taking $P = \max(P_s, P_f, P_{\text{fuse}})$ with ArcFace scale $s=32.0$ allowed Stream B's sensor noise false alarms to contaminate the bona fide scores, collapsing the EER decision threshold to $1.003 \times 10^{-6}$.

#### Milestone 5: The Architecture Fix (Now in `kaggle_train.py`)
- We added `get_margin()` to `ArcFaceHead`, replacing saturated softmax probabilities with continuous cosine margin differences:
  $$\Delta \cos = \cos(\theta_{\text{morph}}) - \cos(\theta_{\text{bona}}) \in [-2, 2]$$
- We upgraded `evaluate_frll()` to evaluate every branch independently (`[Spatial]`, `[Frequency]`, `[Fused Head]`, `[Calibrated Ensemble]`), generating side-by-side comparison tables and multi-curve ROC plots.
- We added an `--eval-only` flag so teammates can run evaluations on Kaggle in just 2 minutes without re-training.

---

## 3. How Teammates Can Run the Code on Kaggle

All code is packaged in the single self-contained script [kaggle_train.py](kaggle_train.py).

### Fast Benchmark Evaluation (2 Minutes)
If you already have `best_model.pt` in `/kaggle/working` or uploaded to Kaggle:
```bash
python kaggle_train.py --eval-only
```
*(Or set `cfg.EVAL_ONLY = True` at line 136 in the script).*

**What this outputs in Kaggle `/kaggle/working`:**
- `frll_results_comparison.csv`: Side-by-side comparison of Spatial vs Frequency vs Fused vs Ensemble across all 15 splits.
- `roc_morph_*.png`: Comparative multi-curve ROC plots showing all streams on one figure.

### Full Training from Scratch (~14 Minutes)
If you want to train from scratch on Dual NVIDIA T4s:
```bash
python kaggle_train.py
```
*(Ensure Kaggle accelerator is set to **GPU T4 x2**).*

---

## 4. Key Files in This Repository

| File | Purpose | Who Should Read It |
|---|---|---|
| **[README.md](README.md)** | Project overview, badges, quickstart, and result highlights | Everyone / GitHub landing page |
| **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)** | **Master academic dossier** (theory, math, results, failure autopsy) | Professor / Advisor / Thesis |
| **[WEEK1_TEAM_HANDOFF.md](WEEK1_TEAM_HANDOFF.md)** | **This document:** Plain-English summary of Week 1 + tasks | All Teammates |
| **[phase1_report.md](phase1_report.md)** | Phase 1 baseline report (Spatial ViT + StyleGAN gap) | Modeling team |
| **[phase2_report.md](phase2_report.md)** | Phase 2 dual-stream report (Frequency residual + autopsy) | Modeling team |
| **[architecture.md](architecture.md)** | Formal tensor shapes, model definitions, and ArcFace loss | Code developers |
| **[plan.md](plan.md)** | Living project plan with multi-week roadmap | Project manager / Team lead |
| **[compiled_result.md](compiled_result.md)** | Summary of 4 foundational literature papers | Literature review / Background |
| **[kaggle_train.py](kaggle_train.py)** | Self-contained training & evaluation script | Everyone running code |
| **`phase2_results/`** | Tracked CSVs and ROC curve plots from the overnight run | Everyone checking results |

---

## 5. Next Weeks Roadmap & Team Task Breakdown

### Week 2: Calibration & Invariance Optimization (Immediate Priority)
- **Goal:** Reach simultaneous >0.99 AUC on landmark morphs (OpenCV) and >0.85 AUC on StyleGAN2.
- **Teammate Task 2.1:** Run `python kaggle_train.py --eval-only` on Kaggle to generate the standalone branch comparison table.
- **Teammate Task 2.2:** Implement Bona Fide Z-score normalization ($Z = (S - \mu_{\text{bf}}) / \sigma_{\text{bf}}$) so Stream B's real camera noise doesn't shift the threshold.
- **Teammate Task 2.3:** Add sensor noise simulation (subtle Gaussian grain) to Stream B's data augmentation to train frequency invariance on real camera noise.

### Week 3: Explainability & Attention Heatmaps
- **Goal:** Project single-pass visual heatmaps showing human border guards where the morph seam or artifact is located.
- **Task:** Extract DINOv2 self-attention tokens to highlight facial landmark blend zones (eyes, nose, mouth) in <50ms without Grad-CAM.

### Week 4: Final Paper Draft & Defense Prep
- **Goal:** Submission-ready paper in IEEE/IAPR format and slide deck for final review.
