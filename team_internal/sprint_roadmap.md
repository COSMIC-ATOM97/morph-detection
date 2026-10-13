# Implementation Plan: DINO-Branch Morph Attack Detection (Kaggle Baseline)

**Target:** Single-stream foundation model baseline for Morph Attack Detection (MAD) trained on SMDD and evaluated cross-dataset on FRLL-Morphs.  
**Scope Decision:** DINO branch only (Stream A: self-supervised ViT backbone + ArcFace classification head). All secondary streams deferred.

---

## 1. Dataset Inventory (Verified)

### SMDD (`SMDD_release_train.zip` -- 4.54 GB)
- **Role:** Training + in-domain validation.
- **40,000** synthetic face images at **256x256 PNG**.
  - Bonafide (`os25k_bf_t/`): **25,000**.
  - Morph attacks (`m15k_t/`): **15,000**.
- **Metadata:** `train.csv` with columns `image_path,label` (`bonafide` | `attack`).
- **Class ratio:** 5:3 bonafide-to-morph. Handled via weighted random sampler during training.

### FRLL-Morphs (`drive-download-20261010T174433Z-1-001.zip` -- 327 MB)
- **Role:** Strict OOD cross-dataset and cross-algorithm test set. Never seen during training.
- **7,274** real images in JPEG:
  - Bonafide (`raw/`): **204** images (1350x1350).
  - OpenCV morphs (`morph_opencv/`): **1,221** (431x513).
  - FaceMorpher (`morph_facemorpher/`): **1,222** (431x513).
  - WebMorph (`morph_webmorph/`): **1,222** (413x531).
  - StyleGAN2 (`morph_stylegan/`): **1,225** (1024x1024).
  - AMSL (`morph_amsl/`): **2,180** (413x531).
- **Protocols:** 795 `.lst` split files in `protocols/`.

---

## 2. Architecture

```
Input (224x224 RGB)
      │
      ▼
┌─────────────────────────────┐
│  DINOv2 ViT-S/14 Backbone   │
│  (frozen or last-N unfreeze) │
│  Output: CLS token (384-d)  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Projection + BN + L2 Norm  │
│  (384 → 256-d embedding)    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  ArcFace Margin Head        │
│  s=30.0, m=0.35, 2 classes  │
│  Loss: CrossEntropy on      │
│    s * cos(theta + m)       │
└──────────────┬──────────────┘
               │
               ▼
      Morph score P in [0,1]
```

- **Default backbone:** `facebook/dinov2-small` (public, no gating, works on Kaggle directly).
- **Optional upgrade:** DINOv3 via `--backbone dinov3` with HuggingFace token.

---

## 3. Training Protocol

### Stage 1 -- Head Warm-up (Epochs 1-3)
- Backbone fully frozen.
- Train projection + ArcFace head only.
- AdamW, lr=1e-3, weight_decay=1e-4.

### Stage 2 -- Differential Fine-tuning (Epochs 4-10)
- Unfreeze last 2 transformer blocks.
- Backbone lr=1e-5, Head lr=1e-4.
- Cosine annealing schedule.
- Mixed precision (AMP) throughout.

### Class Balance
- WeightedRandomSampler during training to equalize bonafide/morph sampling rates (compensates 5:3 imbalance in SMDD).

### Augmentations (Matching architecture.md spec)
- **JPEG compression:** quality 20-70 (extreme low end per architecture spec).
- **Gaussian blur:** kernel 3-7, sigma 0.1-2.0.
- **Motion blur:** kernel 3-7.
- **Print-scan simulation:** downsample 0.5x + JPEG q=40 + Gaussian noise sigma=5-15 + upsample back.
- **Random gamma/brightness** shift.
- **Random horizontal flip** (p=0.5).
- **Color jitter** (brightness, contrast, saturation, hue).
- Resize to 224x224 + ImageNet normalization.

---

## 4. Evaluation Protocol (ISO/IEC 30107-3)

### In-Domain Validation
- 20% stratified holdout from SMDD (5,000 bonafide, 3,000 morph).
- Checkpoint selection: lowest validation EER.

### Cross-Dataset Evaluation (OOD -- No Fine-tuning)
Best checkpoint evaluated on FRLL-Morphs:

| Test Split | Bonafide Source | Attack Source | Attack Count |
|---|---|---|---|
| FRLL-OpenCV | `raw/` (204) | `morph_opencv/` | 1,221 |
| FRLL-FaceMorpher | `raw/` (204) | `morph_facemorpher/` | 1,222 |
| FRLL-WebMorph | `raw/` (204) | `morph_webmorph/` | 1,222 |
| FRLL-StyleGAN2 | `raw/` (204) | `morph_stylegan/` | 1,225 |
| FRLL-AMSL | `raw/` (204) | `morph_amsl/` | 2,180 |

### Degradation Condition Slices
Each FRLL split also evaluated under synthetic degradation applied at test time:
- **Heavy JPEG:** quality 30 applied to test images before inference.
- **Simulated print-scan:** downsample 0.5x + JPEG q=40 + noise + upsample.

### Metrics Per Split
- **EER** (Equal Error Rate).
- **BPCER @ APCER = 1%** (border security operating point).
- **BPCER @ APCER = 5%** (screening operating point).
- **ROC-AUC**.

### Honesty Rule
Any result near 0% EER triggers a data-leakage audit before reporting.

---

## 5. Kaggle Execution

- **GPU:** Single NVIDIA T4 (16 GB VRAM).
- **Batch size:** 64 (fits comfortably in ~5 GB VRAM with AMP).
- **Epoch time:** ~2-3 min on 32k training samples.
- **Total runtime:** ~30-40 min for full 10-epoch run + evaluation.
- **Data path:** `/kaggle/input/smdd-morph/` and `/kaggle/input/frll-morphs/`.
- **Output path:** `/kaggle/working/` for checkpoints, metrics CSV, and ROC plots.

---

## 7. File Structure

```
kaggle_train.py       # Complete self-contained training + evaluation script for Kaggle
```

Single-file design. Kaggle notebooks cannot import from `src/` subdirectories without packaging. Everything lives in one script that runs end-to-end in a single notebook cell.

---

## 8. Empirical Baseline Benchmark Results (Stream A: DINOv2-Patch)

Trained on SMDD (32k samples) across 4 warm-up epochs with multi-scale patch statistics aggregator (`CLS + Patch Mean + Patch Std + Patch Max` -> 1536-d). Evaluated cross-dataset on unseen FRLL-Morphs:

| Morph Generator | Condition | EER | BPCER @ APCER=1% | BPCER @ APCER=5% | ROC-AUC |
|---|---|---|---|---|---|
| **OpenCV** | Clean | **0.0094** | 0.0000 | 0.0000 | **0.9993** |
| **OpenCV** | JPEG (q=30) | **0.0094** | 0.0098 | 0.0000 | **0.9992** |
| **OpenCV** | Print-Scan | **0.0916** | 0.2990 | 0.1569 | **0.9786** |
| **FaceMorpher** | Clean | **0.0245** | 0.1912 | 0.0000 | **0.9958** |
| **FaceMorpher** | JPEG (q=30) | **0.0249** | 0.1618 | 0.0000 | **0.9956** |
| **FaceMorpher** | Print-Scan | **0.0585** | 0.2647 | 0.0735 | **0.9859** |
| **WebMorph** | Clean | **0.0491** | 0.2745 | 0.0490 | **0.9886** |
| **WebMorph** | JPEG (q=30) | **0.0585** | 0.3039 | 0.0735 | **0.9875** |
| **WebMorph** | Print-Scan | **0.2539** | 0.7892 | 0.5833 | **0.8320** |
| **AMSL** | Clean | **0.1320** | 0.7598 | 0.3382 | **0.9461** |
| **AMSL** | JPEG (q=30) | **0.1367** | 0.7647 | 0.3235 | **0.9434** |
| **AMSL** | Print-Scan | **0.3978** | 0.9216 | 0.8039 | **0.6491** |
| **StyleGAN2** | Clean | **0.5100** | 0.9951 | 0.9902 | **0.4988** |
| **StyleGAN2** | JPEG (q=30) | **0.4998** | 0.9951 | 0.9853 | **0.4973** |
| **StyleGAN2** | Print-Scan | **0.4577** | 0.9608 | 0.8824 | **0.5780** |

### Key Diagnostic Insights
1. **Landmark-based Morphs:** DINOv2 patch features achieve near-perfect discrimination on digital landmark-blended morphs (OpenCV: 0.999 AUC, FaceMorpher: 0.996 AUC, WebMorph: 0.989 AUC), with high resilience against aggressive JPEG compression.
2. **Latent Space Morphs (The StyleGAN Gap):** StyleGAN2 morphs collapse to random guess (0.4988 AUC). Because StyleGAN2 interpolates latent vectors in $W^+$ space rather than triangulating landmarks, it produces zero pixel-level alpha blending seams. DINO patch texture statistics alone cannot separate them without frequency analysis (Stream B).
3. **Formal Report:** Complete analysis documented in [week1_report.md](file:///home/crimsond/project/docs/week1_report.md).

---

## 9. Phase 2 Specification: Dual-Stream UID-MAD (Stream A + Stream B)

**Target:** Unify dense spatial patch features (Stream A) with high-pass frequency residual representations (Stream B) to eliminate the StyleGAN domain gap while preserving >0.99 AUC on landmark morphs.

### Architecture Components
1. **Stream A (Spatial):** Frozen DINOv2 ViT-S/14 backbone + dense patch aggregator (`CLS + Mean + Std + Max` -> 1536-d) -> 256-d projector.
2. **Stream B (Frequency Residual):**
   - High-pass residual computation: 3x3 Laplacian kernel on GPU.
   - Frequency encoder: Lightweight 4-stage convolutional block with Batch Normalization and GELU -> 128-d embedding.
3. **Multi-Task Decoupled Heads:**
   - Independent ArcFace heads for `spatial_head`, `freq_head`, and `fused_head` supervised under joint loss: $\mathcal{L} = \mathcal{L}_{\text{fuse}} + 0.5 \mathcal{L}_s + 0.5 \mathcal{L}_f$.

---

## 10. Phase 2 Empirical Outcomes & Autopsy

Executed on Dual NVIDIA T4 GPUs on Kaggle (`results(1).zip`):
- **In-Domain Convergence (SMDD):** 99.30% training accuracy, 0.0001 (0.01%) validation EER, 0.9999 validation ROC-AUC.
- **Cross-Dataset Results on FRLL-Morphs:**
  - OpenCV: 0.7924 ROC-AUC (recovered from 0.2422 in early concat, but lower than 0.9993 baseline).
  - FaceMorpher: 0.8295 ROC-AUC.
  - StyleGAN: 0.5363 ROC-AUC; WebMorph: 0.5444 ROC-AUC; AMSL: 0.5145 ROC-AUC.
- **Root Cause Identified:**
  - Uncalibrated `torch.maximum(P_s, P_f, P_{\text{fuse}})` with saturated ArcFace temperature ($s=32.0$).
  - Synthetic training vs. real camera sensor noise mismatch: Stream B was trained on synthetic SMDD bona fides, and assigned elevated scores to physical PRNU sensor noise in FRLL real camera bona fides. The `max()` operator propagated these false alarms into the bona fide pool, collapsing the decision threshold to $1.003 \times 10^{-6}$.
- **Checkpoint Status:** All 3 heads (`spatial_head`, `freq_head`, `fused_head`) are intact in `results_run2/best_model.pt`.
- **Formal Report:** Complete analysis documented in [week2_report.md](file:///home/crimsond/project/docs/week2_report.md).

---

## 11. Phase 3: Calibrated Decision Architecture & Verification Plan

1. **Standalone Head Evaluation:** Update evaluation harness to evaluate `spatial_head`, `freq_head`, and `fused_head` independently on FRLL to observe isolated performance.
2. **Cosine-Margin Scoring:** Score using raw angular cosine difference $\Delta \cos = \cos(\theta_{\text{morph}}) - \cos(\theta_{\text{bona}})$ instead of saturated probability $P$, eliminating numerical threshold collapse.
3. **Domain-Calibrated Ensemble:** Standardize stream margins against bona fide reference statistics ($Z = (S - \mu_{\text{bf}}) / \sigma_{\text{bf}}$) before soft combination.

---

## 12. Multi-Week Team Roadmap & Milestone Schedule

### Week 1: Foundation, Dual-Stream Exploration & Empirical Diagnosis (COMPLETED)
- [x] Consolidate foundational MAD literature (Papers 1-4) into unified synthesis (`compiled_result.md`).
- [x] Implement and benchmark Phase 1 Single-Stream DINOv2 Spatial Baseline on Kaggle Dual NVIDIA T4s.
- [x] Discover StyleGAN2 spatial domain gap (0.4988 AUC) and formulate high-pass frequency residual hypothesis.
- [x] Implement Phase 2 Dual-Stream architecture (Laplacian filter + 4-stage ConvNet + multi-branch ArcFace).
- [x] Run overnight Kaggle training and perform mathematical autopsy on modality competition and sensor noise domain shift.
- [x] Author comprehensive formal dossiers (`RESEARCH_REPORT.md`, `week1_report.md`, `week2_report.md`, `README.md`).

### Week 2: Score Calibration & Domain Invariance Optimization
- [ ] **Task 2.1 (Evaluation):** Run `--eval-only` on Kaggle with `best_model.pt` to extract isolated benchmark metrics for each branch (`Spatial`, `Frequency`, `Fused Head`, `Cosine Margin Ensemble`).
- [ ] **Task 2.2 (Calibration):** Implement Bona Fide Z-score standardization ($Z = (S - \mu_{\text{bf}}) / \sigma_{\text{bf}}$) to eliminate false alarm spillover from camera sensor noise.
- [ ] **Task 2.3 (Robust Training):** Add sensor noise data augmentation (Gaussian noise + sensor grain simulation) to Stream B to train frequency invariance on real camera noise.
- [ ] **Milestone 2 Target:** Achieve simultaneous >0.99 AUC on landmark morphs (OpenCV, FaceMorpher) and >0.85 AUC on generative morphs (StyleGAN2) under Clean and JPEG conditions.

### Week 3: Single-Pass Explainability & Attention Visualization
- [ ] **Task 3.1 (Attention Map Extraction):** Extract DINOv2 multi-head self-attention rollout maps from last transformer layer to localize facial blending anomalies.
- [ ] **Task 3.2 (Landmark Alignment):** Overlay patch attention heatmaps with 68 facial landmark zones (eyes, nose, mouth boundary) as proposed in Paper 1.
- [ ] **Task 3.3 (Inference Speed):** Ensure explanation heatmaps generate in single forward pass (<50ms per image) without multi-pass occlusion or Grad-CAM computation.
- [ ] **Milestone 3 Target:** Produce visual explanation masks that highlight iris double contours and mask edges for border control officer verification.

### Week 4: Paper Draft, Benchmarking & Final Presentation
- [ ] **Task 4.1 (Paper Authoring):** Compile method, architecture, and empirical tables into a formal conference paper draft (IEEE/IAPR format).
- [ ] **Task 4.2 (Comparative Benchmarking):** Tabulate UID-MAD metrics against published SOTA benchmarks (FaceNet+XQDA, Single-ViT, MorGAN baselines).
- [ ] **Task 4.3 (Team Presentation):** Prepare final slide deck covering problem definition, baseline failure, dual-stream fix, and operational FRONTEX metrics.
- [ ] **Milestone 4 Target:** Complete submission-ready research paper and working interactive demo.



