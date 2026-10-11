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

