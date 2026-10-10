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

## 6. File Structure

```
kaggle_train.py       # Complete self-contained training + evaluation script for Kaggle
```

Single-file design. Kaggle notebooks cannot import from `src/` subdirectories without packaging. Everything lives in one script that runs end-to-end in a single notebook cell.

---

## 7. Follow-Up Prompt

```text
Please proceed with implementing the DINO branch plan from plan.md. Build kaggle_train.py as a single self-contained script.
```
