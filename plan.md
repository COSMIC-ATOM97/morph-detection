# Implementation Plan: DINO-Branch Morph Attack Detection (Kaggle Baseline)

**Target:** Single-stream foundation model baseline for Morph Attack Detection (MAD) trained on SMDD and evaluated cross-dataset on FRLL-Morphs.  
**Scope Decision:** Focused exclusively on the DINO branch (Stream A: self-supervised ViT backbone + ArcFace classification head). All secondary streams (frequency residual stream, U-Net mask decoder, landmark regional attention, adversarial GRL disentanglement, semi-supervised consistency) are deferred to later iterations.

---

## 1. Dataset Verification & Findings

Direct binary inspection of the workspace archives confirms the following distribution:

### SMDD (`SMDD_release_train.zip` - 4.54 GB)
- **Role:** Primary training and in-domain validation.
- **Images:** 40,000 synthetic face images in 256x256 PNG format.
  - Bonafide (`os25k_bf_t/`): 25,000 images.
  - Morph Attacks (`m15k_t/`): 15,000 images.
- **Metadata:** `train.csv` provided with columns `image_path,label` (`bonafide` vs `attack`).

### FRLL-Morphs (`drive-download-20261010T174433Z-1-001.zip` - 327 MB)
- **Role:** Strict Out-of-Distribution (OOD) cross-dataset and cross-algorithm test set.
- **Images:** 7,274 total images in JPEG format.
  - Bonafide (`raw/`): 204 images from London Research Lab (1350x1350 high resolution).
  - OpenCV morphs (`morph_opencv/`): 1,221 images (431x513).
  - FaceMorpher morphs (`morph_facemorpher/`): 1,222 images (431x513).
  - WebMorph morphs (`morph_webmorph/`): 1,222 images (413x531).
  - StyleGAN2 morphs (`morph_stylegan/`): 1,225 images (1024x1024).
  - AMSL morphs (`morph_amsl/`): 2,180 images (413x531).
- **Protocols:** 795 `.lst` files in `protocols/` defining probe and model evaluation splits.

---

## 2. Model Architecture

```
                    ┌──────────────────────────────┐
                    │       Input Face Image       │
                    │      (224x224 RGB Crop)      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      DINO ViT Backbone       │
                    │   - DINOv2 (ViT-S/14) or     │
                    │   - DINOv3 (ViT-S/16)        │
                    │   Feature dim: D = 384       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Embedding Norm         │
                    │       z = L2_norm(CLS)       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   ArcFace Classification     │
                    │   W = L2_norm(Weights)       │
                    │   logits = s * cos(theta + m)│
                    │   s = 30.0, m = 0.35         │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Morph Risk Score        │
                    │          P in [0, 1]         │
                    └──────────────────────────────┘
```

### Components
1. **Backbone:**
   - DINOv2 ViT-S/14 (`facebook/dinov2-small` via HuggingFace or PyTorch Hub): un-gated public weights, works on Kaggle out-of-the-box.
   - DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`): gated HuggingFace weights, supported via user token.
   - Configurable via parameter `--backbone`, defaulting to DINOv2 for reproducible execution.
2. **Head:**
   - Linear projection + batch normalization + L2 normalized representation.
   - ArcFace metric head enforcing angular margin between bonafide and morph classes in hyperspherical space.
   - Binary cross-entropy (BCE) loss on scaled margins.

---

## 3. Training Protocol

### Phase 1: Warm-up (Head Only)
- **Epochs:** 1 - 3
- **Frozen:** Entire ViT backbone.
- **Trained:** Projection layer and ArcFace head.
- **Optimizer:** AdamW, learning rate = 1e-3, weight decay = 1e-4.
- **Purpose:** Prevent noisy initial head gradients from degrading pretrained self-supervised representations.

### Phase 2: Differential Fine-Tuning
- **Epochs:** 4 - 10
- **Unfrozen:** Top 2-4 transformer blocks of the ViT backbone.
- **Optimizer:** AdamW with cosine annealing schedule:
  - Backbone learning rate = 1e-5.
  - Head learning rate = 1e-4.
- **Precision:** Mixed precision training (`torch.cuda.amp.autocast`) to minimize VRAM usage and maximize throughput.

### Augmentations (Anti-Overfitting)
- Simulated JPEG compression (quality range 30 to 90).
- Mild Gaussian blur and Gaussian noise.
- Random horizontal flip (p = 0.5).
- Image resizing to 224x224 and ImageNet normalization.

---

## 4. Evaluation Suite (ISO/IEC 30107-3 Standard)

### In-Domain Validation
- 20% stratified hold-out from SMDD (5,000 bonafide, 3,000 morph).
- Used for checkpointing the model based on lowest validation EER.

### Cross-Dataset / Cross-Algorithm Evaluation (OOD)
The best checkpoint is evaluated on FRLL-Morphs across all 5 distinct generators without fine-tuning:
1. `FRLL-OpenCV`
2. `FRLL-FaceMorpher`
3. `FRLL-WebMorph`
4. `FRLL-StyleGAN2`
5. `FRLL-AMSL`

### Metrics Reported Per Split
- **EER (Equal Error Rate):** Threshold where False Accept Rate equals False Reject Rate.
- **BPCER @ APCER = 1%:** Bona Fide Presentation Classification Error Rate when Attack Presentation Classification Error Rate is held at 1% (strict security).
- **BPCER @ APCER = 5%:** Operational screening threshold.
- **ROC-AUC:** Overall separation quality.

---

## 5. Kaggle Execution Blueprint

- **Hardware Target:** Single NVIDIA T4 GPU (16 GB VRAM).
- **Batch Size:** 64.
- **Expected Speed:** ~2 to 3 minutes per epoch on 32,000 training samples.
- **Memory Footprint:** < 6 GB VRAM with AMP enabled.
- **Storage Strategy:** Datasets mounted read-only from `/kaggle/input/`, checkpoints and metrics written to `/kaggle/working/`.

---

## 6. Implementation File Structure

```
src/
├── dataset.py        # SMDD and FRLL-Morphs PyTorch dataset loaders + augmentations
├── model.py          # DINOv2 / DINOv3 backbone wrapper + ArcFace head
├── metrics.py        # Strict ISO/IEC 30107-3 metrics calculator (EER, BPCER@APCER)
├── train.py          # Two-stage training loop with AMP and checkpointing
└── evaluate.py       # Standalone OOD evaluation on FRLL-Morphs subsets
kaggle_runner.ipynb   # Ready-to-run Kaggle notebook assembling pipeline
```

---

## 7. Follow-Up Execution Prompt

When you are ready to begin implementation, copy and send the following prompt:

```text
Please proceed with implementing the DINO branch plan from plan.md. Build out src/dataset.py, src/model.py, src/metrics.py, src/train.py, src/evaluate.py, and the kaggle_runner notebook, ensuring all unit checks pass.
```
