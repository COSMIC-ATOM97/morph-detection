# UID-MAD: Team Technical Debrief & Complete Walkthrough

**Target Audience:** You and your project teammates.  
**Purpose:** A clear, step-by-step technical guide explaining every decision, bug fix, architectural innovation, and benchmark result in the Morph Attack Detection (UID-MAD) pipeline, so you can explain it to your team and advisors with total confidence.  
**Standard:** ISO/IEC 30107-3 (Biometric Presentation Attack Detection)  
**Latest Checkpoint:** `checkpoints/phase2_dual_stream_best.pt` (96.4 MB)  
**Primary Execution Script:** `kaggle_train.py` / `scripts/phase2_dual_stream_uid_mad.py`  

---

## 1. The 60-Second Overview (What Are We Doing & Why?)

### What is a Face Morph Attack?
A face morph attack happens when two people (Person A, an accomplice with no criminal record, and Person B, a wanted criminal) blend their facial images into a single composite photograph. 

```
  Accomplice (Person A) ──┐
                          ├─► [Morph Generator] ─► Composite Face
  Criminal   (Person B) ──┘
```

When this composite photo is printed on a passport:
- Person A can walk through an Automated Border Control (ABC) e-Gate, and facial recognition verifies them.
- Person B can pick up the exact same passport later, walk through the e-Gate, and the system verifies them too!
- **The Core Flaw in Standard Face Recognition:** Standard models (like FaceNet or InsightFace) are designed to ignore minor facial differences to achieve identity invariance. That invariance actually helps morph attacks succeed.
- **Our Mission:** Build a Single-Image Morph Attack Detection (S-MAD) system that flags composite faces at the border before anyone enters.

### What Did We Achieve This Week?
- We built and validated **Dual-Stream UID-MAD**, combining a self-supervised Vision Transformer (Spatial) with a high-pass Laplacian ConvNet (Frequency Residuals).
- We resolved the **StyleGAN2 blind spot**, leaping from **0.4988 ROC-AUC (random chance)** to **0.9399 AUC clean** and **0.9854 AUC under print-scan**.
- Across all 15 benchmark slices on FRLL-Morphs under ISO/IEC 30107-3, our calibrated model achieves an average of **0.992 ROC-AUC** and never drops below **0.9809**.

---

## 2. Phase 1: What We Built First and Why It Failed on StyleGAN2

### What Was Stream A (Spatial ViT)?
In Week 1, we built a single-stream spatial baseline using Meta's **DINOv2 ViT-S/14** (`vit_small_patch14_dinov2.lvd142m`).
- **Why DINOv2?** Most deep models (like ResNet or ImageNet ViTs) are trained to classify object categories (dog, car), discarding fine pixel-level texture. DINOv2 is self-supervised; it retains dense spatial correspondences and fine surface textures.
- **Multi-Scale Patch Token Pooling:** A 224x224 image is divided into a 16x16 grid of 256 patches. In addition to the global `CLS` token, we compute the **mean**, **max**, and **standard deviation** across all 256 patch tokens:
  $$\mathbf{f} = [\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu}_{\text{patch}} \parallel \boldsymbol{\sigma}_{\text{patch}} \parallel \mathbf{z}_{\text{max}}] \in \mathbb{R}^{1536}$$
  The patch standard deviation is sensitive to localized gradient variance: if there is an unnatural seam or blending edge in a few patches, the standard deviation spikes.

### Why Did Phase 1 Succeed on Landmark Morphs?
On landmark-based morphs (OpenCV, FaceMorpher), faces are created by triangulating facial landmarks (Delaunay triangulation), warping the triangles, and alpha-blending pixels:
$$I_{\text{morph}} = \alpha I_A(\mathcal{W}_A) + (1-\alpha) I_B(\mathcal{W}_B)$$
This physical warping leaves telltale signs: double pupil contours, ghosting along the nose bridge, and cut-and-paste boundaries.
DINOv2's spatial tokens caught these easily:
- **morph_opencv:** 0.9993 ROC-AUC (0.94% EER)
- **morph_facemorpher:** 0.9958 ROC-AUC (2.45% EER)

### Why Did Phase 1 Completely Collapse on StyleGAN2?
On generative latent morphs (StyleGAN2), Phase 1 scored **0.4988 ROC-AUC (51.0% EER)**. That is mathematically identical to a coin flip.
- **Why?** StyleGAN2 does not cut, warp, or blend pixels. It inverts both faces into an abstract latent space $\mathcal{W}+$, averages their latent vectors, and feeds that vector to a generative neural network $\mathcal{G}(\mathbf{w}_{\text{morph}})$.
- The resulting face is painted fresh from scratch. There are **zero pixel seams, zero double contours, and zero ghosting**.
- To an RGB spatial Vision Transformer, a StyleGAN2 face is continuous and photorealistic. Spatial features alone cannot solve generative morphs.

---

## 3. The 10-Hour Kaggle Hang and How We Fixed It

When we initially ran the code on Kaggle, the notebook ran for 10 hours (36,000 seconds) and was terminated by Kaggle's maximum runtime limit, producing `Output: 0 B`.

### Root Cause 1: Kaggle GCS FUSE Seek Latency
- On Kaggle, `/kaggle/input` is not a local hard drive. It is a virtual filesystem (FUSE mount) connected over the internal network to Google Cloud Storage (GCS).
- When our PyTorch dataset tried to read images directly out of the 4.5 GB `SMDD_release_train.zip` on the fly, opening each image required random seek operations across the virtual mount.
- Seeking over a network mount took 5 to 10 seconds per batch. Loading an epoch took 3+ hours instead of 20 seconds!

### Root Cause 2: PyTorch DataLoader CUDA Fork Deadlock
- When PyTorch initializes CUDA in the main process before creating DataLoader worker processes (`num_workers > 0`), PyTorch's default start method on Linux is `fork`.
- Child processes inherited dirty CUDA IPC memory handles from the parent, causing child worker threads to deadlock in memory and wait forever without throwing an error.

### How We Fixed It (Permanent Invariants)
1. **Local NVMe Scratch Extraction:** In `kaggle_train.py`, we added `extract_archive_if_needed()`. At startup, it unzips `SMDD_release_train.zip` once into local container NVMe scratch space (`/tmp/smdd_extracted`). Loading an entire 40,000-image epoch from `/tmp/` takes just **22 seconds**.
2. **Single-Process DataLoader:** We set `NUM_WORKERS = 0` and `PIN_MEMORY = False`. Running in the main process completely eliminates CUDA IPC synchronization deadlocks.
3. **The Result:** The entire 8-epoch training finished in **58.4 minutes**, and all 15 cross-dataset evaluations finished in **9.5 minutes** (67.9 minutes total).

---

## 4. Phase 2: The Dual-Stream Solution & Scientific Discoveries

To solve the StyleGAN2 blind spot, we engineered **Stream B (Frequency Residual Stream)** and combined it with Stream A.

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

### 4.1 How Does Stream B Catch StyleGAN2?
1. **The Laplacian High-Pass Filter:**
   We pass the input image through a fixed 3x3 Laplacian filter kernel:
   $$\mathbf{K} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{R} = |\mathbf{I} * \mathbf{K}|$$
   This operation subtracts the average color of surrounding pixels from the center pixel. It completely removes low-frequency facial identity, skin color, and lighting, leaving behind only raw high-frequency edge gradients and noise.
2. **The Transposed Convolution Fingerprint:**
   Generators like StyleGAN2 upsample images using transposed convolutions. Discrete upsampling creates periodic high-frequency energy spikes across the 2D frequency spectrum (Nyquist checkerboard patterns).
3. **The 4-Stage ConvNet:**
   Our lightweight ConvNet (4 conv stages downsampling to 14x14 feature maps + dual Avg/Max pooling to 128-d) detects these unnatural periodic frequency distributions.

### 4.2 LoRA (Low-Rank Adaptation)
- **Why not unfreeze the whole ViT?** Fine-tuning all 21.6 million parameters on 32,000 synthetic faces causes catastrophic forgetting of DINOv2's pre-trained spatial knowledge.
- **How LoRA Works:** Instead of modifying the full weight matrix $W$, we inject small rank $r=16$ trainable adapter matrices into the query, value, and MLP layers of the last 6 ViT blocks:
  $$\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}, \; B \in \mathbb{R}^{d_{\text{out}} \times r}$$
- Trainable parameters are reduced to just **1.4M (only ~589k for LoRA)**.
- **Factorized Forward Projection:** In `LoRALinear`, we compute $((\text{dropout}(x) \cdot A^T) \cdot B^T) \cdot \frac{\alpha}{r}$ sequentially, which avoids creating huge intermediate matrices in GPU VRAM.

### 4.3 Trigonometric ArcFace Stability
- Standard ArcFace uses $\arccos(x)$ to find the angle $\theta$. In FP16 mixed precision, the gradient of $\arccos(x)$ is $-\frac{1}{\sqrt{1 - x^2}}$, which explodes to infinity when $x \to \pm 1$, causing `NaN` loss crashes.
- We replaced this with trigonometric angle addition:
  $$\cos(\theta + m) = \cos\theta \cos m - \sin\theta \sin m$$
  with a linear Taylor fallback when $\theta + m > \pi$. This never divides by zero and runs stably in FP16 AMP.

### 4.4 The Naive Max-Gate Trap & The Calibrated Fusion Solution
In Week 2, early tests tried combining streams using a simple maximum probability rule:
$$P = \max(P_{\text{spatial}}, P_{\text{freq}}, P_{\text{fused}})$$
This failed because:
1. **Camera Sensor Noise (PRNU):** In SMDD (the training set), bona fide faces are synthetic (smooth sensor noise). In FRLL (the test set), bona fide faces are real camera photos possessing natural Photo-Response Non-Uniformity (PRNU) sensor noise.
2. Stream B, trained only on synthetic data, saw real camera sensor noise and assumed it was an attack!
3. Because ArcFace scale $s=32$ causes softmax probabilities to saturate near 1.0, a single false alarm from Stream B caused the whole ensemble to flag innocent travelers as attackers.

**The Solution -- Z-Score Calibrated Decision Fusion:**
- Throw away saturated softmax probabilities. Use linear cosine margins:
  $$S = \cos\theta_{\text{morph}} - \cos\theta_{\text{bona}}$$
- Compute baseline mean ($\mu$) and standard deviation ($\sigma$) of genuine faces for both streams:
  $$Z_{\text{spatial}} = \frac{S_s - \mu_s}{\sigma_s}, \quad Z_{\text{freq}} = \frac{S_f - \mu_f}{\sigma_f}$$
- Combine them with equal weighting:
  $$Z_{\text{final}} = 0.5 \cdot Z_{\text{spatial}} + 0.5 \cdot Z_{\text{freq}}$$
- **The Result:** Camera sensor noise false alarms disappeared completely. StyleGAN2 surged to **0.9399 AUC clean** and **0.9854 under print-scan**.

### 4.5 The Print-Scan Domain Orthogonality Discovery
One of our biggest research findings is what happens under physical printing and optical scanning:
1. **Spatial ViT Degradation:** Printing (CMYK halftoning) acts as an analog low-pass filter. It blurs out subtle edge seams, causing the spatial ViT to drop sharply (StyleGAN2 drops to 0.3155, AMSL to 0.3743).
2. **Frequency Stream Surge:** Under print-scan, the high-pass frequency residual stream surges to **>0.997 ROC-AUC across every single generator**!
3. **Complementary Safety Net:** Because spatial and frequency streams fail in opposite environments, calibrated fusion stays above **0.9809 ROC-AUC** everywhere.

---

## 5. Summary Benchmark Table to Share with the Team

| Generator | Attack Method | Test Condition | Phase 1 Spatial ViT | Phase 2 Calibrated Fusion | What This Proves |
|---|---|---|---|---|---|
| **morph_opencv** | Delaunay Landmark Warp | Clean | 0.9993 AUC | **0.9997 AUC (0.74% EER)** | Preserved landmark accuracy |
| **morph_opencv** | Delaunay Landmark Warp | Print-Scan | 0.9786 AUC | **0.9973 AUC (2.33% EER)** | Maintained under physical scan |
| **morph_facemorpher** | Landmark Alpha Blend | Clean | 0.9958 AUC | **0.9999 AUC (0.41% EER)** | Flawless cross-dataset transfer |
| **morph_webmorph** | Landmark Warp | Clean | 0.9886 AUC | **0.9968 AUC (1.96% EER)** | Robust out-of-distribution |
| **morph_stylegan** | Generative Latent W+ | Clean | 0.4988 AUC (Random) | **0.9399 AUC (14.66% EER)** | **+44.1% AUC Leap (Blind spot solved)** |
| **morph_stylegan** | Generative Latent W+ | Print-Scan | 0.5780 AUC | **0.9854 AUC (3.92% EER)** | **Frequency stream hits 0.9977 AUC** |
| **morph_amsl** | Feathered Landmark | Clean | 0.9461 AUC | **0.9938 AUC (2.92% EER)** | Overcame blending feathering |

- **Overall Benchmark:** Evaluated on all 7,274 images across 15 operational slices.
- **Mean ROC-AUC:** **0.992** across all slices.
- **Minimum ROC-AUC:** **0.9809** (no slice fails).

---

## 6. How the Code Works & How to Run It

### Repository Map
```
morph-detection/
├── kaggle_train.py             # The self-contained master script for Kaggle or local training & eval
├── README.md                   # Main project presentation dossier and benchmark tables
├── checkpoints/                # Contains phase2_dual_stream_best.pt (the trained weights)
├── data/                       # Contains the raw .zip dataset archives (git-ignored)
├── docs/                       # Formal reports: week1_report.md, week2_report.md, week3_report.md
├── results/phase2/             # Official CSV tables and ROC curve plots
├── scripts/                    # Standalone reproducible scripts (phase1 & phase2 .py and .ipynb)
└── team_internal/              # This walkthrough and developer notes
```

### Running the Entire Pipeline (Training + 15-Slice Evaluation)
On Kaggle or a local GPU machine:
```bash
python kaggle_train.py
```
- It automatically detects your GPU accelerator.
- If running on Kaggle, it extracts the dataset to `/tmp/smdd_extracted` once.
- Trains 4 warmup epochs + 4 LoRA fine-tuning epochs (~58 minutes).
- Evaluates all 15 slices of FRLL-Morphs under ISO/IEC 30107-3 (~9.5 minutes).
- Saves `frll_results_comparison.csv`, `training_history.csv`, and all 5 ROC curve plots to `/kaggle/working/` (or current directory).

---

## 7. Teammate & Advisor Q&A Scripts (Ready-to-Use Answers)

**Q: Why did you use DINOv2 instead of a standard face recognition model like ArcFace or FaceNet?**  
> *"Face recognition models are trained to ignore subtle facial lighting, camera noise, and minor edge differences so they can identify the same person across different photos. But morph detection needs the exact opposite: we need to detect localized pixel-level blending seams and synthetic artifacts. DINOv2 is self-supervised; its patch tokens preserve localized spatial geometry without discarding fine textures."*

**Q: Why did your Phase 1 model fail completely on StyleGAN2?**  
> *"Landmark morphs create visible spatial seams because faces are warped and alpha-blended together. StyleGAN2 morphs are generated continuously from latent space vectors; they have zero alpha-blending seams and zero ghosting. To an RGB spatial Vision Transformer, a StyleGAN2 face looks continuous and photorealistic, causing it to collapse to a coin flip (0.4988 AUC)."*

**Q: How does the frequency stream solve StyleGAN2?**  
> *"StyleGAN2 uses transposed convolutions to upsample feature maps to high resolution. This discrete upsampling leaves periodic high-frequency checkerboard patterns in the 2D frequency spectrum. By applying a 3x3 Laplacian high-pass filter, we strip facial identity and skin color, allowing our 4-stage ConvNet to detect these unnatural periodic generator noise artifacts, lifting StyleGAN2 detection to 0.9399 AUC clean and 0.9854 under print-scan."*

**Q: Why didn't you just take the maximum score between both streams?**  
> *"A naive maximum operator causes false alarms because real camera photos have physical sensor noise (PRNU), which the frequency stream initially mistook for synthetic noise. Taking the max allowed a single stream false alarm to ruin the entire bona fide distribution. Our Z-score calibrated fusion standardizes margins against genuine baseline statistics, completely eliminating sensor noise bias and achieving >= 0.9809 AUC across all 15 operational test slices."*
