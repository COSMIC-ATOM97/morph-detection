# UID-MAD: Master Presentation & Viva Defense Deep-Dive Guide
### Comprehensive Step-by-Step Study Guide for Mid-Semester Evaluation (NIT Kurukshetra)
**Subject:** Mid Sem. Evaluation of B.Tech Project Work (CSPC401 / ITPC401)  
**Authors:** Himank Gupta (123103097), Bablu Prajapati (123103019), Puranpal Singh (123103021)  
**Supervisor:** Dr. Abhilasha Sharma | **Academic Year:** 2026-2027  
**Official Project Title:** Unified Interpretable Dual-Stream Face Morph Attack Detection (UID-MAD) via Spatial Foundation Transformers and Spectral Residuals

---

## Quick Reference Summary Table

| Core Dimension | Technical Implementation | Why This Choice / Impact |
| :--- | :--- | :--- |
| **Problem Solved** | Single-Image Morph Attack Detection (S-MAD) | Prevents criminal passport fraud at issuance & e-Gates |
| **Stream A (Spatial)** | DINOv2 ViT-S/14 (`vit_small_patch14_dinov2`) | Dense local patch geometry without ImageNet object bias |
| **Spatial Aggregator** | $[\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu}_{\text{patch}} \parallel \boldsymbol{\sigma}_{\text{patch}} \parallel \mathbf{z}_{\text{max}}] \in \mathbb{R}^{1536}$ | $\boldsymbol{\sigma}_{\text{patch}}$ acts as explicit mathematical seam detector |
| **Fine-Tuning Method**| Rank-16 LoRA on Attention & MLP (Blocks 6-11) | Adapts forensic anomalies; avoids catastrophic forgetting |
| **Stream B (Spectral)**| Depthwise Laplacian Filter Bank + Haar Wavelet HH | Strips identity/lighting; isolates neural upsampling noise |
| **Frequency ConvNet** | 4-Stage ConvNet ($3 \to 32 \to 64 \to 128 \to 128$) | Extracts 128-d spectral embedding $\mathbf{e}_{\text{freq}}$ in 1.6 ms |
| **Supervision Loss**  | Decoupled Multi-Task ArcFace ($s=32.0, m=0.25$) | Prevents CNN gradients from overpowering ViT (Trial 1 fix) |
| **Inference Fusion**  | Bona Fide Z-Score Calibration ($Z = 0.5 Z_s + 0.5 Z_f$) | Eliminates camera sensor PRNU noise & softmax saturation |
| **Generative Recovery**| StyleGAN2 Clean AUC: **0.4988 $\to$ 0.9399 (+44.1%)** | Solved Phase 1 generative blind spot (coin-flip failure) |
| **Print-Scan Surge**  | Frequency AUC: **$> 0.997$ across all 5 tools** | Physical halftone/ink noise rescues degraded spatial seams |
| **Overall Benchmark** | **Mean 0.992 ROC-AUC** ($\ge 0.9809$ across 15 slices)| Complete cross-generator, cross-condition robustness |
| **Edge Feasibility**  | **17.3 ms single-frame latency** on dual T4 GPUs | Fully satisfies border e-Gate $\le 50$ ms latency SLA |

---

## 1. Problem Identification & Foundations (From Scratch)

### 1.1 What is an Automated Border Control (ABC) e-Gate?
At modern international airports, passengers pass through automated e-Gates rather than speaking to a human passport control officer:
1. The passenger places their biometric passport (known as an **electronic Machine Readable Travel Document / eMRTD**, strictly standardized under **ICAO Doc 9303**) on an optical scanner.
2. An embedded contactless RFID chip inside the passport transmits the officially signed, registered digital portrait photograph.
3. A live camera at the gate captures the passenger's current face.
4. An Automated Facial Recognition System (AFRS) compares the live capture against the passport portrait. If the similarity exceeds a security threshold, the gate opens.

### 1.2 What is a Face Morph Attack?
Consider this scenario:
* **Subject B (The Malicious Actor):** A known criminal or individual on an international security watch list who cannot legally obtain a passport or travel.
* **Subject A (The Accomplice):** An individual with no criminal record who looks somewhat similar to Subject B and is legally entitled to a passport.

**The Attack Pipeline:**
1. Subject A and Subject B merge their facial portraits into a single synthetic hybrid image: $I_{\text{morph}}$.
2. Accomplice A applies for a legitimate biometric passport using $I_{\text{morph}}$. Because Subject A has a clean background, the passport issuance authority legally issues the document.
3. **The Exploit:** Because $I_{\text{morph}}$ contains roughly 50% geometric and photometric features from both individuals, **both Subject A and Subject B can pass through automated border e-Gates using the exact same physical passport!**

```
    Accomplice A (Clean Record)  ──┐
                                   ├─► Morphing Tool ──► I_morph ──► Valid Passport Issued
    Criminal B (Watch-listed)    ──┘                                        │
                                                                            ▼
                                                e-Gate verifies BOTH A and B successfully!
```

### 1.3 The Identity Invariance Paradox: Why Commercial Face Matchers Fail
Commercial face recognition systems (such as ArcFace, FaceNet, and InsightFace) are explicitly engineered to maximize **identity invariance**. They are trained to ignore lighting changes, minor facial hair, makeup, facial expressions, and aging so that an individual can be recognized reliably under varying conditions.

Because a face morph is mathematically engineered to lie right in the overlapping feature space between Subject A and Subject B, **identity-invariant matchers accept both individuals with high confidence (>98%)**. In short: commercial face matchers do not detect morphs because morphs exploit the exact invariant representation the matchers were designed to learn.

### 1.4 S-MAD vs. D-MAD
* **S-MAD (Single-Image Morph Attack Detection):** Inspects a single submitted portrait photograph in isolation with no secondary reference image. This is mandatory for online passport application portals, visa issuance systems, and identity card issuance. **This project addresses S-MAD.**
* **D-MAD (Differential Morph Attack Detection):** Compares the stored passport photo against a trusted, live webcam capture at the border e-Gate. While D-MAD is an easier classification task (because the live capture provides an authentic reference), it cannot protect the initial passport application stage.

---

## 2. The Two Opposite Attack Physics (Why This is Two Problems in One)

Face morphs do not share a single forensic signature. They divide into two fundamentally different mathematical and physical families:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FACE MORPHING FAMILIES                                   │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Digital Landmark-Warped Morphs        │ 2. Generative Latent-Synthesized Morphs     │
│    (OpenCV, FaceMorpher, WebMorph, AMSL) │    (StyleGAN2)                              │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • 68 facial landmarks extracted from A&B │ • Faces inverted into latent space W+       │
│ • Triangulated via Delaunay mesh         │ • Latent vectors averaged: w_M = 0.5(wA+wB) │
│ • Pixels warped and alpha-blended        │ • Re-synthesized from scratch via G(w_M)    │
│ • SIGNATURE: Spatial blending seams,     │ • SIGNATURE: ZERO spatial seams! Flawless   │
│   ghosting along nose bridge, double     │   skin boundaries. BUT leaves periodic      │
│   iris/pupil contours, outer mask edges. │   checkerboard Nyquist upsampling noise.    │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### 2.1 Landmark Blending (Mathematical Formulation)
Given landmark coordinates $\mathcal{P}_A, \mathcal{P}_B \in \mathbb{R}^{68 \times 2}$, an interpolated mesh is computed:
$$\mathcal{P}_M = \alpha \mathcal{P}_A + (1 - \alpha) \mathcal{P}_B$$
Source pixels are warped via affine transformation $\mathcal{W}$ and alpha-blended:
$$I_{\text{morph}} = \alpha \cdot I_A(\mathcal{W}_A) + (1 - \alpha) \cdot I_B(\mathcal{W}_B)$$
This piecewise-affine warping produces physical boundary discontinuities, dual edge contours, and double pupils.

### 2.2 Generative Latent Interpolation (Mathematical Formulation)
Source portraits are mapped into StyleGAN2's intermediate latent space $\mathcal{W}+$:
$$\mathbf{w}_A = \mathcal{E}(I_A), \quad \mathbf{w}_B = \mathcal{E}(I_B)$$
Latent vectors are linearly interpolated:
$$\mathbf{w}_{\text{morph}} = \alpha \mathbf{w}_A + (1 - \alpha) \mathbf{w}_B$$
The composite image is generated entirely through the neural network generator:
$$I_{\text{morph}} = \mathcal{G}(\mathbf{w}_{\text{morph}})$$
Because the face is sampled continuously from a neural manifold, **it contains zero spatial blending seams, zero ghosting, and zero double contours**. However, the generator's transposed convolutions leave periodic high-frequency checkerboard artifacts across the spectral domain.

---

## 3. Standardized Biometric Metrics (ISO/IEC 30107-3)

Never evaluate biometric security using generic accuracy alone. The committee expects ISO/IEC 30107-3 standardized terminology:

1. **APCER (Attack Presentation Classification Error Rate):**
   $$\text{APCER} = \frac{\text{False Positives (Morphs accepted as Genuine)}}{\text{Total Morph Attacks Tested}}$$
   *Security risk:* The proportion of attackers who successfully sneak through the gate.

2. **BPCER (Bona Fide Presentation Classification Error Rate):**
   $$\text{BPCER} = \frac{\text{False Negatives (Genuine travelers flagged as Morphs)}}{\text{Total Bona Fide Presentations Tested}}$$
   *Operational friction:* The proportion of honest citizens wrongly stopped for secondary interrogation.

3. **EER (Equal Error Rate):**
   The operating threshold where $\text{APCER} = \text{BPCER}$. A lower EER indicates higher fundamental discrimination power.

4. **BPCER @ APCER = 1.0% (and 5.0%):**
   The operational border security benchmark. When the border authority calibrates the gate to accept no more than 1% of morph attacks ($\text{APCER} \le 1\%$), what percentage of innocent citizens get falsely flagged?

---

## 4. Phase 1: What We Did First & The Generative Blind Spot

### 4.1 The Phase 1 Baseline Architecture
In Phase 1, we constructed a single-stream spatial baseline deploying a self-supervised Vision Transformer (**DINOv2 ViT-S/14**) with multi-scale token pooling and ArcFace supervision.

### 4.2 The Empirical Results & The Failure Mode
* **On Landmark Morphs:** Outstanding performance.
  * `morph_opencv`: **0.9993 ROC-AUC** ($0.94\%$ EER)
  * `morph_facemorpher`: **0.9958 ROC-AUC** ($2.45\%$ EER)
* **On Generative Morphs (`morph_stylegan`): Complete Structural Collapse!**
  * **0.4988 ROC-AUC** ($51.00\%$ EER, $\text{BPCER@1\%} = 99.50\%$)

```
    Phase 1 ROC Curve on StyleGAN2:
    True Positive Rate
     1.0 ┼
         │
     0.5 │          /  <── Exactly on the 0.50 diagonal!
         │        /        (Pure coin flip / random guessing)
     0.0 ┼──────/─────────
        0.0    0.5    1.0   False Positive Rate
```

### 4.3 Why Did Phase 1 Collapse on StyleGAN2?
An ROC-AUC of 0.4988 is mathematically identical to a random coin flip. Because StyleGAN2 creates faces from latent space vectors, **there are no physical pixel seams, edge cutouts, or double irises for a spatial transformer to see**. A spatial-only model is fundamentally blind to generative morphs.

This empirical finding directly motivated our **Phase 2 Dual-Stream Architecture (UID-MAD)**.

---

## 5. Phase 2: UID-MAD Complete System Architecture

To detect both attack families simultaneously without compromise, we designed a decoupled dual-stream network:

```
                            INPUT IMAGE  I ∈ R^(224×224×3)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         STREAM A (SPATIAL)                              STREAM B (FREQUENCY)
         DINOv2 ViT-S/14 Backbone                        Non-Trainable Filter Bank
         + Rank-16 LoRA (Blocks 6-11)                    (Laplacian + Haar HH Wavelet)
         Multi-Scale Patch Aggregator                    4-Stage ConvNet (32→64→128→128)
         e_spatial ∈ R^256                               e_freq ∈ R^128
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                             CROSS-STREAM GATED FUSION
                             f_aligned = W_align · e_freq + b_align  (128 → 256)
                             gate g = σ(W_g [e_spatial || f_aligned] + b_g)
                             z_fused = e_spatial + g ⊙ f_aligned
                             e_fuse = FusionMLP(z_fused) ∈ R^256
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            SPATIAL HEAD              FREQ HEAD               FUSED HEAD
          ArcFace(256 → 2)        ArcFace(128 → 2)        ArcFace(256 → 2)
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          ▼
                          DECOUPLED MULTI-TASK ARCFACE LOSS
                    L_total = L_fuse + 0.5·L_spatial + 0.5·L_freq
                                          │
                                          ▼ (At Test Time)
                          BONA FIDE Z-SCORE DECISION CALIBRATION
                              Z_final = 0.5·Z_s + 0.5·Z_f
```

---

### 5.1 Stream A (Spatial Pathway) Detailed Mechanics

#### Foundation Backbone: DINOv2 ViT-S/14
* Model: `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, embedding dimension $D = 384$, patch size $14 \times 14$).
* For a $224 \times 224$ image:
  $$\text{Patch Grid} = \left(\frac{224}{14}\right) \times \left(\frac{224}{14}\right) = 16 \times 16 = 256 \text{ spatial tokens } \{\mathbf{z}_i\}_{i=1}^{256} + 1 \text{ class token } \mathbf{z}_{\text{cls}}$$
* **Why DINOv2 instead of supervised ViT or ResNet?**
  Supervised models trained on ImageNet classify whole objects (dog, car, plane) and learn to discard subtle local pixel variations. DINOv2 was trained using self-supervised patch-level student-teacher distillation and masked image modeling. Its patch tokens preserve dense, localized pixel geometry and fine boundary structures.

#### Multi-Scale Token Aggregator (The Mathematical Seam Detector)
Standard models perform global average pooling ($\boldsymbol{\mu}_{\text{patch}}$) across spatial tokens. **Global average pooling washes out localized blending seams**, diluting a sharp artifact across 256 patches.

UID-MAD constructs an explicit 1536-dimensional descriptor concatenating four statistics:
$$\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}$$
Where:
* $\mathbf{z}_{\text{cls}} \in \mathbb{R}^{384}$: Global semantic scene summary.
* $\boldsymbol{\mu}_{\text{patch}} = \frac{1}{N}\sum_{i=1}^N \mathbf{z}_i \in \mathbb{R}^{384}$: Background spatial representation.
* $\boldsymbol{\sigma}_{\text{patch}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{z}_i - \boldsymbol{\mu}_{\text{patch}})^2 + \varepsilon} \in \mathbb{R}^{384}$: **The Mathematical Seam Detector.** Blending seams create abrupt feature differences between adjacent patches. Computing inter-patch standard deviation explicitly captures localized spatial variance spikes.
* $\mathbf{z}_{\text{max}} = \max_{1 \le i \le N} \mathbf{z}_i \in \mathbb{R}^{384}$: Preserves the single strongest localized anomaly (e.g., a double pupil or cutout boundary) that averaging would dilute.

The vector $\mathbf{f} \in \mathbb{R}^{1536}$ is projected via a two-layer bottleneck MLP ($1536 \to 512 \to 256$) with GELU and BatchNorm to produce $\mathbf{e}_{\text{spatial}} \in \mathbb{R}^{256}$.

#### Parameter-Efficient LoRA (Low-Rank Adaptation)
* **The Catastrophic Forgetting Hazard:** Fully fine-tuning all 21.6M parameters of DINOv2 on 32,000 synthetic images would overwrite the general visual representations learned during pre-training.
* **The LoRA Mechanism:** LoRA freezes base weights $\mathbf{W}_0$ and injects trainable rank decomposition matrices:
  $$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{A} \cdot \mathbf{B})$$
  Where $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ (initialized with Gaussian noise) and $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ (initialized to zero, ensuring $\Delta \mathbf{W} = 0$ at training onset).
* **Configuration:** Rank $r = 16$, scaling factor $\alpha = 32.0$ (scaling ratio $\frac{\alpha}{r} = 2.0$), dropout $0.1$.
* **Target Blocks:** Injected into attention projections (`qkv`, `proj`) and feed-forward MLP projections (`fc1`, `fc2`) across **Blocks 6 through 11 only**.
* **Why Only Blocks 6-11?** Early transformer blocks (0-5) encode universal low-level visual primitives (edges, gradients, primary textures). Adapting only the higher semantic blocks preserves the foundational visual manifold while tuning the top layers to detect morphing boundary anomalies. LoRA adds only 0.59M trainable parameters (2.6% of the backbone).

---

### 5.2 Stream B (Frequency Residual Pathway) Detailed Mechanics

Stream B completely ignores facial identity, skin color, and global morphology. It isolates high-frequency residual energy and upsampling noise.

#### 1. Spatial Filter Bank Residual ($R$)
The RGB image is convolved with a fixed, non-trainable $3 \times 3$ kernel combining an 8-neighbor Laplacian, horizontal second-difference, and vertical second-difference:
$$\mathbf{K}_{\text{Lap}} = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}, \quad \mathbf{K}_h = \begin{bmatrix} 0 & 0 & 0 \\ 1 & -2 & 1 \\ 0 & 0 & 0 \end{bmatrix}, \quad \mathbf{K}_v = \begin{bmatrix} 0 & 1 & 0 \\ 0 & -2 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$
$$\mathbf{K} = \frac{1}{3} (\mathbf{K}_{\text{Lap}} + \mathbf{K}_h + \mathbf{K}_v), \quad \mathbf{R} = |\mathbf{I} * \mathbf{K}|$$
This step strips out low-frequency illumination and smooth facial gradients, isolating high-frequency noise.

#### 2. 2D Haar Wavelet High-High (HH) Subband
Transposed convolutions in generative models (like StyleGAN2) leave periodic diagonal checkerboard artifacts. A 2D Haar wavelet decomposition splits the image into four subbands:
* **LL (Low-Low):** Low-pass approximation.
* **LH (Low-High):** Horizontal high-frequency edges.
* **HL (High-Low):** Vertical high-frequency edges.
* **HH (High-High):** Diagonal high-frequency components:
  $$HH(x, y) = \frac{1}{2} \Big( I_{2x, 2y} - I_{2x, 2y+1} - I_{2x+1, 2y} + I_{2x+1, 2y+1} \Big)$$

The rectified subband $|HH|$ is bilinearly upsampled back to $224 \times 224$.

#### 3. Composite Residual Map
The spatial residual and wavelet diagonal residual are blended:
$$\mathbf{C} = 0.5 \mathbf{R} + 0.5 |HH|\uparrow \;\in \mathbb{R}^{224 \times 224 \times 3}$$

#### 4. Lightweight Frequency ConvNet
The composite map $\mathbf{C}$ passes through a 4-stage convolutional encoder:
$$\text{ConvNet: } 3 \to 32 \to 64 \to 128 \to 128 \text{ channels}$$
Each stage contains `Conv2D(stride=2)` $\to$ `BatchNorm` $\to$ `GELU`. Dual adaptive pooling ($[\text{AvgPool} \parallel \text{MaxPool}]$) produces a compact 128-dimensional spectral vector: $\mathbf{e}_{\text{freq}} \in \mathbb{R}^{128}$ (only 0.22M parameters, running in 1.6 ms).

---

## 6. Training Strategy, Decoupled Supervision, & Options Considered

### 6.1 The Modality Competition Trap: Why Early Concatenation Failed
In our initial Phase 2 design (Trial 1), we directly concatenated $\mathbf{e}_{\text{spatial}}$ and $\mathbf{e}_{\text{freq}}$ into a single linear classifier.

**The Catastrophic Failure:** `morph_opencv` detection collapsed from **0.9993 AUC down to 0.2422 AUC** (worse than random chance; complete feature inversion).

**Why Did This Happen?**
* The 4-stage ConvNet has only 0.22M parameters and learns rapidly within the first 1-2 epochs, sending large backpropagation gradients.
* The Vision Transformer with LoRA requires steady, subtle gradient updates to avoid disturbing the pre-trained manifold.
* In a shared classification head, the aggressive convolutional gradients dominated the shared weights, overwriting and corrupting the spatial transformer representations.

### 6.2 The Solution: Decoupled Multi-Task ArcFace Supervision
Instead of sharing a single head, UID-MAD trains **three independent classification heads** concurrently:
1. `spatial_head`: ArcFace($\mathbf{e}_{\text{spatial}} \in \mathbb{R}^{256} \to 2$)
2. `freq_head`: ArcFace($\mathbf{e}_{\text{freq}} \in \mathbb{R}^{128} \to 2$)
3. `fused_head`: ArcFace($\mathbf{e}_{\text{fuse}} \in \mathbb{R}^{256} \to 2$)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{fuse}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{spatial}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{freq}}, y)$$
By weighting auxiliary single-stream heads at 0.5, each stream receives its own gradient signal. The CNN cannot overwrite the ViT, and both streams converge stably.

### 6.3 What is ArcFace and Why Choose It Over Cross-Entropy?
Standard Binary Cross-Entropy (BCE) merely seeks a separating hyperplane. Once classes are linearly separated, training loss plateaus, leaving the decision boundary sensitive to slight out-of-distribution shifts.

**ArcFace (Additive Angular Margin Loss):**
1. $L_2$-normalizes both feature embeddings $\hat{\mathbf{e}}$ and class prototype weights $\hat{\mathbf{w}}_j$:
   $$\|\hat{\mathbf{e}}\| = 1, \quad \|\hat{\mathbf{w}}_j\| = 1 \implies \hat{\mathbf{w}}_j^T \hat{\mathbf{e}} = \cos \theta_j$$
2. Adds a strict angular margin $m = 0.25$ radians to the ground-truth target angle:
   $$\cos(\theta_y + m) = \cos \theta_y \cos m - \sin \theta_y \sin m$$
3. Scales logits by radius $s = 32.0$:
   $$\mathcal{L}_{\text{ArcFace}} = -\log \frac{\exp(s \cdot \cos(\theta_y + m))}{\exp(s \cdot \cos(\theta_y + m)) + \sum_{j \neq y} \exp(s \cdot \cos \theta_j)}$$

ArcFace forces genuine and morph representations into tightly clustered cones on a unit hypersphere, maintaining clean separation when evaluated on unseen attacks.

---

## 7. Inference: Bona Fide Z-Score Decision Calibration

### 7.1 Why Softmax Probability Fusion Fails (The Max-OR Trap)
In ablation experiments, we evaluated standard softmax probability fusion:
$$P_{\text{morph}} = \max(P_{\text{spatial}}, P_{\text{freq}})$$
**Failure Mode:** Out-of-distribution bona fide portraits triggered false positives ($\text{BPCER@1\%} = 100\%$, meaning every honest passenger was flagged).

**Root Cause:**
Different camera sensors introduce distinct Photo-Response Non-Uniformity (PRNU) patterns. When an unseen camera's sensor noise passes through an uncalibrated softmax layer, the unnormalized logit shifts, saturating the sigmoid/softmax to 1.0.

### 7.2 The Proposed Solution: Bona Fide Z-Score Standardization
Instead of using non-linear softmax probabilities, UID-MAD evaluates linear cosine margins:
$$S_s = \cos \theta_{\text{morph}, s} - \cos \theta_{\text{bf}, s}, \quad S_f = \cos \theta_{\text{morph}, f} - \cos \theta_{\text{bf}, f}$$
It standardizes these margins against the empirical mean ($\mu_{\text{bf}}$) and standard deviation ($\sigma_{\text{bf}}$) of bona fide presentations in the active test environment:
$$Z_s = \frac{S_s - \mu_{\text{bf}, s}}{\sigma_{\text{bf}, s} + \varepsilon}, \quad Z_f = \frac{S_f - \mu_{\text{bf}, f}}{\sigma_{\text{bf}, f} + \varepsilon}$$
$$Z_{\text{final}} = 0.5 \cdot Z_s + 0.5 \cdot Z_f$$

### 7.3 Why This Calibration Works
1. Genuine presentations are anchored around $Z = 0.0$ (with standard deviation 1.0).
2. Attack presentation scores reflect standard deviations away from genuine faces. An attack score of $Z = +3.12$ means the sample sits 3.12 standard deviations away from genuine portraits.
3. Camera sensor noise offsets shift both $\mu_{\text{bf}}$ and $S$ equally, canceling out sensor-specific bias.

---

## 8. Experimental Results & Discoveries

### 8.1 Training Configuration on Kaggle
* **Training Dataset:** SMDD Synthetic Morph Dataset (32,000 images: 20,030 bona fide, 11,970 morphs) split 80/20 into 25,600 training and 6,400 validation samples.
* **Hardware:** Dual NVIDIA T4 GPUs (16GB VRAM each) using PyTorch `DataParallel` and FP16 Automatic Mixed Precision (AMP).
* **Engineering Optimizations:**
  * Extracted the 4.5GB dataset archive directly to local NVMe `/tmp` scratch storage to bypass FUSE network filesystem latency.
  * Configured `NUM_WORKERS = 0` to prevent PyTorch CUDA IPC deadlocks and OpenMP CPU oversubscription.
* **Two-Stage Training Schedule (8 Epochs, 58.4 minutes total):**
  * *Stage 1 (Epochs 1-4, Head Warmup):* ViT backbone and LoRA frozen; only projection heads trained.
  * *Stage 2 (Epochs 5-8, LoRA Fine-Tuning):* LoRA adapters unfrozen; trained end-to-end.

```
+-------+--------------------------+------------+------------+-----------+-----------+-------------+
| Epoch | Stage                    | Train Loss | Train Acc  | Val Loss  | Val EER   | Val ROC-AUC |
+-------+--------------------------+------------+------------+-----------+-----------+-------------+
|   1   | Stage 1: Warmup          |   2.7199   |   92.58%   | 3.28e-03  |  0.0201%  |   0.99999   |
|   2   | Stage 1: Warmup          |   1.4864   |   95.96%   | 7.11e-05  | < 0.001%  |   1.00000   |
|   3   | Stage 1: Warmup          |   1.1097   |   96.63%   | 4.47e-03  |  0.0201%  |   0.99999   |
|   4   | Stage 1: Warmup          |   0.8437   |   96.71%   | 3.04e-06  | < 0.001%  |   1.00000   |
|   5   | Stage 2: LoRA Fine-Tune  |   0.6428   |   97.74%   | 7.50e-08  | < 0.001%  |   1.00000   |
|   6   | Stage 2: LoRA Fine-Tune  |   0.5726   |   98.39%   | 4.19e-08  | < 0.001%  |   1.00000   |
|   7   | Stage 2: LoRA Fine-Tune  |   0.5241   |   98.83%   | 2.92e-08  | < 0.001%  |   1.00000   |
|   8   | Stage 2: LoRA Fine-Tune  |   0.5039   |   99.12%   | 2.50e-08  | < 0.001%  |   1.00000   |
+-------+--------------------------+------------+------------+-----------+-----------+-------------+
```

---

### 8.2 Discovery 1: Resolving the Generative Blind Spot
On the zero-shot FRLL-Morphs benchmark (7,274 images), the dual-stream architecture resolved the Phase 1 StyleGAN2 failure:
```
    StyleGAN2 Clean ROC-AUC:      0.4988 (Phase 1) ──► 0.9399 (Phase 2)  [+44.11% Surge!]
    StyleGAN2 Clean EER:          51.00% (Phase 1) ──► 14.66% (Phase 2)  [36.34% Reduction]
    StyleGAN2 Print-Scan ROC-AUC: 0.5780 (Phase 1) ──► 0.9854 (Phase 2)  [3.92% EER]
```

---

### 8.3 Discovery 2: The Print-Scan Orthogonal Frequency Surge
In real-world passport issuance, digital photos are printed onto security paper and scanned back into the system. This analog print-scan channel acts as an optical low-pass filter:
* It blurs subtle blending edges, causing Stream A's spatial AUC to drop:
  * `morph_stylegan` spatial AUC dropped to **0.3155** ($63.94\%$ EER)
  * `morph_webmorph` spatial AUC dropped to **0.6355** ($41.50\%$ EER)
  * `morph_amsl` spatial AUC dropped to **0.3743** ($59.43\%$ EER)
* **The Breakthrough:** Physical printer halftone dot screening, microscopic ink dispersion, and scanner CCD noise create distinct, periodic high-frequency patterns. Stream B's frequency filter amplified these cues, driving frequency stream AUC to **$> 0.997$ across all five generator families**:
  * `morph_opencv` (Print-Scan): **0.9992 AUC** ($1.47\%$ EER)
  * `morph_facemorpher` (Print-Scan): **0.9996 AUC** ($1.06\%$ EER)
  * `morph_webmorph` (Print-Scan): **0.9976 AUC** ($2.49\%$ EER)
  * `morph_stylegan` (Print-Scan): **0.9977 AUC** ($1.92\%$ EER)
  * `morph_amsl` (Print-Scan): **0.9981 AUC** ($1.52\%$ EER)

**Core Takeaway:** Spatial and frequency streams are orthogonally complementary. When physical analog degradation destroys spatial blending cues, frequency residuals rescue the classification decision.

---

### 8.4 The Complete 15-Slice FRLL-Morphs Benchmark Record
Evaluated zero-shot under official ISO/IEC 30107-3 protocols (`results/phase2/frll_results_comparison.csv`):

```
+---------------+------------+-------------+-------------+----------------+----------------+------------------+
| Generator     | Condition  | Spatial AUC | Freq AUC    | Calibrated AUC | Calibrated EER | BPCER @ 1% APCER |
+---------------+------------+-------------+-------------+----------------+----------------+------------------+
| OpenCV        | Clean      |   0.9991    |   0.8564    |     0.9997     |     0.74%      |      0.00%       |
| OpenCV        | JPEG (q=30)|   0.9990    |   0.8576    |     0.9996     |     0.25%      |      0.00%       |
| OpenCV        | Print-Scan |   0.9143    |   0.9992    |     0.9973     |     2.33%      |      2.45%       |
| FaceMorpher   | Clean      |   0.9994    |   0.9506    |     0.9999     |     0.41%      |      0.00%       |
| FaceMorpher   | JPEG (q=30)|   0.9993    |   0.9507    |     0.9999     |     0.49%      |      0.00%       |
| FaceMorpher   | Print-Scan |   0.9342    |   0.9996    |     0.9994     |     1.02%      |      1.47%       |
| WebMorph      | Clean      |   0.9963    |   0.7395    |     0.9968     |     1.96%      |     14.71%       |
| WebMorph      | JPEG (q=30)|   0.9961    |   0.7426    |     0.9967     |     2.21%      |     13.73%       |
| WebMorph      | Print-Scan |   0.6355    |   0.9976    |     0.9893     |     3.39%      |      7.84%       |
| StyleGAN2     | Clean      |   0.9116    |   0.7644    |     0.9399     |    14.66%      |     55.88%       |
| StyleGAN2     | JPEG (q=30)|   0.9083    |   0.7640    |     0.9384     |    14.33%      |     56.37%       |
| StyleGAN2     | Print-Scan |   0.3155    |   0.9977    |     0.9854     |     3.92%      |     17.65%       |
| AMSL          | Clean      |   0.9982    |   0.6605    |     0.9938     |     2.92%      |     16.18%       |
| AMSL          | JPEG (q=30)|   0.9979    |   0.6634    |     0.9935     |     3.23%      |     16.18%       |
| AMSL          | Print-Scan |   0.3743    |   0.9981    |     0.9809     |     6.73%      |     14.22%       |
+---------------+------------+-------------+-------------+----------------+----------------+------------------+
```
* **Performance Floor:** $\ge 0.9809$ ROC-AUC across all 15 operational splits.
* **Overall Mean Performance:** **0.992 ROC-AUC**.

---

### 8.5 Architectural Ablation Matrix (Why Other Options Were Rejected)

```
+-----------------------------------+-----------------+------------------+--------------+---------------------------------------------+
| Architectural Variant             | StyleGAN2 Clean | StyleGAN2 P-Scan | OpenCV Clean | Empirical Finding / Reason Rejected         |
+-----------------------------------+-----------------+------------------+--------------+---------------------------------------------+
| 1. Phase 1 Spatial ViT Baseline   |     0.4988      |      0.5780      |    0.9993    | Generative blind spot (pure random guess).  |
| 2. Phase 2 Early Feature Concat   |     0.8525      |        --        |    0.2422    | Gradient dominance & feature inversion.     |
| 3. Phase 2 Max-OR Softmax Fusion  |     0.5363      |      0.6489      |    0.7924    | Probability saturation (BPCER@1% = 100%).   |
| 4. Phase 2 Spatial Head Alone     |     0.9116      |      0.3155      |    0.9991    | Fails under analog print-scan low-pass blur.|
| 5. Phase 2 Frequency Head Alone   |     0.7644      |      0.9977      |    0.8564    | Sub-optimal on clean digital landmark morphs|
| 6. Phase 2 Fused ArcFace Alone    |     0.1305      |      0.3952      |    0.2937    | Uncalibrated joint logits drift under OOD.  |
| 7. Proposed Z-Score Calibrated    |   0.9399 (Ours) |   0.9854 (Ours)  | 0.9997 (Ours)| Robust zero-shot generalization (0.992 mean)|
+-----------------------------------+-----------------+------------------+--------------+---------------------------------------------+
```

---

## 9. Computational Complexity & Edge Latency Profile

Automated border gates have strict performance requirements:
* Total passenger processing budget: **2.0 to 3.0 seconds**.
* Face Morph Detection sub-module allocation: **$\le 50\text{ ms}$**.
* Privacy regulations (**ICAO Doc 9303 / GDPR**): Biometric data cannot be transmitted to external cloud APIs; inference must execute locally on the gate's edge computer.

```
+-------------------------------------------+------------+--------------------+-----------+---------------------+
| Architectural Module                      | Total Param| Trainable (Stage 2)| Share (%) | Single-Image Latency|
+-------------------------------------------+------------+--------------------+-----------+---------------------+
| Stream A: DINOv2 ViT-S/14 Backbone        |  21.60 M   |  0.59 M (LoRA r=16)|   96.5%   |      14.2 ms        |
| Spatial Aggregator & Projector            |   0.46 M   |  0.46 M            |    2.1%   |       0.6 ms        |
| Stream B: Non-Trainable Filter Bank       |   0.00 M   |  0.00 M (Fixed)    |    0.0%   |       0.4 ms        |
| Stream B: Residual 4-Stage ConvNet        |   0.22 M   |  0.22 M            |    1.0%   |       1.6 ms        |
| Cross-Stream Gated Alignment              |   0.10 M   |  0.10 M            |    0.4%   |       0.4 ms        |
| Decoupled ArcFace Heads (3 x 2)           |   0.001 M  |  0.001 M           |   < 0.1%  |       0.1 ms        |
+-------------------------------------------+------------+--------------------+-----------+---------------------+
| TOTAL UID-MAD ARCHITECTURE                |  22.38 M   |  1.37 M (6.1%)     |  100.0%   |      17.3 ms        |
+-------------------------------------------+------------+--------------------+-----------+---------------------+
```
* **Latency Result:** **17.3 ms per image** on dual NVIDIA T4 GPUs, well within the 50 ms operational budget.
* **Trainable Footprint:** Only 1.37M parameters (6.1% of the model) are trained, keeping adaptation lightweight.

---

## 10. Future Work Roadmap (Phases 3 & 4)

Your model architecture, LoRA adaptation, and benchmark validation are complete. The remaining scope focuses on production packaging and edge deployment:

```
    Weeks 9-11: Phase 3 Edge Optimization
    ┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
    │  ONNX Runtime Export     │ ──► │  TensorRT Kernel Fusion  │ ──► │  INT8 Calibration        │
    │  Static graph, opset 17  │     │  Conv+BN+GELU layer fuse │     │  < 10ms latency, < 25MB  │
    └──────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
                                                  │
                                                  ▼
    Weeks 12-14: Phase 4 Border e-Gate Application & Explainability
    ┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
    │  Spatial Grad-CAM        │     │  2D Fourier Residual     │     │  Inspector Desktop UI    │
    │  Visualizes iris/seams   │     │  Visualizes GAN peaks    │     │  Live camera & passport  │
    └──────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
```

1. **Phase 3: Embedded Edge Optimization (Weeks 9-11):**
   * **ONNX Export (`torch.onnx.export`):** Remove Python runtime dependencies and export a static computational graph.
   * **TensorRT Kernel Fusion:** Merge Conv, BatchNorm, and GELU layers into single fused execution blocks, eliminating GPU memory read/write cycles.
   * **INT8 Quantization:** Quantize weights and activations to 8-bit integers via KL-divergence calibration on genuine samples, reducing model size to $<25$\,MB and lowering latency to $<10$\,ms with $<0.1\%$ change in ROC-AUC.
2. **Phase 4: Border e-Gate Application & Visual Explainability (Weeks 12-14):**
   * **Spatial Grad-CAM Overlays:** Provide real-time visual heatmaps highlighting double pupil contours and boundary blending seams for border control officers.
   * **2D Fourier Spectral Saliency:** Render residual frequency spectra highlighting checkerboard spikes when generative attacks are flagged.
   * **Operator UI Dashboard:** Build an operator interface that ingests passport photos, checks ICAO Doc 9303 compliance, displays calibrated Z-scores, and provides an ACCEPT/REJECT verdict.

---

## 11. Slide-by-Slide Presentation Speaking Script (Exact 13-Slide Final PPT)

### Slide 1: Title (Dual-Stream Face Morph Attack Detection)
* **Time:** 15 to 20 seconds
* **Speaker Script:**
> "Good morning, respected committee members and faculty. Today, we present our mid-semester evaluation on Dual-Stream Face Morph Attack Detection, presenting our framework, UID-MAD. In this project, we have designed, trained, and benchmarked a dual-stream architecture capable of intercepting both digital and physical face morph attacks for Automated Border Control e-Gates under official international ISO standards."

### Slide 2: Team Members & Supervision
* **Time:** 10 seconds
* **Speaker Script:**
> "This project is carried out by Bablu Prajapati, Puranpal Singh, and Himank Gupta, under the esteemed guidance of Dr. Abhilasha Sharma, Assistant Professor in the Department of Computer Engineering."

### Slide 3: 1. Introduction
* **Time:** 45 seconds
* **Speaker Script:**
> "Automated Border Control e-Gates verify travelers against the digital portrait stored in the RFID chip of their biometric passport, standardized under ICAO Doc 9303. A face morph attack compromises this verification model: an accomplice with a clean record and a criminal combine their facial portraits into a single composite image. The accomplice legally obtains a passport using this photo, and because it contains biometric features of both individuals, both can pass through automated e-Gates using the exact same physical credential. We focus on Single-Image Morph Attack Detection (S-MAD), which is mandatory during passport issuance and online applications where no live reference image is available."

### Slide 4: 2. Problem Description (Two Attack Paradigms)
* **Time:** 45 seconds
* **Speaker Script:**
> "Face morphing splits into two fundamentally opposite attack paradigms. The first family is digital landmark-warped morphs, produced by tools like OpenCV or FaceMorpher. These average 68 landmark points and apply piecewise-affine warping, leaving physical cutouts, double iris contours, and blending seams. The second family is generative latent morphs, produced by StyleGAN2. Here, source faces are inverted into the intermediate latent space W+, interpolated, and re-synthesized from scratch. Because the entire face is generated on a continuous neural manifold, there are zero physical blending seams. However, neural upsampling leaves periodic Nyquist checkerboard patterns in the high-frequency spectrum."

### Slide 5: Critical Flaws in Existing Detection Methods
* **Time:** 45 seconds
* **Speaker Script:**
> "Existing detection methods suffer from three critical flaws that our project directly set out to solve:
> First is the Generative Blind Spot: detectors trained solely on spatial image seams collapse to 0.4988 ROC-AUC (pure random chance) on StyleGAN2 morphs, because GAN faces have no spatial cutouts.
> Second is Modality Competition: when combining spatial and frequency streams into a single classifier, fast-converging convolutional branches send overpowering gradients that disrupt and invert the slower-adapting Vision Transformer representations.
> Third is the Print-Scan Degradation Trap: physical passport printing and scanning acts as an analog low-pass filter, smudging out spatial blending seams and causing spatial detectors to fail."

### Slide 6: 3. Related Work & Research Gaps (Part 1)
* **Time:** 30 seconds
* **Speaker Script:**
> "Reviewing prior literature: classical handcrafted texture descriptors like LBP and SIFT fail to generalize across different camera sensors. Standard deep CNNs like ResNet discard fine-grained pixel details in favor of high-level object semantics, missing subtle alpha-blending discontinuities. In contrast, we adopt DINOv2, a self-supervised Vision Transformer trained via patch distillation and masked patch modeling, which retains fine-grained geometric correspondences and edge statistics optimal for detecting localized blending boundaries."

### Slide 7: Related Work & Research Gaps (Part 2)
* **Time:** 30 seconds
* **Speaker Script:**
> "Continuing with research gaps: prior work proved that neural upsampling introduces periodic spectral peaks that can be isolated using high-pass filters and wavelet decompositions. However, existing approaches suffer from major limitations: they use computationally prohibitive occlusion heatmaps that violate e-Gate latency budgets, rely on flawed face recognition backbones like FaceNet that discard forensic cues, or evaluate on tiny datasets without testing cross-generator generalization."

### Slide 8: 4. Proposed Work (Feature Extractors)
* **Time:** 1 minute
* **Speaker Script:**
> "Our proposed UID-MAD framework uses two specialized feature extractors:
> In Stream A, our spatial pathway uses DINOv2. Rather than using standard average pooling (which washes out localized blending seams), we concatenate the class token, patch mean, patch standard deviation, and patch maximum. The patch standard deviation explicitly measures feature variance across the face, turning it into a mathematical seam detector. To fine-tune without catastrophic forgetting, we inject rank-16 LoRA adapters into blocks 6 through 11, training only 0.59 million parameters.
> In Stream B, our spectral pathway completely ignores facial identity. It filters the image through an 8-neighbor Laplacian kernel and a 2D Haar Wavelet High-High subband to isolate diagonal Nyquist checkerboard noise from GAN transposed convolutions. This residual map is encoded by a lightweight 4-stage ConvNet into a 128-dimensional spectral embedding."

### Slide 9: Proposed Work (Decoupled Heads & Z-Score Calibration)
* **Time:** 45 seconds
* **Speaker Script:**
> "To train and evaluate the architecture stably:
> First, to solve the modality competition problem from our early trials, we supervise the network with Decoupled Multi-Task ArcFace. By training three independent heads simultaneously (Spatial, Frequency, and Fused), the fast-learning convolutional updates have their own dedicated head and cannot overwrite the spatial transformer. ArcFace enforces an angular margin of 0.25 radians on a unit hypersphere, ensuring compact intra-class clustering.
> Second, at test time, raw softmax probabilities saturate due to camera sensor pattern noise. We discard softmax and compute linear cosine margins, standardizing them via bona fide Z-score calibration. Genuine faces anchor at zero, and attack scores reflect standard deviations away from genuine faces, making the decision boundary invariant to sensor shifts."

### Slide 10: UID-MAD Architecture (Block Diagram)
* **Time:** 1 minute
* **Speaker Script:**
> "Here is our complete UID-MAD architecture diagram. 
> Tracing from the left: an input facial portrait is split into two pathways. Stream A at the top feeds the RGB image into our frozen DINOv2 backbone with rank-16 LoRA adapters. The spatial tokens pass through our multi-scale aggregator to extract the 1536-dimensional descriptor, producing a 256-dimensional spatial embedding e_spatial.
> Simultaneously, Stream B at the bottom passes the image through our fixed Laplacian and Haar wavelet filters, stripping out identity and isolating periodic checkerboard noise. Our 4-stage ConvNet extracts a 128-dimensional frequency embedding e_freq.
> In the center, a channel gating network dynamically weights and aligns the features into a fused embedding e_fuse.
> During training, shown in red, three decoupled ArcFace heads optimize spatial, frequency, and fused representations simultaneously under our joint loss. 
> Finally, at test time, shown in yellow, linear cosine margins are standardized via bona fide Z-score calibration to output Z_final under strict ISO/IEC 30107-3 biometric standards."

### Slide 11: 5. Result (Protocol, Blind Spot, & Print-Scan Discovery)
* **Time:** 1 minute
* **Speaker Script:**
> "Slide 11 presents our core empirical breakthroughs. 
> We trained on the synthetic SMDD dataset and evaluated zero-shot under ISO/IEC 30107-3 across all 15 operational slices of the real-world FRLL-Morphs benchmark. The model was never exposed to FRLL during training.
> Our first breakthrough was resolving the generative blind spot: on StyleGAN2, our Phase 1 baseline collapsed to 0.4988 AUC (a pure coin flip). With our dual-stream frequency pipeline, clean StyleGAN2 detection leaped to 0.9399 AUC, reducing EER from 51% down to 14.66%.
> Our second breakthrough was the discovery of Orthogonal Domain Complementarity under print-scan degradation. When photos are physically printed and scanned, ink dispersion blurs spatial blending seams, degrading spatial accuracy. However, halftone screening dots and paper grain create periodic high-frequency patterns. Our frequency stream amplified these cues, surging to over 0.997 AUC across every single generator family. This proves that when physical degradation destroys spatial features, frequency residuals rescue the decision."

### Slide 12: Result (Overall Performance & Latency)
* **Time:** 30 seconds
* **Speaker Script:**
> "Across all 15 operational slices (including clean digital, JPEG compressed, and physical print-scan across five diverse morphing tools), UID-MAD achieved an overall mean ROC-AUC of 0.992, never falling below 0.9809 on any split.
> In terms of deployment readiness, single-image inference runs in 17.3 milliseconds using FP16 precision on an NVIDIA T4 GPU. This comfortably meets the strict sub-50 millisecond operational budget required for real-time Automated Border Control e-Gates."

### Slide 13: Conclusion & Thanks
* **Time:** 15 seconds
* **Speaker Script:**
> "In conclusion, our mid-semester work identified the theoretical blind spot of spatial models, designed a dual-stream architecture with LoRA adaptation, and proved its zero-shot robustness across international benchmark standards. In the second half of the semester, we will export to ONNX and TensorRT for edge hardware and build the real-time border inspection interface. 
> Thank you, and we are now ready to take your questions."

---

## 12. Master Viva Defense Bank: 20 Hard Questions & Winning Answers

### Q1: Why did standard face recognition systems fail to detect face morphs?
> *"Face recognition models like ArcFace or FaceNet are trained to maximize identity invariance, meaning they intentionally ignore small textural shifts, lighting changes, and aging to verify a person reliably. Because a morph is engineered to sit directly in the overlapping feature space of two individuals, identity-invariant matchers match both identities with high confidence (>98%). In essence, morphing exploits the fundamental objective of face recognition."*

### Q2: What caused the 0.4988 AUC failure in Phase 1 on StyleGAN2?
> *"In Phase 1, our spatial DINOv2 baseline focused entirely on RGB patch boundaries. Landmark-based morphs (OpenCV, FaceMorpher) leave visible pixel seams and double iris cutouts from Delaunay triangulation warping, which the transformer easily caught (0.9993 AUC). However, StyleGAN2 synthesizes faces continuously from the intermediate latent space $\mathcal{W}+$. It contains zero physical blending seams or boundary cutouts. Because the spatial transformer had no physical artifacts to detect, its performance was equivalent to random guessing (0.4988 AUC, 51% EER)."*

### Q3: Why use LoRA instead of fine-tuning the entire Vision Transformer?
> *"Fully fine-tuning all 21.6M parameters of DINOv2 on 32,000 synthetic images risks catastrophic forgetting, destroying the generalized visual representations learned from millions of images during pre-training. LoRA freezes the base weights $\mathbf{W}_0$ and injects low-rank trainable matrices ($r=16, \alpha=32$) into attention and MLP projections across the final six blocks. This trains only 0.59M parameters (2.6% of the backbone), adapting the model to morph artifacts while preserving its underlying visual representations."*

### Q4: Why apply LoRA only to Blocks 6-11, leaving Blocks 0-5 frozen?
> *"Early transformer blocks (0-5) capture universal low-level visual features like edges, corners, and color distributions, which should remain intact. The final blocks (6-11) model semantic relationships and task-specific patterns. Adapting only the upper layers allows the network to learn forensic anomaly detection without destabilizing low-level feature extraction."*

### Q5: In your patch aggregator, why concatenate $[\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu}_{\text{patch}} \parallel \boldsymbol{\sigma}_{\text{patch}} \parallel \mathbf{z}_{\text{max}}]$?
> *"Standard Vision Transformers use global average pooling ($\boldsymbol{\mu}_{\text{patch}}$), which averages out localized blending seams across all 256 patches. Our concatenation retains key local signals: $\boldsymbol{\sigma}_{\text{patch}}$ (patch standard deviation) measures feature variance across the face, acting as an explicit seam detector since boundary cutouts produce localized variance spikes. Additionally, $\mathbf{z}_{\text{max}}$ preserves the single most prominent local artifact (such as a double pupil) that averaging would dilute."*

### Q6: Why did early feature concatenation fail with a 0.2422 AUC in Trial 1?
> *"Early concatenation suffered from modality competition and gradient dominance. The 4-stage convolutional frequency stream is lightweight (0.22M parameters) and converges in 1-2 epochs, producing large backpropagation gradients. In contrast, the Vision Transformer with LoRA adapts more gradually. When both streams fed into a shared classification head, the CNN's aggressive updates dominated the shared weights, disrupting spatial learning and causing feature inversion."*

### Q7: How does decoupled ArcFace solve that gradient dominance problem?
> *"We separate optimization into three independent ArcFace heads: a spatial head, a frequency head, and a fused head, using a joint loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fuse}} + 0.5 \mathcal{L}_{\text{spatial}} + 0.5 \mathcal{L}_{\text{freq}}$. The auxiliary single-stream heads provide dedicated gradient paths for each stream, insulating the ViT from CNN updates and allowing both pathways to converge stably."*

### Q8: Why did you pick ArcFace over standard Binary Cross-Entropy?
> *"Binary cross-entropy only seeks a linear separating boundary. Once classes are separated, loss stabilizes, leaving the boundary susceptible to domain shifts. ArcFace normalizes features and weights onto a unit hypersphere and enforces an explicit angular margin ($m=0.25$ radians, $s=32.0$). This compacts intra-class clusters and maximizes the gap between bona fide and morph distributions, improving zero-shot generalization on unseen attacks."*

### Q9: Why does the frequency stream surge to $>0.997$ AUC under print-scan conditions while the spatial stream collapses?
> *"Physical printing and scanning act as an optical low-pass filter: ink bleeding and paper texture blur localized blending seams, causing spatial AUC to drop (e.g., to 0.3155 on StyleGAN2). However, halftone screening dots, ink dispersion, and scanner CCD sampling introduce periodic high-frequency noise. Stream B's Laplacian and Haar wavelet filters amplify these high-frequency patterns, driving frequency detection to $>0.997$ AUC across all five generator families. This confirms that the two streams are orthogonally complementary."*

### Q10: Why did you implement Bona Fide Z-score calibration instead of using softmax probabilities?
> *"Raw softmax probabilities saturate under domain shifts. Differences in camera sensor pattern noise (PRNU) shift logit distributions, pushing softmax outputs to 0.0 or 1.0 and causing high false rejection rates ($\text{BPCER@1\%} = 100\%$). Our calibration evaluates linear cosine margins ($S = \cos \theta_{\text{morph}} - \cos \theta_{\text{bf}}$) and standardizes them using bona fide presentations from the active environment ($Z = (S - \mu_{\text{bf}}) / \sigma_{\text{bf}}$). Genuine faces are anchored at $Z = 0.0$, and attacks are scored by their standard deviations from genuine samples, removing camera-specific bias."*

### Q11: Your in-domain SMDD validation error was near zero. Did your model overfit?
> *"In-domain synthetic validation data (SMDD) is known to be easily separable for foundation models because training and validation sets share the same generator distributions. The true evaluation of robustness is our zero-shot out-of-distribution benchmark on FRLL-Morphs, which contains real human subjects and unseen generator families. On FRLL, the model achieved realistic operational error rates (such as 14.66% EER on clean StyleGAN2 and 2.45% to 17.65% BPCER under print-scan), demonstrating genuine generalization without data leakage."*

### Q12: Why did you choose DINOv2 over ResNet50 or EfficientNet?
> *"Convolutional models like ResNet50 possess inductive biases (translation equivariance, local receptive fields) that work well for object classification, but struggle to model long-range correlations between distant facial regions (such as symmetry between the left and right eye, or cheek-to-neck blending transitions). DINOv2's self-attention models pairwise relationships across all 256 patches simultaneously. Furthermore, self-supervised pre-training retains fine-grained surface geometry that supervised ImageNet CNNs discard."*

### Q13: What is the purpose of the Haar wavelet High-High (HH) subband?
> *"A 2D Haar wavelet decomposition separates an image into approximation and directional details. The HH subband applies high-pass filtering along both horizontal and vertical axes, isolating diagonal high-frequency energy. Generative networks rely on transposed convolutions for upsampling, which leave characteristic 2D diagonal checkerboard patterns. Rectifying the HH subband captures these generator-specific artifacts directly."*

### Q14: What is the difference between APCER and BPCER in ISO/IEC 30107-3?
> *"APCER (Attack Presentation Classification Error Rate) is the false acceptance rate for morph attacks - the proportion of fraudulent credentials accepted as genuine. BPCER (Bona Fide Presentation Classification Error Rate) is the false rejection rate for genuine travelers - the proportion of honest citizens wrongly flagged. In border control, APCER must remain low to stop impostors, while BPCER must remain low to prevent terminal bottlenecks."*

### Q15: How does this model satisfy border control latency requirements?
> *"Airport e-Gates allocate an operational latency budget of $\le 50\text{ ms}$ for morph detection. Our complete dual-stream network executes in 17.3 ms on dual NVIDIA T4 GPUs. In Phase 3, exporting to ONNX and compiling via NVIDIA TensorRT with FP16/INT8 quantization will reduce latency below 10 ms, fitting comfortably within the operational window."*

### Q16: What is the SMDD dataset and how was it partitioned?
> *"SMDD (Synthetic Morphing Attack Detection Development dataset) contains 32,000 synthetic facial portraits (20,030 bona fide and 11,970 morph attacks). We partitioned it 80/20 into 25,600 training samples and 6,400 validation samples, using a WeightedRandomSampler to eliminate class imbalance."*

### Q17: What was the zero-shot test set?
> *"The FRLL-Morphs benchmark contains 7,274 images across 5 distinct morph generators (OpenCV, FaceMorpher, WebMorph, AMSL, StyleGAN2) tested across 3 conditions (Clean, JPEG $q=30$, and Print-Scan). The model was never exposed to FRLL during training."*

### Q18: What data augmentations did you use to prevent shortcut learning?
> *"We applied an Albumentations pipeline including random JPEG compression ($q \in [20, 80]$), Gaussian and motion blur, synthetic print-scan downscaling with noise injection, and random grayscale conversion. Converting to grayscale forced the network to learn structural facial geometry and spectral noise rather than color tone shortcuts."*

### Q19: What engineering optimizations were required to train on Kaggle?
> *"We unpacked the 4.5 GB dataset archive directly into the local NVMe `/tmp` directory, bypassing network-attached virtual drive bottlenecks. We also set `NUM_WORKERS = 0` to prevent PyTorch CUDA IPC deadlocks, and utilized Automatic Mixed Precision (FP16) across dual NVIDIA T4 GPUs. Training completed in 58.4 minutes."*

### Q20: What are your deliverables for the second half of the semester?
> *"Our remaining scope focuses on production deployment: exporting the model to ONNX, compiling TensorRT execution engines with INT8 quantization for sub-10ms edge latency, building a standalone Automated Border Control desktop application, and integrating Grad-CAM and 2D Fourier visual explanation overlays for border guards."*
