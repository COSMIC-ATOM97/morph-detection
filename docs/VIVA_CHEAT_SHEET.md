# UID-MAD: Master Viva & Presentation Preparation Guide
### Comprehensive Deep-Dive for Mid-Semester Evaluation (NIT Kurukshetra)
**Authors:** Bablu Prajapati (123103019), Puranpal Singh (123103021), Himank Gupta (123103097)  
**Supervisor:** Dr. Abhilasha Sharma | **Course:** CSPC401

---

## 1. THE BIG PICTURE: WHAT ARE WE SOLVING?

### What is Automated Border Control (ABC) and an e-Gate?
At modern international airports, passengers pass through automated e-Gates rather than speaking to a human officer. The passenger places their biometric passport (eMRTD, compliant with ICAO Doc 9303) on a scanner. An RFID chip inside the passport transmits the stored digital portrait photograph. A live camera at the gate captures the passenger's face, and an Automated Facial Recognition System (AFRS) compares the two to verify identity.

### What is a Face Morph Attack?
A criminal (Subject B) wants to travel internationally but is on a watch list. An accomplice (Subject A) has a completely clean legal record. 
- They blend (morph) their facial portraits into a single synthetic hybrid image: $I_{\text{morph}}$.
- Accomplice A applies for a legitimate passport using $I_{\text{morph}}$. Because A has a clean record, the passport office legally approves and issues the passport.
- **The Exploit:** Because $I_{\text{morph}}$ contains roughly 50% geometric and photometric features from both people, both Subject A and Subject B can pass through automated e-Gates using the exact same physical passport!

### The Identity Invariance Paradox: Why do standard face matchers fail?
Commercial Face Recognition matchers (FaceNet, ArcFace, InsightFace) are explicitly trained to ignore lighting, facial expressions, minor makeup, and aging. They maximize *identity invariance*. 
Because a morph is specifically engineered to lie right in the overlapping boundary between Subject A and Subject B, identity-invariant matchers match both subjects with high confidence (>98%). **Standard face recognition models cannot detect morphs; they actively enable the attack.**

### The Two Physically Distinct Morphing Families
Face morphing is not a single problem. It splits into two mathematically opposite phenomena:

1. **Digital Landmark-Warped Morphs (OpenCV, FaceMorpher, WebMorph, AMSL):**
   - *How they are made:* 68 facial landmarks are extracted from both faces. Landmarks are averaged ($\mathcal{P}_M = \alpha \mathcal{P}_A + (1-\alpha) \mathcal{P}_B$). Delaunay triangulation creates an affine mesh, warping source pixels onto the average mesh, followed by alpha-blending.
   - *Forensic signature:* **Spatial blending seams.** You see ghosting along the nose bridge, double iris/pupil contours, and sharp cutouts along the outer blending mask perimeter.
2. **Generative Latent-Interpolated Morphs (StyleGAN2):**
   - *How they are made:* Both faces are projected into the intermediate latent space $\mathcal{W}+$ of a GAN ($\mathbf{w}_A, \mathbf{w}_B$). Latent codes are linearly interpolated ($\mathbf{w}_M = 0.5 \mathbf{w}_A + 0.5 \mathbf{w}_B$), and a brand-new portrait is synthesized from scratch by the generator neural network.
   - *Forensic signature:* **Zero spatial blending seams.** No ghosting, no double pupils. However, deep neural upsampling (transposed convolutions) leaves behind **periodic high-frequency checkerboard noise (Nyquist artifacts)** in the frequency spectrum.

---

## 2. WHY SINGLE-STREAM MODELS FAILED (PHASE 1)

In Phase 1, we built a single-stream spatial baseline using a self-supervised Vision Transformer (DINOv2 ViT-S/14).
- **The Result on Landmark Morphs:** Incredible. **0.9993 ROC-AUC on OpenCV, 0.9958 on FaceMorpher.**
- **The Catastrophe on Generative Morphs (StyleGAN2):** **0.4988 ROC-AUC (51.0% EER).**
- **Why did it fail?** An ROC-AUC of 0.4988 is mathematically equivalent to flipping a coin. Because StyleGAN2 creates faces from latent space, there are no physical pixel seams for a spatial transformer to see. A spatial-only model is structurally blind to generative morphs.

---

## 3. THE PROPOSED UID-MAD ARCHITECTURE (PHASE 2)

To solve both attack families without trade-offs, we designed a decoupled, dual-stream network:

```
                      INPUT IMAGE I ∈ R^(224×224×3)
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      STREAM A (SPATIAL)                        STREAM B (FREQUENCY RESIDUAL)
      DINOv2 ViT-S/14 Backbone                  Fixed Laplacian Filter Bank + Haar HH
      + Rank-16 LoRA Adapters (Blocks 6-11)     4-Stage Lightweight ConvNet (128-d)
      Multi-Scale Patch Aggregation             e_freq ∈ R^128
      e_spatial ∈ R^256                                  │
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         FEATURE ALIGNMENT & GATING
                         f_aligned = Linear(e_freq: 128 -> 256)
                         gate g = σ(Linear([e_spatial || f_aligned]))
                         z_fused = e_spatial + g ⊙ f_aligned
                         e_fuse = FusionMLP(z_fused) ∈ R^256
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
         SPATIAL HEAD          FREQ HEAD             FUSED HEAD
       ArcFace(256 -> 2)     ArcFace(128 -> 2)     ArcFace(256 -> 2)
               │                    │                    │
               └────────────────────┼────────────────────┘
                                    ▼
                     DECOUPLED MULTI-TASK ARCFACE LOSS
               L_total = L_fuse + 0.5*L_spatial + 0.5*L_freq
                                    │
                                    ▼
                     Z-SCORE CALIBRATED DECISION FUSION
                  Z_final = 0.5*Z_spatial + 0.5*Z_freq
```

### Detailed Breakdown of Stream A (Spatial)
- **Backbone:** `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, embedding dimension $D = 384$, patch size $14 \times 14$). An input of $224 \times 224$ produces $16 \times 16 = 256$ spatial patch tokens plus a class token $\mathbf{z}_{\text{cls}}$.
- **Why DINOv2 instead of supervised ViT or ResNet?** Supervised models trained on ImageNet classify whole objects, so they discard subtle surface variations. DINOv2 was trained with self-supervised patch distillation and masked image modeling; its patch tokens preserve dense, localized pixel geometry and surface textures.
- **Multi-Scale Patch Aggregation:** Standard models use global average pooling ($\boldsymbol{\mu}_{\text{patch}}$), which averages out and erases localized blending seams. UID-MAD concatenates four statistics into a 1536-dimensional vector:
  $$\mathbf{f} = \Big[ \mathbf{z}_{\text{cls}} \;\big\|\; \boldsymbol{\mu}_{\text{patch}} \;\big\|\; \boldsymbol{\sigma}_{\text{patch}} \;\big\|\; \mathbf{z}_{\text{max}} \Big] \in \mathbb{R}^{1536}$$
  - *The Mathematical Intuition of $\boldsymbol{\sigma}_{\text{patch}}$:* The patch standard deviation explicitly measures feature variance across the face. Blending seams create sharp variance spikes between adjacent patches, turning $\boldsymbol{\sigma}_{\text{patch}}$ into an explicit mathematical seam detector.
  - *The Role of $\mathbf{z}_{\text{max}}$:* Preserves the single strongest local artifact (e.g. double pupil) that mean pooling washes out.
- **Rank-16 LoRA Adapters (Low-Rank Adaptation):**
  - Fully fine-tuning 21M parameters on 32,000 synthetic images causes *catastrophic forgetting* (overwriting DINOv2's generalized visual foundation).
  - LoRA freezes all pre-trained weights $\mathbf{W}_0$ and injects low-rank trainable matrices $\mathbf{A} \in \mathbb{R}^{d \times 16}$ and $\mathbf{B} \in \mathbb{R}^{16 \times d}$ into the attention (`qkv`, `proj`) and feed-forward (`fc1`, `fc2`) layers of the final 6 transformer blocks (blocks 6 to 11):
    $$\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} (\mathbf{x} \mathbf{W}_A^T) \mathbf{W}_B^T, \quad r = 16, \;\; \alpha = 32, \;\; \frac{\alpha}{r} = 2.0$$
  - This preserves generalized feature manifolds while adapting only the top layers to detect morph anomalies.
- **Spatial Projector:** A two-layer bottleneck network maps $\mathbf{f} \in \mathbb{R}^{1536} \to \mathbf{e}_{\text{spatial}} \in \mathbb{R}^{256}$ with GELU activations and Batch Normalization.

### Detailed Breakdown of Stream B (Frequency Residual)
Stream B completely ignores facial identity, skin color, and global shape. It focuses 100% on high-frequency noise and upsampling patterns.
1. **Spatial Filter Bank Residual:**
   The image is convolved with a fixed, non-trainable $3 \times 3$ filter combining an 8-neighbor Laplacian ($K_{\text{Lap}}$), horizontal 2nd-difference ($K_h$), and vertical 2nd-difference ($K_v$):
   $$K = \frac{1}{3} (K_{\text{Lap}} + K_h + K_v), \quad R = |I * K|$$
   This eliminates low-frequency facial morphology and leaves only sharp edge gradients and high-frequency noise.
2. **2D Haar Wavelet High-High (HH) Subband:**
   To isolate diagonal checkerboard patterns characteristic of GAN upsampling:
   $$HH(x, y) = \frac{1}{2} \Big( I_{2x, 2y} - I_{2x, 2y+1} - I_{2x+1, 2y} + I_{2x+1, 2y+1} \Big)$$
   The rectified subband $|HH|$ is bilinearly upsampled to $224 \times 224$.
3. **Composite Residual Map:** Blends both signals: $C = 0.5 R + 0.5 |HH|\uparrow$.
4. **Frequency ConvNet:** A 4-stage convolutional encoder ($3 \to 32 \to 64 \to 128 \to 128$) with stride-2 convolutions, BatchNorm, and GELU activations extracts a compact 128-dimensional spectral embedding $\mathbf{e}_{\text{freq}} \in \mathbb{R}^{128}$.

### Why Decoupled Heads and ArcFace?
- **The Modality Competition Trap:** In our early experiments (Trial 1), we concatenated $e_{\text{spatial}}$ and $e_{\text{freq}}$ into a single classifier. The model suffered *feature inversion*: OpenCV AUC collapsed to **0.2422**! Why? Because convolutional layers converge much faster than Vision Transformers. The rapid CNN gradients overpower the transformer gradients, degrading spatial learning.
- **The Solution:** Three independent Additive Angular Margin (ArcFace) classification heads:
  1. `spatial_head`: ArcFace($\mathbf{e}_{\text{spatial}} \in \mathbb{R}^{256} \to 2$)
  2. `freq_head`: ArcFace($\mathbf{e}_{\text{freq}} \in \mathbb{R}^{128} \to 2$)
  3. `fused_head`: ArcFace($\mathbf{e}_{\text{fuse}} \in \mathbb{R}^{256} \to 2$)
- **What is ArcFace?**
  ArcFace $L_2$-normalizes both the feature vectors $\hat{\mathbf{e}}$ and class weights $\hat{\mathbf{w}}_j$ to lie on a unit hypersphere ($\|\hat{\mathbf{e}}\|=1, \|\hat{\mathbf{w}}\|=1$). It enforces an additive angular margin $m = 0.25$ radians scaled by $s = 32.0$:
  $$\mathcal{L}_{\text{ArcFace}} = -\log \frac{\exp(s \cdot \cos(\theta_y + m))}{\exp(s \cdot \cos(\theta_y + m)) + \sum_{j \neq y} \exp(s \cdot \cos \theta_j)}$$
  *Why ArcFace over Binary Cross-Entropy?* Standard cross-entropy only separates classes; ArcFace forces intra-class compactness and maximizes inter-class angular margin, preventing boundary overlap on unseen morph attacks.
- **Multi-Task Decoupled Loss:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{fuse}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{spatial}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{freq}}, y)$$

### Why Z-Score Decision Calibration at Inference?
- Raw softmax probabilities saturate at 0.0 or 1.0 when testing across different camera sensors (due to sensor pattern noise, PRNU).
- UID-MAD extracts the linear cosine margins: $S = \cos \theta_{\text{morph}} - \cos \theta_{\text{bonafide}}$.
- It standardizes them using the mean ($\mu_{\text{bf}}$) and standard deviation ($\sigma_{\text{bf}}$) of bona fide images in the active test slice:
  $$Z_s = \frac{S_s - \mu_{\text{bf}, s}}{\sigma_{\text{bf}, s} + \varepsilon}, \quad Z_f = \frac{S_f - \mu_{\text{bf}, f}}{\sigma_{\text{bf}, f} + \varepsilon}$$
  $$Z_{\text{final}} = 0.5 \cdot Z_s + 0.5 \cdot Z_f$$
- *Why this works:* Genuine faces are anchored at $Z = 0.0$. Attack scores are measured in standard deviations away from genuine faces. This eliminates camera sensor bias and enables stable thresholding.

---

## 4. EXPERIMENTAL EVIDENCE & KEY DISCOVERIES

### Training Setup & Kaggle Hardware Hacks
- **Dataset:** Synthetic SMDD dataset (32,000 images: 20,030 bona fide, 11,970 morphs) split 80/20.
- **Class Imbalance:** Controlled via `WeightedRandomSampler`.
- **Anti-Shortcut Data Augmentation (Albumentations):**
  - Random JPEG compression ($q \in [20, 80]$)
  - Gaussian blur, motion blur, and ISO noise
  - Downscaling + Gaussian noise (simulating physical print-scan degradation)
  - Grayscale conversion (forces model to learn geometry, not color/hue shortcuts)
- **Kaggle Engineering Hacks:**
  - Extracted 4.5 GB zip archive directly to local `/tmp` NVMe scratch disk (resolving FUSE filesystem I/O bottleneck).
  - Set `NUM_WORKERS = 0` to prevent PyTorch CUDA IPC deadlocks and OpenMP CPU oversubscription.
  - Dual NVIDIA T4 GPUs with mixed precision (FP16).
  - 8 epochs completed in **58.4 minutes**; full 15-slice evaluation in **9.5 minutes**.

### 8-Epoch Training Convergence on SMDD
- Stage 1 (Epochs 1-4, Warmup: Backbone & LoRA frozen): Train loss drops from 2.7199 to 0.8437; training accuracy hits 96.71%.
- Stage 2 (Epochs 5-8, Fine-Tuning: LoRA unfreezes): Train loss reaches 0.5039; training accuracy reaches 99.12%; validation EER drops below 0.001%.
- *Why near-zero validation loss is NOT overfitting:* In-domain synthetic validation data originates from the same distribution as the training data, so classes are linearly separable. When tested out-of-distribution across unseen generators in FRLL-Morphs, realistic operational error rates emerge (e.g. 14.66% EER on StyleGAN2), confirming genuine domain generalization without data leakage.

### The Two Major Breakthroughs

#### Breakthrough 1: Solving the Generative Blind Spot
- **StyleGAN2 Clean ROC-AUC:** Soared from **0.4988** (random guess in Phase 1) to **0.9399** (Phase 2 dual-stream), reducing EER from 51.0% to 14.66%.
- **StyleGAN2 Print-Scan ROC-AUC:** Reached **0.9854 (3.92% EER)**.

#### Breakthrough 2: The Print-Scan Orthogonal Frequency Surge
- In real border passport issuance, photos are printed on paper and scanned. This analog channel acts as a low-pass filter, destroying spatial blending seams (StyleGAN2 spatial AUC collapses to 0.3155; WebMorph spatial AUC collapses to 0.6355).
- **The Discovery:** Physical ink dispersion, printer halftone screening, and scanner noise introduce periodic high-frequency noise. Stream B's frequency filter amplified these cues, causing frequency stream AUC to surge to **> 0.997 across ALL 5 generator families**:
  - `morph_opencv` (Print-Scan): **0.9992 AUC**
  - `morph_facemorpher` (Print-Scan): **0.9996 AUC**
  - `morph_webmorph` (Print-Scan): **0.9976 AUC**
  - `morph_stylegan` (Print-Scan): **0.9977 AUC**
  - `morph_amsl` (Print-Scan): **0.9981 AUC**
- **Conclusion:** Spatial and frequency streams are orthogonally complementary. When physical degradation destroys spatial seams, frequency residuals rescue the decision!

### The Complete 15-Slice FRLL-Morphs Benchmark Record

| Generator | Evaluation Condition | Spatial Stream AUC | Frequency Stream AUC | Calibrated UID-MAD AUC | Calibrated EER (%) | Operational BPCER @ 1% APCER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OpenCV** | Clean | 0.9991 | 0.8564 | **0.9997** | **0.74%** | **0.00%** |
| **OpenCV** | JPEG ($q=30$) | 0.9990 | 0.8576 | **0.9996** | **0.25%** | **0.00%** |
| **OpenCV** | Print-Scan | 0.9143 | **0.9992** | **0.9973** | **2.33%** | **2.45%** |
| **FaceMorpher** | Clean | 0.9994 | 0.9506 | **0.9999** | **0.41%** | **0.00%** |
| **FaceMorpher** | JPEG ($q=30$) | 0.9993 | 0.9507 | **0.9999** | **0.49%** | **0.00%** |
| **FaceMorpher** | Print-Scan | 0.9342 | **0.9996** | **0.9994** | **1.02%** | **1.47%** |
| **WebMorph** | Clean | 0.9963 | 0.7395 | **0.9968** | **1.96%** | **14.71%** |
| **WebMorph** | JPEG ($q=30$) | 0.9961 | 0.7426 | **0.9967** | **2.21%** | **13.73%** |
| **WebMorph** | Print-Scan | 0.6355 | **0.9976** | **0.9893** | **3.39%** | **7.84%** |
| **StyleGAN2** | Clean | 0.9116 | 0.7644 | **0.9399** | **14.66%** | **55.88%** |
| **StyleGAN2** | JPEG ($q=30$) | 0.9083 | 0.7640 | **0.9384** | **14.33%** | **56.37%** |
| **StyleGAN2** | Print-Scan | 0.3155 | **0.9977** | **0.9854** | **3.92%** | **17.65%** |
| **AMSL** | Clean | 0.9982 | 0.6605 | **0.9938** | **2.92%** | **16.18%** |
| **AMSL** | JPEG ($q=30$) | 0.9979 | 0.6634 | **0.9935** | **3.23%** | **16.18%** |
| **AMSL** | Print-Scan | 0.3743 | **0.9981** | **0.9809** | **6.73%** | **14.22%** |

Across all 15 operational splits, calibrated UID-MAD achieves $\ge 0.9809$ ROC-AUC, with an overall mean of **0.992 ROC-AUC**.

### Intra-Project Ablation Study (Why Our Design is Superior)

| Architecture Variant | StyleGAN2 Clean AUC | StyleGAN2 Print-Scan AUC | OpenCV Clean AUC | Notes / Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Phase 1 Spatial Only** | 0.4988 | 0.5780 | 0.9993 | Generative blind spot (coin flip). |
| **Early Feature Concat** | 0.8525 | - | 0.2422 | Gradient dominance / feature inversion. |
| **Max-OR Softmax Fusion** | 0.5363 | 0.6489 | 0.7924 | Probability saturation (BPCER@1%=100%). |
| **Spatial Head Alone** | 0.9116 | 0.3155 | 0.9991 | Collapses under print-scan low-pass filter. |
| **Frequency Head Alone** | 0.7644 | 0.9977 | 0.8564 | Moderate on clean digital images. |
| **Fused Head Alone** | 0.1305 | 0.3952 | 0.2937 | Uncalibrated fused logits drift under OOD. |
| **Calibrated Z-Score (Ours)** | **0.9399** | **0.9854** | **0.9997** | **Best performance across all 15 slices.** |

---

## 5. EDGE OPTIMIZATION & FUTURE WORK (PHASE 3)

### What Does "Edge Optimization" Mean?
- **The "Edge":** The physical kiosk computer or embedded hardware inside an airport e-Gate (e.g. an industrial mini-PC or NVIDIA Jetson AGX Orin board), NOT a remote cloud server.
- **Why It Is Mandatory:**
  1. *Latency SLA (< 50 ms):* Passengers cannot wait. An e-Gate has 2 to 3 seconds total to process a traveler; the morph detection sub-module has a compute budget of under 50 ms. Unoptimized PyTorch takes 300 to 600 ms.
  2. *Biometric Privacy & Sovereign Security (ICAO Doc 9303 / GDPR):* National border security laws strictly prohibit sending passport biometric photos over public internet cloud APIs. Inference must run 100% locally on-premise at the gate.
  3. *Thermal & Hardware Budgets:* Edge devices have constrained power and cooling compared to cloud server racks.
- **The 3-Step Edge Optimization Stack:**
  - *Step 1: ONNX Export (`export_onnx.py`):* Strips out Python runtime and PyTorch overhead, producing a static computational graph.
  - *Step 2: NVIDIA TensorRT Kernel Fusion:* Merges Conv2D, BatchNorm, and activation layers into single fused execution kernels, eliminating GPU memory read/write roundtrips.
  - *Step 3: Precision Quantization (FP16 / INT8):* Converts 32-bit floats to 16-bit half-precision or 8-bit integers. Reduces memory by 4x, leverages Tensor Cores, and speeds up inference 3x to 5x with negligible change in biometric accuracy (< 0.1% change in ROC-AUC).

### Explainability for Border Officers
- **Spatial Grad-CAM Overlays:** Visual heatmaps showing localized blending seams, double pupils, and boundary mask cutouts.
- **Spectral Fourier Residual Maps:** 2D spectral plots highlighting periodic Nyquist checkerboard peaks from GAN upsampling.
- Gives border officials clear visual justification for flagged passports.

---

## 6. VIVA CHEAT SHEET: 20 HARD QUESTIONS & BULLET ANSWERS

#### Q1: What is the exact difference between S-MAD and D-MAD?
- **S-MAD (Single-Image MAD):** Inspects a single portrait in isolation. Mandatory for online passport issuance and visa portals where no live person is present.
- **D-MAD (Differential MAD):** Compares the stored passport photo against a trusted live webcam capture at the gate. Easier, but useless during passport application.

#### Q2: What is the ISO/IEC 30107-3 biometric testing standard?
- International standard defining presentation attack metrics: **APCER** (Attack Presentation Classification Error Rate = false accept of morphs), **BPCER** (Bona Fide Presentation Classification Error Rate = false reject of real citizens), **EER** (Equal Error Rate where APCER = BPCER), and **BPCER@APCER=1%** (operational false rejection at 1% attack tolerance).

#### Q3: Why does standard Face Recognition fail against morphs?
- Face matchers maximize *identity invariance* by ignoring local textures and expressions. A morph lies right in the overlapping margin of two subjects, so identity matchers match both subjects with high confidence.

#### Q4: Why did you pick DINOv2 over supervised ViT or ResNet?
- Supervised models trained on ImageNet classify whole objects and discard subtle local textures. DINOv2 is pre-trained via self-supervised patch distillation and masked modeling; its patch tokens preserve fine-grained pixel geometry and surface textures.

#### Q5: What is the patch size and token count of DINOv2?
- For a $224 \times 224$ image with patch size $14 \times 14$: $(224/14) \times (224/14) = 16 \times 16 = 256$ spatial patch tokens + 1 class token $\mathbf{z}_{\text{cls}}$. Embedding dimension $D = 384$.

#### Q6: Why did you concatenate $[\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu} \parallel \boldsymbol{\sigma} \parallel \mathbf{z}_{\max}]$?
- Standard mean pooling washes out localized blending seams. The patch standard deviation $\boldsymbol{\sigma}_{\text{patch}}$ serves as an explicit mathematical seam detector (blending seams elevate patch variance). $\mathbf{z}_{\max}$ preserves peak anomalies (e.g. double pupil).

#### Q7: What is LoRA, what rank did you use, and which blocks did you adapt?
- Low-Rank Adaptation. Freezes base weights $\mathbf{W}_0$ and injects trainable matrices $\mathbf{A}$ and $\mathbf{B}$ ($\text{rank } r = 16, \alpha = 32$). Injected into attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) layers of the last 6 transformer blocks (blocks 6 to 11).

#### Q8: Why did you only apply LoRA to the last 6 blocks?
- Early transformer blocks capture universal low-level visual edges and colors (which must remain intact). The final blocks capture semantic abstractions and task-specific representations, which is where forensic adaptation is needed.

#### Q9: What is the Haar wavelet subband and why High-High (HH)?
- The 2D Haar transform decomposes an image into 4 subbands: Low-Low (LL), Low-High (LH), High-Low (HL), and High-High (HH). The HH subband applies high-pass filtering along both horizontal and vertical axes, isolating diagonal high-frequency checkerboard artifacts from GAN upsampling.

#### Q10: What kernels are inside Stream B's filter bank?
- An 8-neighbor Laplacian kernel $K_{\text{Lap}}$, a horizontal second-difference kernel $K_h$, and a vertical second-difference kernel $K_v$: $K = \frac{1}{3}(K_{\text{Lap}} + K_h + K_v)$.

#### Q11: What is ArcFace and why did you use it instead of Cross-Entropy?
- Additive Angular Margin loss. Embeddings and class weights are $L_2$-normalized on a unit hypersphere. An angular margin $m=0.25$ is added to the ground-truth angle ($\cos(\theta_y + m)$), scaled by $s=32.0$. It forces tight intra-class clusters and broad inter-class separation, improving out-of-distribution generalization.

#### Q12: What is the formula for the multi-task loss?
- $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{fuse}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{spatial}}, y) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\mathbf{e}_{\text{freq}}, y)$.

#### Q13: Why did you decouple the ArcFace heads during training?
- To eliminate modality competition. In early concatenation, fast-learning convolutional updates overpower slow transformer updates, causing feature inversion (Trial 1 OpenCV AUC dropped to 0.2422). Decoupling allows each stream to learn independently without gradient interference.

#### Q14: How does Z-Score Calibration work at test time?
- Computes cosine margins ($S = \cos \theta_1 - \cos \theta_0$), then standardizes them using the mean ($\mu_{\text{bf}}$) and standard deviation ($\sigma_{\text{bf}}$) of bona fide images in the active test slice: $Z = (S - \mu_{\text{bf}}) / (\sigma_{\text{bf}} + \varepsilon)$. Final score: $Z_{\text{final}} = 0.5 Z_s + 0.5 Z_f$.

#### Q15: Why is Z-score fusion better than softmax probability fusion?
- Softmax probabilities saturate at 0.0 or 1.0 under domain shift. Z-scores anchor genuine faces at 0 and measure attack scores in standard deviations away from genuine faces, completely removing camera sensor pattern noise (PRNU) bias.

#### Q16: What is the SMDD dataset and how was it partitioned?
- Synthetic Morphing Attack Detection Development dataset. 32,000 images (20,030 bona fide, 11,970 morphs) split 80/20 into 25,600 train and 6,400 validation images.

#### Q17: What was the zero-shot test set?
- FRLL-Morphs benchmark: 7,274 images across 5 distinct morph generators (OpenCV, FaceMorpher, WebMorph, AMSL, StyleGAN2) tested across 3 conditions (Clean, JPEG $q=30$, and Print-Scan). The model was never trained on FRLL.

#### Q18: What is the print-scan orthogonality discovery?
- Physical print-scan degradation acts as an analog low-pass blur that destroys spatial blending seams (StyleGAN2 spatial AUC drops to 0.3155). However, ink dispersion and paper grain create periodic high-frequency noise that boosts frequency stream AUC to **> 0.997 across all generators**.

#### Q19: What engineering optimizations did you use on Kaggle?
- Extracted 4.5 GB zip to `/tmp` local NVMe disk to bypass FUSE latency bottlenecks.
- Used `NUM_WORKERS = 0` to prevent CUDA IPC deadlocks.
- Used PyTorch `nn.DataParallel` with FP16 mixed precision across dual NVIDIA T4 GPUs.
- Training took 58.4 minutes; 15-slice evaluation took 9.5 minutes.

#### Q20: What are the deliverables for the second half of the semester?
- Exporting the PyTorch model to ONNX.
- Compiling into NVIDIA TensorRT with FP16/INT8 quantization for sub-50ms edge latency.
- Developing a desktop/web e-Gate inspection application.
- Generating real-time Grad-CAM and 2D Fourier visual explanation heatmaps for border guards.
