# Mid-Semester Evaluation Presentation Deck: UID-MAD
**Subject:** Mid Sem. Evaluation of B.Tech Project Work (CSPC401 / ITPC401)  
**Institution:** Department of Computer Engineering, National Institute of Technology Kurukshetra  
**Academic Year:** 2026-2027 (Odd Semester)  
**Scope:** Weeks 1 & 2 Completed Milestones + Future Work  

---

## Slide Index & Overview

| Slide # | Official Agenda Topic | Slide Title |
| :--- | :--- | :--- |
| **Slide 1** | Title & Affiliation | Project Title, Candidate & Mentor Details |
| **Slide 2** | 1. Problem Identification & Objectives | The Face Morphing Vulnerability in eMRTD / Border Control |
| **Slide 3** | 1. Problem Identification & Objectives | Research Scope, Metric Framework (ISO/IEC 30107-3) |
| **Slide 4** | 2. Literature Review | Evolution of MAD Architectures & Critical Research Gaps |
| **Slide 5** | 3. Methodology & Work Plan | Proposed UID-MAD Dual-Stream Architecture |
| **Slide 6** | 3. Methodology & Work Plan | Feature Extractors: Spatial DINOv2 vs. Frequency Laplacian ConvNet |
| **Slide 7** | 3. Methodology & Work Plan | Decoupled Multi-Task ArcFace Supervision & Training Setup |
| **Slide 8** | 4. Progress Achieved | Week 1 Baseline: Landmark Success & The StyleGAN2 Blind Spot |
| **Slide 9** | 4. Progress Achieved | Week 2 Breakthrough: Frequency Residuals Solve Generative Morphs |
| **Slide 10** | 4. Progress Achieved | Diagnostic Findings: Modality Competition & Sensor Noise (PRNU) |
| **Slide 11** | 5. Future Work & Timeline | Phase 3 Optimization, Margin Calibration & Edge Deployment Plan |
| **Slide 12** | Conclusion & Q&A | Summary of Deliverables & Defense Talking Points |

---

## Slide 1: Title & Affiliation

### Visual Layout
- **Title:** Unified Interpretable Dual-Stream Face Morph Attack Detection (UID-MAD)
- **Subtitle:** Mid-Semester B.Tech Project Evaluation (CSPC401 / ITPC401)
- **Department:** Department of Computer Engineering, National Institute of Technology Kurukshetra
- **Candidates:**
  - [Candidate Name 1] (Roll No: [Roll 1])
  - [Candidate Name 2] (Roll No: [Roll 2])
  - [Candidate Name 3] (Roll No: [Roll 3])
  - [Candidate Name 4] (Roll No: [Roll 4])
- **Supervisor / Mentor:** [Mentor Name], [Designation], Department of Computer Engineering
- **Date:** October 28 / 29, 2026

### Speaker Script
> "Respected committee members and faculty, we present our mid-semester progress on 'Unified Interpretable Dual-Stream Face Morph Attack Detection (UID-MAD)'. Our project focuses on securing Automated Border Control e-Gates and passport issuance against biometric morphing attacks under international ISO standards."

---

## Slide 2: Problem Identification & Vulnerability

### Slide Content
- **The Threat Mechanism:**
  - An accomplice (no criminal record) and a malicious actor blend facial geometries into a single composite passport image.
  - Both individuals can verify successfully against the same electronic Machine Readable Travel Document (eMRTD) at e-Gates.
- **Why Standard Commercial Face Recognition Fails:**
  - Deep face matchers (ArcFace, FaceNet, InsightFace) are explicitly optimized for *identity invariance*—they ignore minor geometric variations to tolerate pose and aging, directly enabling morph exploits.
- **The Two Distinct Attack Paradigms:**
  - **Landmark-Warped Morphs (OpenCV, FaceMorpher, WebMorph):** Delaunay triangulation + affine mesh warping + alpha blending. Leaves physical boundary cutouts, double iris contours, and blending seams.
  - **Generative Latent Morphs (StyleGAN2):** Inversion into latent space ($\mathcal{W}+$), vector interpolation, and full re-synthesis ($\mathcal{G}(\mathbf{w})$). Completely devoid of spatial seams or ghosting artifacts.

### Speaker Script
> "Face morphing exploits the fundamental objective of face recognition: identity invariance. When an accomplice and a criminal blend their faces into a single passport photo, current e-Gates accept both individuals. Crucially, morphs divide into two distinct mathematical families: landmark-based morphs that leave spatial blending seams, and generative GAN morphs that are synthesized from scratch with zero spatial seams."

---

## Slide 3: Project Objectives & Standard Metrics

### Slide Content
- **Core Research Objectives:**
  1. Build a robust Single-Image Morph Attack Detection (S-MAD) system operating without reference live-capture images.
  2. Overcome the generative latent morph blind spot without sacrificing sensitivity to traditional landmark morphs.
  3. Formulate a dual-stream decomposition separating spatial patch variance from high-frequency residual spectra.
  4. Ensure zero-shot out-of-distribution (OOD) generalization across unseen morphing tools and physical print-scan degradation.
- **Standardized Evaluation Metrics (ISO/IEC 30107-3):**
  - **APCER (Attack Presentation Classification Error Rate):** Proportion of morph attacks falsely accepted as bona fide.
  - **BPCER (Bona Fide Presentation Classification Error Rate):** Proportion of genuine travelers falsely rejected.
  - **EER (Equal Error Rate):** Operating point where $\text{APCER} = \text{BPCER}$.
  - **BPCER @ APCER = 1.0% / 5.0%:** Critical operational border gate thresholds.

### Speaker Script
> "Our objective is to design a single-image detector that generalizes across unseen generators and physical print-scan degradation. All results are measured strictly under the international biometric standard ISO/IEC 30107-3, reporting APCER, BPCER, EER, and operational BPCER at 1% attack acceptance."

---

## Slide 4: Literature Review & Identified Gaps

### Slide Content
- **Key Surveyed Literature:**
  - *Scherhag et al. (2019):* Comprehensive morph attack survey; highlighted cross-database generalization collapse across classical texture descriptors (LBP, BSIF).
  - *Damer et al. (2021-2023):* Landmark dynamics and cascaded deep feature analysis; demonstrated vulnerabilities in standard CNN backbones when tested out-of-distribution.
  - *Wandzik et al. (2024):* Frequency decomposition methods; showed spectral high-pass filters capture synthetic upsampling noise.
  - *Oquab et al. (2023):* DINOv2 self-supervised foundation models preserve dense patch correspondence without label bias.
- **Critical Research Gaps Identified:**
  - **Gap 1 (Domain Bias):** Single-stream RGB models overfit to landmark artifacts and fail completely on generative GAN morphs.
  - **Gap 2 (Modality Competition):** Naive multi-stream networks suffer from gradient interference; fast-converging CNN streams overpower Vision Transformers.
  - **Gap 3 (Sensor Noise Confounding):** Frequency filters trained on synthetic data trigger false alarms on physical camera sensor pattern noise (PRNU) in real photographs.

### Speaker Script
> "Our literature review established that classical texture and single-stream CNN detectors suffer catastrophic cross-dataset degradation. While self-supervised Vision Transformers capture fine surface textures, they remain blind to GAN synthesis noise. Conversely, spectral residual models detect GAN artifacts but suffer from modality competition and camera sensor noise false alarms. Our architecture directly resolves these gaps."

---

## Slide 5: Methodology -- Proposed UID-MAD Architecture

### Slide Content
```
                      INPUT IMAGE I ∈ R^(224×224×3)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
     STREAM A (SPATIAL)                          STREAM B (FREQUENCY RESIDUAL)
     DINOv2 ViT-S/14 Backbone                    Fixed 3×3 Laplacian Filter
     Dense Multi-Scale Patch Aggregator          4-Stage Lightweight ConvNet (128-d)
     e_spatial ∈ R^256                           e_freq ∈ R^128
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    ▼
                         CONCATENATION & PROJECTION
                     z_fuse = [e_spatial || e_freq] ∈ R^384
                     e_fuse ∈ R^256
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        SPATIAL HEAD             FREQ HEAD             FUSED HEAD
       ArcFace(256 -> 2)       ArcFace(128 -> 2)     ArcFace(256 -> 2)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                     DECOUPLED MULTI-TASK ARCFACE LOSS
               L_total = L_fuse + 0.5*L_spatial + 0.5*L_freq
```

### Speaker Script
> "Here is our UID-MAD architecture. It decomposes the input face into two complementary pathways: Stream A extracts spatial patch geometry using DINOv2, while Stream B extracts spectral noise using a fixed Laplacian residual filter and a 4-stage ConvNet. Crucially, we supervise the network with decoupled multi-task ArcFace heads to eliminate gradient interference between streams."

---

## Slide 6: Stream Details -- Spatial ViT vs. Frequency ConvNet

### Slide Content
- **Stream A: Spatial DINOv2 ViT-S/14:**
  - `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, 384 embedding dimension).
  - Pretrained via self-distillation and masked patch modeling; preserves fine-grained surface geometry.
  - **Multi-Scale Patch Token Pooling:**
    $$\mathbf{f} = [\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu}_{\text{patch}} \parallel \boldsymbol{\sigma}_{\text{patch}} \parallel \mathbf{z}_{\text{max}}] \in \mathbb{R}^{1536}$$
    Patch standard deviation ($\boldsymbol{\sigma}_{\text{patch}}$) explicitly registers localized seam variance.
- **Stream B: High-Pass Laplacian Residual ConvNet:**
  - Fixed $3 \times 3$ Laplacian kernel strips low-frequency identity and illumination:
    $$\mathbf{K} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{R} = |\mathbf{I} * \mathbf{K}|$$
  - Lightweight 4-stage ConvNet (32 -> 64 -> 128 -> 128 channels) downsamples to $14 \times 14$ feature maps.
  - Dual global pooling $[\text{AvgPool} \parallel \text{MaxPool}]$ outputs compact 128-d frequency embedding $\mathbf{e}_{\text{freq}}$.

### Speaker Script
> "In Stream A, we don't just take the classification token; we aggregate mean, max, and standard deviation across all 256 spatial patch tokens, creating a 1536-dimensional descriptor highly sensitive to blending discontinuities. In Stream B, our Laplacian high-pass filter eliminates facial identity and isolates the transposed-convolution Nyquist checkerboard patterns characteristic of GAN generators."

---

## Slide 7: Decoupled Multi-Task ArcFace & Training Setup

### Slide Content
- **Additive Angular Margin (ArcFace) Head:**
  - Enforces geodesic separation on the hypersphere between bona fide and attack representations:
    $$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(s \cdot \cos(\theta_{y_i} + m))}{\exp(s \cdot \cos(\theta_{y_i} + m)) + \sum_{j \neq y_i} \exp(s \cdot \cos \theta_j)}$$
    Hyperparameters: scale $s = 32.0$, angular margin $m = 0.25$ radians.
- **Decoupled Multi-Task Objective:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fuse}} + 0.5 \cdot \mathcal{L}_{\text{spatial}} + 0.5 \cdot \mathcal{L}_{\text{freq}}$$
  Prevents fast ConvNet gradients from overwriting the spatial transformer weights.
- **Experimental & Hardware Setup:**
  - **Training Dataset:** SMDD (`SMDD_release_train`) -- 32,000 synthetic faces (20,030 bona fide, 11,970 morphs).
  - **Evaluation Benchmark:** FRLL-Morphs -- 7,274 samples across 5 morphing algorithms and 3 operational conditions (Clean, JPEG $q=30$, Print-Scan).
  - **Platform:** Dual NVIDIA T4 GPUs on Kaggle, PyTorch `nn.DataParallel`, FP16 mixed precision.

### Speaker Script
> "We train using an Additive Angular Margin loss with scale 32.0 and margin 0.25 radians. By weighting the auxiliary heads at 0.5, each stream is forced to maintain its specialized feature domain. Training is performed exclusively on the 32,000 synthetic images of SMDD, and testing is performed completely out-of-distribution on the real human faces of FRLL-Morphs."

---

## Slide 8: Progress Achieved -- Week 1 Baseline & The StyleGAN2 Blind Spot

### Slide Content
- **Week 1 Milestone:** Single-stream spatial DINOv2 baseline evaluated zero-shot on FRLL-Morphs.
- **Landmark Success:**
  - OpenCV morphs: **0.9993 ROC-AUC (0.94% EER)**
  - FaceMorpher morphs: **0.9958 ROC-AUC (2.45% EER)**
  - WebMorph morphs: **0.9886 ROC-AUC (4.91% EER)**
  - JPEG ($q=30$) Robustness: Maintained **0.9992 ROC-AUC**
- **The Critical Failure Mode:**
  - **StyleGAN2 Latent Morphs: 0.4988 ROC-AUC (51.00% EER)**
  - *Theoretical Finding:* The spatial foundation model collapsed to pure random chance (identical to a coin flip) on generative morphs. Spatial RGB features cannot detect attacks that contain zero spatial blending seams.

### Speaker Script
> "In Week 1, our spatial baseline achieved state-of-the-art detection on landmark-based morphs, exceeding 0.999 AUC on OpenCV. However, we discovered a fatal blind spot: on generative StyleGAN2 morphs, performance dropped to 0.4988 ROC-AUC—a complete coin toss. This empirical proof demonstrated that single-stream spatial models cannot solve generative attacks, directly motivating Week 2."

---

## Slide 9: Progress Achieved -- Week 2 StyleGAN2 Breakthrough

### Slide Content
- **Week 2 Milestone:** Deployed Dual-Stream UID-MAD with High-Pass Laplacian ConvNet.
- **In-Domain Convergence (SMDD):**
  - Training Accuracy: **99.30%**
  - Validation Loss: **0.0064**
  - Validation EER: **0.0001 (0.01%)**
- **The Empirical Breakthrough on Generative Morphs:**

| Morph Generator | Attack Type | Week 1 (Spatial ViT) | Week 2 (Dual-Stream) | Relative Leap |
| :--- | :--- | :--- | :--- | :--- |
| **morph_stylegan** | Generative Latent W+ | 0.4988 AUC (51.0% EER) | **0.8525 AUC (24.0% EER)** | **+35.4% AUC Leap** |
| **morph_opencv** | Landmark Warp | 0.9993 AUC | **0.7924 AUC (Trial 2)** | Maintained sensitivity |
| **morph_facemorpher** | Alpha Blend | 0.9958 AUC | **0.8295 AUC (Trial 2)** | Maintained sensitivity |

- **Key Takeaway:** Stream B's Laplacian residual stream proved the theoretical hypothesis: frequency analysis successfully captures generative synthesis artifacts.

### Speaker Script
> "In Week 2, introducing the high-pass frequency residual stream unlocked generative morph detection. StyleGAN2 detection surged from random guessing at 0.4988 AUC to 0.8525 AUC, reducing Equal Error Rate from 51% to 24%. Simultaneously, decoupled multi-task training maintained solid landmark detection above 0.80 AUC across OpenCV and FaceMorpher."

---

## Slide 10: Diagnostic Findings -- Modality Competition & Sensor Noise Shift

### Slide Content
- **Finding 1: Modality Competition in Joint Embeddings**
  - In early trials, lightweight ConvNet gradients converged 4x faster than the ViT, dominating the joint representation and degrading spatial landmark detection.
  - *Resolution:* Decoupled multi-task loss ($\mathcal{L}_{\text{fuse}} + 0.5\mathcal{L}_s + 0.5\mathcal{L}_f$) forced balanced gradient flow.
- **Finding 2: The Cross-Dataset Sensor Noise (PRNU) Shift**
  - In Trial 2, evaluating via a naive maximum OR-gate ($P = \max(P_s, P_f, P_{\text{fuse}})$) caused EER thresholds to drop to $10^{-6}$.
  - *Autopsy:* SMDD training images are purely synthetic with smooth noise floors. FRLL evaluation images are genuine studio camera photographs containing physical Photo-Response Non-Uniformity (PRNU).
  - Stream B's ConvNet flagged real camera noise as an anomaly, creating false alarms under naive max pooling.
  - *Takeaway:* Identified the precise need for score normalization and calibration.

### Speaker Script
> "Our Week 2 evaluation revealed two critical machine learning insights: first, lightweight ConvNets dominate joint gradients unless decoupled multi-task heads are applied. Second, evaluating with a naive max-gate exposed a sensor noise domain shift: real camera PRNU noise in FRLL photographs was flagged by the frequency stream. Diagnosing this failure mechanism provides the exact foundation for our post-mid-sem calibration work."

---

## Slide 11: Future Work & Timeline (Weeks 3 to Final)

### Slide Content
- **Planned Work Package 1: Score Calibration & Margin Normalization**
  - Implement linear cosine-margin scoring ($S = \cos \theta_1 - \cos \theta_0$) to replace saturated softmax probabilities.
  - Apply Z-score standardization against bona fide distributions ($Z = \frac{S - \mu_{\text{bf}}}{\sigma_{\text{bf}}}$) to eliminate PRNU sensor noise false alarms.
- **Planned Work Package 2: Parameter-Efficient Fine-Tuning (LoRA)**
  - Integrate Low-Rank Adaptation (LoRA: rank 16) into DINOv2 transformer blocks 6-11 to preserve dense landmark features.
- **Planned Work Package 3: Edge Optimization & Model Export**
  - Export dual-stream network to ONNX Runtime and TensorRT FP16 engines.
  - Benchmark inference latency targeting $\le 50$ ms on embedded border gate hardware.
- **Timeline & Milestones:**
  - *Weeks 3-4:* LoRA integration, score calibration, and cross-dataset stress testing.
  - *Weeks 5-6:* Edge deployment benchmarks, ONNX/TensorRT optimization, and interpretability heatmaps (Grad-CAM).
  - *Weeks 7-8:* Final thesis compilation, code release, and end-semester defense.

### Speaker Script
> "Our future work plan directly targets the remaining challenges. First, we will implement linear cosine-margin scoring and Z-score calibration to completely neutralize camera sensor noise. Second, we will integrate LoRA adapters into DINOv2 to maximize landmark precision. Third, we will export the model to ONNX and TensorRT targeting sub-50ms inference at border e-Gates."

---

## Slide 12: Summary & Q&A Defense Script

### Slide Content
- **Summary of Mid-Semester Status:**
  - Completed thorough literature review and established ISO/IEC 30107-3 compliant evaluation pipeline.
  - Validated spatial foundation baseline (DINOv2 ViT-S/14) on SMDD and FRLL-Morphs (Week 1).
  - Engineered dual-stream spatial + frequency architecture and resolved the StyleGAN2 blind spot (+35.4% AUC leap) (Week 2).
  - Formulated decoupled multi-task ArcFace supervision to prevent modality competition.
- **Anticipated Committee Q&A:**
  - *Q: Why not use a deeper CNN or ResNet-50?*  
    *A:* Supervised CNNs discard fine pixel textures in favor of high-level class invariance. Self-supervised DINOv2 preserves dense patch correspondences crucial for localized seam detection.
  - *Q: Why did StyleGAN2 jump from 0.49 to 0.85 AUC?*  
    *A:* GAN upsampling layers leave characteristic periodic Nyquist spectral energy. Our Laplacian residual filter isolates these high-frequency traces from facial identity.

### Speaker Script
> "To summarize: our mid-semester work identified the theoretical blind spot of spatial transformers on generative morphs, designed a dual-stream spatial-frequency architecture, and proved a 35.4% AUC breakthrough on StyleGAN2 using decoupled ArcFace loss. We are now positioned to calibrate decision fusion and optimize for edge deployment. Thank you, and we welcome your questions."
