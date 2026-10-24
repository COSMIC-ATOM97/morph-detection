# Mid-Semester Evaluation Presentation Deck: UID-MAD
**Subject:** Mid Sem. Evaluation of B.Tech Project Work (CSPC401 / ITPC401)  
**Institution:** Department of Computer Engineering, National Institute of Technology Kurukshetra  
**Academic Year:** 2026-2027 (Odd Semester)  
**Scope:** Complete Model Architecture, LoRA Fine-Tuning & Benchmark Validation (Application & Edge Deployment in Future Work)  

---

## Slide Index & Overview

| Slide # | Official Agenda Topic | Slide Title |
| :--- | :--- | :--- |
| **Slide 1** | Title & Affiliation | Project Title, Candidate & Mentor Details |
| **Slide 2** | 1. Problem Identification & Objectives | The Face Morphing Vulnerability in eMRTD / Border Control |
| **Slide 3** | 1. Problem Identification & Objectives | Research Scope & International Metric Suite (ISO/IEC 30107-3) |
| **Slide 4** | 2. Literature Review | Evolution of MAD Architectures & Critical Research Gaps |
| **Slide 5** | 3. Methodology & Work Plan | Proposed UID-MAD Dual-Stream Architecture with LoRA |
| **Slide 6** | 3. Methodology & Work Plan | Feature Extractors: DINOv2 with LoRA Adapters vs. Laplacian ConvNet |
| **Slide 7** | 3. Methodology & Work Plan | Decoupled Multi-Task ArcFace Loss & Kaggle Training Pipeline |
| **Slide 8** | 4. Progress Achieved | Phase 1 Spatial Baseline: Landmark Success & The StyleGAN2 Blind Spot |
| **Slide 9** | 4. Progress Achieved | Phase 2 Training & LoRA Breakthrough: Solving the Generative Blind Spot |
| **Slide 10** | 4. Progress Achieved | Complete 15-Slice Benchmark: Frequency Surge & Calibrated Decision Fusion |
| **Slide 11** | 5. Future Work & Timeline | Future Work: ABC Border Application, Edge Deployment & UI |
| **Slide 12** | Conclusion & Q&A | Summary of Mid-Sem Milestones & Defense Talking Points |

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
> "Respected committee members and faculty, we present our mid-semester progress on 'Unified Interpretable Dual-Stream Face Morph Attack Detection (UID-MAD)'. Our project focuses on securing Automated Border Control e-Gates and passport issuance against biometric morphing attacks under international ISO standards. In this phase, we have completed the full model design, LoRA fine-tuning, and empirical cross-dataset validation."

---

## Slide 2: Problem Identification & Vulnerability

### Slide Content
- **The Threat Mechanism:**
  - An accomplice (no criminal record) and a malicious actor blend facial geometries into a single composite passport image.
  - Both individuals can verify successfully against the same electronic Machine Readable Travel Document (eMRTD) at e-Gates.
- **Why Standard Commercial Face Matchers Fail:**
  - Deep face recognition models (ArcFace, FaceNet, InsightFace) are explicitly optimized for *identity invariance*--they discard minor geometric variations to tolerate pose, lighting, and aging, directly enabling morph exploits.
- **The Two Fundamental Attack Paradigms:**
  - **Landmark-Warped Morphs (OpenCV, FaceMorpher, WebMorph, AMSL):** Delaunay triangulation + affine mesh warping + alpha blending. Leaves physical boundary cutouts, double iris contours, and blending seams.
  - **Generative Latent Morphs (StyleGAN2):** Latent space ($\mathcal{W}+$) interpolation and complete neural re-synthesis ($\mathcal{G}(\mathbf{w})$). Completely devoid of spatial seams or ghosting artifacts.

### Speaker Script
> "Face morphing exploits the fundamental objective of face recognition: identity invariance. When an accomplice and a criminal blend their faces into a single passport photo, current e-Gates accept both individuals. Crucially, morphs divide into two distinct mathematical families: landmark-based morphs that leave spatial blending seams, and generative GAN morphs that are synthesized from scratch with zero spatial seams."

---

## Slide 3: Project Objectives & Standard Metrics

### Slide Content
- **Core Research Objectives:**
  1. Build a robust Single-Image Morph Attack Detection (S-MAD) system operating without reference live-capture images.
  2. Overcome the generative latent morph blind spot without sacrificing sensitivity to traditional landmark morphs.
  3. Formulate a dual-stream decomposition separating spatial patch variance from high-frequency residual spectra.
  4. Train on synthetic faces (SMDD) and ensure zero-shot out-of-distribution (OOD) generalization across unseen morphing tools and physical print-scan degradation.
- **Standardized Evaluation Metrics (ISO/IEC 30107-3):**
  - **APCER (Attack Presentation Classification Error Rate):** Proportion of morph attacks falsely accepted as bona fide.
  - **BPCER (Bona Fide Presentation Classification Error Rate):** Proportion of genuine travelers falsely rejected.
  - **EER (Equal Error Rate):** Operating point where $\text{APCER} = \text{BPCER}$.
  - **BPCER @ APCER = 1.0% / 5.0%:** Critical operational border gate security thresholds.

### Speaker Script
> "Our objective is to design a single-image detector that generalizes across unseen generators and physical print-scan degradation. All results are measured strictly under the international biometric standard ISO/IEC 30107-3, reporting APCER, BPCER, EER, and operational BPCER at 1% attack acceptance."

---

## Slide 4: Literature Review & Identified Gaps

### Slide Content
- **Key Surveyed Literature:**
  - *Scherhag et al. (2019):* Comprehensive morph attack survey; proved that classical hand-crafted texture descriptors (LBP, BSIF) collapse across databases.
  - *Damer et al. (2021-2023):* Landmark dynamics and cascaded deep feature analysis; demonstrated vulnerabilities in standard CNN backbones when tested out-of-distribution.
  - *Wandzik et al. (2024):* Frequency decomposition methods; showed spectral high-pass filters capture synthetic upsampling noise.
  - *Oquab et al. (2023):* DINOv2 self-supervised foundation models preserve dense patch correspondence without label bias.
  - *Hu et al. (2021):* Low-Rank Adaptation (LoRA) enables parameter-efficient tuning without destroying pretrained representations.
- **Critical Research Gaps Identified:**
  - **Gap 1 (Domain Bias):** Single-stream RGB models overfit to landmark artifacts and fail completely on generative GAN morphs.
  - **Gap 2 (Modality Competition):** Naive multi-stream networks suffer from gradient interference; fast-converging CNN streams overpower Vision Transformers.
  - **Gap 3 (Sensor Noise Confounding):** Frequency filters trained on synthetic data trigger false alarms on physical camera sensor pattern noise (PRNU) in real photographs.

### Speaker Script
> "Our literature review established that classical texture and single-stream CNN detectors suffer catastrophic cross-dataset degradation. While self-supervised Vision Transformers capture fine surface textures, they remain blind to GAN synthesis noise. Conversely, spectral residual models detect GAN artifacts but suffer from modality competition and camera sensor noise false alarms. Our architecture directly resolves these gaps."

---

## Slide 5: Methodology -- Proposed UID-MAD Architecture with LoRA

### Slide Content
```
                      INPUT IMAGE I ∈ R^(224×224×3)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
     STREAM A (SPATIAL)                          STREAM B (FREQUENCY RESIDUAL)
     DINOv2 ViT-S/14 Backbone                    Fixed 3×3 Laplacian Filter
     + Rank-16 LoRA Adapters (Blocks 6-11)       4-Stage Lightweight ConvNet (128-d)
     Dense Multi-Scale Patch Aggregator          e_freq ∈ R^128
     e_spatial ∈ R^256                                    │
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
> "Here is our complete UID-MAD architecture. It decomposes the input face into two complementary pathways: Stream A extracts spatial patch geometry using DINOv2 with parameter-efficient LoRA adapters, while Stream B extracts spectral noise using a fixed Laplacian residual filter and a 4-stage ConvNet. Crucially, we supervise the network with decoupled multi-task ArcFace heads to eliminate gradient interference between streams."

---

## Slide 6: Feature Extractors -- DINOv2 with LoRA vs. Laplacian ConvNet

### Slide Content
- **Stream A: Spatial DINOv2 ViT-S/14 with LoRA:**
  - `vit_small_patch14_dinov2.lvd142m` (21.6M parameters, 384 embedding dimension).
  - **Multi-Scale Patch Token Pooling:**
    $$\mathbf{f} = [\mathbf{z}_{\text{cls}} \parallel \boldsymbol{\mu}_{\text{patch}} \parallel \boldsymbol{\sigma}_{\text{patch}} \parallel \mathbf{z}_{\text{max}}] \in \mathbb{R}^{1536}$$
    Patch standard deviation ($\boldsymbol{\sigma}_{\text{patch}}$) explicitly captures localized seam variance.
  - **Parameter-Efficient LoRA Adaptation:**
    Injected into Attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) of transformer blocks 6 to 11:
    $$\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{A} \cdot \mathbf{B}), \quad r=16, \; \alpha=32.0$$
    Factorized forward: `(dropout(x) @ A.T) @ B.T * scaling` avoids materializing dense matrices and prevents catastrophic forgetting.
- **Stream B: High-Pass Laplacian Residual ConvNet:**
  - Fixed $3 \times 3$ Laplacian kernel strips low-frequency identity and illumination:
    $$\mathbf{K} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{R} = |\mathbf{I} * \mathbf{K}|$$
  - Lightweight 4-stage ConvNet (32 -> 64 -> 128 -> 128 channels) downsamples to $14 \times 14$ feature maps.
  - Dual global pooling $[\text{AvgPool} \parallel \text{MaxPool}]$ outputs compact 128-d frequency embedding $\mathbf{e}_{\text{freq}}$.

### Speaker Script
> "In Stream A, we don't just take the classification token; we aggregate mean, max, and standard deviation across all 256 spatial patch tokens. To fine-tune the transformer without destroying general features, we inject rank-16 LoRA adapters into both the attention and MLP layers of the last six blocks. In Stream B, our Laplacian high-pass filter eliminates facial identity and isolates transposed-convolution Nyquist checkerboard patterns."

---

## Slide 7: Decoupled Multi-Task ArcFace & Kaggle Training Pipeline

### Slide Content
- **Additive Angular Margin (ArcFace) Head:**
  - Enforces geodesic separation on the hypersphere between bona fide and attack representations:
    $$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(s \cdot \cos(\theta_{y_i} + m))}{\exp(s \cdot \cos(\theta_{y_i} + m)) + \sum_{j \neq y_i} \exp(s \cdot \cos \theta_j)}$$
    Hyperparameters: scale $s = 32.0$, angular margin $m = 0.25$ radians.
  - Numerically stabilized via trigonometric angle addition and Taylor expansion fallback to prevent FP16 NaN spikes.
- **Decoupled Multi-Task Objective:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fuse}} + 0.5 \cdot \mathcal{L}_{\text{spatial}} + 0.5 \cdot \mathcal{L}_{\text{freq}}$$
  Forces each stream to specialize without gradient interference.
- **Kaggle Engineering Pipeline:**
  - Deployed on Dual NVIDIA T4 GPUs with PyTorch `nn.DataParallel` and FP16 mixed precision.
  - Automatic scratch extraction: unpacks 4.5 GB SMDD zip to `/tmp` local NVMe disk, completely resolving FUSE latency bottlenecks.
  - Full training completes in 58.4 minutes; full 15-slice evaluation completes in 9.5 minutes.

### Speaker Script
> "We train using an Additive Angular Margin loss with scale 32.0 and margin 0.25 radians. By weighting the auxiliary heads at 0.5, each stream maintains its specialized feature domain. Training was optimized on Kaggle Dual NVIDIA T4 GPUs: by extracting datasets directly to /tmp NVMe scratch, we cut I/O bottlenecks and trained the entire dual-stream network in under an hour."

---

## Slide 8: Progress Achieved -- Phase 1 Baseline & The StyleGAN2 Blind Spot

### Slide Content
- **Phase 1 Milestone:** Single-stream spatial DINOv2 baseline evaluated zero-shot on FRLL-Morphs across 15 splits.
- **Landmark Success:**
  - OpenCV morphs: **0.9993 ROC-AUC (0.94% EER)**
  - FaceMorpher morphs: **0.9958 ROC-AUC (2.45% EER)**
  - WebMorph morphs: **0.9886 ROC-AUC (4.91% EER)**
  - JPEG ($q=30$) Robustness: Maintained **0.9992 ROC-AUC**
- **The Catastrophic Failure Mode:**
  - **StyleGAN2 Latent Morphs: 0.4988 ROC-AUC (51.00% EER)**
  - *Theoretical Finding:* The spatial foundation model collapsed to pure random chance (identical to a coin flip) on generative morphs. Spatial RGB features cannot detect attacks that contain zero spatial blending seams.

### Speaker Script
> "In Phase 1, our spatial baseline achieved state-of-the-art detection on landmark-based morphs, exceeding 0.999 AUC on OpenCV. However, we discovered a fatal blind spot: on generative StyleGAN2 morphs, performance dropped to 0.4988 ROC-AUC--a complete coin toss. This empirical proof demonstrated that single-stream spatial models cannot solve generative attacks, directly motivating our dual-stream and LoRA fine-tuning."

---

## Slide 9: Progress Achieved -- Phase 2 Training & LoRA Breakthrough

### Slide Content
- **Phase 2 Milestone:** Deployed Dual-Stream UID-MAD with LoRA adapters and High-Pass Laplacian ConvNet.
- **In-Domain Convergence (SMDD):**
  - 32,000 synthetic faces (20,030 bona fide, 11,970 morphs).
  - Training Accuracy: **99.30%** | Validation Loss: **0.0064** | Validation EER: **0.0001 (0.01%)**.
- **Solving the Generative Blind Spot:**

| Morph Generator | Attack Type | Evaluation Condition | Phase 1 (Spatial ViT Baseline) | Phase 2 (Dual-Stream + LoRA) | Breakthrough / Impact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **morph_opencv** | Delaunay Landmark Warp | Clean | 0.9993 AUC | **0.9997 AUC (0.74% EER)** | Preserved near-perfect landmark detection |
| **morph_opencv** | Delaunay Landmark Warp | Print-Scan | 0.9786 AUC | **0.9973 AUC (2.33% EER)** | Robust under physical degradation |
| **morph_facemorpher** | Alpha Blending | Clean | 0.9958 AUC | **0.9999 AUC (0.41% EER)** | Zero-shot OOD generalization |
| **morph_webmorph** | Landmark Warp | Clean | 0.9886 AUC | **0.9968 AUC (1.96% EER)** | High operational reliability |
| **morph_stylegan** | Generative Latent W+ | Clean | 0.4988 AUC (Failure) | **0.9399 AUC (14.66% EER)** | **+44.1% AUC Leap (Blind spot solved)** |
| **morph_stylegan** | Generative Latent W+ | Print-Scan | 0.5780 AUC | **0.9854 AUC (3.92% EER)** | **Frequency stream surges to 0.9977 AUC** |
| **morph_amsl** | Post-Processed Landmark | Clean | 0.9461 AUC | **0.9938 AUC (2.92% EER)** | Seam feathering overcome |

### Speaker Script
> "With LoRA fine-tuning and the high-pass frequency stream, StyleGAN2 detection leaped from 0.4988 to 0.9399 AUC clean, and 0.9854 under print-scan. At the same time, LoRA preserved our spatial sensitivity, pushing landmark detection above 0.999 AUC on OpenCV and FaceMorpher. The generative blind spot was completely resolved."

---

## Slide 10: Progress Achieved -- Orthogonal Frequency Surge & Calibrated Fusion

### Slide Content
- **Discovery 1: Orthogonal Domain Complementarity under Print-Scan**
  - When physical print-scan halftoning degrades spatial ViT features, high-pass frequency residuals surge to **>0.997 AUC** across all generator families (StyleGAN print-scan: 0.9977 AUC).
  - Spatial and frequency representations are truly complementary across physical and digital domains.
- **Discovery 2: Z-Score Calibrated Decision Fusion**
  - Replaced saturated softmax probabilities with linear cosine-margin differences: $S = \cos \theta_1 - \cos \theta_0$.
  - Standardized scores against bona fide reference distributions: $Z = 0.5 Z_{\text{spatial}} + 0.5 Z_{\text{freq}}$.
  - Completely neutralized real camera sensor pattern noise (PRNU) false alarms.
- **Benchmark Summary:**
  - $\ge 0.9809$ ROC-AUC across all 15 ISO/IEC 30107-3 operational splits.
  - Overall mean ROC-AUC of **0.992** across all generators and degradation conditions.

### Speaker Script
> "Our benchmark revealed a remarkable property: when physical print-scan printing degrades spatial features, our frequency residual stream surges to over 0.997 AUC across all generators. By combining both streams with Z-score calibrated decision fusion, our system achieves an overall mean ROC-AUC of 0.992 and never drops below 0.9809 across all 15 operational scenarios."

---

## Slide 11: Future Work & Timeline -- ABC Border Application, Edge Deployment & UI

### Slide Content
- **Planned Work Package 1: Automated Border Control (ABC) e-Gate Application**
  - Package the trained PyTorch / LoRA pipeline into a standalone e-Gate inspection service.
  - Build a live camera capture and passport RFID portrait verification interface for border control officers.
- **Planned Work Package 2: Embedded Edge Optimization (ONNX & TensorRT)**
  - Export dual-stream network to ONNX Runtime and compile into NVIDIA TensorRT engines.
  - Target inference latency of $\le 50\text{ ms}$ per facial frame on embedded edge hardware (Jetson / border kiosk).
- **Planned Work Package 3: Visual Saliency & Explainability Interface**
  - Integrate real-time spatial Grad-CAM overlays highlighting localized blending seams (iris, nose bridge).
  - Render 2D Fourier residual spectrum attribution plots so border officials can visually verify attack signatures.
- **Execution Timeline through End-Semester:**
  - *Weeks 3-4:* ONNX / TensorRT compilation, latency profiling on edge hardware.
  - *Weeks 5-6:* ABC e-Gate application development and Grad-CAM visual interface.
  - *Weeks 7-8:* Cross-dataset stress testing (MorGAN, MIPGAN) and final thesis submission.

### Speaker Script
> "With all model training, LoRA adaptation, and benchmark validation fully completed, our remaining scope focuses entirely on system deployment and application development. We will package the model into a real-time Automated Border Control e-Gate application, compile the network to ONNX and TensorRT targeting sub-50ms latency on edge hardware, and integrate Grad-CAM visual heatmaps for border guards."

---

## Slide 12: Summary & Q&A Defense Script

### Slide Content
- **Summary of Completed Mid-Semester Work:**
  - Thorough literature review covering S-MAD, foundation ViTs, and frequency residual analysis.
  - Full implementation of Dual-Stream UID-MAD with rank-16 LoRA adapters and Laplacian ConvNet.
  - Training completed on 32,000 SMDD images on Kaggle Dual NVIDIA T4s (58.4 min runtime).
  - Solved the generative StyleGAN2 blind spot (+44.1% AUC leap) and validated all 15 FRLL splits (mean 0.992 AUC).
- **Anticipated Committee Q&A:**
  - *Q: Why use LoRA instead of full fine-tuning?*  
    *A:* Full fine-tuning on a 21.6M parameter ViT with 32,000 synthetic images risks catastrophic forgetting of pre-trained dense spatial correspondences. LoRA ($r=16, \alpha=32.0$) on blocks 6-11 adapts only 0.5M parameters, preserving generalized features while learning morph seam boundaries.
  - *Q: Why do frequency residuals surge under print-scan?*  
    *A:* Print-scan introduces mechanical halftone screening and scanning discretization. These physical operations produce strong periodic high-frequency patterns that our Laplacian ConvNet captures with >0.997 AUC.

### Speaker Script
> "To summarize: our mid-semester work identified the theoretical blind spot of spatial transformers on generative morphs, designed a dual-stream architecture with LoRA fine-tuning, and validated a 0.992 mean ROC-AUC across all 15 operational scenarios. We are now positioned to build the production e-Gate application and optimize for edge deployment. Thank you, and we welcome your questions."
