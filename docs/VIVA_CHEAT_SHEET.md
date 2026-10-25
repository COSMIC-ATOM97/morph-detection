# Mid-Sem Evaluation Quick Cheat-Sheet (Phone-Friendly)
## UID-MAD: Face Morph Attack Detection

---

### 1. The 30-Second Elevator Pitch
> "Our project tackles Face Morph Attacks at automated airport e-Gates, where two people combine their faces into one passport to bypass border control.  
> Existing models look only at spatial blending seams, which means they completely fail against AI-generated morphs like StyleGAN2.  
> We built **UID-MAD**, a dual-stream model: Stream A uses a DINOv2 Vision Transformer with LoRA for spatial seams, and Stream B uses high-pass Laplacian and wavelet filters for generative frequency noise. Across all 15 operational conditions in our benchmark, our model achieved a **0.992 mean ROC-AUC**."

---

### 2. The 5 Killer Numbers to Know Cold

| Metric / Fact | The Number | Why It Matters |
| :--- | :---: | :--- |
| **Phase 1 Spatial Failure** | **0.4988 ROC-AUC (51.0% EER)** | On StyleGAN2 morphs, spatial-only models are equivalent to a coin toss. |
| **Phase 2 Breakthrough** | **0.9399 ROC-AUC (14.66% EER)** | Adding the frequency stream recovered StyleGAN2 detection on clean images. |
| **Print-Scan Frequency Surge** | **> 0.997 ROC-AUC** | When physical print-scan destroys spatial seams, frequency residuals rescue the decision. |
| **Overall System Performance** | **0.992 Mean ROC-AUC** | Across all 15 test slices of the standard FRLL-Morphs benchmark. |
| **Training Efficiency** | **58.4 minutes** | Trained on 32,000 images (SMDD dataset) on dual NVIDIA T4 GPUs over 8 epochs. |

---

### 3. The 6 Questions the Professor Will Ask & Exact Answers

#### Q1: "Why do standard Face Recognition systems (like FaceNet or InsightFace) fail against morphs?"
- **Answer:** Face recognition models are trained to maximize *identity invariance*. They ignore minor lighting, expression, and skin differences so they can match you when you smile or change hairstyle.
- Because a morphed face intentionally looks similar to both contributing subjects, identity matchers verify both of them.
- We use self-supervised DINOv2 foundation features because it analyzes raw patch geometry and texture variance instead of facial identity.

#### Q2: "Why do you need two streams instead of a single network?"
- **Answer:** Morph attacks come from two fundamentally different physical processes:
  1. *Landmark-warped morphs (OpenCV, FaceMorpher)*: Alpha-blending leaves physical edge seams, double pupil contours, and mask borders (handled by Spatial Stream A).
  2. *Generative latent morphs (StyleGAN2)*: Generated purely by neural upsampling with zero physical seams, but neural upsampling leaves periodic high-frequency checkerboard noise (handled by Frequency Stream B).
- Forcing one network to learn both causes feature competition and degrades accuracy.

#### Q3: "What is LoRA and why did you use it here?"
- **Answer:** Low-Rank Adaptation. DINOv2 has 21 million parameters. Fully fine-tuning it on our 32,000 synthetic morph images would cause *catastrophic forgetting* of its pre-trained visual knowledge.
- LoRA freezes the backbone weights and injects trainable rank-16 adapter matrices into the last 6 transformer blocks ($r=16, \alpha=32$).
- This adapts the foundation model to subtle forensic artifacts while preserving its robust geometric representations.

#### Q4: "What happens when someone prints the passport photo on paper and scans it?"
- **Answer:** Physical printing and scanning act like an analog low-pass blur filter that erases spatial blending seams (our spatial model dropped to 0.31 AUC on StyleGAN2).
- However, ink dispersion and paper grain create periodic high-frequency noise.
- Our frequency stream exploits this, surging to **> 0.997 AUC across every generator family**.
- This proves that spatial and frequency streams are orthogonally complementary.

#### Q5: "What is your training and evaluation dataset? Did you test cross-dataset?"
- **Answer:**
  - **Training:** Synthetic SMDD dataset (32,000 images: 20,030 bona fide, 11,970 morphs) split 80/20.
  - **Testing:** 100% zero-shot out-of-distribution evaluation on FRLL-Morphs (7,274 images across 5 distinct generator families and 3 conditions: Clean, JPEG, and Print-Scan). We never trained on FRLL.

#### Q6: "What is left for the second half of the semester (Future Work)?"
- **Answer:**
  - The model engineering, training, and 15-slice benchmark validation are 100% complete.
  - The second half focuses on **Edge Optimization**: exporting to ONNX and compiling with NVIDIA TensorRT (FP16/INT8 quantization) to hit the **< 50 ms latency SLA** for airport e-Gates.
  - We will also build a desktop e-Gate inspection GUI and add Grad-CAM / Fourier visual heatmaps for border guards.

---

### 4. Recommended Team Division

- **Person 1 (e.g., Bablu): Problem Statement, Vulnerabilities, and Datasets**
  - Explains how face morphing bypasses e-Gates.
  - Explains the difference between landmark morphs and StyleGAN2.
  - Explains the SMDD training corpus and zero-shot FRLL-Morphs evaluation splits.

- **Person 2 (e.g., Puranpal): System Architecture, Stream A (DINOv2 + LoRA), and Stream B (Frequency)**
  - Explains the dual-stream block diagram.
  - Explains multi-scale patch pooling ($\mu, \sigma, \max$) and rank-16 LoRA adapters.
  - Explains Laplacian and Haar wavelet filters and ArcFace loss.

- **Person 3 (e.g., Himank): Experimental Results, Key Insights, and Future Work**
  - Explains the Phase 1 StyleGAN2 failure (0.4988 AUC) vs Phase 2 recovery (0.9399 AUC).
  - Explains the print-scan frequency surge (>0.997 AUC).
  - Explains the Edge Optimization / TensorRT roadmap and e-Gate application goals.
