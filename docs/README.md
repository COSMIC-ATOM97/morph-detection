# UID-MAD: Formal Documentation & Reports

This directory stores the formal academic research reports, system architecture specifications, and empirical phase evaluations.

---

## 📂 Document Index

1. **[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md):**
   - **The Master Academic Dossier & Research Paper:** Formulates the problem under ISO/IEC 30107-3 standards, outlines the mathematical models for Stream A and Stream B, presents comprehensive cross-dataset evaluation matrices across 15 splits, details the failure autopsies, and outlines the operational path to deployment.
2. **[`architecture.md`](architecture.md):**
   - **System Architecture & Tensor Dimensions:** Detailed specification of the DINOv2 dense patch aggregator, Laplacian high-pass filter, 4-stage convolutional encoder, and multi-branch ArcFace classification heads.
3. **[`phase1_report.md`](phase1_report.md):**
   - **Phase 1 Evaluation Report:** Establishes the single-stream spatial baseline (0.9993 AUC on OpenCV, 0.9958 AUC on FaceMorpher) and presents the diagnostic autopsy of why spatial models collapse on StyleGAN2 morphs (0.4988 AUC).
4. **[`phase2_report.md`](phase2_report.md):**
   - **Phase 2 Evaluation Report:** Evaluates the dual-stream frequency residual stream, documenting the StyleGAN breakthrough (AUC surged to 0.8525), the modality competition phenomenon, and the sensor noise domain shift.
