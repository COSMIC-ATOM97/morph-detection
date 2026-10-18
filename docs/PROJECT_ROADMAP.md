# UID-MAD: Strategic Master Plan & Project Roadmap

**Project:** Unified Interpretable Face Morph Attack Detection (UID-MAD)  
**Standard:** ISO/IEC 30107-3 (Biometric Presentation Attack Detection) / ICAO Doc 9303  
**Status:** Phase 1 & 2 Completed (Empirically Validated) | Transitioning to Phase 3  

---

## 1. Executive Master Roadmap (Phases 1 – 4)

| Phase | Milestone | Primary Architecture / Focus | Key Empirical Output | Status |
|---|---|---|---|---|
| **Phase 1** | Spatial Baseline Pipeline | DINOv2 ViT-S/14 + Multi-Scale Patch Pooling (1536-d) + ArcFace ($s=30, m=0.35$) | 0.9993 AUC on OpenCV; discovered StyleGAN2 collapse (0.4988 AUC) | **Complete** |
| **Phase 2** | Dual-Stream UID-MAD | Spatial ViT (LoRA $r=16$) + Laplacian ConvNet (128-d) + Calibrated Fusion | StyleGAN2 surged to 0.9399 AUC clean / 0.9854 print-scan; mean AUC 0.992 | **Complete** |
| **Phase 3** | Explainability & Deployment | Seam Heatmaps (Grad-CAM + Attention Rollout) + ONNX / TensorRT Latency (<50ms) | Morph seam localization exhibits & e-Gate compliance profile | **Active** |
| **Phase 4** | Publication & Oral Defense | Comprehensive Academic Manuscript (IEEE TBIOM / WACV) + Presentation Slide Deck | Conference submission draft, thesis dossier, and public release | **Upcoming** |

---

## 2. Standardized Repository Structure

```
morph-detection/
├── README.md                          # Primary research dossier, architecture & benchmarks
├── LICENSE                            # Open-source research license (MIT)
├── requirements.txt                   # Production package dependencies
├── kaggle_train.py                    # Self-contained Kaggle runner (Dual T4 optimized)
│
├── src/                               # MODULAR CORE SOURCE CODE
│   ├── models/                        # Neural network architectures
│   │   ├── spatial_vit.py             # DINOv2 backbone with LoRA low-rank adapters
│   │   ├── frequency_stream.py        # Laplacian filter + 4-stage residual ConvNet
│   │   ├── dual_stream.py             # Unified Dual-Stream model wrapper
│   │   └── layers/                    # Factorized LoRA and trigonometric ArcFace
│   ├── data/                          # Dataset loaders & Albumentations pipelines
│   ├── fusion/                        # Z-score margin calibration & decision rules
│   └── explainability/                # Grad-CAM, token attention rollout & seam visualizer
│
├── scripts/                           # REPRODUCIBLE EXECUTION ENTRY POINTS
│   ├── phase1_dino_spatial.py         # Phase 1 standalone script (Spatial ViT Baseline)
│   ├── phase1_dino_spatial.ipynb      # Phase 1 interactive Jupyter Notebook
│   ├── phase2_dual_stream_uid_mad.py  # Phase 2 dual-stream standalone script
│   ├── phase2_dual_stream_uid_mad.ipynb # Phase 2 dual-stream Jupyter Notebook
│   ├── phase3_explainability.py       # Phase 3 standalone: Heatmaps & forensic exhibits
│   ├── export_onnx.py                 # Phase 3: ONNX & TensorRT model export
│   └── benchmark_latency.py           # Phase 3: Forward inference latency profiler
│
├── docs/                              # FORMAL ACADEMIC RESEARCH & DEFENSE DOSSIERS
│   ├── README.md                      # Index of all technical reports
│   ├── PROJECT_ROADMAP.md             # Living sprint tracker & milestone plan
│   ├── architecture.md                # Mathematical formulations & tensor dimensions
│   ├── RESEARCH_REPORT.md             # Comprehensive problem formulation & standards
│   ├── phase1_comprehensive_briefing.md # Phase 1 defense guide & faculty Q&A matrix
│   ├── week1_report.md                # Week 1 report: Spatial ViT baseline
│   ├── week2_report.md                # Week 2 report: Dual-stream design & PRNU autopsy
│   └── week3_report.md                # Week 3 report: Calibrated fusion & StyleGAN2 fix
│
├── results/                           # QUANTITATIVE ARTIFACTS (METRICS & PLOTS)
│   ├── README.md                      # Directory overview & benchmark descriptions
│   ├── phase1/                        # Phase 1 evaluation metrics
│   ├── phase2/                        # Phase 2 dual-stream metrics (15-slice tables & ROC curves)
│   └── phase3/                        # Phase 3 explainability & latency outputs
│
├── checkpoints/                       # MODEL WEIGHTS & CHECKPOINTS (Git Ignored)
│   ├── phase1_spatial_baseline.pt     # Phase 1 trained weights (91.9 MB)
│   ├── phase2_dual_stream_best.pt     # Phase 2 best validation model (96.4 MB)
│   └── phase2_dual_stream_final.pt    # Phase 2 final epoch model (96.4 MB)
│
├── literature/                        # FOUNDATIONAL BACKGROUND LITERATURE
│   ├── literature_synthesis.md        # Synthesis of 4 foundational MAD papers
│   └── paper[1-4]_*.md                # Detailed deep-dives on reference literature
│
├── team_internal/                     # DEVELOPER WORKSPACE & HANDOFF NOTES
└── .okf/                              # PROJECT DURABLE MEMORY BUNDLE
```

---

## 3. Phase 3 Sprint Plan: Explainability & Deployment

### Sprint 3.1: Morph Seam Localization & Forensic Explainability
- **Objective:** Provide visual justification for morph attack predictions to support forensic border control inspections.
- **Components:**
  1. **Self-Attention Rollout (Spatial Stream):** Track token attention matrices from DINOv2 blocks 10-11 to project localized spatial weighting onto the original 224x224 RGB image, highlighting warping seams and cutouts.
  2. **Grad-CAM (Frequency Residual Stream):** Compute gradient backpropagation from frequency logits to stage-4 convolutional residual feature maps, highlighting periodic transposed convolution synthesis noise.
  3. **Multi-Panel Forensic Exhibit:** Generate 4-panel visual inspection figures: `[Input Image] | [Laplacian Residual] | [Attention Heatmap Overlay] | [Score Dashboard]`.

### Sprint 3.2: Model Serialization & e-Gate Latency Benchmarking
- **Objective:** Verify compliance with the ICAO Doc 9303 border control latency requirement (<50ms per verification).
- **Components:**
  1. **ONNX Export:** Export dual-stream graph via `torch.onnx.export` with dynamic batching.
  2. **Half-Precision & Quantization:** Test FP16 and INT8 Post-Training Quantization (PTQ) via TensorRT / ONNX Runtime.
  3. **Latency Benchmarking:** Measure latency distributions (mean, p95, p99) across batch sizes {1, 4, 8} on GPU and CPU.

---

## 4. Phase 4 Sprint Plan: Paper Manuscript & Oral Defense

1. **Target Venues:** IEEE Transactions on Biometrics, Identity, and Behavior (TBIOM) / IEEE/CVF WACV.
2. **Core Narrative:**
   - Proof of the representation boundary of spatial-only ViTs (collapse on StyleGAN2).
   - Discovery of print-scan domain orthogonality (spatial ViT attenuates; frequency stream surges to >0.997 AUC).
   - Late Z-score calibrated decision fusion as a generalizable solution for heterogeneous biometric streams.
3. **Defense Preparation:** Structured slide deck, Q&A defense matrix, and interactive presentation demo.
