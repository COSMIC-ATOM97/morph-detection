
# Student-Scoped UID-MAD: Practical Unified Interpretable Dual-Stream Morph Attack Detection

**Version:** 1.0 (College-Student Feasible)  
**Date:** September 2026  
**Goal:** A clean, modern, high-impact MAD architecture that a motivated college student can implement in one semester, while remaining competitive with 2025–2026 published methods on honest cross-dataset evaluation.

---

## 1. Executive Summary

We keep the strongest, most practical ideas from the original four-paper synthesis and deliberately drop the high-difficulty components (full U-Net with offline pseudo-mask teacher, adversarial GRL disentanglement, and large-scale semi-supervised consistency training).

**Core Architecture (Student-Scoped UID-MAD):**
- Shared modern self-supervised backbone (DINOv3 small / DINOv2 ViT-S or B)
- Dual-stream design: Spatial (backbone) + Frequency Residual stream
- Simple landmark-guided regional attention
- Classification head with ArcFace-style margin loss
- Single-pass inference
- Strict Leave-One-Out (LOO) cross-dataset + cross-algorithm evaluation with ISO/IEC 30107-3 metrics

This version prioritizes **generalization**, **frequency sensitivity**, **modern features**, and **honest evaluation** — the exact combination that current top papers (FD-MAD 2026, SelfMAD 2025, MADation 2025) show is most effective.

---

## 2. What We Kept vs. What We Simplified

| Original Component                  | Decision          | Reason |
|-------------------------------------|-------------------|--------|
| DINOv2 / DINOv3 backbone            | **Keep** (small variants) | Best dense features available in 2026 |
| Frequency / residual stream         | **Keep** (lightweight) | Highest impact for cross-dataset gains (see FD-MAD 2026) |
| Classification + ArcFace / margin   | **Keep**          | Simple and effective |
| Landmark regional attention         | **Keep** (simplified) | Easy win, matches paper1 + paper3/4 convergence |
| Strict LOO + BPCER@APCER reporting  | **Keep** (mandatory) | Non-negotiable for credibility |
| Full U-Net mask head + offline teacher | **Defer**       | High implementation + validation cost |
| Adversarial GRL disentanglement     | **Defer**         | Unstable and time-consuming |
| Full semi-supervised consistency    | **Defer**         | Requires large clean unlabeled pool + careful tuning |
| Optional Siamese D-MAD head         | **Optional later**| Build pure S-MAD first |

---

## 3. Final Recommended Architecture — Student-Scoped UID-MAD

```
                    ┌─────────────────────────────────────────────┐
  INPUT IMAGE ────► │  PREPROCESSING                              │
                    │  Face detect + align (RetinaFace/MTCNN)     │
                    │  Quality filter + ICAO-style checks         │
                    │  Heavy augmentation (train only)            │
                    │  Residual map R = I − simple_denoise(I)     │
                    └──────────────────┬──────────────────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
   ┌──────────────────────────────┐           ┌──────────────────────────────┐
   │ STREAM A — SPATIAL           │           │ STREAM B — FREQUENCY         │
   │ DINOv3-S / DINOv2 ViT-S/B    │           │ DCT or DWT on residual map   │
   │ (frozen or lightly fine-tuned)│           │ + small CNN / MLP encoder    │
   │ + simple landmark attention  │           │ (very lightweight)           │
   │   (eyes / nose / mouth)      │           └──────────────┬───────────────┘
   └──────────────┬───────────────┘                          │
                  └──────────────────┬───────────────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │ FUSION                     │
                      │ Concat / gated /            │
                      │ simple cross-attention      │
                      └─────────────┬──────────────┘
                                    ▼
                      ┌────────────────────────────┐
                      │ HEAD                       │
                      │ ArcFace / Margin + BCE     │
                      │ → REAL / MORPHED score     │
                      └────────────────────────────┘
```

### 3.1 Component Details

**Preprocessing**
- Face detection + 5-point landmark alignment
- Optional quality filtering (blur, resolution, pose)
- Residual map: simple high-pass or light denoising residual (cheap and effective)

**Stream A – Spatial**
- Backbone: DINOv3 ViT-S/16 or ViT-B/16 (or DINOv2 equivalent). Prefer the distilled small models released with DINOv3 (Aug 2025).
- Landmark attention: extract features from eye, nose, and mouth regions (using the same landmarks from alignment) and weight or concatenate them. Very low overhead.

**Stream B – Frequency Residual**
- Compute residual map once.
- Apply 2D DCT or multi-level DWT on the residual (or on luma channel).
- Feed selected frequency bands / patches into a tiny CNN or MLP (2–4 layers). This follows the successful spirit of FD-MAD (2026).

**Fusion**
- Simple concatenation + linear layer, or a lightweight gated fusion, or one cross-attention block. Keep it minimal.

**Head**
- ArcFace (or CosFace / AdaFace) margin loss on the fused features + binary cross-entropy.
- Output: single morph score (higher = more likely morph).

**Inference**
- Completely single-pass. No multi-pass occlusion, no extra teachers at test time.

---

## 4. Training Strategy (Simplified)

1. **Data**
   - Labeled pool only for the first version (SMDD + any public morph sets you have access to).
   - Hold out entire generators and entire datasets for LOO evaluation from day 1.
   - Generate extra morphs in-house with algorithms that will **never** appear in the test sets.

2. **Augmentation (critical)**
   - Extreme JPEG (quality 20–60)
   - Gaussian blur, motion blur
   - Simulated print-scan (downsample + JPEG + mild noise)
   - Random illumination / gamma
   - Light beautification filters
   - Partial crops / small occlusions

3. **Optimization**
   - Freeze backbone first → warm-up head → lightly unfreeze last blocks if needed.
   - Loss = ArcFace-margin classification loss + optional small BCE term.
   - Standard AdamW + cosine schedule.

4. **What we deliberately skip for now**
   - Pseudo-mask generation and mask loss
   - Gradient-reversal layers
   - Consistency losses on unlabeled data

---

## 5. Evaluation Protocol (Non-Negotiable)

| Requirement              | Specification |
|--------------------------|---------------|
| Cross-dataset            | Leave-One-Out (e.g., train on SMDD / AMSL → test FRLL-Morph and reverse) |
| Cross-algorithm          | Hold out entire morph generators |
| Metrics                  | D-EER / EER, AUC, **BPCER @ APCER = 1% and 5%** (ISO/IEC 30107-3) |
| Extra slices             | Print-scan, heavy JPEG, low-resolution |
| Honesty rule             | Any result near 0% EER triggers automatic data-leakage audit |
| Ablations                | Spatial-only vs Dual-stream, with/without landmark attention |

**Target to beat:** Published LOO numbers from FD-MAD, SelfMAD, MADation, and a re-implemented simple FaceNet/HASC baseline under the **same** protocol.

---

## 6. Implementation Roadmap (Student Timeline)

### Phase 0 – Evaluation Harness & Data Pipeline (2–3 weeks)
- LOO dataset loaders
- Full metric suite (D-EER, BPCER@APCER 1%/5%)
- Residual map generation
- Heavy augmentation pipeline
- Experiment logging (W&B or simple CSV + TensorBoard)
- **Exit criterion:** You can train a linear probe on frozen DINOv3 and get honest LOO numbers.

### Phase 1 – Spatial Baseline (2–3 weeks)
- DINOv3-S / DINOv2 ViT-S/B + ArcFace head
- Frozen → light fine-tune
- Full LOO evaluation
- **Exit criterion:** Solid spatial-only numbers that already beat simple handcrafted or old CNN baselines.

### Phase 2 – Frequency Residual Stream + Fusion (2 weeks)
- Add Stream B
- Simple fusion
- Ablation study
- **Exit criterion:** Clear gain on cross-dataset and JPEG/print-scan slices.

### Phase 3 – Landmark Attention & Polish (1–2 weeks)
- Add simple regional attention
- Hyper-parameter sweep
- Final LOO tables + ablations
- **Exit criterion:** Clean, reproducible results ready for a report or workshop paper.

### Phase 4 – Optional Extensions (later)
- Lightweight decoder for rough localization maps
- Siamese D-MAD head
- Distillation to MobileViT / EfficientViT
- Limited semi-supervised experiments

---

## 7. Expected Strengths of This Scoped Version

- Strong generalization (self-supervised backbone + frequency residual + heavy augmentation)
- Very practical latency (small DINOv3 + tiny frequency branch)
- Easy to implement and debug
- Fully aligned with 2025–2026 SOTA trends
- Honest evaluation from day one
- Clear path to later upgrades (mask head, disentanglement, etc.) without rewriting the core

---

## 8. Final Recommendation

**Adopt this Student-Scoped UID-MAD as the working architecture.**

It removes the highest-risk, highest-effort components while preserving the ideas that actually drive performance in 2026: modern self-supervised features, frequency residual analysis, landmark focus, and rigorous LOO evaluation.

**Next concrete steps:**
1. Implement Phase 0 evaluation harness.
2. Choose exact backbone (recommend starting with DINOv3 ViT-S/16 or ViT-B/16).
3. Build residual map + frequency stream prototype.
4. Run the first honest LOO baseline.

This version is ready to code.

