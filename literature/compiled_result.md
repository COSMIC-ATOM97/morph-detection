# Final Report: All Learnings + Architecture Pick

## Scope
Join all four papers. Pick final morph detection architecture.

## Executive Summary
Current MAD fail real world. Three reasons:
1. Black box.
2. Overfit tiny data.
3. Learn dataset tricks, not true morph signs.

| Paper | Angle | Need |
| --- | --- | --- |
| paper1 | Explain | Post-hoc heatmaps slow + indirect → need built-in single-pass mask |
| paper2 | Survey | Generalization die on new data → need self-supervised attack-agnostic features + unlabeled data |
| paper3 | Baseline | 0.0% EER on ~38 images = memorize → need frequency stream, disentangle, ArcFace, augmentation |
| paper4 | Deep-dive | Miss FRONTEX metrics, O(D^3) slow, ignore covariates → need strict ISO eval, light dual-stream, LOO |

**Decision:** One unified arch = paper1/2 (DINOv2 encoder + cls head + U-Net mask, semi-supervised) + paper3/4 (spatial-freq dual stream, landmark attention, covariate disentangle, residual, ArcFace, hard cross-dataset eval).

---

## All Learnings

### Findings
**paper1:**
- Explain ViT morph decision by block regions. Watch confidence drop.
- Two blocks fuse: grid (coarse) + landmark (eyes, nose, mouth) → composite heatmap.
- Landmark zones = blend zones. Human + model agree.

**paper2:**
- MAD = S-MAD (one image) vs D-MAD (compare two).
- CNN, GAN, ViT catch blends but die on NIST FATE MORPH / FVC-onGoing.
- Face recognition features raise morph risk. Feature pathway between two identities.
- Train one morph method. Fail new methods (GAN, diffusion).

**paper3:**
- Baseline: FaceNet 128-d + HASC-Beta → 284-d → XQDA. Also LBP, BSIF, DCT.
- Attacks: OpenCV/FaceFusion, StyleGAN, MorGAN.
- Metric: MMPMR.
- D-MAD >> S-MAD: 3.36% vs 37.10% D-EER print-scan.

**paper4:**
- Full math: FaceNet triplet + HASC-Beta 156-d → L2 norm → 284-d → XQDA eigenvalue.
- Feature domains: texture (LBP/BSIF/SURF), freq (DCT, PRNU), residual (I - denoised).
- Standards: ISO/IEC 30107-3 (APCER, BPCER), MMPMR, Bologna-SOTAMD.
- Claim 0.0% EER / 1.0 AUC vs real lit (0.5-15% EER) = fake.

### Pros - Keep
| # | Strength | Source |
| --- | --- | --- |
| P1 | Post-hoc heatmaps: model-agnostic, no pixel labels | paper1 |
| P2 | Landmark regions = semantically clear | paper1 |
| P3 | Deep crush hand-made features on known data | paper2 |
| P4 | Many arch work: small CNN, Siamese, GAN demorph | paper2 |
| P5 | D-MAD >> S-MAD (3.36% vs 37.10% D-EER) | paper3 |
| P6 | Semantic + texture fusion = global + local errors | paper3, paper4 |
| P7 | XQDA = solid class separation math | paper4 |
| P8 | Residual + DCT/PRNU catch compression + generative freq errors | paper3, paper4 |
| P9 | MMPMR + APCER/BPCER = trustable metric stack | paper3, paper4 |

### Cons - Kill
| # | Weakness | Source | Sev |
| --- | --- | --- | --- |
| C1 | Occlusion heatmaps = multi-pass. Too slow for gates | paper1 | High |
| C2 | Localization indirect. Never learn artifact shape | paper1 | High |
| C3 | Grid block coarse. Miss sub-pixel blend edges | paper1 | High |
| C4 | ImageNet pretrain = macro objects, not micro errors | paper1 | Med |
| C5 | Black box (binary or low-res Grad-CAM) bad for security | paper2 | High |
| C6 | Overfit small data. Memorize light/camera noise | paper2,3,4 | Critical |
| C7 | Morph feature pathway. Classifier can't break it | paper2 | High |
| C8 | 0.0% EER on 38 images = memorize | paper3,4 | Critical |
| C9 | No cross-dataset / cross-algorithm test | paper3,4 | Critical |
| C10 | Fail unseen attacks, JPEG, low res, print-scan | paper3,4 | High |
| C11 | Ignore covariates: aging, beautify, twins, partial morph | paper3,4 | High |
| C12 | Slow: HASC + XQDA O(D^3) | paper3,4 | High |
| C13 | Miss BPCER @ APCER 1%/5% (FRONTEX) | paper4 | High |
| C14 | FaceNet old. No margin loss | paper4 | Med |

### Gaps → Fixes
| Gap | Fix | Source |
| --- | --- | --- |
| G1 | Intrinsic explain: native pixel mask | paper1,2 |
| G2 | Single-pass real-time: decision + explain same time | paper1 |
| G3 | Attack-agnostic features: self-supervised patch geometry, not tool print | paper1,2 |
| G4 | Semi-supervised: use unlabeled. Escape label bottleneck | paper1,2 |
| G5 | Frequency stream: DCT/DWT catch periodic/generative freq errors | paper3,4 |
| G6 | Landmark attention: weight eyes/nose/mouth. Fight partial morph | paper3,4 (+paper1) |
| G7 | Covariate disentangle: strip aging/light/JPEG | paper3,4 |
| G8 | Residual: R = I - denoise(I), local patches | paper3,4 |
| G9 | Robust train: extreme augment (blur, JPEG, print-scan) + ArcFace | paper3,4 |
| G10 | Honest eval: LOO cross-dataset, cross-algorithm holdout, D-EER + BPCER@APCER | paper3,4 |
| G11 | Modern backbone: kill FaceNet/XQDA | paper4 |

---

## All Four Agree
1. Status quo die outside training data. All four say so.
2. Generalization = main problem. Not raw accuracy.
3. Explain built-in, not bolt-on.
4. Landmark zones (eyes, nose, mouth) = key evidence. Independent find. Bake into arch.
5. Data scarcity: attack both sides. Unlabeled data (paper1/2) + heavy augment labeled (paper3/4).
6. Latency matter. Kill multi-pass occlusion. Kill O(D^3) XQDA. Border gate = hard limit.

## Tensions
| Tension | paper1/2 | paper3/4 | Solve |
| --- | --- | --- | --- |
| Single vs dual stream | DINOv2 alone | Spatial + frequency | Keep both. DINOv2 + light freq stream, fuse |
| Supervision | Semi-supervised | Supervised + augment | Self-supervised pretrain → supervised fine-tune + augment |
| Explain | U-Net mask central | Not said | Adopt mask head (paper1/2) |
| Freq/residual | Not said | Central | Adopt (paper3/4) |
| S-MAD vs D-MAD | Implied S-MAD | D-MAD much better | Open. Rec: S-MAD core + optional Siamese D-MAD head, same encoder |
| Mask labels | "No pixel labels" but U-Net need targets | — | Offline occlusion/landmark teacher → pseudo-masks, train only |

---

## Decision Criteria
| # | Criterion | Why | Weight |
| --- | --- | --- | --- |
| D1 | Cross-dataset + cross-algorithm generalization | Existential fail all 4 (C6-C10) | Must |
| D2 | Intrinsic single-pass explain (pixel mask) | paper1/2 core (G1, G2) | Must |
| D3 | Attack-agnostic learning | Unseen GAN/diffusion (G3) | Must |
| D4 | Real-time single-pass latency | Gates (C1, C12) | Must |
| D5 | ISO/IEC 30107-3 + FRONTEX reporting | paper4 (C13) | Must |
| D6 | Covariate robustness (twins, aging, JPEG, print-scan) | C10, C11, G7 | High |
| D7 | Data efficiency, label scarcity | G4, G9 | High |
| D8 | Freq/residual sensitivity | G5, G8 | High |
| D9 | Landmark focus | Convergent (§3) | High |
| D10 | Light modern backbone | C12, C14, G11 | Med |

---

## Final Arch: UID-MAD (Unified Interpretable Dual-Stream MAD)

> paper1/2 system (DINOv2 encoder → cls head + U-Net, semi-supervised labeled+unlabeled) + paper3/4 system (dual-stream spatial-freq, landmark attention, adversarial disentangle, residual, ArcFace).

```
ALL IMAGES (labeled + unlabeled)
  → DATA LAYER: align → augment (JPEG/blur/print-scan/aging/beautify) → residual R(x,y)
  → split:
      STREAM A - SPATIAL: DINOv2/ViT shared encoder (self-supervised)
                    + landmark attention eyes/nose/mouth
      STREAM B - FREQUENCY: DCT/DWT on luma + residual patches, light CNN/MLP
  → CROSS-ATTENTION FUSION (spatial ↔ frequency)
  → three heads:
      HEAD 1: Classification REAL/MORPHED (single pass)
      HEAD 2: U-Net mask decoder (pixel region = intrinsic explain)
      HEAD 3 (optional): Siamese D-MAD comparator (same encoder, suspect + reference)
  → ADVERSARIAL COVARIATE DISENTANGLE (GRL on illumination, JPEG, age)
  → LOSSES: ArcFace/BCE + mask + adv (GRL) + consistency (semi-sup.)
```

### Component ↔ Fix
| Component | Fixes | Source |
| --- | --- | --- |
| DINOv2/ViT shared encoder | G3, G4, C4, C14, G11 | paper1,2 |
| Cls head single pass | G2, C1 | paper1,2 |
| U-Net mask decoder | G1, C2, C5 | paper1,2 |
| Labeled + unlabeled ingest | G4, C6 | paper1,2 |
| Freq stream (DCT/DWT) | G5, C10 | paper3,4 |
| Residual R(x,y) input | G8 | paper3,4 |
| Landmark attention | G6, §3 convergence | paper1+3+4 |
| Cross-attention fusion | C7 (break identity pathway) | paper2+3/4 |
| Adversarial disentangle (GRL) | G7, C11 | paper3,4 |
| ArcFace margin + BCE | G9, C14 | paper3,4 |
| Heavy augmentation | G9, C6, C10 | paper3,4 |
| Optional Siamese head | D-MAD edge (P5) | paper3,4 |
| Strict LOO eval | G10, C9, C13 | paper3,4 |

---

## Training

### Data + Augment
- Pool 1 (labeled): real vs morph. Make morphs in-house with held-out algos (warp, StyleGAN, diffusion). Train never see eval generators.
- Pool 2 (unlabeled): rest of faces → DINOv2 self-supervised pretrain + consistency pseudo-labels (confidence threshold).
- Augment (C6, C10, G9): Gaussian blur, extreme JPEG (20-60), print-scan sim, low res, random light, beautify sim, partial crop.

### Weak Mask Labels (solve tension)
paper1 say "no pixel labels" but U-Net need targets. Fix: offline teacher. Run paper1 occlusion + landmark blend dilation once, offline, train images only → pseudo-masks. Inference stays single-pass. Slow loop never touch deploy.

### Loss
```
L_total = L_cls(ArcFace-margin CE + BCE)
        + λ_m · L_mask (BCE + Dice on pseudo-masks)
        + λ_a · L_adv (GRL covariate classifier)
        + λ_c · L_consistency (labeled ↔ unlabeled agree)
```
ArcFace = hard latent split real vs morph (kill C7 pathway). Mask loss = localization skill (G1). Adv loss = strip covariates (G7).

### Schedule
1. Self-supervised pretrain encoder (or load DINOv2).
2. Warm heads on labeled, freeze encoder.
3. Joint fine-tune. Turn on GRL after warm-up (stabilize).
4. Semi-supervised consistency on unlabeled.

---

## Eval Protocol (non-negotiable — G10, C9, C13)
| Need | Spec |
| --- | --- |
| Cross-dataset | LOO: train AMSL → test FRLL, reverse (baseline never did) |
| Cross-algorithm | Hold out whole generators (train warp+GAN, test diffusion) |
| External bench | Bologna-SOTAMD, NIST FATE MORPH, FVC-onGoing |
| Core metrics | D-EER, EER, AUC, MMPMR |
| Compliance | BPCER @ APCER 1% + 5% (FRONTEX/ISO 30107-3) — must |
| Covariate slices | Separate EER: print-scan, JPEG ≤40, low res, aging/beautified, twin |
| Honesty rule | Near 0% EER → data-leak audit first (C8) |
| Ablations | Kill each: freq stream, mask head, GRL, landmark attention, semi-sup stage |

**Must beat:** lit EER 0.5-15% (Dargaud/Borghi/Colbois), Bologna D-MAD 3.36% D-EER, FaceNet+HASC+XQDA reimpl under proper LOO (its 0% claim die).

---

## Decision Matrix
| Option | Desc | D1 | D2 | D3 | D4 | D5 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | FaceNet+HASC+XQDA baseline | ✗ | ✗ | ✗ | ✗ | ✗ | Reject |
| B | ViT + post-hoc heatmaps | ~ | ✓ but slow | ✗ | ✗ | ~ | Reject primary. Keep as offline pseudo-mask teacher + audit |
| C | DINOv2 + cls + U-Net only | ✓ | ✓ | ✓ | ✓ | ~ | Incomplete. No freq (D8), no disentangle (D6) |
| D | Two-stream + ArcFace, no mask | ✓ | ✗ | ✓ | ✓ | ✓ | Incomplete. Fail D2 (black box) |
| **E = UID-MAD** | C ∪ D | ✓ | ✓ | ✓ | ✓ | ✓ | **Recommended** |

### Open: S-MAD vs D-MAD
paper3/4: D-MAD easier (3.36% vs 37%). paper1/2: explain one image = S-MAD. **Rec:** UID-MAD S-MAD first (harder, more general; mask only makes sense one image). Add Head 3 Siamese for optional D-MAD when reference exist. One encoder, both modes, no fork.

---

## Risks
| Risk | Fix |
| --- | --- |
| Pseudo-mask bad | Teacher = paper1 landmark+grid occlusion, dilated. Validate mask head with pointing-game/IOU small hand-audited set |
| GRL unstable | Delay GRL till heads converge. Tune λ_a with gradient-norm monitor |
| Freq branch slow | Keep DCT/DWT + tiny CNN. Target <5 ms. INT8 quantize for gate |
| DINOv2 heavy on edge | Phase 2: distill to MobileViT (paper4 nominate it), keep mask head |
| Semi-sup label noise | Confidence threshold + consistency loss. Pseudo-labels never enter mask branch |
| Overfit creep back (C8) | LOO in CI. Near-0% EER auto-block report |

---

## Roadmap
1. **Phase 0 — Harness:** LOO cross-dataset train/eval + augment + metrics (D-EER, BPCER@APCER 1/5%, MMPMR) BEFORE arch code. (Answer C9/C13.)
2. **Phase 1 — Core UID-MAD (S-MAD):** DINOv2 encoder + cls head + ArcFace/BCE. Set cross-dataset baseline.
3. **Phase 2 — Freq + residual stream:** Add Stream B + fusion. Ablation D8.
4. **Phase 3 — Mask head:** Offline occlusion teacher → U-Net → intrinsic single-pass explain (G1/G2).
5. **Phase 4 — Disentangle + landmark attention:** GRL covariate + eyes/nose/mouth attention (G6/G7).
6. **Phase 5 — Semi-sup scale-up:** Unlabeled pretrain + consistency (G4).
7. **Phase 6 — Optional D-MAD head:** Siamese comparator (P5).
8. **Phase 7 — Edge optimize:** Distill/quantize for real-time gates (D4).

Each phase end with full §8 protocol. Ship only if no cross-dataset D-EER or BPCER@APCER regress.

---

## Final Rec
**Adopt Option E — UID-MAD.** Unified, interpretable, dual-stream, semi-supervised multi-task.

- Only option hits ALL ten criteria (§5).
- Every con from four papers maps to specific component/loss/protocol rule (§6.1, §2.3).
- paper1/2 alone = no frequency robustness. paper3/4 alone = black box. **Union mandatory.**
- paper1 slow occlusion not thrown away — demoted to offline teacher/audit. Its one strength (semantic landmark heatmaps) becomes training signal for fast U-Net head.

**Decide:** (a) approve UID-MAD, (b) confirm S-MAD-first + optional D-MAD head, (c) start Phase 0 eval harness.
