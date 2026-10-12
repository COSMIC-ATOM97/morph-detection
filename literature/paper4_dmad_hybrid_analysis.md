# Paper 4: D-MAD Hybrid Deep-Dive

## Main Idea
- S-MAD: one suspect image, CNN (VGG19, ResNet).
- D-MAD: compare suspect vs trusted live. Siamese, ArcFace, GAN demorph.
- Feature types:
  - Texture: LBP, BSIF, SURF.
  - Frequency: DCT, PRNU. Catch double-compression, sensor noise.
  - Residual: original minus CNN-denoised. Catch pixel breaks.
- Paper's method: images → 160x160 → FaceNet 128-d (triplet loss) + HASC-Beta 156-d (HSV, YCbCr) → join 284-d → XQDA. XQDA uses eigenvalue math.

## Standards
- ISO/IEC 30107-3: APCER, BPCER. MMPMR = weakness metric.
- Bologna-SOTAMD baselines: D-MAD 3.36% D-EER. S-MAD 37.10% / 38.99%.
- Paper claims 0.0% EER, 1.0 AUC. Literature: Dargaud 7-15%, Borghi 1.3-2.9%, Colbois 0.5-1.2% EER. Claim unrealistic.

## Good
- Math fusion: deep semantic + shallow texture. Global shape + local blend errors.
- XQDA solid class separation.

## Bad
- Overfit. Only 38 real images (AMSL), 43 (FRLL). Memorize, not learn.
- No cross-dataset test. No cross-algorithm test.
- Missing FRONTEX numbers: BPCER at 1% / 5% APCER.
- Ignore real threats: twins, look-alikes.
- Ignore physical: print-scan, JPEG, partial morph, aging, beautify.
- Slow: HASC mutual info + XQDA O(D^3). Border gate too slow.
- FaceNet old. No margin loss.

## Need Better - Custom Arch
1. Two streams: MobileViT/DenseNet spatial + DCT/DWT frequency. Catch blend ghosting + GAN/diffusion spectral noise.
2. ViT attention: weight eyes, nose, mouth. Fight partial morph. Prioritize edge breaks.
3. Adversarial layer: strip covariates (gate light, aging, beautify) from morph signs.
4. Do it:
   - Data loader: heavy augmentation. Gaussian blur, extreme JPEG, print-scan sim.
   - Residual: high-freq residual maps in local patches. No global pooling.
   - Loss: ArcFace margin + BCE. Separate real vs morph hard.
   - Eval: Leave-One-Out cross-dataset. Report D-EER + BPCER at 1%/5% APCER.
