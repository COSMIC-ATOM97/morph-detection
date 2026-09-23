# Paper 3: FaceNet + HASC + XQDA Baseline

## Main Idea
- Test morph detection two ways: S-MAD (one image) and D-MAD (compare images).
- Features: FaceNet 128-d embedding + HASC-Beta texture. Join → 284-d vector.
- XQDA reduce dimensions. Separate classes.
- Other features: LBP, BSIF, DCT (frequency).
- Attacks tested: OpenCV, FaceFusion warp; StyleGAN, MorGAN.
- Weakness metric: MMPMR formula.

## Good
- D-MAD much better than S-MAD. D-EER 3.36% vs 37.10% print-scan.
- Hybrid features catch big face shape + small blend texture.

## Bad
- Overfit catastrophe. Claim 0.0% EER, 1.0 AUC. Only 38 real training images. Fake good.
- No cross-dataset test. Train and test same data.
- Fail real world: unseen attacks, heavy JPEG, low res, print-scan.
- Ignore covariates: aging, beautify filters, identical twins, partial morph.
- Slow. Not fit border gate real-time.

## Need Better - Custom Arch
- Two streams: DenseNet spatial (keep high-freq detail) + DCT frequency (catch generative spectral errors).
- ViT attention on eyes, nose, mouth. Prioritize edge breaks over background.
- Adversarial layer: separate morph errors from natural covariates (aging, light).
- Residual: R = original - denoised. Analyze blend breaks in local patches.
- Heavy augmentation: Gaussian blur, print-scan degrade. ArcFace margin loss. Force real vs morph apart in latent space.
