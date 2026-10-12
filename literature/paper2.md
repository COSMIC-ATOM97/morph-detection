# Paper 2: Deep Learning Surveys - Morph Attack Detection

## Main Idea
- Surveys show neural nets vs face morph attacks.
- Two types: S-MAD (one image) and D-MAD (compare two images).
- Models: CNN, GAN, ViT. Find blend artifacts.
- Good on own data. Bad on NIST FATE MORPH / FVC-onGoing benchmarks.
- Face recognition features actually help morph attacks. Morph create "pathway" between two identities.
- Train on one morph method. Fail on new method (GAN, diffusion).

## Good
- Deep models beat old hand-made features (LBP, HOG) on known data.
- Many architectures work: small CNN, Siamese networks.

## Bad
- Black box. Only binary score or blurry Grad-CAM. Not enough for security.
- Small dirty datasets. Models memorize lighting/camera noise. Not real morph signs.
- Morph makes feature bridge between identities. Classifier can't break bridge. False accepts.

## Need Better
- Use unlabeled data. Learn face structure without full labels.
- Mask must be built-in. Not bolt-on after.
- Learn attack-agnostic features. Not one tool's fingerprint. Self-supervised patch geometry.

## Proposed Fix
- ALL images: labeled + unlabeled. Semi-supervised. No dataset bottleneck.
- DINOv2 encoder: self-supervised. Sees local patch relations. Catches unseen morphs better.
- Two heads one pass:
  1. Classification head: REAL / MORPHED.
  2. U-Net decoder: pixel mask. Show exactly where morph is. Real-time explain.
