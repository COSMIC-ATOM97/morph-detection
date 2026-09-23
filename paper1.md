# Paper 1: Composite Heatmaps - Morph Attack Detection

## Main Idea
- Deep model = black box. Hard know why model say "morph" or "real face."
- Method: block parts of face image. Watch confidence drop. Face parts that drop confidence = important parts.

## Two Ways Block Face
- Grid block: cover face in squares. Coarse.
- Landmark block: cover eyes, nose, mouth. Smart - these = morph blend zones.

## Result
- Fuse both drops → one heatmap. Heatmap show face parts model care about.

## Good
- Work on any trained ViT. No retrain needed.
- Landmark regions match human gut feeling.
- No pixel annotations needed. Only need yes/no labels.

## Bad
- Very slow. Many passes through model. Not real-time.
- Model never learn what morph look like. Only guess by score drop.
- Grid too coarse. Miss small blend edges.
- ImageNet pretrain too big-picture. Miss tiny structure errors.

## Need Better
- Model must output mask directly. No guessing after.
- One pass only. Decision + explanation same time.
- Better features: self-supervised models (DINOv2).
- Use unlabeled data too. Not only labeled.

## Proposed Fix
- Ingest ALL images: labeled + unlabeled.
- DINOv2/ViT shared encoder → visual features.
- Two heads one pass:
  1. Classification head: REAL or MORPHED.
  2. U-Net decoder: output region mask. Instant explain.
