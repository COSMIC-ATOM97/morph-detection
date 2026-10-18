#!/usr/bin/env python3
"""
Phase 3: Explainable AI (XAI) & Morph Seam Localization Pipeline
Generates multi-panel forensic exhibits combining:
  1. Input RGB Image
  2. Laplacian High-Pass Residual Map
  3. DINOv2 Spatial ViT Attention Rollout Heatmap
  4. Frequency Stream Grad-CAM Spectral Activation Heatmap
  5. Multi-Branch Margin & Calibrated Decision Dashboard
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dual_stream import DualStreamUIDMAD
from src.explainability.token_attention import DINOv2AttentionRollout
from src.explainability.gradcam import FrequencyGradCAM
from src.explainability.visualizer import ForensicVisualizer
from src.data.augmentations import get_val_transforms

def load_model(checkpoint_path: str, device: torch.device) -> DualStreamUIDMAD:
    print(f"[*] Loading Dual-Stream UID-MAD from: {checkpoint_path}")
    model = DualStreamUIDMAD()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Clean potential DataParallel 'module.' prefix
    cleaned_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        cleaned_dict[new_k] = v

    model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def process_single_image(image_path: str, model: DualStreamUIDMAD,
                         rollout: DINOv2AttentionRollout, gradcam: FrequencyGradCAM,
                         visualizer: ForensicVisualizer, device: torch.device,
                         output_name: str, true_label: str = "Unknown"):
    transform = get_val_transforms(img_size=224)
    pil_img = Image.open(image_path).convert("RGB")
    raw_np = np.array(pil_img.resize((224, 224))) / 255.0

    transformed = transform(image=np.array(pil_img))
    tensor_img = transformed["image"].unsqueeze(0).to(device)

    # 1. Compute Laplacian residual map
    with torch.no_grad():
        lap_residual = model.freq_stream.high_pass(tensor_img).mean(dim=1).squeeze().cpu().numpy()

    # 2. Compute Spatial ViT Attention Rollout Heatmap
    spatial_heatmap = rollout.compute_rollout(tensor_img)

    # 3. Compute Frequency Stream Grad-CAM
    tensor_img.requires_grad = True
    freq_cam = gradcam.compute_cam(tensor_img, target_class=1)

    # 4. Compute Model Decision Scores
    with torch.no_grad():
        out = model(tensor_img)
        norm_w_s = F.normalize(model.spatial_head.weight, p=2, dim=1)
        norm_e_s = F.normalize(out["emb_spatial"], p=2, dim=1)
        cos_s = F.linear(norm_e_s, norm_w_s)[0]
        s_margin = float((cos_s[1] - cos_s[0]).cpu())

        norm_w_f = F.normalize(model.freq_head.weight, p=2, dim=1)
        norm_e_f = F.normalize(out["emb_freq"], p=2, dim=1)
        cos_f = F.linear(norm_e_f, norm_w_f)[0]
        f_margin = float((cos_f[1] - cos_f[0]).cpu())

        # Estimated calibration offset from validation distribution
        calibrated_score = 0.5 * (s_margin - 0.15) + 0.5 * (f_margin - 0.10)

    scores = {
        "spatial_margin": s_margin,
        "freq_margin": f_margin,
        "calibrated_score": calibrated_score
    }

    save_path = visualizer.generate_exhibit(
        raw_img=raw_np,
        laplacian_residual=lap_residual,
        spatial_heatmap=spatial_heatmap,
        freq_cam=freq_cam,
        scores=scores,
        save_name=output_name,
        true_label=true_label
    )
    print(f"  [+] Saved forensic exhibit: {save_path} (Calibrated Score: {calibrated_score:+.3f})")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="UID-MAD Phase 3: Forensic Explainability & Seam Localization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2_dual_stream_best.pt",
                        help="Path to trained DualStreamUIDMAD weights.")
    parser.add_argument("--output-dir", type=str, default="results/phase3/heatmaps",
                        help="Directory to save generated forensic exhibits.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device (cuda or cpu).")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        print(f"[!] Warning: Checkpoint {args.checkpoint} not found. Run training or provide weights.")
        return

    model = load_model(args.checkpoint, device)
    rollout = DINOv2AttentionRollout(model, start_block=6)
    gradcam = FrequencyGradCAM(model)
    visualizer = ForensicVisualizer(output_dir=args.output_dir)

    print(f"[*] Phase 3 Forensic Explainability pipeline initialized on {device}.")
    print(f"[*] Output directory ready: {args.output_dir}")

if __name__ == "__main__":
    main()
