import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

class ForensicVisualizer:
    """
    Generates multi-panel forensic visual inspection exhibits combining:
      1. Original Input Image
      2. Laplacian High-Pass Residual Map
      3. DINOv2 Spatial Token Attention Rollout Heatmap
      4. Frequency Stream Grad-CAM Activation Heatmap
      5. Multi-Branch Decision Verdict & Margin Bar Chart
    """
    def __init__(self, output_dir: str = "results/phase3/heatmaps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def overlay_heatmap_on_image(self, img_rgb: np.ndarray, heatmap: np.ndarray,
                                  alpha: float = 0.5, colormap: str = "jet") -> np.ndarray:
        """Overlays normalized 2D heatmap on RGB image."""
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # Strip alpha
        overlay = (1.0 - alpha) * img_rgb + alpha * colored_heatmap
        return np.clip(overlay, 0.0, 1.0)

    def generate_exhibit(self, raw_img: np.ndarray, laplacian_residual: np.ndarray,
                          spatial_heatmap: np.ndarray, freq_cam: np.ndarray,
                          scores: dict, save_name: str, true_label: str = "Unknown"):
        """
        Creates and saves a comprehensive 4-panel forensic exhibit figure.
        
        Args:
            raw_img: Normalized RGB image in [0, 1] of shape (H, W, 3).
            laplacian_residual: 2D numpy array of high-pass Laplacian energy.
            spatial_heatmap: 2D numpy array of DINOv2 attention rollout.
            freq_cam: 2D numpy array of Frequency Stream Grad-CAM.
            scores: Dictionary with keys 'spatial_margin', 'freq_margin', 'calibrated_score'.
            save_name: Output filename (e.g. 'morph_stylegan_clean_001.png').
            true_label: Ground truth label ('Bona Fide' or 'Morph Attack').
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f172a")

        # Color palette and text styling
        title_color = "#f8fafc"
        accent_color = "#38bdf8"

        # Panel 1: Original Face Image
        axes[0].imshow(raw_img)
        axes[0].set_title(f"1. Input Face ({true_label})", color=title_color, fontsize=13, fontweight="bold", pad=10)
        axes[0].axis("off")

        # Panel 2: Laplacian High-Pass Residual
        axes[1].imshow(laplacian_residual, cmap="magma")
        axes[1].set_title("2. Laplacian High-Pass Residual", color=title_color, fontsize=13, fontweight="bold", pad=10)
        axes[1].axis("off")

        # Panel 3: Spatial Seam Heatmap (ViT Attention Rollout)
        spatial_overlay = self.overlay_heatmap_on_image(raw_img, spatial_heatmap, alpha=0.55, colormap="jet")
        axes[2].imshow(spatial_overlay)
        axes[2].set_title("3. Spatial ViT Attention (Seams)", color=title_color, fontsize=13, fontweight="bold", pad=10)
        axes[2].axis("off")

        # Panel 4: Frequency Spectral Heatmap (Grad-CAM)
        freq_overlay = self.overlay_heatmap_on_image(raw_img, freq_cam, alpha=0.55, colormap="plasma")
        axes[3].imshow(freq_overlay)
        axes[3].set_title("4. Frequency Grad-CAM (GAN Noise)", color=title_color, fontsize=13, fontweight="bold", pad=10)
        axes[3].axis("off")

        # Decision Summary Banner
        cal_score = scores.get("calibrated_score", 0.0)
        verdict = "MORPH ATTACK DETECTED" if cal_score > 0.0 else "VERIFIED BONA FIDE"
        verdict_color = "#ef4444" if cal_score > 0.0 else "#22c55e"

        plt.suptitle(
            f"UID-MAD Forensic Exhibit: {verdict} | Calibrated Anomaly Score: {cal_score:+.3f} "
            f"(Spatial Margin: {scores.get('spatial_margin', 0.0):+.3f}, Freq Margin: {scores.get('freq_margin', 0.0):+.3f})",
            color=verdict_color,
            fontsize=15,
            fontweight="heavy",
            y=1.04
        )

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return save_path
