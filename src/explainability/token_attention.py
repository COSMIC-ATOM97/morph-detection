import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOv2AttentionRollout:
    """
    Computes Attention Rollout (Abnar & Zuidema, 2020) across DINOv2 Vision Transformer blocks.
    Traces information flow from the CLS token to spatial patch tokens (16x16 grid),
    isolating localized blending seams, ghosting edges, and warping discontinuities.
    """
    def __init__(self, model: nn.Module, start_block: int = 6):
        self.model = model
        self.start_block = start_block
        self.attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Access the ViT backbone from DualStreamUIDMAD or SpatialDINOv2
        if hasattr(self.model, "spatial_stream"):
            backbone = self.model.spatial_stream.backbone
        elif hasattr(self.model, "backbone"):
            backbone = self.model.backbone
        else:
            backbone = self.model

        for idx, block in enumerate(backbone.blocks):
            if idx >= self.start_block:
                hook = block.attn.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, inputs, output):
        # Extract attention weights from Multi-Head Self-Attention
        # In timm ViT, attn forward computes Q, K, V
        x = inputs[0]
        B, N, C = x.shape
        # Reconstruct attention weights from Q and K
        # If timm exposes attention_weights, use it; otherwise compute from module
        if hasattr(module, "get_attention_map"):
            attn = module.get_attention_map(x)
        else:
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
        self.attentions.append(attn.detach())

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_rollout(self, x: torch.Tensor) -> np.ndarray:
        """
        Executes forward pass and returns spatial attention heatmap of shape (H, W).
        """
        self.attentions = []
        with torch.no_grad():
            _ = self.model(x)

        if not self.attentions:
            raise RuntimeError("No attention maps captured. Verify backbone hooks.")

        # Compute recursive attention rollout
        result = torch.eye(self.attentions[0].size(-1), device=x.device).unsqueeze(0)
        for attn in self.attentions:
            # Average attention across heads
            attn_heads_fused = attn.mean(dim=1)
            # Add identity matrix to account for residual connections
            I = torch.eye(attn_heads_fused.size(-1), device=x.device).unsqueeze(0)
            a = 0.5 * attn_heads_fused + 0.5 * I
            # Renormalize rows
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)

        # Extract CLS token attention to all spatial patches
        # Token 0 is CLS; tokens 1..256 are spatial patches (16x16)
        cls_attention = result[0, 0, 1:]
        grid_size = int(math.sqrt(cls_attention.size(0))) if "math" in globals() else 16
        heatmap_2d = cls_attention.reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)

        # Upsample to input resolution (e.g. 224x224)
        heatmap_up = F.interpolate(heatmap_2d, size=(x.shape[-2], x.shape[-1]), mode="bicubic", align_corners=False)
        heatmap_norm = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min() + 1e-7)

        return heatmap_norm.squeeze().cpu().numpy()
