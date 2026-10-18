import timm
import torch
import torch.nn as nn
from src.models.layers.lora import LoRALinear

class SpatialDINOv2(nn.Module):
    """
    Stream A: DINOv2 ViT-S/14 spatial stream with multi-scale patch token pooling
    (CLS + Mean + Std + Max) and LoRA adapters on the last N transformer blocks.
    """
    def __init__(self, backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
                 out_features: int = 256, lora_r: int = 16, lora_alpha: float = 32.0,
                 num_lora_blocks: int = 6):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, dynamic_img_size=True)

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA to last N blocks
        if lora_r > 0 and num_lora_blocks > 0:
            total_blocks = len(self.backbone.blocks)
            start_block = max(0, total_blocks - num_lora_blocks)
            for b_idx in range(start_block, total_blocks):
                block = self.backbone.blocks[b_idx]
                block.attn.qkv = LoRALinear(block.attn.qkv, r=lora_r, lora_alpha=lora_alpha)
                block.attn.proj = LoRALinear(block.attn.proj, r=lora_r, lora_alpha=lora_alpha)
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, r=lora_r, lora_alpha=lora_alpha)
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, r=lora_r, lora_alpha=lora_alpha)

        d_model = self.backbone.embed_dim  # 384 for vit_small
        # Multi-scale pooling: CLS (384) + Mean (384) + Std (384) + Max (384) = 1536-d
        self.projector = nn.Sequential(
            nn.Linear(d_model * 4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        mean_patch = patch_tokens.mean(dim=1)
        std_patch = torch.sqrt(patch_tokens.var(dim=1, unbiased=False) + 1e-7)
        max_patch, _ = patch_tokens.max(dim=1)

        aggregated = torch.cat([cls_token, mean_patch, std_patch, max_patch], dim=1)
        return self.projector(aggregated)
