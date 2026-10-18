import torch
import torch.nn as nn
from src.models.spatial_vit import SpatialDINOv2
from src.models.frequency_stream import FrequencyResidualStream
from src.models.layers.arcface import ArcFaceHead

class DualStreamUIDMAD(nn.Module):
    """
    Unified Interpretable Face Morph Attack Detection (UID-MAD)
    Dual-stream architecture fusing Spatial Vision Transformer (Stream A)
    and High-Pass Frequency Residual ConvNet (Stream B) with decoupled multi-task ArcFace heads.
    """
    def __init__(self, backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
                 spatial_dim: int = 256, freq_dim: int = 128, fused_dim: int = 256,
                 lora_r: int = 16, lora_alpha: float = 32.0, num_lora_blocks: int = 6,
                 arcface_s: float = 32.0, arcface_m: float = 0.25):
        super().__init__()
        self.spatial_stream = SpatialDINOv2(
            backbone_name=backbone_name,
            out_features=spatial_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_lora_blocks=num_lora_blocks
        )
        self.freq_stream = FrequencyResidualStream(out_features=freq_dim)

        # Gated reinforcement fusion
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, fused_dim),
            nn.BatchNorm1d(fused_dim)
        )

        # Independent ArcFace classification heads
        self.spatial_head = ArcFaceHead(spatial_dim, num_classes=2, s=arcface_s, m=arcface_m)
        self.freq_head = ArcFaceHead(freq_dim, num_classes=2, s=arcface_s, m=arcface_m)
        self.fused_head = ArcFaceHead(fused_dim, num_classes=2, s=arcface_s, m=arcface_m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        e_spatial = self.spatial_stream(x)
        e_freq = self.freq_stream(x)

        e_concat = torch.cat([e_spatial, e_freq], dim=1)
        e_fused = self.fusion(e_concat)

        logits_spatial = self.spatial_head(e_spatial, labels)
        logits_freq = self.freq_head(e_freq, labels)
        logits_fused = self.fused_head(e_fused, labels)

        return {
            "logits_fused": logits_fused,
            "logits_spatial": logits_spatial,
            "logits_freq": logits_freq,
            "emb_spatial": e_spatial,
            "emb_freq": e_freq,
            "emb_fused": e_fused
        }
