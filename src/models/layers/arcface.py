import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) classification head.
    Formulated via trigonometric angle addition identities:
      cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    with a linear Taylor fallback for theta + m > pi, preventing NaN overflows in FP16 AMP.
    """
    def __init__(self, in_features: int, num_classes: int = 2, s: float = 32.0, m: float = 0.25):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # L2-normalize embeddings and weights onto unit hypersphere
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(norm_emb, norm_w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cos_theta * self.s

        # Stable trigonometric identity
        sin_theta = torch.sqrt((1.0 - torch.square(cos_theta)).clamp(min=1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Safe monotonic fallback for theta + m > pi
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        return output * self.s
