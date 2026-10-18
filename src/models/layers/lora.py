import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping a frozen Linear layer.
    Uses factorized low-rank forward projection: (dropout(x) @ A.t()) @ B.t() * scaling.
    Avoids materializing large d_out x d_in dense weight matrices.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Factorized low-rank forward pass
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out
