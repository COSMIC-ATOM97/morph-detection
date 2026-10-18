import torch
import torch.nn as nn
import torch.nn.functional as F

class HighPassLaplacianResidual(nn.Module):
    """
    Fixed 3x3 Laplacian high-pass filter that strips low-frequency identity/illumination
    and isolates high-frequency generator artifacts and blending edge gradients.
    """
    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        kernel = laplacian_kernel.repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Grouped conv2d applying channel-wise Laplacian filter
        residual = F.conv2d(x, self.kernel, padding=1, groups=3)
        return torch.abs(residual)

class FrequencyResidualStream(nn.Module):
    """
    Stream B: Lightweight 4-stage convolutional residual encoder extracting
    spectral generator noise (StyleGAN2 transposed convolution artifacts).
    """
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.high_pass = HighPassLaplacianResidual()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.projector = nn.Sequential(
            nn.Linear(256, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.high_pass(x)
        feat = self.encoder(res)
        avg_pool = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        max_pool = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.projector(pooled)
