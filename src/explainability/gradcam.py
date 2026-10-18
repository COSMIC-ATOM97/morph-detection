import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyGradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM) on the final
    convolutional feature map of Stream B (Frequency Residual Stream).
    Highlights spatial regions producing periodic spectral synthesis noise (StyleGAN2 checkerboard artifacts).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_target_layer()

    def _register_target_layer(self):
        # Locate the final conv layer in frequency stream
        if hasattr(self.model, "freq_stream"):
            target_layer = self.model.freq_stream.encoder[-2]  # Final Conv2d (index 9 in sequential)
        else:
            target_layer = self.model.encoder[-2]

        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h_f = target_layer.register_forward_hook(forward_hook)
        h_b = target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h_f, h_b])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def compute_cam(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for target_class (1 = Morph Attack).
        Returns normalized 2D numpy array of shape (H, W).
        """
        self.model.zero_grad()
        outputs = self.model(x)

        # Select target logit from frequency head or fused head
        if "logits_freq" in outputs:
            logit = outputs["logits_freq"][0, target_class]
        else:
            logit = outputs["logits_fused"][0, target_class]

        logit.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured during backward pass.")

        # Global average pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam_up = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        cam_min, cam_max = cam_up.min(), cam_up.max()
        cam_norm = (cam_up - cam_min) / (cam_max - cam_min + 1e-7)

        return cam_norm.squeeze().cpu().numpy()
