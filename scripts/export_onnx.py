#!/usr/bin/env python3
"""
Phase 3: Model Graph Serialization (PyTorch -> ONNX)
Exports the Dual-Stream UID-MAD model to an optimized ONNX computational graph
with dynamic batch dimension support for Automated Border Control (ABC) deployment.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.dual_stream import DualStreamUIDMAD

class DualStreamInferenceWrapper(nn.Module):
    """
    Wraps DualStreamUIDMAD for streamlined inference and ONNX serialization,
    outputting linear cosine margins for spatial, frequency, and fused heads.
    """
    def __init__(self, model: DualStreamUIDMAD):
        super().__init__()
        self.spatial_stream = model.spatial_stream
        self.freq_stream = model.freq_stream
        self.fusion = model.fusion
        self.spatial_head = model.spatial_head
        self.freq_head = model.freq_head
        self.fused_head = model.fused_head

    def forward(self, x: torch.Tensor):
        e_spatial = self.spatial_stream(x)
        e_freq = self.freq_stream(x)
        e_concat = torch.cat([e_spatial, e_freq], dim=1)
        e_fused = self.fusion(e_concat)

        # Compute cosine similarities against ArcFace weight vectors
        norm_w_s = F.normalize(self.spatial_head.weight, p=2, dim=1)
        norm_e_s = F.normalize(e_spatial, p=2, dim=1)
        cos_spatial = F.linear(norm_e_s, norm_w_s)
        margin_spatial = cos_spatial[:, 1] - cos_spatial[:, 0]

        norm_w_f = F.normalize(self.freq_head.weight, p=2, dim=1)
        norm_e_f = F.normalize(e_freq, p=2, dim=1)
        cos_freq = F.linear(norm_e_f, norm_w_f)
        margin_freq = cos_freq[:, 1] - cos_freq[:, 0]

        norm_w_fuse = F.normalize(self.fused_head.weight, p=2, dim=1)
        norm_e_fuse = F.normalize(e_fused, p=2, dim=1)
        cos_fused = F.linear(norm_e_fuse, norm_w_fuse)
        margin_fused = cos_fused[:, 1] - cos_fused[:, 0]

        # Calibrated decision score
        calibrated_score = 0.5 * margin_spatial + 0.5 * margin_freq

        return {
            "calibrated_score": calibrated_score,
            "margin_spatial": margin_spatial,
            "margin_freq": margin_freq,
            "margin_fused": margin_fused
        }

def export_to_onnx(checkpoint_path: str, output_onnx_path: str, opset_version: int = 17):
    print(f"[*] Initializing model for ONNX export...")
    model = DualStreamUIDMAD()
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading checkpoint weights: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    else:
        print(f"[!] Checkpoint not found at {checkpoint_path}. Exporting base model architecture.")

    model.eval()
    wrapper = DualStreamInferenceWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)
    print(f"[*] Exporting ONNX graph to: {output_onnx_path} (Opset {opset_version})...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_face"],
        output_names=["calibrated_score", "margin_spatial", "margin_freq", "margin_fused"],
        dynamic_axes={
            "input_face": {0: "batch_size"},
            "calibrated_score": {0: "batch_size"},
            "margin_spatial": {0: "batch_size"},
            "margin_freq": {0: "batch_size"},
            "margin_fused": {0: "batch_size"}
        }
    )

    print(f"[+] Successfully exported ONNX graph to: {output_onnx_path}")
    print(f"    Graph file size: {os.path.getsize(output_onnx_path) / (1024*1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="UID-MAD ONNX Serialization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2_dual_stream_best.pt",
                        help="Path to trained weights checkpoint.")
    parser.add_argument("--output", type=str, default="checkpoints/uid_mad_dual_stream.onnx",
                        help="Target output ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, opset_version=args.opset)

if __name__ == "__main__":
    main()
