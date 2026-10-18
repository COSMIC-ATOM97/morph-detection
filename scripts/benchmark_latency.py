#!/usr/bin/env python3
"""
Phase 3: Operational Profiling & e-Gate Compliance Latency Benchmark
Profiles forward pass inference latency across batch sizes {1, 4, 8} on available hardware
and verifies compliance with the ICAO Doc 9303 / Frontex <50ms e-Gate threshold.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.dual_stream import DualStreamUIDMAD

def benchmark_inference(model: torch.nn.Module, device: torch.device,
                        batch_sizes=(1, 4, 8), num_warmup: int = 15,
                        num_trials: int = 100) -> pd.DataFrame:
    records = []
    print(f"[*] Running latency benchmark on {device} ({num_trials} trials per batch size)...")

    for b in batch_sizes:
        dummy = torch.randn(b, 3, 224, 224, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed trials
        latencies = []
        with torch.no_grad():
            for _ in range(num_trials):
                t0 = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms per batch

        latencies = np.array(latencies)
        per_image_latencies = latencies / b

        mean_lat = float(np.mean(per_image_latencies))
        p50_lat = float(np.percentile(per_image_latencies, 50))
        p95_lat = float(np.percentile(per_image_latencies, 95))
        p99_lat = float(np.percentile(per_image_latencies, 99))
        throughput_fps = float(1000.0 / mean_lat)
        compliant = "PASS (<50ms)" if p95_lat < 50.0 else "EXCEEDS (>50ms)"

        records.append({
            "batch_size": b,
            "mean_latency_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "p99_ms": round(p99_lat, 2),
            "throughput_fps": round(throughput_fps, 1),
            "icao_compliance": compliant
        })

        print(f"  -> Batch {b:2d}: Mean={mean_lat:6.2f} ms | p95={p95_lat:6.2f} ms | FPS={throughput_fps:6.1f} | ICAO Status: {compliant}")

    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="UID-MAD Operational Latency Profiler")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2_dual_stream_best.pt",
                        help="Path to trained model weights.")
    parser.add_argument("--output-csv", type=str, default="results/phase3/latency_benchmark.csv",
                        help="Path to save latency benchmarks.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device for profiling.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = DualStreamUIDMAD()
    if os.path.exists(args.checkpoint):
        print(f"[*] Loading model checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
    else:
        print(f"[!] Warning: Checkpoint not found at {args.checkpoint}. Profiling uninitialized architecture.")

    model.to(device)
    model.eval()

    df = benchmark_inference(model, device)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[+] Benchmark complete. Saved results to: {args.output_csv}")

if __name__ == "__main__":
    main()
