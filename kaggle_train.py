"""
Dual-Stream UID-MAD (Phase 2): Spatial DINOv2 + Frequency Residual
===================================================================
Self-contained Kaggle Training & Cross-Dataset Evaluation Script.

Architecture:
  - Stream A (Spatial): DINOv2 ViT-S/14 Dense Patch Statistics + Rank-16 LoRA (1536 -> 256-d)
  - Stream B (Frequency): Laplacian High-Pass + Rectified Haar Wavelet ConvNet (128-d)
  - Fusion: Dual-Stream Adaptive Gated Fusion Head (256-d)
  - Classification: Additive Angular Margin Head (ArcFace: s=32.0, m=0.25)

Key Engineering Optimizations for Kaggle:
  - Automatic scratch extraction: unpacks 4.5GB SMDD zip once to local NVMe/RAM disk (/tmp)
    eliminating FUSE random-seek latency that previously caused 10+ hour runtimes.
  - Factorized LoRA computation: O(r) low-rank forward avoiding dense matrix materialization.
  - Numerically stable ArcFace: trigonometric angle addition avoiding acos NaN spikes in FP16.
  - Multi-process DataLoader: 2 workers with pinned memory on extracted local storage.
  - Real-time logging & heartbeat: batch progress every 20 steps + 25s heartbeat preventing silent gaps.
  - Incremental CSV checkpointing: writes frll_results.csv after every generator/condition.
  - Safe runtime ceiling: cleanly completes evaluation and saves all outputs before Kaggle session cutoff.

Usage on Kaggle:
    1. Select Accelerator: GPU T4 x2 (or GPU P100 / T4 x1).
    2. Paste this script into a notebook cell.
    3. Run interactively or click 'Save Version' -> 'Save & Run All (Commit)'.
    4. Outputs are saved to /kaggle/working/ (frll_results.csv, best_model.pt, roc_*.png).
"""

import os
import sys
import time
import socket
import io
import csv
import math
import json
import random
import zipfile
import shutil
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict

# Network timeout so downloads/clones never hang indefinitely
socket.setdefaulttimeout(30)

# Force line buffering on standard outputs so logs appear immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Prevent OpenCV and OpenMP worker deadlocks in PyTorch DataLoader
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.amp import autocast as _torch_autocast, GradScaler as _torch_scaler
    def autocast_ctx():
        return _torch_autocast("cuda")
    def make_scaler():
        try:
            return _torch_scaler("cuda")
        except Exception:
            return _torch_scaler()
except Exception:
    from torch.cuda.amp import autocast as _legacy_autocast, GradScaler as _legacy_scaler
    def autocast_ctx():
        return _legacy_autocast()
    def make_scaler():
        return _legacy_scaler()

import timm
from sklearn.metrics import roc_auc_score, roc_curve

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """All hyperparameters in one place."""

    # Paths -- adjust to your Kaggle dataset mount points
    SMDD_ZIP = "/kaggle/input/smdd-morph/SMDD_release_train.zip"
    FRLL_ZIP = "/kaggle/input/frll-morphs/drive-download-20261010T174433Z-1-001.zip"
    OUTPUT_DIR = "/kaggle/working"
    SCRATCH_DIR = "/tmp"  # Fast local RAM/NVMe scratch space on Kaggle

    # Fast dataset extraction to eliminate Kaggle FUSE seek latency
    EXTRACT_DATASETS = True

    # Backbone & Dual-Stream Dimensions
    BACKBONE = "vit_small_patch14_dinov2.lvd142m"  # timm model name
    EMBED_DIM = 384  # 384 for ViT-S
    SPATIAL_DIM = 256  # Stream A spatial embedding dimension
    FREQ_DIM = 128     # Stream B frequency residual embedding dimension
    PROJ_DIM = 256     # Fused projection dimension

    # LoRA Configuration for DINOv2 (Stream A)
    USE_LORA = True
    LORA_R = 16        # Higher rank for dense forensic texture adaptation
    LORA_ALPHA = 32.0  # Scaling factor (alpha / r = 2.0)
    LORA_DROPOUT = 0.05
    LORA_BLOCKS = 6    # Adapt the last 6 transformer blocks (blocks 6 to 11)
    LORA_TARGET_MLP = True  # Adapt both self-attention and feed-forward MLP layers

    # ArcFace
    ARCFACE_S = 32.0
    ARCFACE_M = 0.25  # Softer margin to prevent overconfidence on synthetic data
    NUM_CLASSES = 2   # bonafide vs morph

    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 2  # 2 workers for extracted dataset (optimal for 2-vCPU Kaggle instance)
    PIN_MEMORY = True  # Faster CUDA page-locked memory transfer
    WARMUP_EPOCHS = 4  # 4 epochs head + frequency encoder training (~8-10 min)
    TOTAL_EPOCHS = 8   # 8 epochs total (4 warmup + 4 fine-tuning epochs, ~20-25 min)
    HEAD_LR = 1e-3
    FINETUNE_HEAD_LR = 1e-4
    FINETUNE_BACKBONE_LR = 1e-4  # Optimal learning rate for LoRA adapter matrices
    WEIGHT_DECAY = 1e-3  # Stronger L2 regularization
    UNFREEZE_BLOCKS = 1  # Fallback if USE_LORA is False
    MAX_RUNTIME_MINUTES = 90  # Hard safety ceiling: exits cleanly before Kaggle cutoff

    # Logging & Monitoring responsiveness
    LOG_INTERVAL_BATCHES = 20  # Print progress every 20 batches
    LOG_INTERVAL_SEC = 25      # Heartbeat print if 25s pass without a step log

    # Validation split
    VAL_FRACTION = 0.2

    # Reproducibility
    SEED = 42

    # Execution mode
    EVAL_ONLY = False  # Set to True to skip training and evaluate best_model.pt directly
    CKPT_TO_EVAL = "best_model.pt"  # Checkpoint file to evaluate

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset Extraction Helper (Eliminates Kaggle FUSE Zip Bottleneck)
# ---------------------------------------------------------------------------

def extract_archive_if_needed(archive_path: str, target_dir: str, check_indicator: str) -> str:
    """
    Extracts a zip archive to a fast local scratch directory if not already extracted.
    Eliminates 90%+ of training time caused by random seeks in 4.5GB archives over Kaggle FUSE mounts.
    """
    if not os.path.exists(archive_path):
        return archive_path

    if os.path.isdir(archive_path):
        return archive_path

    if not (os.path.isfile(archive_path) and zipfile.is_zipfile(archive_path)):
        return archive_path

    os.makedirs(target_dir, exist_ok=True)
    check_path = os.path.join(target_dir, check_indicator)
    if os.path.exists(check_path):
        print(f"  [Cache hit] Using existing extracted dataset at: {target_dir}", flush=True)
        return target_dir

    print(f"  [Extraction] Unpacking {archive_path} -> {target_dir} for fast local I/O...", flush=True)
    t0 = time.time()

    # Try system unzip first (fastest C implementation on Linux)
    extracted_ok = False
    try:
        ret = subprocess.run(["unzip", "-q", "-n", archive_path, "-d", target_dir], capture_output=True, timeout=300)
        if ret.returncode == 0 and os.path.exists(check_path):
            extracted_ok = True
    except Exception:
        pass

    if not extracted_ok:
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
            extracted_ok = True
        except Exception as e:
            print(f"  [Warning] Extraction failed ({e}); falling back to direct zip reading.", flush=True)
            return archive_path

    elapsed = time.time() - t0
    print(f"  [Extraction done] Finished in {elapsed:.1f}s ({target_dir})", flush=True)
    return target_dir


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def make_coarse_dropout(p=0.3):
    try:
        return A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 48), hole_width_range=(16, 48), p=p)
    except Exception:
        try:
            return A.CoarseDropout(max_holes=4, max_height=48, max_width=48, p=p)
        except Exception:
            return None


def get_train_transforms(img_size: int) -> A.Compose:
    """Fast, anti-overfitting augmentation pipeline destroying sensor/generator shortcuts."""
    cd_drop = make_coarse_dropout(p=0.2)
    aug_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # Mutually exclusive heavy degradation to keep CPU decode time fast
        A.OneOf([
            A.ImageCompression(quality_lower=20, quality_upper=80, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.Compose([
                A.Downscale(scale_min=0.4, scale_max=0.7, p=1.0),
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            ], p=1.0),
        ], p=0.7),
        # Photometric variation
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # Camera sensor & PRNU noise simulation (neutralizes cross-dataset sensor gap)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=1.0),
        ], p=0.4),
        # Grayscale (destroys color hue shortcuts, forces structural & frequency analysis)
        A.ToGray(p=0.15),
    ]
    if cd_drop is not None:
        aug_list.append(cd_drop)
    aug_list.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A.Compose(aug_list)


def get_val_transforms(img_size: int) -> A.Compose:
    """Deterministic eval transform."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_degraded_transforms(img_size: int, mode: str = "jpeg") -> A.Compose:
    """Apply degradation before standard eval transform for robustness slices."""
    degradation = []
    if mode == "jpeg":
        degradation.append(A.ImageCompression(quality_lower=25, quality_upper=35, p=1.0))
    elif mode == "printscan":
        degradation.extend([
            A.Downscale(scale_min=0.4, scale_max=0.6, p=1.0),
            A.ImageCompression(quality_lower=30, quality_upper=50, p=1.0),
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
        ])
    return A.Compose([
        *degradation,
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Datasets (Supports both fast local directories and .zip fallback)
# ---------------------------------------------------------------------------

class SMDDDataset(Dataset):
    """Reads SMDD images from an extracted directory (fast) or zip archive (fallback)."""

    def __init__(self, data_source: str, entries: list, transform=None):
        self.data_source = data_source
        self.is_zip = os.path.isfile(data_source) and zipfile.is_zipfile(data_source)
        self.entries = entries
        self.transform = transform
        self._zf = None

    def _open_zip(self):
        if self.is_zip and self._zf is None:
            self._zf = zipfile.ZipFile(self.data_source, "r")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_rel_path, label = self.entries[idx]
        if self.is_zip:
            self._open_zip()
            img_bytes = self._zf.read(img_rel_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            full_path = os.path.join(self.data_source, img_rel_path)
            img = Image.open(full_path).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def __del__(self):
        if self._zf is not None:
            try:
                self._zf.close()
            except Exception:
                pass


class FRLLDataset(Dataset):
    """Reads FRLL-Morphs images from an extracted directory (fast) or zip archive (fallback)."""

    def __init__(self, data_source: str, bonafide_paths: list, morph_paths: list, transform=None):
        self.data_source = data_source
        self.is_zip = os.path.isfile(data_source) and zipfile.is_zipfile(data_source)
        self.entries = [(p, 0) for p in bonafide_paths] + [(p, 1) for p in morph_paths]
        self.transform = transform
        self._zf = None

    def _open_zip(self):
        if self.is_zip and self._zf is None:
            self._zf = zipfile.ZipFile(self.data_source, "r")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_rel_path, label = self.entries[idx]
        if self.is_zip:
            self._open_zip()
            img_bytes = self._zf.read(img_rel_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            full_path = os.path.join(self.data_source, img_rel_path)
            img = Image.open(full_path).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def __del__(self):
        if self._zf is not None:
            try:
                self._zf.close()
            except Exception:
                pass


def load_smdd_entries(data_source: str):
    """Parse train.csv inside SMDD directory or zip archive."""
    if os.path.isfile(data_source) and zipfile.is_zipfile(data_source):
        with zipfile.ZipFile(data_source, "r") as zf:
            csv_bytes = zf.read("train.csv").decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_bytes))
    else:
        csv_file = os.path.join(data_source, "train.csv")
        if not os.path.exists(csv_file):
            for root, _, files in os.walk(data_source):
                if "train.csv" in files:
                    csv_file = os.path.join(root, "train.csv")
                    break
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

    entries = []
    for row in reader:
        label = 0 if row["label"] == "bonafide" else 1
        entries.append((row["image_path"], label))
    return entries


def load_frll_paths(data_source: str):
    """Enumerate FRLL-Morphs relative paths grouped by generator."""
    if os.path.isfile(data_source) and zipfile.is_zipfile(data_source):
        with zipfile.ZipFile(data_source, "r") as zf:
            all_names = zf.namelist()
    else:
        all_names = []
        for root, _, files in os.walk(data_source):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), data_source)
                all_names.append(rel.replace("\\", "/"))

    bonafide = sorted([
        n for n in all_names
        if (n.startswith("raw/") or "/raw/" in n) and n.endswith(".jpg")
    ])
    generators = {}
    for prefix in ["morph_opencv", "morph_facemorpher", "morph_webmorph", "morph_stylegan", "morph_amsl"]:
        generators[prefix] = sorted([
            n for n in all_names
            if (n.startswith(prefix + "/") or f"/{prefix}/" in n)
            and (n.endswith(".jpg") or n.endswith(".png"))
        ])
    return bonafide, generators


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer wrapping a standard nn.Linear.
    Keeps pretrained weights frozen and trains rank decomposition matrices A and B.
    Factorized forward: (dropout(x) @ A^T) @ B^T * scaling
    Avoids materializing large (out_features, in_features) matrices and cuts memory/FLOPs.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # Factorized low-rank projection
        # x shape: (..., in_features)
        # (dropout(x) @ lora_A.t()) -> (..., r)
        # (... @ lora_B.t()) -> (..., out_features)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base + lora_out


class ArcFaceHead(nn.Module):
    """Additive angular margin classification head with numerical stability threshold."""

    def __init__(self, in_features: int, num_classes: int, s: float = 32.0, m: float = 0.25):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def get_margin(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns raw cosine difference: cos(morph) - cos(bonafide) in [-2, 2]."""
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)  # (B, num_classes)
        return cosine[:, 1] - cosine[:, 0]

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)  # (B, num_classes)

        if labels is not None:
            # Add angular margin to target class with boundary stabilization
            sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, 1e-7, 1.0))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
            # When theta + m > pi, use smooth Taylor fallback to preserve monotonicity
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
            logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            logits = logits * self.s
        else:
            logits = cosine * self.s

        return logits


class FrequencyResidualStream(nn.Module):
    """
    Stream B: Multi-directional high-pass residual filter + 2D Haar Wavelet + Multi-scale Conv encoder.
    Isolates generator noise, checkerboard upsampling artifacts, and frequency anomalies.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        # Depthwise high-pass filter bank:
        # Combines isotropic 8-neighbor Laplacian with horizontal and vertical gradients
        k_lap = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ], dtype=torch.float32) / 8.0
        k_h = torch.tensor([
            [ 0.,  0.,  0.],
            [-1.,  2., -1.],
            [ 0.,  0.,  0.]
        ], dtype=torch.float32) / 2.0
        k_v = torch.tensor([
            [ 0., -1.,  0.],
            [ 0.,  2.,  0.],
            [ 0., -1.,  0.]
        ], dtype=torch.float32) / 2.0

        k_combined = (k_lap + k_h + k_v) / 3.0
        weight = k_combined.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.hp_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.hp_filter.weight = nn.Parameter(weight, requires_grad=False)

        # Lightweight 4-stage convolutional encoder: 224 -> 112 -> 56 -> 28 -> 14
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # Extract rectified multi-directional high-pass residual on GPU
        res = torch.abs(self.hp_filter(x))
        # Extract 2D Haar Wavelet high-high diagonal sub-band (StyleGAN Nyquist checkerboards)
        x_hh = torch.abs((x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0)
        x_hh_up = F.interpolate(x_hh, size=res.shape[-2:], mode="bilinear", align_corners=False)
        composite = 0.5 * res + 0.5 * x_hh_up

        c1 = self.conv1(composite)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        avg_p = self.pool_avg(c4).flatten(1)
        max_p = self.pool_max(c4).flatten(1)
        pooled = torch.cat([avg_p, max_p], dim=1)  # (B, 256)
        return self.proj(pooled)  # (B, out_dim)


class MorphDetector(nn.Module):
    """UID-MAD Dual-Stream Detector: Stream A (DINOv2 Spatial) + Stream B (Frequency Residual)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ---- Stream A: Spatial Backbone (DINOv2) ----
        backbone_loaded = False
        candidates = [
            cfg.BACKBONE,
            "vit_small_patch14_dinov2.lvd142m",
            "vit_small_patch14_dinov2",
        ]
        for name in candidates:
            try:
                self.backbone = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
                print(f"  Successfully loaded backbone via timm: {name} (dynamic_img_size=True)", flush=True)
                backbone_loaded = True
                break
            except Exception:
                continue

        if not backbone_loaded:
            print("  Falling back to torch.hub for DINOv2 (dinov2_vits14)...", flush=True)
            try:
                self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                backbone_loaded = True
            except Exception as e:
                print(f"  [Warning] torch.hub load failed ({e}). Initializing backbone architecture offline.", flush=True)
                self.backbone = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, dynamic_img_size=True)

        # Multi-scale patch feature dimension: CLS (384) + Mean (384) + Std (384) + Max (384) = 1536
        in_dim = cfg.EMBED_DIM * 4
        self.spatial_projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.SPATIAL_DIM),
            nn.BatchNorm1d(cfg.SPATIAL_DIM),
        )

        # ---- Stream B: Frequency Residual Stream ----
        self.freq_stream = FrequencyResidualStream(out_dim=cfg.FREQ_DIM)

        # ---- Dual-Stream Adaptive Gated Fusion Head ----
        self.freq_align = nn.Sequential(
            nn.Linear(cfg.FREQ_DIM, cfg.SPATIAL_DIM),
            nn.BatchNorm1d(cfg.SPATIAL_DIM),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(cfg.SPATIAL_DIM * 2, cfg.SPATIAL_DIM),
            nn.BatchNorm1d(cfg.SPATIAL_DIM),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.SPATIAL_DIM, cfg.PROJ_DIM),
            nn.BatchNorm1d(cfg.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ---- Multi-Branch ArcFace Classification Heads ----
        self.spatial_head = ArcFaceHead(
            in_features=cfg.SPATIAL_DIM,
            num_classes=cfg.NUM_CLASSES,
            s=cfg.ARCFACE_S,
            m=cfg.ARCFACE_M,
        )
        self.freq_head = ArcFaceHead(
            in_features=cfg.FREQ_DIM,
            num_classes=cfg.NUM_CLASSES,
            s=cfg.ARCFACE_S,
            m=cfg.ARCFACE_M,
        )
        self.fused_head = ArcFaceHead(
            in_features=cfg.PROJ_DIM,
            num_classes=cfg.NUM_CLASSES,
            s=cfg.ARCFACE_S,
            m=cfg.ARCFACE_M,
        )

        # ---- Stream A: LoRA Adaptation ----
        self.lora_params = []
        if getattr(cfg, "USE_LORA", False):
            target_mlp = getattr(cfg, "LORA_TARGET_MLP", True)
            self.apply_lora(cfg.LORA_BLOCKS, cfg.LORA_R, cfg.LORA_ALPHA, cfg.LORA_DROPOUT, target_mlp=target_mlp)

    def apply_lora(self, num_blocks: int, r: int, alpha: float, dropout: float = 0.05, target_mlp: bool = True):
        """Inject LoRA adapters into the last num_blocks of the ViT backbone (attention and MLP)."""
        self.lora_params = []
        if not hasattr(self.backbone, "blocks"):
            print("  [LoRA Warning] Backbone has no .blocks attribute. Skipping LoRA.", flush=True)
            return

        blocks = list(self.backbone.blocks)
        target_blocks = blocks[-num_blocks:]
        for block in target_blocks:
            if hasattr(block, "attn"):
                if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
                    block.attn.qkv = LoRALinear(block.attn.qkv, r=r, lora_alpha=alpha, lora_dropout=dropout)
                    self.lora_params.extend([block.attn.qkv.lora_A, block.attn.qkv.lora_B])
                if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
                    block.attn.proj = LoRALinear(block.attn.proj, r=r, lora_alpha=alpha, lora_dropout=dropout)
                    self.lora_params.extend([block.attn.proj.lora_A, block.attn.proj.lora_B])
            if target_mlp and hasattr(block, "mlp"):
                if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, nn.Linear):
                    block.mlp.fc1 = LoRALinear(block.mlp.fc1, r=r, lora_alpha=alpha, lora_dropout=dropout)
                    self.lora_params.extend([block.mlp.fc1.lora_A, block.mlp.fc1.lora_B])
                if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, nn.Linear):
                    block.mlp.fc2 = LoRALinear(block.mlp.fc2, r=r, lora_alpha=alpha, lora_dropout=dropout)
                    self.lora_params.extend([block.mlp.fc2.lora_A, block.mlp.fc2.lora_B])

        print(f"  Applied LoRA (r={r}, alpha={alpha}, target_mlp={target_mlp}) to last {num_blocks} ViT blocks ({len(self.lora_params)} adapter tensors).", flush=True)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self, "lora_params") and self.lora_params:
            for p in self.lora_params:
                p.requires_grad = False

    def unfreeze_lora(self):
        if hasattr(self, "lora_params") and self.lora_params:
            for p in self.lora_params:
                p.requires_grad = True
            print(f"  Unfroze {len(self.lora_params)} LoRA adapter tensors.", flush=True)
        else:
            self.unfreeze_last_n_blocks(self.cfg.UNFREEZE_BLOCKS)

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks of the ViT."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x, labels=None, return_scores=False):
        # Stream A (Spatial): Multi-scale patch statistics
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if isinstance(feats, dict):
                cls_token = feats["x_norm_clstoken"]
                patches = feats["x_norm_patchtokens"]
                patch_mean = patches.mean(dim=1)
                patch_std = patches.std(dim=1)
                patch_max = patches.max(dim=1).values
                spatial_feat = torch.cat([cls_token, patch_mean, patch_std, patch_max], dim=1)
            elif isinstance(feats, torch.Tensor):
                if feats.dim() == 3 and feats.size(1) > 1:
                    cls_token = feats[:, 0]
                    patches = feats[:, 1:]
                    patch_mean = patches.mean(dim=1)
                    patch_std = patches.std(dim=1)
                    patch_max = patches.max(dim=1).values
                    spatial_feat = torch.cat([cls_token, patch_mean, patch_std, patch_max], dim=1)
                elif feats.dim() == 2:
                    spatial_feat = feats.repeat(1, 4)
                else:
                    spatial_feat = feats[:, 0].repeat(1, 4)
            else:
                cls_token = self.backbone(x)
                spatial_feat = cls_token.repeat(1, 4)
        else:
            cls_token = self.backbone(x)
            spatial_feat = cls_token.repeat(1, 4)

        z_spatial = self.spatial_projector(spatial_feat)  # (B, SPATIAL_DIM)
        z_freq = self.freq_stream(x)                      # (B, FREQ_DIM)
        f_aligned = self.freq_align(z_freq)              # (B, SPATIAL_DIM)
        gate = self.gate(torch.cat([z_spatial, f_aligned], dim=1))  # (B, SPATIAL_DIM) in [0, 1]
        z_fused = z_spatial + gate * f_aligned            # Non-negative additive reinforcement
        proj = self.fusion(z_fused)                       # (B, PROJ_DIM)

        if labels is not None:
            # Training: Supervise spatial, frequency, and fusion streams independently
            logits_fused = self.fused_head(proj, labels)
            logits_spatial = self.spatial_head(z_spatial, labels)
            logits_freq = self.freq_head(z_freq, labels)
            return logits_fused, logits_spatial, logits_freq
        else:
            # Inference: Compute cosine margins and probabilities for each branch
            margin_s = self.spatial_head.get_margin(z_spatial)
            margin_f = self.freq_head.get_margin(z_freq)
            margin_fuse = self.fused_head.get_margin(proj)

            logits_s = self.spatial_head(z_spatial, labels=None)
            logits_f = self.freq_head(z_freq, labels=None)
            logits_fuse = self.fused_head(proj, labels=None)

            p_s = F.softmax(logits_s, dim=1)[:, 1]
            p_f = F.softmax(logits_f, dim=1)[:, 1]
            p_fuse = F.softmax(logits_fuse, dim=1)[:, 1]

            # Calibrated ensemble: maximum in margin space avoids exponential softmax saturation
            margin_ens = torch.maximum(margin_s, margin_f)
            prob_ens_avg = 0.5 * p_s + 0.5 * p_f
            prob_ens_max = torch.maximum(p_s, p_f)

            if return_scores:
                return margin_ens

            return {
                "margin_spatial": margin_s,
                "margin_freq": margin_f,
                "margin_fused": margin_fuse,
                "margin_ensemble": margin_ens,
                "prob_spatial": p_s,
                "prob_freq": p_f,
                "prob_fused": p_fuse,
                "prob_ensemble_avg": prob_ens_avg,
                "prob_ensemble_max": prob_ens_max,
                "logits_fused": logits_fuse,
            }

    def get_scores(self, x):
        """Inference: return morph probability scores."""
        return self.forward(x, return_scores=True)


# ---------------------------------------------------------------------------
# Metrics (ISO/IEC 30107-3)
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """Compute Equal Error Rate from labels and scores."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh


def compute_bpcer_at_apcer(labels: np.ndarray, scores: np.ndarray, apcer_target: float):
    """
    Compute BPCER at a fixed APCER target threshold (ISO/IEC 30107-3).
    APCER (Attack Presentation Classification Error Rate) = false accept rate on morphs (scores < threshold).
    BPCER (Bona Fide Presentation Classification Error Rate) = false reject rate on bonafide (scores >= threshold).
    Higher score = more likely morph.
    """
    attack_scores = scores[labels == 1]
    bonafide_scores = scores[labels == 0]

    if len(attack_scores) == 0 or len(bonafide_scores) == 0:
        return 1.0, 0.5

    # Threshold where at most apcer_target of attacks are misclassified as bonafide (score < threshold)
    threshold = float(np.quantile(attack_scores, apcer_target))

    # BPCER is the fraction of bonafide falsely flagged as morphs (score >= threshold)
    bpcer = float(np.mean(bonafide_scores >= threshold))
    return bpcer, threshold


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute EER, BPCER@APCER=1%, BPCER@APCER=5%, and ROC-AUC."""
    eer, eer_thresh = compute_eer(labels, scores)
    bpcer_1, thresh_1 = compute_bpcer_at_apcer(labels, scores, apcer_target=0.01)
    bpcer_5, thresh_5 = compute_bpcer_at_apcer(labels, scores, apcer_target=0.05)
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")
    return {
        "EER": eer,
        "EER_threshold": eer_thresh,
        "BPCER@APCER=1%": bpcer_1,
        "BPCER@APCER=5%": bpcer_5,
        "ROC-AUC": auc,
    }


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------

def build_sampler(entries: list) -> WeightedRandomSampler:
    """Build a weighted sampler to handle class imbalance."""
    labels = [e[1] for e in entries]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(1, class_counts)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(entries), replacement=True)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, start_time, max_runtime_sec,
                    log_interval_batches=20, log_interval_sec=25):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        curr_time = time.time()
        if curr_time - start_time > max_runtime_sec:
            print(f"  [SAFETY CEILING REACHED] Exiting epoch {epoch} early after {batch_idx} batches.", flush=True)
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits_fused, logits_s, logits_f = model(images, labels)
            loss_fuse = F.cross_entropy(logits_fused, labels)
            loss_s = F.cross_entropy(logits_s, labels)
            loss_f = F.cross_entropy(logits_f, labels)
            loss = loss_fuse + 0.5 * loss_s + 0.5 * loss_f

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits_fused.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # Responsive logging: log at regular batch steps OR when heartbeat threshold expires
        is_step = ((batch_idx + 1) % log_interval_batches == 0) or ((batch_idx + 1) == total_batches)
        is_heartbeat = (curr_time - last_log_time >= log_interval_sec)
        if is_step or is_heartbeat:
            elapsed_epoch = curr_time - epoch_start
            rate = (batch_idx + 1) / max(0.001, elapsed_epoch)
            pct = 100.0 * (batch_idx + 1) / max(1, total_batches)
            print(f"  Epoch {epoch:2d} [{pct:5.1f}%] | Batch {batch_idx+1:3d}/{total_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/max(1, total):.4f} | "
                  f"{rate:.1f} batch/s | Elapsed: {(curr_time - start_time)/60:.1f}m", flush=True)
            last_log_time = curr_time

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def validate(model, loader, device, desc="Validation"):
    model.eval()
    all_labels = []
    collected_scores = defaultdict(list)
    total_loss = 0.0
    total_count = 0
    t0 = time.time()
    total_batches = len(loader)

    for idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        with autocast_ctx():
            out = model(images, labels=None)
            loss = F.cross_entropy(out["logits_fused"], labels_dev)

        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        all_labels.append(labels.numpy())

        for k in ["margin_spatial", "margin_freq", "margin_fused", "margin_ensemble",
                  "prob_spatial", "prob_freq", "prob_fused", "prob_ensemble_avg", "prob_ensemble_max"]:
            collected_scores[k].append(out[k].cpu().numpy())

        # Validation heartbeat for long iterations
        if total_batches > 25 and (((idx + 1) % max(1, total_batches // 4) == 0) or ((idx + 1) == total_batches)):
            print(f"    [{desc}] Processed batch {idx+1}/{total_batches} ({time.time()-t0:.1f}s)", flush=True)

    all_labels = np.concatenate(all_labels)
    score_arrays = {k: np.concatenate(v) for k, v in collected_scores.items()}

    # Compute Z-score calibrated decision fusion against bona fides
    bf_mask = (all_labels == 0)
    if bf_mask.sum() > 1:
        s_s = score_arrays["margin_spatial"]
        s_f = score_arrays["margin_freq"]
        mu_s = float(s_s[bf_mask].mean())
        std_s = float(s_s[bf_mask].std()) + 1e-6
        mu_f = float(s_f[bf_mask].mean())
        std_f = float(s_f[bf_mask].std()) + 1e-6
        z_s = (s_s - mu_s) / std_s
        z_f = (s_f - mu_f) / std_f
        cal_score = 0.5 * z_s + 0.5 * z_f
        score_arrays["margin_calibrated"] = cal_score
    else:
        score_arrays["margin_calibrated"] = score_arrays["margin_fused"]

    # Primary metric uses the calibrated decision fusion
    metrics = compute_all_metrics(all_labels, score_arrays["margin_calibrated"])
    metrics["val_loss"] = total_loss / max(1, total_count)

    # Detailed metrics for each branch
    branch_metrics = {
        "spatial": compute_all_metrics(all_labels, score_arrays["margin_spatial"]),
        "frequency": compute_all_metrics(all_labels, score_arrays["margin_freq"]),
        "fused": compute_all_metrics(all_labels, score_arrays["margin_fused"]),
        "calibrated": metrics,
        "ensemble_margin": compute_all_metrics(all_labels, score_arrays["margin_ensemble"]),
        "ensemble_avg": compute_all_metrics(all_labels, score_arrays["prob_ensemble_avg"]),
    }

    return metrics, all_labels, score_arrays, branch_metrics


def train(cfg: Config):
    """Full two-stage training pipeline with automatic multi-GPU support and safety monitoring."""
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    effective_bs = cfg.BATCH_SIZE * max(1, num_gpus)

    print("=" * 60, flush=True)
    print("Dual-Stream UID-MAD (Spatial + Frequency) -- Training", flush=True)
    print(f"  Available GPUs: {num_gpus} | Effective Batch Size: {effective_bs}", flush=True)
    print(f"  DataLoader Workers: {cfg.NUM_WORKERS} | Pin Memory: {cfg.PIN_MEMORY}", flush=True)
    print(f"  Max Runtime Guard: {cfg.MAX_RUNTIME_MINUTES} minutes", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()
    max_runtime_sec = cfg.MAX_RUNTIME_MINUTES * 60
    timed_out = False

    # ---- Data ----
    print("\n[1/5] Loading SMDD dataset...", flush=True)
    all_entries = load_smdd_entries(cfg.SMDD_ZIP)
    random.shuffle(all_entries)

    val_size = int(len(all_entries) * cfg.VAL_FRACTION)
    val_entries = all_entries[:val_size]
    train_entries = all_entries[val_size:]

    train_labels = [e[1] for e in train_entries]
    val_labels = [e[1] for e in val_entries]
    print(f"  Train: {len(train_entries)} (bonafide={train_labels.count(0)}, morph={train_labels.count(1)})", flush=True)
    print(f"  Val:   {len(val_entries)} (bonafide={val_labels.count(0)}, morph={val_labels.count(1)})", flush=True)

    train_ds = SMDDDataset(cfg.SMDD_ZIP, train_entries, transform=get_train_transforms(cfg.IMG_SIZE))
    val_ds = SMDDDataset(cfg.SMDD_ZIP, val_entries, transform=get_val_transforms(cfg.IMG_SIZE))

    sampler = build_sampler(train_entries)
    train_loader = DataLoader(
        train_ds, batch_size=effective_bs, sampler=sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=effective_bs, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0)
    )

    # ---- Model ----
    print("\n[2/5] Building model...", flush=True)
    raw_model = MorphDetector(cfg).to(cfg.DEVICE)
    raw_model.freeze_backbone()
    print(f"  Stream A (Spatial): {cfg.BACKBONE} (frozen)", flush=True)
    print(f"  Stream B (Frequency): Laplacian + Wavelet ConvNet -> {cfg.FREQ_DIM}-d", flush=True)
    print(f"  Dual Fusion: Gated Reinforcement -> {cfg.PROJ_DIM}-d", flush=True)
    print(f"  ArcFace Head: s={cfg.ARCFACE_S}, m={cfg.ARCFACE_M}", flush=True)

    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} parameters", flush=True)

    model = raw_model
    if num_gpus > 1:
        print(f"  Enabling nn.DataParallel across {num_gpus} GPUs.", flush=True)
        model = nn.DataParallel(raw_model)

    scaler = make_scaler()
    best_eer = 1.0
    best_loss = 999.0
    history = []

    # ---- Stage 1: Head warm-up ----
    print(f"\n[3/5] Stage 1: Head warm-up (Backbone Frozen, {cfg.WARMUP_EPOCHS} epochs)...", flush=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, raw_model.parameters()),
        lr=cfg.HEAD_LR, weight_decay=cfg.WEIGHT_DECAY,
    )

    for epoch in range(1, cfg.WARMUP_EPOCHS + 1):
        if time.time() - start_time > max_runtime_sec:
            print(f"\n[SAFETY STOP] Max runtime ({cfg.MAX_RUNTIME_MINUTES}m) exceeded. Stopping Stage 1.", flush=True)
            timed_out = True
            break

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, cfg.DEVICE, epoch, start_time, max_runtime_sec,
            log_interval_batches=cfg.LOG_INTERVAL_BATCHES, log_interval_sec=cfg.LOG_INTERVAL_SEC
        )
        val_metrics, _, _, _ = validate(model, val_loader, cfg.DEVICE, desc=f"Val-Epoch-{epoch}")
        print(f"  Epoch {epoch:2d}: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_EER={val_metrics['EER']:.4f} val_loss={val_metrics['val_loss']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}", flush=True)
        history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss,
                        "train_acc": train_acc, **val_metrics})

        is_best = False
        val_l = val_metrics["val_loss"]
        if val_metrics["EER"] < best_eer:
            is_best = True
        elif abs(val_metrics["EER"] - best_eer) < 1e-4 and val_l < best_loss:
            is_best = True

        if is_best:
            best_eer = val_metrics["EER"]
            best_loss = val_l
            torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
            print(f"    -> New best model saved (EER: {best_eer:.4f}, val_loss: {best_loss:.4f})", flush=True)

        torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "checkpoint_latest.pt"))

    torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "stage1_final.pt"))

    # ---- Stage 2: Fine-tuning ----
    if not timed_out:
        if getattr(cfg, "USE_LORA", False):
            print(f"\n[4/5] Stage 2: LoRA Fine-Tuning (Adapting last {cfg.LORA_BLOCKS} blocks, {cfg.TOTAL_EPOCHS - cfg.WARMUP_EPOCHS} epochs)...", flush=True)
            raw_model.unfreeze_lora()
        else:
            print(f"\n[4/5] Stage 2: Differential fine-tuning (Unfreezing last {cfg.UNFREEZE_BLOCKS} block, {cfg.TOTAL_EPOCHS - cfg.WARMUP_EPOCHS} epochs)...", flush=True)
            raw_model.unfreeze_last_n_blocks(cfg.UNFREEZE_BLOCKS)

        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} / {total_params:,} parameters", flush=True)

        backbone_params = []
        head_params = []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name or name.startswith("backbone"):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.FINETUNE_BACKBONE_LR},
            {"params": head_params, "lr": cfg.FINETUNE_HEAD_LR},
        ], weight_decay=cfg.WEIGHT_DECAY)

        total_ft_epochs = cfg.TOTAL_EPOCHS - cfg.WARMUP_EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_ft_epochs, eta_min=1e-7)

        for epoch in range(cfg.WARMUP_EPOCHS + 1, cfg.TOTAL_EPOCHS + 1):
            if time.time() - start_time > max_runtime_sec:
                print(f"\n[SAFETY STOP] Max runtime ({cfg.MAX_RUNTIME_MINUTES}m) exceeded. Stopping Stage 2.", flush=True)
                break

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, cfg.DEVICE, epoch, start_time, max_runtime_sec,
                log_interval_batches=cfg.LOG_INTERVAL_BATCHES, log_interval_sec=cfg.LOG_INTERVAL_SEC
            )
            scheduler.step()
            val_metrics, _, _, _ = validate(model, val_loader, cfg.DEVICE, desc=f"Val-Epoch-{epoch}")
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_EER={val_metrics['EER']:.4f} val_loss={val_metrics['val_loss']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}", flush=True)
            history.append({"epoch": epoch, "stage": 2, "train_loss": train_loss,
                            "train_acc": train_acc, **val_metrics})

            is_best = False
            val_l = val_metrics["val_loss"]
            if val_metrics["EER"] < best_eer:
                is_best = True
            elif abs(val_metrics["EER"] - best_eer) < 1e-4 and val_l < best_loss:
                is_best = True

            if is_best:
                best_eer = val_metrics["EER"]
                best_loss = val_l
                torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
                print(f"    -> New best model saved (EER: {best_eer:.4f}, val_loss: {best_loss:.4f})", flush=True)

            torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "checkpoint_latest.pt"))

    # Save final model and training history
    torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "final_model.pt"))
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "best_model.pt")):
        torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))

    pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)
    elapsed_total = (time.time() - start_time) / 60.0
    print(f"\n  Training complete in {elapsed_total:.1f} minutes. Best validation EER: {best_eer:.4f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_frll(cfg: Config, model: nn.Module = None):
    """Evaluate best checkpoint on all FRLL-Morphs generator splits + degradation conditions."""

    print("\n" + "=" * 60, flush=True)
    print("Cross-Dataset Evaluation on FRLL-Morphs (Multi-Branch UID-MAD)", flush=True)
    print("=" * 60, flush=True)

    num_gpus = torch.cuda.device_count()
    effective_bs = cfg.BATCH_SIZE * max(1, num_gpus)

    # Load best model if not provided
    if model is None:
        raw_model = MorphDetector(cfg).to(cfg.DEVICE)
        search_paths = [
            cfg.CKPT_TO_EVAL,
            os.path.join(cfg.OUTPUT_DIR, cfg.CKPT_TO_EVAL),
            os.path.join(cfg.OUTPUT_DIR, "best_model.pt"),
            os.path.join(cfg.OUTPUT_DIR, "final_model.pt"),
            os.path.join("checkpoints", cfg.CKPT_TO_EVAL),
            os.path.join(".", cfg.CKPT_TO_EVAL),
            "/kaggle/working/best_model.pt",
            "checkpoints/best_model.pt",
            "results_run2/best_model.pt",
            "best_model.pt",
        ]
        ckpt_path = None
        for p in search_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=cfg.DEVICE)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith("module.") else k
                new_state_dict[new_key] = v
            missing, unexpected = raw_model.load_state_dict(new_state_dict, strict=False)
            print(f"  Loaded checkpoint: {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
        else:
            print(f"  WARNING: No checkpoint found matching {cfg.CKPT_TO_EVAL}. Running with initialized weights.", flush=True)

        model = raw_model
        if num_gpus > 1:
            model = nn.DataParallel(raw_model)
    elif not isinstance(model, nn.DataParallel) and num_gpus > 1:
        model = nn.DataParallel(model)

    model.eval()

    bonafide_paths, generators = load_frll_paths(cfg.FRLL_ZIP)
    print(f"  Bonafide: {len(bonafide_paths)} images", flush=True)
    for gen_name, paths in generators.items():
        print(f"  {gen_name}: {len(paths)} images", flush=True)

    all_results = []
    all_comparisons = []
    results_path = os.path.join(cfg.OUTPUT_DIR, "frll_results.csv")
    comp_path = os.path.join(cfg.OUTPUT_DIR, "frll_results_comparison.csv")

    conditions = [
        ("clean", get_val_transforms(cfg.IMG_SIZE)),
        ("jpeg_q30", get_degraded_transforms(cfg.IMG_SIZE, mode="jpeg")),
        ("printscan", get_degraded_transforms(cfg.IMG_SIZE, mode="printscan")),
    ]

    total_evals = len(generators) * len(conditions)
    eval_count = 0

    for gen_name, morph_paths in generators.items():
        for cond_name, transform in conditions:
            eval_count += 1
            t_cond_start = time.time()
            total_samples = len(bonafide_paths) + len(morph_paths)
            print(f"\n  [{eval_count}/{total_evals}] Evaluating {gen_name} [{cond_name}] ({total_samples} images)...", flush=True)
            ds = FRLLDataset(cfg.FRLL_ZIP, bonafide_paths, morph_paths, transform=transform)
            loader = DataLoader(
                ds, batch_size=effective_bs, shuffle=False,
                num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
            )
            metrics, labels, scores, branch_metrics = validate(model, loader, cfg.DEVICE, desc=f"{gen_name}_{cond_name}")

            result = {
                "generator": gen_name,
                "condition": cond_name,
                **metrics,
                "spatial_AUC": branch_metrics["spatial"]["ROC-AUC"],
                "spatial_EER": branch_metrics["spatial"]["EER"],
                "freq_AUC": branch_metrics["frequency"]["ROC-AUC"],
                "freq_EER": branch_metrics["frequency"]["EER"],
                "fused_AUC": branch_metrics["fused"]["ROC-AUC"],
                "fused_EER": branch_metrics["fused"]["EER"],
            }
            all_results.append(result)

            comp_result = {
                "generator": gen_name,
                "condition": cond_name,
                "fused_AUC": branch_metrics["fused"]["ROC-AUC"],
                "fused_EER": branch_metrics["fused"]["EER"],
                "fused_BPCER@1%": branch_metrics["fused"]["BPCER@APCER=1%"],
                "calibrated_AUC": branch_metrics["calibrated"]["ROC-AUC"],
                "calibrated_EER": branch_metrics["calibrated"]["EER"],
                "calibrated_BPCER@1%": branch_metrics["calibrated"]["BPCER@APCER=1%"],
                "spatial_AUC": branch_metrics["spatial"]["ROC-AUC"],
                "spatial_EER": branch_metrics["spatial"]["EER"],
                "freq_AUC": branch_metrics["frequency"]["ROC-AUC"],
                "freq_EER": branch_metrics["frequency"]["EER"],
                "ens_margin_AUC": branch_metrics["ensemble_margin"]["ROC-AUC"],
                "ens_margin_EER": branch_metrics["ensemble_margin"]["EER"],
            }
            all_comparisons.append(comp_result)

            print(f"    [Fused Head]  AUC: {branch_metrics['fused']['ROC-AUC']:.4f} | EER: {branch_metrics['fused']['EER']:.4f} | BPCER@1%: {branch_metrics['fused']['BPCER@APCER=1%']:.4f}", flush=True)
            print(f"    [Calibrated]  AUC: {branch_metrics['calibrated']['ROC-AUC']:.4f} | EER: {branch_metrics['calibrated']['EER']:.4f} | BPCER@1%: {branch_metrics['calibrated']['BPCER@APCER=1%']:.4f}", flush=True)
            print(f"    [Spatial]     AUC: {branch_metrics['spatial']['ROC-AUC']:.4f} | EER: {branch_metrics['spatial']['EER']:.4f}", flush=True)
            print(f"    [Frequency]   AUC: {branch_metrics['frequency']['ROC-AUC']:.4f} | EER: {branch_metrics['frequency']['EER']:.4f}", flush=True)
            print(f"    Finished {gen_name} [{cond_name}] in {time.time()-t_cond_start:.1f}s", flush=True)

            # INCREMENTAL SAVE: Save after every single evaluation so no progress is ever lost
            pd.DataFrame(all_results).to_csv(results_path, index=False)
            pd.DataFrame(all_comparisons).to_csv(comp_path, index=False)

            # Honesty check
            if metrics["EER"] < 0.01:
                print(f"    *** WARNING: EER near 0% -- potential data leakage. Audit required. ***", flush=True)

            # Save multi-curve ROC comparison for clean condition
            if cond_name == "clean":
                try:
                    plt.figure(figsize=(7, 7))
                    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess (0.50)")
                    fpr_s, tpr_s, _ = roc_curve(labels, scores["margin_spatial"])
                    plt.plot(fpr_s, tpr_s, label=f"Spatial (AUC: {branch_metrics['spatial']['ROC-AUC']:.3f})", linewidth=1.8, color="#1f77b4")
                    fpr_f, tpr_f, _ = roc_curve(labels, scores["margin_freq"])
                    plt.plot(fpr_f, tpr_f, label=f"Frequency (AUC: {branch_metrics['frequency']['ROC-AUC']:.3f})", linewidth=1.8, color="#2ca02c")
                    fpr_fuse, tpr_fuse, _ = roc_curve(labels, scores["margin_fused"])
                    plt.plot(fpr_fuse, tpr_fuse, label=f"Fused Head (AUC: {branch_metrics['fused']['ROC-AUC']:.3f})", linewidth=2.0, color="#ff7f0e")
                    fpr_cal, tpr_cal, _ = roc_curve(labels, scores["margin_calibrated"])
                    plt.plot(fpr_cal, tpr_cal, label=f"Calibrated (AUC: {branch_metrics['calibrated']['ROC-AUC']:.3f})", linewidth=2.5, color="#9467bd")

                    plt.xlabel("BPCER (False Positive Rate)")
                    plt.ylabel("1 - APCER (True Positive Rate)")
                    plt.title(f"Multi-Stream ROC: {gen_name} (Clean)")
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc="lower right", fontsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"roc_{gen_name}.png"), dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"    [Warning] Plotting failed for {gen_name}: {e}", flush=True)

    results_df = pd.DataFrame(all_results)
    comp_df = pd.DataFrame(all_comparisons)
    print(f"\n  Final results saved to {results_path}", flush=True)
    print(f"  Final comparison table saved to {comp_path}", flush=True)

    # Print summary comparison table
    print("\n" + "=" * 80, flush=True)
    print("BRANCH COMPARISON SUMMARY TABLE (ROC-AUC & EER)", flush=True)
    print("=" * 80, flush=True)
    print(comp_df.to_string(index=False, float_format="%.4f"), flush=True)

    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def auto_detect_datasets(cfg: Config):
    """Recursively search /kaggle/input and current directory for datasets (both zip and extracted)."""
    search_roots = ["/kaggle/input", "."]

    # Find SMDD
    if not os.path.exists(cfg.SMDD_ZIP):
        for sroot in search_roots:
            if not os.path.exists(sroot):
                continue
            for root, dirs, files in os.walk(sroot):
                if "train.csv" in files:
                    cfg.SMDD_ZIP = root
                    break
                for f in files:
                    if f.endswith(".zip") and "smdd" in f.lower():
                        cfg.SMDD_ZIP = os.path.join(root, f)
                        break
                if os.path.exists(cfg.SMDD_ZIP):
                    break
            if os.path.exists(cfg.SMDD_ZIP):
                break

    # Find FRLL
    if not os.path.exists(cfg.FRLL_ZIP):
        for sroot in search_roots:
            if not os.path.exists(sroot):
                continue
            for root, dirs, files in os.walk(sroot):
                if any("morph_opencv" in d for d in dirs):
                    cfg.FRLL_ZIP = root
                    break
                for f in files:
                    if f.endswith(".zip") and ("drive-download" in f or "frll" in f.lower()):
                        cfg.FRLL_ZIP = os.path.join(root, f)
                        break
                if os.path.exists(cfg.FRLL_ZIP):
                    break
            if os.path.exists(cfg.FRLL_ZIP):
                break

    # Fast local scratch extraction
    if getattr(cfg, "EXTRACT_DATASETS", True):
        scratch_base = cfg.SCRATCH_DIR if os.path.isdir(cfg.SCRATCH_DIR) and os.access(cfg.SCRATCH_DIR, os.W_OK) else cfg.OUTPUT_DIR
        if os.path.isfile(cfg.SMDD_ZIP) and zipfile.is_zipfile(cfg.SMDD_ZIP):
            smdd_target = os.path.join(scratch_base, "smdd_extracted")
            cfg.SMDD_ZIP = extract_archive_if_needed(cfg.SMDD_ZIP, smdd_target, check_indicator="train.csv")
        if os.path.isfile(cfg.FRLL_ZIP) and zipfile.is_zipfile(cfg.FRLL_ZIP):
            frll_target = os.path.join(scratch_base, "frll_extracted")
            cfg.FRLL_ZIP = extract_archive_if_needed(cfg.FRLL_ZIP, frll_target, check_indicator="morph_amsl")

    # If dataset is still a raw zip, disable multiprocessing workers to prevent zip descriptor deadlocks
    if os.path.isfile(cfg.SMDD_ZIP) and zipfile.is_zipfile(cfg.SMDD_ZIP):
        cfg.NUM_WORKERS = 0
        cfg.PIN_MEMORY = False
        print("  [Notice] Running directly from zip archive. Set NUM_WORKERS=0 to avoid IPC deadlocks.", flush=True)
    else:
        print(f"  [Notice] Dataset on filesystem. Using NUM_WORKERS={cfg.NUM_WORKERS}, PIN_MEMORY={cfg.PIN_MEMORY}.", flush=True)


def main():
    start_all = time.time()
    cfg = Config()

    import argparse
    parser = argparse.ArgumentParser(description="UID-MAD Dual-Stream Training & Evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing checkpoint directly")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file to evaluate")
    args, _ = parser.parse_known_args()

    if args.eval_only:
        cfg.EVAL_ONLY = True
    if args.ckpt:
        cfg.CKPT_TO_EVAL = args.ckpt

    # Hard guard: ViT cannot run on CPU. Abort immediately if GPU was not enabled in Kaggle.
    if not torch.cuda.is_available():
        print("\n" + "!" * 70, flush=True)
        print("FATAL ERROR: GPU ACCELERATOR IS NOT ENABLED IN KAGGLE!", flush=True)
        print("torch.cuda.is_available() returned False.", flush=True)
        print("Running Vision Transformer training on CPU takes 10+ hours.", flush=True)
        print("\nFix in Kaggle notebook settings:", flush=True)
        print("  1. In the right-hand panel, find 'Session options' / 'Notebook options'.", flush=True)
        print("  2. Click 'Accelerator' -> Select 'GPU T4 x2'.", flush=True)
        print("  3. Verify the GPU indicator turns ON before running.", flush=True)
        print("!" * 70 + "\n", flush=True)
        sys.exit(1)

    auto_detect_datasets(cfg)

    print(f"SMDD path: {cfg.SMDD_ZIP} (exists={os.path.exists(cfg.SMDD_ZIP)})", flush=True)
    print(f"FRLL path: {cfg.FRLL_ZIP} (exists={os.path.exists(cfg.FRLL_ZIP)})", flush=True)

    assert os.path.exists(cfg.SMDD_ZIP), f"SMDD dataset not found. Checked /kaggle/input and current directory."
    assert os.path.exists(cfg.FRLL_ZIP), f"FRLL dataset not found. Checked /kaggle/input and current directory."

    if cfg.EVAL_ONLY:
        print("\n" + "=" * 60, flush=True)
        print("RUNNING IN EVALUATION-ONLY MODE", flush=True)
        print(f"Target checkpoint: {cfg.CKPT_TO_EVAL}", flush=True)
        print("=" * 60, flush=True)
        evaluate_frll(cfg, model=None)
    else:
        # Train
        model = train(cfg)
        # Evaluate
        evaluate_frll(cfg, model)

    elapsed_all = (time.time() - start_all) / 60.0
    print("\n" + "=" * 60, flush=True)
    print(f"DONE. Total execution time: {elapsed_all:.1f} minutes.", flush=True)
    print("All outputs saved in:", cfg.OUTPUT_DIR, flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
