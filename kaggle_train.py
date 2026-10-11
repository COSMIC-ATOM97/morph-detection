"""
DINO-Branch Morph Attack Detection -- Kaggle Training & Evaluation Script
=========================================================================
Single-file, self-contained implementation.

Usage on Kaggle:
    1. Upload SMDD and FRLL-Morphs as Kaggle Datasets.
    2. Paste this file into a Code cell and run.
    3. Results are written to /kaggle/working/.

Requires (all pre-installed on Kaggle):
    torch, torchvision, timm, albumentations, scikit-learn, matplotlib, pandas
"""

import os
import sys
import time

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

import io
import csv
import math
import json
import random
import zipfile
import warnings
from pathlib import Path
from collections import defaultdict

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

    # Backbone
    BACKBONE = "vit_small_patch14_dinov2.lvd142m"  # timm model name
    EMBED_DIM = 384  # 384 for ViT-S, 768 for ViT-B
    PROJ_DIM = 256

    # ArcFace
    ARCFACE_S = 32.0
    ARCFACE_M = 0.25  # Softer margin to prevent overconfidence on synthetic data
    NUM_CLASSES = 2  # bonafide vs morph

    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 0  # 0 workers completely eliminates multiprocessing IPC deadlocks on Kaggle
    PIN_MEMORY = False  # Avoids CUDA allocator deadlocks in multithreading
    WARMUP_EPOCHS = 4  # 4 epochs head warmup on frozen backbone (~6-7 min)
    TOTAL_EPOCHS = 8  # 8 epochs total (4 warmup + 4 fine-tune, ~13-14 min)
    HEAD_LR = 1e-3
    FINETUNE_HEAD_LR = 1e-4
    FINETUNE_BACKBONE_LR = 1e-6  # Conservative fine-tune rate to preserve DINOv2 generalization
    WEIGHT_DECAY = 1e-3  # Stronger L2 regularization
    UNFREEZE_BLOCKS = 1  # Unfreeze only the final block (block 11)
    MAX_RUNTIME_MINUTES = 30  # Hard safety ceiling: exits cleanly if training ever exceeds 30m

    # Validation split
    VAL_FRACTION = 0.2

    # Reproducibility
    SEED = 42

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
    """Anti-overfitting augmentation pipeline destroying sensor/generator shortcuts."""
    cd_drop = make_coarse_dropout(p=0.3)
    aug_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # Severe JPEG compression
        A.ImageCompression(quality_lower=15, quality_upper=85, p=0.6),
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.5)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.5),
        # Aggressive print-scan simulation (downsample + JPEG + Gaussian noise + upsample)
        A.Compose([
            A.Downscale(scale_min=0.3, scale_max=0.6, p=1.0),
            A.ImageCompression(quality_lower=20, quality_upper=50, p=1.0),
            A.GaussNoise(var_limit=(10.0, 35.0), p=1.0),
        ], p=0.4),
        # Photometric variation
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        # Grayscale / desaturation (destroys color hue shortcuts, forces structural analysis)
        A.ToGray(p=0.2),
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
# Datasets (Supports both .zip archives and extracted folders)
# ---------------------------------------------------------------------------

class SMDDDataset(Dataset):
    """Reads SMDD images from a zip archive or an extracted directory using train.csv metadata."""

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
    """Reads FRLL-Morphs images from a zip archive or an extracted directory for evaluation."""

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
    """Parse train.csv inside the SMDD zip or directory and return list of (rel_path, label_int)."""
    if os.path.isfile(data_source) and zipfile.is_zipfile(data_source):
        with zipfile.ZipFile(data_source, "r") as zf:
            csv_bytes = zf.read("train.csv").decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_bytes))
    else:
        csv_file = os.path.join(data_source, "train.csv")
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

class ArcFaceHead(nn.Module):
    """Additive angular margin classification head."""

    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        # L2-normalize weight and embedding
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, w)  # (B, num_classes)

        if labels is not None:
            # Add angular margin to the target class
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
            logits = torch.cos(theta + one_hot * self.m)
            logits = logits * self.s
        else:
            logits = cosine * self.s

        return logits


class MorphDetector(nn.Module):
    """DINOv2 backbone + projection + ArcFace head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Load backbone (try timm variants, fallback to torch.hub)
        backbone_loaded = False
        candidates = [
            cfg.BACKBONE,
            "vit_small_patch14_dinov2.lvd142m",
            "vit_small_patch14_dinov2",
        ]
        for name in candidates:
            try:
                self.backbone = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
                print(f"  Successfully loaded backbone via timm: {name} (dynamic_img_size=True)")
                backbone_loaded = True
                break
            except Exception:
                continue

        if not backbone_loaded:
            print("  Falling back to torch.hub for DINOv2 (dinov2_vits14)...")
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        # Multi-scale feature dimension: CLS (384) + Patch Mean (384) + Patch Std (384) + Patch Max (384)
        in_dim = cfg.EMBED_DIM * 4
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.PROJ_DIM),
            nn.BatchNorm1d(cfg.PROJ_DIM),
        )

        # ArcFace head
        self.arc_head = ArcFaceHead(
            in_features=cfg.PROJ_DIM,
            num_classes=cfg.NUM_CLASSES,
            s=cfg.ARCFACE_S,
            m=cfg.ARCFACE_M,
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks of the ViT."""
        # timm DINOv2 stores blocks in self.backbone.blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze the final norm layer
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x, labels=None, return_scores=False):
        # Extract patch and CLS tokens from backbone
        if hasattr(self.backbone, "forward_features"):
            tokens = self.backbone.forward_features(x)
            if tokens.dim() == 3 and tokens.size(1) > 1:
                cls_token = tokens[:, 0]
                patches = tokens[:, 1:]
                patch_mean = patches.mean(dim=1)
                patch_std = patches.std(dim=1)
                patch_max = patches.max(dim=1).values
                features = torch.cat([cls_token, patch_mean, patch_std, patch_max], dim=1)
            elif tokens.dim() == 2:
                features = tokens.repeat(1, 4)
            else:
                features = tokens[:, 0].repeat(1, 4)
        else:
            cls_token = self.backbone(x)
            features = cls_token.repeat(1, 4)

        proj = self.projector(features)
        if return_scores:
            logits = self.arc_head(proj, labels=None)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1]  # morph class probability
        logits = self.arc_head(proj, labels)  # (B, NUM_CLASSES)
        return logits

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
    """Build a weighted sampler to handle 5:3 class imbalance in SMDD."""
    labels = [e[1] for e in entries]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(entries), replacement=True)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, start_time, max_runtime_sec):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        if time.time() - start_time > max_runtime_sec:
            print(f"  [SAFETY CEILING REACHED] Exiting epoch {epoch} early after {batch_idx} batches.", flush=True)
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed_epoch = time.time() - epoch_start
            rate = (batch_idx + 1) / max(0.001, elapsed_epoch)
            print(f"  Epoch {epoch:2d} | Batch {batch_idx+1:3d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/max(1, total):.4f} | "
                  f"{rate:.1f} it/s | Total Elapsed: {(time.time() - start_time)/60:.1f}m", flush=True)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    total_loss = 0.0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model(images, labels=None)
            loss = F.cross_entropy(logits, labels_dev)
            scores = F.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    metrics = compute_all_metrics(all_labels, all_scores)
    metrics["val_loss"] = total_loss / max(1, total_count)
    return metrics, all_labels, all_scores


def train(cfg: Config):
    """Full two-stage training pipeline with automatic multi-GPU support."""
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    effective_bs = cfg.BATCH_SIZE * max(1, num_gpus)

    print("=" * 60, flush=True)
    print("DINO-Branch Morph Attack Detection -- Training", flush=True)
    print(f"  Available GPUs: {num_gpus} | Effective Batch Size: {effective_bs}", flush=True)
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
    train_loader = DataLoader(train_ds, batch_size=effective_bs, sampler=sampler,
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=effective_bs, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    # ---- Model ----
    print("\n[2/5] Building model...", flush=True)
    raw_model = MorphDetector(cfg).to(cfg.DEVICE)
    raw_model.freeze_backbone()
    print(f"  Backbone: {cfg.BACKBONE} (frozen)", flush=True)
    print(f"  Projection: {cfg.EMBED_DIM}*4 -> {cfg.PROJ_DIM}", flush=True)
    print(f"  ArcFace: s={cfg.ARCFACE_S}, m={cfg.ARCFACE_M}", flush=True)

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

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg.DEVICE, epoch, start_time, max_runtime_sec)
        val_metrics, _, _ = validate(model, val_loader, cfg.DEVICE)
        print(f"  Epoch {epoch:2d}: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_EER={val_metrics['EER']:.4f} val_loss={val_metrics['val_loss']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}", flush=True)
        history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss,
                        "train_acc": train_acc, **val_metrics})

        is_best = False
        val_l = val_metrics["val_loss"]
        if val_metrics["EER"] < best_eer:
            is_best = True
        elif abs(val_metrics["EER"] - best_eer) < 1e-6 and val_l < best_loss:
            is_best = True

        if is_best:
            best_eer = val_metrics["EER"]
            best_loss = val_l
            torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
            print(f"    -> New best model saved (EER: {best_eer:.4f}, val_loss: {best_loss:.4f})", flush=True)

    # ---- Stage 2: Differential fine-tuning ----
    if not timed_out:
        print(f"\n[4/5] Stage 2: Differential fine-tuning (Unfreezing last {cfg.UNFREEZE_BLOCKS} block, {cfg.TOTAL_EPOCHS - cfg.WARMUP_EPOCHS} epochs)...", flush=True)
        raw_model.unfreeze_last_n_blocks(cfg.UNFREEZE_BLOCKS)

        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"  Trainable: {trainable:,} / {total_params:,} parameters", flush=True)

        backbone_params = []
        head_params = []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone"):
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

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg.DEVICE, epoch, start_time, max_runtime_sec)
            scheduler.step()
            val_metrics, _, _ = validate(model, val_loader, cfg.DEVICE)
            print(f"  Epoch {epoch:2d}: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_EER={val_metrics['EER']:.4f} val_loss={val_metrics['val_loss']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}", flush=True)
            history.append({"epoch": epoch, "stage": 2, "train_loss": train_loss,
                            "train_acc": train_acc, **val_metrics})

            is_best = False
            val_l = val_metrics["val_loss"]
            if val_metrics["EER"] < best_eer:
                is_best = True
            elif abs(val_metrics["EER"] - best_eer) < 1e-6 and val_l < best_loss:
                is_best = True

            if is_best:
                best_eer = val_metrics["EER"]
                best_loss = val_l
                torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
                print(f"    -> New best model saved (EER: {best_eer:.4f}, val_loss: {best_loss:.4f})", flush=True)

    # Save final model and training history
    torch.save(raw_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "final_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)
    elapsed_total = (time.time() - start_time) / 60.0
    print(f"\n  Training complete in {elapsed_total:.1f} minutes. Best validation EER: {best_eer:.4f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_frll(cfg: Config, model: nn.Module = None):
    """Evaluate best checkpoint on all FRLL-Morphs generator splits + degradation conditions."""

    print("\n" + "=" * 60)
    print("Cross-Dataset Evaluation on FRLL-Morphs")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    effective_bs = cfg.BATCH_SIZE * max(1, num_gpus)

    # Load best model
    if model is None:
        raw_model = MorphDetector(cfg).to(cfg.DEVICE)
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
        if os.path.exists(ckpt_path):
            raw_model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
            print(f"  Loaded checkpoint: {ckpt_path}")
        model = raw_model
        if num_gpus > 1:
            model = nn.DataParallel(raw_model)
    elif not isinstance(model, nn.DataParallel) and num_gpus > 1:
        model = nn.DataParallel(model)

    model.eval()

    bonafide_paths, generators = load_frll_paths(cfg.FRLL_ZIP)
    print(f"  Bonafide: {len(bonafide_paths)} images")
    for gen_name, paths in generators.items():
        print(f"  {gen_name}: {len(paths)} images")

    all_results = []
    conditions = [
        ("clean", get_val_transforms(cfg.IMG_SIZE)),
        ("jpeg_q30", get_degraded_transforms(cfg.IMG_SIZE, mode="jpeg")),
        ("printscan", get_degraded_transforms(cfg.IMG_SIZE, mode="printscan")),
    ]

    for gen_name, morph_paths in generators.items():
        for cond_name, transform in conditions:
            print(f"  Evaluating {gen_name} [{cond_name}]...", flush=True)
            ds = FRLLDataset(cfg.FRLL_ZIP, bonafide_paths, morph_paths, transform=transform)
            loader = DataLoader(ds, batch_size=effective_bs, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
            metrics, labels, scores = validate(model, loader, cfg.DEVICE)

            result = {"generator": gen_name, "condition": cond_name, **metrics}
            all_results.append(result)
            print(f"    EER: {metrics['EER']:.4f} | AUC: {metrics['ROC-AUC']:.4f} | "
                  f"BPCER@1%: {metrics['BPCER@APCER=1%']:.4f} | BPCER@5%: {metrics['BPCER@APCER=5%']:.4f}", flush=True)

            # Honesty check
            if metrics["EER"] < 0.01:
                print(f"    *** WARNING: EER near 0% -- potential data leakage. Audit required. ***", flush=True)

            # Save ROC curve for clean condition
            if cond_name == "clean":
                fpr, tpr, _ = roc_curve(labels, scores)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, linewidth=2)
                plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
                plt.xlabel("FPR (BPCER)")
                plt.ylabel("TPR (1 - APCER)")
                plt.title(f"ROC: {gen_name}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"roc_{gen_name}.png"), dpi=150)
                plt.close()

    # Save results table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "frll_results.csv"), index=False)
    print(f"\n  Results saved to {cfg.OUTPUT_DIR}/frll_results.csv", flush=True)

    # Print summary table
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY TABLE", flush=True)
    print("=" * 60, flush=True)
    print(results_df.to_string(index=False, float_format="%.4f"), flush=True)

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


def main():
    start_all = time.time()
    cfg = Config()
    auto_detect_datasets(cfg)

    print(f"SMDD path: {cfg.SMDD_ZIP} (exists={os.path.exists(cfg.SMDD_ZIP)})", flush=True)
    print(f"FRLL path: {cfg.FRLL_ZIP} (exists={os.path.exists(cfg.FRLL_ZIP)})", flush=True)

    assert os.path.exists(cfg.SMDD_ZIP), f"SMDD dataset not found. Checked /kaggle/input and current directory."
    assert os.path.exists(cfg.FRLL_ZIP), f"FRLL dataset not found. Checked /kaggle/input and current directory."

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
