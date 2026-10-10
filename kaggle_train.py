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
from torch.cuda.amp import autocast, GradScaler

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
    BACKBONE = "dinov2_vits14"  # timm model name; or "dinov2_vitb14" for ViT-B
    EMBED_DIM = 384  # 384 for ViT-S, 768 for ViT-B
    PROJ_DIM = 256

    # ArcFace
    ARCFACE_S = 30.0
    ARCFACE_M = 0.35
    NUM_CLASSES = 2  # bonafide vs morph

    # Training
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    WARMUP_EPOCHS = 3
    TOTAL_EPOCHS = 10
    HEAD_LR = 1e-3
    FINETUNE_HEAD_LR = 1e-4
    FINETUNE_BACKBONE_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    UNFREEZE_BLOCKS = 2  # number of last transformer blocks to unfreeze in stage 2

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

def get_train_transforms(img_size: int) -> A.Compose:
    """Heavy augmentation pipeline matching architecture.md spec."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # Simulated JPEG compression (quality 20-70)
        A.ImageCompression(quality_lower=20, quality_upper=70, p=0.5),
        # Gaussian blur
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.3),
        # Motion blur
        A.MotionBlur(blur_limit=(3, 7), p=0.2),
        # Simulated print-scan: downsample + JPEG + noise + upsample
        A.Compose([
            A.Downscale(scale_min=0.4, scale_max=0.6, p=1.0),
            A.ImageCompression(quality_lower=30, quality_upper=50, p=1.0),
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
        ], p=0.2),
        # Random gamma / brightness
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # Color jitter
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        # Normalize + tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


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

        # Load backbone via timm
        self.backbone = timm.create_model(
            cfg.BACKBONE,
            pretrained=True,
            num_classes=0,  # remove classifier head, get features
        )

        # Projection
        self.projector = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.PROJ_DIM),
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

    def forward(self, x, labels=None):
        features = self.backbone(x)  # (B, EMBED_DIM)
        proj = self.projector(features)  # (B, PROJ_DIM)
        logits = self.arc_head(proj, labels)  # (B, NUM_CLASSES)
        return logits

    def get_scores(self, x):
        """Inference: return morph probability scores."""
        features = self.backbone(x)
        proj = self.projector(features)
        logits = self.arc_head(proj, labels=None)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # morph class probability


# ---------------------------------------------------------------------------
# Metrics (ISO/IEC 30107-3)
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """Compute Equal Error Rate from labels and scores."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Find the point where FPR ~ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def compute_bpcer_at_apcer(labels: np.ndarray, scores: np.ndarray, apcer_target: float):
    """
    Compute BPCER at a fixed APCER threshold.

    APCER = fraction of attacks misclassified as bonafide (FNR on attacks).
    BPCER = fraction of bonafide misclassified as attacks (FPR on bonafide).

    Higher score = more likely morph.
    """
    attack_scores = scores[labels == 1]
    bonafide_scores = scores[labels == 0]

    # Sort thresholds: we want to find the threshold where APCER <= apcer_target
    thresholds = np.sort(np.unique(scores))

    best_bpcer = 1.0
    for t in thresholds:
        # Attacks classified as bonafide (score < threshold): APCER
        apcer = np.mean(attack_scores < t)
        # Bonafide classified as attack (score >= threshold): BPCER
        bpcer = np.mean(bonafide_scores >= t)

        if apcer <= apcer_target:
            best_bpcer = bpcer
            break

    return best_bpcer


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute EER, BPCER@APCER=1%, BPCER@APCER=5%, and ROC-AUC."""
    eer, eer_thresh = compute_eer(labels, scores)
    bpcer_1 = compute_bpcer_at_apcer(labels, scores, apcer_target=0.01)
    bpcer_5 = compute_bpcer_at_apcer(labels, scores, apcer_target=0.05)
    try:
        auc = roc_auc_score(labels, scores)
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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_labels = []
    all_scores = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            scores = model.get_scores(images)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    metrics = compute_all_metrics(all_labels, all_scores)
    return metrics, all_labels, all_scores


def train(cfg: Config):
    """Full two-stage training pipeline."""
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("DINO-Branch Morph Attack Detection -- Training")
    print("=" * 60)

    # ---- Data ----
    print("\n[1/5] Loading SMDD dataset...")
    all_entries = load_smdd_entries(cfg.SMDD_ZIP)
    random.shuffle(all_entries)

    val_size = int(len(all_entries) * cfg.VAL_FRACTION)
    val_entries = all_entries[:val_size]
    train_entries = all_entries[val_size:]

    train_labels = [e[1] for e in train_entries]
    val_labels = [e[1] for e in val_entries]
    print(f"  Train: {len(train_entries)} (bonafide={train_labels.count(0)}, morph={train_labels.count(1)})")
    print(f"  Val:   {len(val_entries)} (bonafide={val_labels.count(0)}, morph={val_labels.count(1)})")

    train_ds = SMDDDataset(cfg.SMDD_ZIP, train_entries, transform=get_train_transforms(cfg.IMG_SIZE))
    val_ds = SMDDDataset(cfg.SMDD_ZIP, val_entries, transform=get_val_transforms(cfg.IMG_SIZE))

    sampler = build_sampler(train_entries)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ---- Model ----
    print("\n[2/5] Building model...")
    model = MorphDetector(cfg).to(cfg.DEVICE)
    model.freeze_backbone()
    print(f"  Backbone: {cfg.BACKBONE} (frozen)")
    print(f"  Projection: {cfg.EMBED_DIM} -> {cfg.PROJ_DIM}")
    print(f"  ArcFace: s={cfg.ARCFACE_S}, m={cfg.ARCFACE_M}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} parameters")

    scaler = GradScaler()
    best_eer = 1.0
    history = []

    # ---- Stage 1: Head warm-up ----
    print("\n[3/5] Stage 1: Head warm-up...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.HEAD_LR, weight_decay=cfg.WEIGHT_DECAY,
    )

    for epoch in range(1, cfg.WARMUP_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg.DEVICE, epoch)
        val_metrics, _, _ = validate(model, val_loader, cfg.DEVICE)
        print(f"  Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_EER={val_metrics['EER']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}")
        history.append({"epoch": epoch, "stage": 1, "train_loss": train_loss,
                        "train_acc": train_acc, **val_metrics})

        if val_metrics["EER"] < best_eer:
            best_eer = val_metrics["EER"]
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
            print(f"    -> New best EER: {best_eer:.4f}")

    # ---- Stage 2: Differential fine-tuning ----
    print(f"\n[4/5] Stage 2: Unfreezing last {cfg.UNFREEZE_BLOCKS} blocks...")
    model.unfreeze_last_n_blocks(cfg.UNFREEZE_BLOCKS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total_params:,} parameters")

    # Differential LR: backbone params get lower LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_ft_epochs)

    for epoch in range(cfg.WARMUP_EPOCHS + 1, cfg.TOTAL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg.DEVICE, epoch)
        scheduler.step()
        val_metrics, _, _ = validate(model, val_loader, cfg.DEVICE)
        print(f"  Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_EER={val_metrics['EER']:.4f} val_AUC={val_metrics['ROC-AUC']:.4f}")
        history.append({"epoch": epoch, "stage": 2, "train_loss": train_loss,
                        "train_acc": train_acc, **val_metrics})

        if val_metrics["EER"] < best_eer:
            best_eer = val_metrics["EER"]
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pt"))
            print(f"    -> New best EER: {best_eer:.4f}")

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)
    print(f"\n  Training complete. Best validation EER: {best_eer:.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_frll(cfg: Config, model: MorphDetector = None):
    """Evaluate best checkpoint on all FRLL-Morphs generator splits + degradation conditions."""

    print("\n" + "=" * 60)
    print("Cross-Dataset Evaluation on FRLL-Morphs")
    print("=" * 60)

    # Load best model
    if model is None:
        model = MorphDetector(cfg).to(cfg.DEVICE)
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.DEVICE))
        print(f"  Loaded checkpoint: {ckpt_path}")
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
            ds = FRLLDataset(cfg.FRLL_ZIP, bonafide_paths, morph_paths, transform=transform)
            loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)
            metrics, labels, scores = validate(model, loader, cfg.DEVICE)

            result = {"generator": gen_name, "condition": cond_name, **metrics}
            all_results.append(result)
            print(f"\n  {gen_name} [{cond_name}]:")
            print(f"    EER:             {metrics['EER']:.4f}")
            print(f"    BPCER@APCER=1%:  {metrics['BPCER@APCER=1%']:.4f}")
            print(f"    BPCER@APCER=5%:  {metrics['BPCER@APCER=5%']:.4f}")
            print(f"    ROC-AUC:         {metrics['ROC-AUC']:.4f}")

            # Honesty check
            if metrics["EER"] < 0.01:
                print(f"    *** WARNING: EER near 0% -- potential data leakage. Audit required. ***")

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
    print(f"\n  Results saved to {cfg.OUTPUT_DIR}/frll_results.csv")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="%.4f"))

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
    cfg = Config()
    auto_detect_datasets(cfg)

    print(f"SMDD path: {cfg.SMDD_ZIP} (exists={os.path.exists(cfg.SMDD_ZIP)})")
    print(f"FRLL path: {cfg.FRLL_ZIP} (exists={os.path.exists(cfg.FRLL_ZIP)})")

    assert os.path.exists(cfg.SMDD_ZIP), f"SMDD dataset not found. Checked /kaggle/input and current directory."
    assert os.path.exists(cfg.FRLL_ZIP), f"FRLL dataset not found. Checked /kaggle/input and current directory."

    # Train
    model = train(cfg)

    # Evaluate
    evaluate_frll(cfg, model)

    print("\n" + "=" * 60)
    print("DONE. All outputs in:", cfg.OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
