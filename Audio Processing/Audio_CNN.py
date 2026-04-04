"""
EP Lab Procedure Phase Classification via Audio Spectrograms
=============================================================
This module implements a CNN-based classifier that takes log-mel spectrograms
derived from OR/EP lab audio recordings and predicts the current procedural phase.

Pipeline:
    Raw Audio (.wav) → Log-Mel Spectrogram → CNN → Phase Label

EP Lab Phases modelled:
    0 - Room Setup & Equipment Check
    1 - Patient Preparation & Sedation
    2 - Vascular Access (sheath insertion)
    3 - Catheter Navigation & Mapping
    4 - Ablation
    5 - Verification / Pacing
    6 - Catheter & Sheath Removal
    7 - Post-Procedure / Recovery

Dependencies:
    pip install torch torchvision torchaudio librosa numpy scikit-learn
                matplotlib seaborn tqdm
"""

import os
import json
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # --- Audio preprocessing ---
    SAMPLE_RATE: int = 22050          # Hz — resample all audio to this rate
    CLIP_DURATION: float = 5.0        # seconds per spectrogram window
    HOP_DURATION: float = 2.5         # seconds between windows (50% overlap)
    N_FFT: int = 1024                 # FFT window size
    HOP_LENGTH: int = 512             # STFT hop length (samples)
    N_MELS: int = 128                 # number of mel filterbanks
    FMIN: float = 20.0                # lowest mel frequency (Hz)
    FMAX: float = 8000.0              # highest mel frequency (Hz)
    TOP_DB: float = 80.0              # dynamic range for power_to_db

    # --- Derived dimensions (do not edit) ---
    @classmethod
    def frames(cls) -> int:
        """Number of time frames in one fixed-length clip."""
        samples = int(cls.CLIP_DURATION * cls.SAMPLE_RATE)
        return 1 + samples // cls.HOP_LENGTH  # librosa convention

    # --- Model ---
    NUM_CLASSES: int = 8              # number of EP lab phases
    DROPOUT: float = 0.4

    # --- Training ---
    EPOCHS: int = 50
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.10
    SEED: int = 42

    # --- Paths ---
    DATA_DIR: str = "data/audio"      # root folder; subfolders = phase names
    CHECKPOINT_DIR: str = "checkpoints"
    BEST_MODEL_PATH: str = "checkpoints/best_ep_lab_cnn.pt"

    # --- Device ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase label map (folder name → index)
    PHASE_LABELS: Dict[str, int] = {
        "room_setup":            0,
        "patient_prep":          1,
        "vascular_access":       2,
        "catheter_navigation":   3,
        "ablation":              4,
        "verification_pacing":   5,
        "catheter_removal":      6,
        "post_procedure":        7,
    }

    PHASE_NAMES: List[str] = [
        "Room Setup",
        "Patient Prep",
        "Vascular Access",
        "Catheter Navigation",
        "Ablation",
        "Verification / Pacing",
        "Catheter Removal",
        "Post-Procedure",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUDIO → SPECTROGRAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file and resample to target sample rate."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
    n_fft: int = Config.N_FFT,
    hop_length: int = Config.HOP_LENGTH,
    n_mels: int = Config.N_MELS,
    fmin: float = Config.FMIN,
    fmax: float = Config.FMAX,
    top_db: float = Config.TOP_DB,
) -> np.ndarray:
    """
    Compute a log-scaled mel spectrogram.

    Returns:
        Spectrogram array of shape (n_mels, time_frames), values in dB.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, top_db=top_db)
    return log_mel  # shape: (n_mels, T)


def slice_audio_to_clips(
    audio: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
    clip_duration: float = Config.CLIP_DURATION,
    hop_duration: float = Config.HOP_DURATION,
) -> List[np.ndarray]:
    """
    Slice a long recording into fixed-length overlapping clips.

    Clips shorter than clip_duration are zero-padded on the right.
    Returns a list of 1-D numpy arrays each of length (sr * clip_duration).
    """
    clip_samples = int(clip_duration * sr)
    hop_samples = int(hop_duration * sr)
    clips = []
    start = 0
    while start < len(audio):
        end = start + clip_samples
        clip = audio[start:end]
        if len(clip) < clip_samples:
            # Pad with silence
            clip = np.pad(clip, (0, clip_samples - len(clip)))
        clips.append(clip)
        start += hop_samples
    return clips


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Per-clip mean/std normalization so each spectrogram has zero mean,
    unit variance. Handles edge case where std ≈ 0.
    """
    mean = spec.mean()
    std = spec.std() + 1e-8
    return (spec - mean) / std


def audio_file_to_tensors(path: str) -> List[torch.Tensor]:
    """
    Full preprocessing pipeline for one audio file.
    Returns a list of (1, n_mels, T) float32 tensors ready for the model.
    """
    audio = load_audio(path)
    clips = slice_audio_to_clips(audio)
    tensors = []
    for clip in clips:
        spec = compute_log_mel_spectrogram(clip)
        spec = normalize_spectrogram(spec)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        tensors.append(tensor)
    return tensors


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class EPLabSpectrogramDataset(Dataset):
    """
    Dataset that loads audio files from a directory tree:

        data/audio/
            room_setup/          ← phase folder name must match Config.PHASE_LABELS
                rec_001.wav
                rec_002.wav
            patient_prep/
                ...

    Each file is sliced into fixed-length clips. Every clip becomes one sample.
    Spectrograms are computed on-the-fly and cached in memory as float32 tensors.
    """

    def __init__(
        self,
        data_dir: str = Config.DATA_DIR,
        phase_labels: Dict[str, int] = Config.PHASE_LABELS,
        augment: bool = False,
    ):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        self.augment = augment
        self._build_dataset(data_dir, phase_labels)

    def _build_dataset(self, data_dir: str, phase_labels: Dict[str, int]) -> None:
        data_path = Path(data_dir)
        for phase_name, label in phase_labels.items():
            phase_dir = data_path / phase_name
            if not phase_dir.exists():
                print(f"  [WARN] Phase directory not found: {phase_dir}")
                continue
            wav_files = list(phase_dir.glob("*.wav")) + list(phase_dir.glob("*.mp3"))
            for wav_path in wav_files:
                try:
                    tensors = audio_file_to_tensors(str(wav_path))
                    for t in tensors:
                        self.samples.append((t, label))
                except Exception as e:
                    print(f"  [ERROR] Could not process {wav_path}: {e}")
        print(f"Dataset built: {len(self.samples)} clips across {len(phase_labels)} classes.")

    # ── Augmentations ───────────────────────────────────────────────────────

    def _time_mask(self, spec: torch.Tensor, max_width: int = 30) -> torch.Tensor:
        """Mask a random contiguous band of time frames with zeros (SpecAugment)."""
        _, _, T = spec.shape
        width = random.randint(0, max_width)
        start = random.randint(0, max(0, T - width))
        spec = spec.clone()
        spec[:, :, start:start + width] = 0.0
        return spec

    def _freq_mask(self, spec: torch.Tensor, max_width: int = 20) -> torch.Tensor:
        """Mask a random contiguous band of mel frequency bins with zeros."""
        _, F, _ = spec.shape
        width = random.randint(0, max_width)
        start = random.randint(0, max(0, F - width))
        spec = spec.clone()
        spec[:, start:start + width, :] = 0.0
        return spec

    def _add_noise(self, spec: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
        """Add Gaussian noise to simulate recording variability."""
        noise = torch.randn_like(spec) * noise_level
        return spec + noise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spec, label = self.samples[idx]
        if self.augment:
            if random.random() > 0.5:
                spec = self._time_mask(spec)
            if random.random() > 0.5:
                spec = self._freq_mask(spec)
            if random.random() > 0.5:
                spec = self._add_noise(spec)
        return spec, label


def build_dataloaders(
    data_dir: str = Config.DATA_DIR,
    batch_size: int = Config.BATCH_SIZE,
    val_split: float = Config.VAL_SPLIT,
    test_split: float = Config.TEST_SPLIT,
    seed: int = Config.SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset into train / validation / test DataLoaders.
    Augmentation is applied only to the training split.
    """
    full_dataset = EPLabSpectrogramDataset(data_dir=data_dir, augment=False)
    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Enable augmentation on train subset by wrapping it
    train_ds.dataset.augment = False  # we wrap manually below
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Split sizes — Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    A two-layer residual block with batch normalization.
    If in_channels != out_channels, a 1×1 projection shortcut is used.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention.
    Recalibrates channel-wise feature responses adaptively.
    Particularly useful for spectrograms where different frequency bands
    carry phase-specific signatures (e.g., RF energy during ablation).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class EPLabCNN(nn.Module):
    """
    Residual CNN for EP lab procedural phase classification from log-mel spectrograms.

    Architecture overview:
        Stem conv (7×7, stride 2)
        → 4 residual stages with channel doubling
        → Channel attention (SE) after each stage
        → Global average pooling
        → Dropout + FC classifier

    Input:  (B, 1, n_mels, T)  — single-channel spectrogram
    Output: (B, NUM_CLASSES)   — raw logits
    """

    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        dropout: float = Config.DROPOUT,
        base_channels: int = 32,
    ):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ───────────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels,      base_channels,      stride=1),
            ResidualBlock(base_channels,      base_channels,      stride=1),
        )
        self.se1 = ChannelAttention(base_channels)

        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels,      base_channels * 2,  stride=2),
            ResidualBlock(base_channels * 2,  base_channels * 2,  stride=1),
        )
        self.se2 = ChannelAttention(base_channels * 2)

        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 2,  base_channels * 4,  stride=2),
            ResidualBlock(base_channels * 4,  base_channels * 4,  stride=1),
        )
        self.se3 = ChannelAttention(base_channels * 4)

        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 4,  base_channels * 8,  stride=2),
            ResidualBlock(base_channels * 8,  base_channels * 8,  stride=1),
        )
        self.se4 = ChannelAttention(base_channels * 8)

        # ── Classifier head ───────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_channels * 8, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv layers; bias=0 for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.stem(x)            # (B, 32, n_mels/4, T/4)

        x = self.stage1(x)
        x = self.se1(x)             # (B, 32, ...)

        x = self.stage2(x)
        x = self.se2(x)             # (B, 64, ...)

        x = self.stage3(x)
        x = self.se3(x)             # (B, 128, ...)

        x = self.stage4(x)
        x = self.se4(x)             # (B, 256, ...)

        x = self.global_pool(x)     # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.dropout(x)
        return self.classifier(x)   # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Convenience wrapper for inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training if validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in tqdm(loader, desc="  Train", leave=False):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)
    return total_loss / total, correct / total


def train(
    data_dir: str = Config.DATA_DIR,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    device: str = Config.DEVICE,
    checkpoint_dir: str = Config.CHECKPOINT_DIR,
) -> EPLabCNN:
    """
    Full training routine.

    1. Build dataloaders
    2. Instantiate model, loss, optimizer, scheduler
    3. Train with early stopping
    4. Save best checkpoint
    5. Return trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(Config.SEED)

    train_loader, val_loader, _ = build_dataloaders(data_dir=data_dir)

    model = EPLabCNN().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Training on: {device}\n")

    # Class-weighted cross entropy to handle potential phase imbalance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    early_stopping = EarlyStopping(patience=8)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} | "
            f"LR: {lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_acc": val_acc},
                Config.BEST_MODEL_PATH,
            )
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Save training history
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, checkpoint_dir)

    # Reload best weights
    ckpt = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nBest model from epoch {ckpt['epoch']} loaded (val_acc={ckpt['val_acc']:.3f}).")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = Config.DEVICE,
    phase_names: List[str] = Config.PHASE_NAMES,
) -> None:
    """Print classification report and plot confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    for specs, labels in test_loader:
        specs = specs.to(device)
        preds = model(specs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=phase_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=phase_names, yticklabels=phase_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("EP Lab Phase Classification — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("checkpoints/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to checkpoints/confusion_matrix.png")


def visualize_spectrogram(
    audio_path: str,
    predicted_label: Optional[int] = None,
    phase_names: List[str] = Config.PHASE_NAMES,
) -> None:
    """Plot the log-mel spectrogram of an audio file with an optional phase label."""
    audio = load_audio(audio_path)
    clips = slice_audio_to_clips(audio)
    spec = compute_log_mel_spectrogram(clips[0])

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spec,
        sr=Config.SAMPLE_RATE,
        hop_length=Config.HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=Config.FMIN,
        fmax=Config.FMAX,
    )
    plt.colorbar(format="%+2.0f dB")
    title = "Log-Mel Spectrogram"
    if predicted_label is not None:
        title += f" — Predicted: {phase_names[predicted_label]}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class EPLabClassifier:
    """
    High-level inference wrapper for deployment.

    Usage:
        classifier = EPLabClassifier("checkpoints/best_ep_lab_cnn.pt")
        results = classifier.predict_file("patient_recording.wav")
        for phase, confidence, start_time in results:
            print(f"  {start_time:.1f}s → {phase} ({confidence:.1%})")
    """

    def __init__(self, checkpoint_path: str, device: str = Config.DEVICE):
        self.device = device
        self.model = EPLabCNN().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Loaded model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")

    def predict_file(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """
        Predict phase labels across a full recording.

        Returns a list of (phase_name, confidence, start_time_seconds) tuples,
        one per clip window.
        """
        audio = load_audio(audio_path)
        clips = slice_audio_to_clips(audio)
        results = []
        hop = Config.HOP_DURATION

        for i, clip in enumerate(clips):
            spec = compute_log_mel_spectrogram(clip)
            spec = normalize_spectrogram(spec)
            tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(tensor)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            phase_name = Config.PHASE_NAMES[pred_idx]
            start_time = i * hop
            results.append((phase_name, confidence, start_time))

        return results

    def predict_clip(self, audio_clip: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict phase label for a single pre-sliced audio clip (numpy array).
        Useful for real-time streaming inference.

        Returns (phase_name, confidence, all_probabilities).
        """
        spec = compute_log_mel_spectrogram(audio_clip)
        spec = normalize_spectrogram(spec)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(tensor)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        return Config.PHASE_NAMES[pred_idx], float(probs[pred_idx]), probs


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EP Lab Phase CNN — Train or Infer")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--data_dir", default=Config.DATA_DIR)
    parser.add_argument("--checkpoint", default=Config.BEST_MODEL_PATH)
    parser.add_argument("--audio_file", default=None, help="Path to WAV file for inference")
    args = parser.parse_args()

    if args.mode == "train":
        print("=" * 60)
        print("  EP Lab Procedural Phase CNN — Training")
        print("=" * 60)
        model = train(data_dir=args.data_dir)

        # Final evaluation on held-out test set
        _, _, test_loader = build_dataloaders(data_dir=args.data_dir)
        evaluate_test_set(model, test_loader)

    elif args.mode == "eval":
        print("Evaluating model on test set...")
        model = EPLabCNN().to(Config.DEVICE)
        ckpt = torch.load(args.checkpoint, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        _, _, test_loader = build_dataloaders(data_dir=args.data_dir)
        evaluate_test_set(model, test_loader)

    elif args.mode == "infer":
        if args.audio_file is None:
            print("Please provide --audio_file for inference mode.")
        else:
            classifier = EPLabClassifier(args.checkpoint)
            results = classifier.predict_file(args.audio_file)
            print(f"\nPhase predictions for: {args.audio_file}")
            print(f"{'Time (s)':<12} {'Phase':<28} {'Confidence'}")
            print("-" * 55)
            for phase, conf, t in results:
                bar = "█" * int(conf * 20)
                print(f"{t:<12.1f} {phase:<28} {conf:.1%}  {bar}")