"""
PFA EP Lab Phase Classification — Transfer Learning with Pretrained CNN14 (PANNs)
==================================================================================
Uses CNN14 pretrained on AudioSet (2M clips, 527 classes) as a feature extractor,
then fine-tunes a new classification head for Pulse-Field Ablation (PFA) procedure
phases in Electrophysiology labs.

Why CNN14 / PANNs?
  - Pretrained on AudioSet: broad acoustic event coverage including electrical
    equipment, alarms, speech, and environmental sounds — all present in EP labs
  - Spectrogram-native: the model was designed for log-mel spectrograms, so the
    feature representations transfer well without architecture changes
  - Strong transfer learning benchmark across medical audio tasks

PFA Procedure Phases:
  0 - Room Setup & Equipment Check
  1 - Patient Prep & Sedation
  2 - Vascular Access (femoral sheath insertion)
  3 - Transseptal Puncture
  4 - 3D Electroanatomical Mapping
  5 - PFA Energy Delivery (pulse-field application)
  6 - Post-Ablation Verification & Pacing
  7 - Sheath Removal & Haemostasis
  8 - Recovery & Handover

Strategy:
  Phase 1 — Feature extraction: freeze CNN14 backbone, train only the new head
             on whatever labeled data you have (even synthetic/simulated)
  Phase 2 — Fine-tuning: unfreeze deeper layers with a very small LR to adapt
             the representations to EP lab acoustics

Dependencies:
  pip install torch torchaudio librosa numpy scikit-learn matplotlib seaborn tqdm
  pip install panns-inference          # PANNs pretrained models from Qiuqiang Kong et al.
"""

import os
import json
import random
import warnings
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
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Audio ─────────────────────────────────────────────────────────────────
    # CNN14 was trained at 32 kHz — we match that for best transfer
    SAMPLE_RATE: int = 32000
    CLIP_DURATION: float = 5.0       # seconds per window
    HOP_DURATION: float = 2.5        # seconds between windows (50% overlap)

    # CNN14 spectrogram settings (must match pretraining exactly)
    N_FFT: int = 1024
    HOP_LENGTH: int = 320            # 10 ms at 32 kHz (standard for AudioSet)
    N_MELS: int = 64                 # CNN14 uses 64 mel bands
    FMIN: float = 50.0
    FMAX: float = 14000.0
    TOP_DB: float = 80.0

    # ── PFA Phases ────────────────────────────────────────────────────────────
    NUM_CLASSES: int = 9
    PHASE_LABELS: Dict[str, int] = {
        "room_setup":            0,
        "patient_prep":          1,
        "vascular_access":       2,
        "transseptal_puncture":  3,
        "mapping":               4,
        "pfa_delivery":          5,
        "verification_pacing":   6,
        "sheath_removal":        7,
        "recovery":              8,
    }
    PHASE_NAMES: List[str] = [
        "Room Setup",
        "Patient Prep",
        "Vascular Access",
        "Transseptal Puncture",
        "3D Mapping",
        "PFA Delivery",
        "Verification / Pacing",
        "Sheath Removal",
        "Recovery",
    ]

    # ── Transfer Learning Strategy ────────────────────────────────────────────
    # Phase 1: train head only (backbone frozen)
    PHASE1_EPOCHS: int = 20
    PHASE1_LR: float = 1e-3

    # Phase 2: unfreeze deeper layers, fine-tune end-to-end
    PHASE2_EPOCHS: int = 30
    PHASE2_LR: float = 5e-5          # much smaller to avoid catastrophic forgetting
    UNFREEZE_FROM_LAYER: str = "conv_block6"  # unfreeze this layer and everything after

    # ── General Training ──────────────────────────────────────────────────────
    BATCH_SIZE: int = 32
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.4
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.10
    SEED: int = 42

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = "data/pfa_audio"
    CHECKPOINT_DIR: str = "checkpoints"
    BEST_MODEL_PATH: str = "checkpoints/best_pfa_cnn14.pt"

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN14 BACKBONE  (PANNs — AudioSet pretrained)
# ─────────────────────────────────────────────────────────────────────────────
# We replicate the CNN14 architecture here so we can load pretrained weights
# and modify the classifier head without depending on the full PANNs repo.
# Weights are downloaded from the official PANNs release on Zenodo.

class ConvBlock(nn.Module):
    """Standard double-conv block used throughout CNN14."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type="avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


class CNN14Backbone(nn.Module):
    """
    CNN14 feature extractor as defined in:
        Kong et al. (2020) "PANNs: Large-Scale Pretrained Audio Neural Networks
        for Audio Pattern Recognition." IEEE/ACM TASLP.

    Input:  (B, 1, 64, T)  — single-channel log-mel spectrogram
    Output: (B, 2048)      — global embedding vector

    The original 527-class AudioSet head is removed. We attach our own
    PFA-specific classification head instead.
    """

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, F, T)  where F=64 mel bands
        returns: (B, 2048) embedding
        """
        # Batch-norm input along frequency axis (matches PANNs training)
        x = x.squeeze(1).transpose(1, 2)   # (B, T, F)
        x = self.bn0(x.unsqueeze(1).transpose(2, 3))  # (B, 1, F, T)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        # Global average + max pooling → concatenate
        x1 = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        x2 = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        x = x1 + x2
        x = x.view(x.size(0), -1)             # (B, 2048)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))              # (B, 2048)
        return x


def load_cnn14_pretrained_weights(
    backbone: CNN14Backbone,
    weights_path: Optional[str] = None,
    auto_download: bool = True,
) -> CNN14Backbone:
    """
    Load AudioSet-pretrained CNN14 weights into the backbone.

    If weights_path is None and auto_download is True, the weights are
    downloaded from the official PANNs Zenodo release (~200 MB).

    The original classifier head keys (fc_audioset.*) are skipped
    automatically since our backbone has no such layer.
    """
    ZENODO_URL = (
        "https://zenodo.org/record/3987831/files/"
        "Cnn14_mAP%3D0.431.pth?download=1"
    )
    DEFAULT_CACHE = os.path.join(
        torch.hub.get_dir(), "panns", "Cnn14_mAP=0.431.pth"
    )

    if weights_path is None:
        if auto_download:
            if not os.path.exists(DEFAULT_CACHE):
                print(f"Downloading CNN14 pretrained weights from Zenodo…")
                os.makedirs(os.path.dirname(DEFAULT_CACHE), exist_ok=True)
                torch.hub.download_url_to_file(ZENODO_URL, DEFAULT_CACHE)
                print(f"Saved to {DEFAULT_CACHE}")
            weights_path = DEFAULT_CACHE
        else:
            print("[WARN] No weights path provided and auto_download=False. "
                  "Backbone will be randomly initialized.")
            return backbone

    checkpoint = torch.load(weights_path, map_location="cpu")
    # PANNs checkpoints store weights under "model" key
    state_dict = checkpoint.get("model", checkpoint)

    # Filter out keys that belong to the original 527-class head
    backbone_keys = set(backbone.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in backbone_keys}

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    print(f"Pretrained weights loaded — "
          f"matched: {len(filtered)}, "
          f"missing: {len(missing)}, "
          f"unexpected (skipped): {len(unexpected)}")
    return backbone


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL MODEL  (CNN14 backbone + PFA head)
# ─────────────────────────────────────────────────────────────────────────────

class PFAPhaseClassifier(nn.Module):
    """
    CNN14 backbone + lightweight PFA-specific classification head.

    The head uses two fully-connected layers with batch normalisation and
    dropout to adapt the 2048-dimensional AudioSet embeddings to our 9 PFA phases.

    Input:  (B, 1, 64, T)   log-mel spectrogram
    Output: (B, NUM_CLASSES) raw logits
    """

    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        dropout: float = Config.DROPOUT,
        pretrained_weights: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = CNN14Backbone()
        self.backbone = load_cnn14_pretrained_weights(
            self.backbone,
            weights_path=pretrained_weights,
            auto_download=auto_download,
        )

        # ── PFA Classification Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Initialize only the new head; backbone keeps pretrained weights
        self._init_head()

        # Start with backbone frozen (Phase 1 training)
        self.freeze_backbone()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Freeze / Unfreeze API ─────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1: head-only training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Training head only.")

    def unfreeze_from(self, layer_name: str = Config.UNFREEZE_FROM_LAYER) -> None:
        """
        Unfreeze backbone layers from `layer_name` onward (Phase 2: fine-tuning).

        Layers earlier than layer_name remain frozen to preserve low-level
        AudioSet features and reduce the risk of catastrophic forgetting.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then unfreeze from the target layer onward
        reached = False
        for name, module in self.backbone.named_children():
            if name == layer_name:
                reached = True
            if reached:
                for param in module.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Unfrozen backbone from '{layer_name}' onward. "
              f"Trainable backbone params: {trainable:,}")

    def unfreeze_all(self) -> None:
        """Unfreeze the entire backbone (use with a very small LR)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Full backbone unfrozen.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 2048)
        return self.head(features)    # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = Config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute log-mel spectrogram matching CNN14 pretraining settings.
    Returns array of shape (64, T) in dB.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmin=Config.FMIN,
        fmax=Config.FMAX,
    )
    log_mel = librosa.power_to_db(mel, top_db=Config.TOP_DB)
    return log_mel  # (64, T)


def slice_audio(audio: np.ndarray) -> List[np.ndarray]:
    """Slice a recording into fixed-length overlapping clips."""
    clip_len = int(Config.CLIP_DURATION * Config.SAMPLE_RATE)
    hop_len = int(Config.HOP_DURATION * Config.SAMPLE_RATE)
    clips, start = [], 0
    while start < len(audio):
        clip = audio[start : start + clip_len]
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))
        clips.append(clip)
        start += hop_len
    return clips


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Per-clip mean/std normalization."""
    return (spec - spec.mean()) / (spec.std() + 1e-8)


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Convert (64, T) numpy array to (1, 64, T) float32 tensor."""
    return torch.tensor(normalize_spectrogram(spec), dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET  (with SpecAugment)
# ─────────────────────────────────────────────────────────────────────────────

class PFADataset(Dataset):
    """
    Loads audio clips from a directory tree:
        data/pfa_audio/
            pfa_delivery/
                case_001_ablation_segment.wav
            mapping/
                case_002_mapping.wav
            ...

    Each clip is a 5-second window. If a WAV file is longer it is automatically
    sliced with 50% overlap. Label is inferred from the subfolder name.

    NOTE ON DATA COLLECTION:
    Even 5–10 minutes of labeled audio per phase (30–60 clips) is enough to
    fine-tune the head in Phase 1. For Phase 2 fine-tuning aim for 100+ clips
    per phase. You can annotate recordings from real cases by timestamping phases
    in a spreadsheet, then use FFmpeg to extract the segments.
    """

    def __init__(
        self,
        data_dir: str = Config.DATA_DIR,
        augment: bool = False,
    ):
        self.augment = augment
        self.samples: List[Tuple[torch.Tensor, int]] = []
        self._build(data_dir)

    def _build(self, data_dir: str) -> None:
        root = Path(data_dir)
        for phase_name, label in Config.PHASE_LABELS.items():
            phase_dir = root / phase_name
            if not phase_dir.exists():
                print(f"  [WARN] Missing phase directory: {phase_dir}")
                continue
            files = list(phase_dir.glob("*.wav")) + list(phase_dir.glob("*.mp3"))
            for f in files:
                try:
                    audio = load_audio(str(f))
                    for clip in slice_audio(audio):
                        spec = compute_log_mel_spectrogram(clip)
                        self.samples.append((spec_to_tensor(spec), label))
                except Exception as e:
                    print(f"  [ERROR] {f.name}: {e}")
        print(f"Dataset: {len(self.samples)} clips | "
              f"{len(Config.PHASE_LABELS)} phases")

    # ── SpecAugment augmentations ─────────────────────────────────────────────

    def _time_mask(self, t: torch.Tensor, max_w: int = 40) -> torch.Tensor:
        _, _, T = t.shape
        w = random.randint(0, min(max_w, T // 4))
        s = random.randint(0, max(0, T - w))
        t = t.clone(); t[:, :, s:s + w] = 0.0
        return t

    def _freq_mask(self, t: torch.Tensor, max_w: int = 15) -> torch.Tensor:
        _, F, _ = t.shape
        w = random.randint(0, min(max_w, F // 4))
        s = random.randint(0, max(0, F - w))
        t = t.clone(); t[:, s:s + w, :] = 0.0
        return t

    def _gaussian_noise(self, t: torch.Tensor, std: float = 0.015) -> torch.Tensor:
        return t + torch.randn_like(t) * std

    def _time_shift(self, t: torch.Tensor, max_shift: int = 20) -> torch.Tensor:
        """Roll the spectrogram along the time axis."""
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(t, shift, dims=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spec, label = self.samples[idx]
        if self.augment:
            if random.random() > 0.5: spec = self._time_mask(spec)
            if random.random() > 0.5: spec = self._freq_mask(spec)
            if random.random() > 0.4: spec = self._gaussian_noise(spec)
            if random.random() > 0.6: spec = self._time_shift(spec)
        return spec, label


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(Config.SEED)
    full_ds = PFADataset(augment=False)
    n = len(full_ds)
    n_test = int(n * Config.TEST_SPLIT)
    n_val = int(n * Config.VAL_SPLIT)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(Config.SEED)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)
    # Wrap train split in an augmenting dataset view
    train_ds.dataset.augment = False  # base; we apply augmentation below via wrapper
    kw = dict(num_workers=2, pin_memory=(Config.DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, **kw)
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best, self.should_stop = 0, float("inf"), False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best - self.min_delta:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def one_epoch_train(model, loader, opt, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def run_training_phase(
    model: PFAPhaseClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    phase_label: str,
    history: Dict,
    best_val_loss: float,
    device: str,
) -> Tuple[float, Dict]:
    """Generic training loop used for both Phase 1 and Phase 2."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)
    stopper = EarlyStopping(patience=7)

    print(f"\n{'─'*60}")
    print(f"  {phase_label}  |  trainable params: {model.trainable_params():,}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = one_epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "val_loss": val_loss, "val_acc": val_acc},
                Config.BEST_MODEL_PATH,
            )
            flag = "  ✓ saved"

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
              f"{flag}")

        stopper.step(val_loss)
        if stopper.should_stop:
            print(f"  Early stop at epoch {epoch}.")
            break

    return best_val_loss, history


# ─────────────────────────────────────────────────────────────────────────────
# 7. TWO-PHASE TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def train_pfa_classifier(
    pretrained_weights: Optional[str] = None,
    auto_download: bool = True,
) -> PFAPhaseClassifier:
    """
    Run the full two-phase transfer learning routine.

    Phase 1 — Head only (fast convergence, few epochs needed)
        The backbone is frozen. Only the 3-layer classification head is trained.
        This quickly adapts the AudioSet embeddings to PFA phase discrimination
        without risk of overfitting or distorting pretrained features.

    Phase 2 — Fine-tuning deeper layers (optional, needs more data)
        Unfreezes conv_block6 and fc1 in the backbone with a very small LR.
        This lets the model adapt mid/high-level feature detectors to EP lab
        acoustics (RF generator hum, PFA pulse signatures, alarm patterns).
    """
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    torch.manual_seed(Config.SEED)
    device = Config.DEVICE

    train_loader, val_loader, test_loader = build_dataloaders()

    model = PFAPhaseClassifier(
        pretrained_weights=pretrained_weights,
        auto_download=auto_download,
    ).to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    # ── Phase 1: Head only ────────────────────────────────────────────────────
    best_val_loss, history = run_training_phase(
        model, train_loader, val_loader,
        epochs=Config.PHASE1_EPOCHS,
        lr=Config.PHASE1_LR,
        phase_label="PHASE 1 — Head-only Training",
        history=history,
        best_val_loss=best_val_loss,
        device=device,
    )

    # ── Phase 2: Fine-tune deeper layers ──────────────────────────────────────
    model.unfreeze_from(Config.UNFREEZE_FROM_LAYER)
    best_val_loss, history = run_training_phase(
        model, train_loader, val_loader,
        epochs=Config.PHASE2_EPOCHS,
        lr=Config.PHASE2_LR,
        phase_label="PHASE 2 — Fine-Tuning (conv_block6 + head)",
        history=history,
        best_val_loss=best_val_loss,
        device=device,
    )

    # ── Save history and plots ────────────────────────────────────────────────
    with open(os.path.join(Config.CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, Config.CHECKPOINT_DIR)

    # ── Reload best checkpoint ────────────────────────────────────────────────
    ckpt = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nBest model reloaded  val_loss={ckpt['val_loss']:.4f}  "
          f"val_acc={ckpt['val_acc']:.3f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    evaluate_test_set(model, test_loader, device)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION & VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: Dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    phase1_end = Config.PHASE1_EPOCHS

    for ax, key_train, key_val, title in [
        (axes[0], "train_loss", "val_loss", "Loss"),
        (axes[1], "train_acc",  "val_acc",  "Accuracy"),
    ]:
        ax.plot(history[key_train], label="Train", color="#0D9488")
        ax.plot(history[key_val],   label="Val",   color="#F59E0B")
        ax.axvline(phase1_end, color="#6B7280", linestyle="--", alpha=0.6,
                   label="Phase 1→2 boundary")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("PFA Phase Classifier — Training Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curves saved → {path}")


@torch.no_grad()
def evaluate_test_set(model, test_loader, device=Config.DEVICE) -> None:
    model.eval()
    all_preds, all_labels = [], []
    for x, y in test_loader:
        preds = model(x.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    print("\n── Classification Report ──────────────────────────────────────────")
    print(classification_report(
        all_labels, all_preds, target_names=Config.PHASE_NAMES, digits=3
    ))

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=Config.PHASE_NAMES,
        yticklabels=Config.PHASE_NAMES,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("PFA Phase Classification — Normalised Confusion Matrix", fontsize=13)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    path = os.path.join(Config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


def visualize_spectrogram_with_prediction(
    audio_path: str,
    model: Optional[PFAPhaseClassifier] = None,
    device: str = Config.DEVICE,
) -> None:
    """Plot spectrogram of a file's first clip, optionally with model prediction."""
    audio = load_audio(audio_path)
    clip = slice_audio(audio)[0]
    spec = compute_log_mel_spectrogram(clip)

    fig, axes = plt.subplots(1, 2 if model else 1, figsize=(14 if model else 7, 4))
    ax_spec = axes[0] if model else axes

    librosa.display.specshow(
        spec, sr=Config.SAMPLE_RATE, hop_length=Config.HOP_LENGTH,
        x_axis="time", y_axis="mel", fmin=Config.FMIN, fmax=Config.FMAX, ax=ax_spec,
    )
    ax_spec.set_title("Log-Mel Spectrogram (first 5 s clip)")
    plt.colorbar(ax_spec.collections[0], ax=ax_spec, format="%+2.0f dB")

    if model:
        tensor = spec_to_tensor(spec).unsqueeze(0).to(device)
        probs = model.predict_proba(tensor)[0].cpu().numpy()
        ax_bar = axes[1]
        colors = ["#0D9488" if i != probs.argmax() else "#F59E0B"
                  for i in range(len(probs))]
        ax_bar.barh(Config.PHASE_NAMES, probs, color=colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Confidence")
        ax_bar.set_title(f"Predicted: {Config.PHASE_NAMES[probs.argmax()]} "
                         f"({probs.max():.1%})")
        ax_bar.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. REAL-TIME INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PFARealTimeClassifier:
    """
    Deployment wrapper for sliding-window real-time inference.

    Usage:
        clf = PFARealTimeClassifier("checkpoints/best_pfa_cnn14.pt")

        # Full file prediction with timeline
        timeline = clf.predict_file("case_042.wav")
        for phase, conf, t in timeline:
            print(f"{t:6.1f}s  {phase:<28}  {conf:.1%}")

        # Single 5-second audio chunk (real-time streaming)
        phase, conf, probs = clf.predict_chunk(audio_chunk_np)
    """

    def __init__(self, checkpoint_path: str, device: str = Config.DEVICE):
        self.device = device
        self.model = PFAPhaseClassifier(auto_download=False).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Model loaded  val_acc={ckpt.get('val_acc', '?'):.3f}")

    def predict_file(self, path: str) -> List[Tuple[str, float, float]]:
        audio = load_audio(path)
        clips = slice_audio(audio)
        results = []
        for i, clip in enumerate(clips):
            phase, conf, _ = self.predict_chunk(clip)
            results.append((phase, conf, i * Config.HOP_DURATION))
        return results

    def predict_chunk(
        self, audio_clip: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """Predict phase for a single audio array of length CLIP_DURATION * SR."""
        spec = compute_log_mel_spectrogram(audio_clip)
        t = spec_to_tensor(spec).unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(t)[0].cpu().numpy()
        idx = int(probs.argmax())
        return Config.PHASE_NAMES[idx], float(probs[idx]), probs

    def print_timeline(self, path: str) -> None:
        results = self.predict_file(path)
        print(f"\n{'Time':>8}  {'Phase':<28}  {'Conf':>6}  Bar")
        print("─" * 65)
        for phase, conf, t in results:
            bar = "█" * int(conf * 24) + "░" * (24 - int(conf * 24))
            print(f"{t:>7.1f}s  {phase:<28}  {conf:>5.1%}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PFA EP Lab Phase Classifier")
    subparsers = parser.add_subparsers(dest="command")

    # -- train --
    t_parser = subparsers.add_parser("train", help="Run two-phase transfer learning")
    t_parser.add_argument("--data_dir", default=Config.DATA_DIR)
    t_parser.add_argument("--weights", default=None,
                          help="Path to CNN14 pretrained .pth (auto-downloaded if omitted)")
    t_parser.add_argument("--no_download", action="store_true",
                          help="Disable auto-download of pretrained weights")

    # -- infer --
    i_parser = subparsers.add_parser("infer", help="Predict phases for a WAV file")
    i_parser.add_argument("audio_file", help="Path to .wav recording")
    i_parser.add_argument("--checkpoint", default=Config.BEST_MODEL_PATH)

    args = parser.parse_args()

    if args.command == "train":
        Config.DATA_DIR = args.data_dir
        train_pfa_classifier(
            pretrained_weights=args.weights,
            auto_download=not args.no_download,
        )

    elif args.command == "infer":
        clf = PFARealTimeClassifier(args.checkpoint)
        clf.print_timeline(args.audio_file)

    else:
        parser.print_help()