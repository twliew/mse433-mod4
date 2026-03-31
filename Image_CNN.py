# medical_cnn_multitask.py
"""
Multi-output CNN for medical procedure analysis using pre-trained ResNet
Outputs:
1. Repositioning time (regression)
2. Idle time (regression)
3. Phase transition delay (regression)
4. Procedure phase (classification)
5. Motion intensity (regression)
"""

import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# =========================
# Dataset
# =========================
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, labels_dict, transform=None):
        """
        Args:
            root_dir: folder with images
            labels_dict: dict {image_path: {'reposition_time': float,
                                           'idle_time': float,
                                           'phase_delay': float,
                                           'phase': int,
                                           'motion_intensity': float}}
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = list(labels_dict.items())
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Regression outputs
        reposition_time = torch.tensor(labels['reposition_time'], dtype=torch.float32)
        idle_time = torch.tensor(labels['idle_time'], dtype=torch.float32)
        phase_delay = torch.tensor(labels['phase_delay'], dtype=torch.float32)
        motion_intensity = torch.tensor(labels['motion_intensity'], dtype=torch.float32)
        # Classification output
        phase = torch.tensor(labels['phase'], dtype=torch.long)

        return img, (reposition_time, idle_time, phase_delay, motion_intensity, phase)

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# =========================
# Multi-output model
# =========================
class MultiOutputResNet(nn.Module):
    def __init__(self, num_phases=4):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove last FC
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Output heads
        self.reposition_head = nn.Linear(512, 1)
        self.idle_head = nn.Linear(512, 1)
        self.phase_delay_head = nn.Linear(512, 1)
        self.motion_head = nn.Linear(512, 1)
        self.phase_head = nn.Linear(512, num_phases)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        reposition_time = self.reposition_head(x)
        idle_time = self.idle_head(x)
        phase_delay = self.phase_delay_head(x)
        motion_intensity = self.motion_head(x)
        phase_logits = self.phase_head(x)
        return reposition_time, idle_time, phase_delay, motion_intensity, phase_logits

# =========================
# Example usage
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputResNet().to(device)
model.eval()

# Dummy labels dictionary for demonstration
labels_dict = {
    "dataset/patient_001/frame_0001.png": {
        "reposition_time": 1.2,
        "idle_time": 0.5,
        "phase_delay": 0.3,
        "phase": 2,  # 0: Prep, 1: Mapping, 2: Ablation, 3: Closure
        "motion_intensity": 0.8
    }
}

dataset = MedicalImageDataset("dataset", labels_dict, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Forward pass example
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        print("Outputs:")
        print(f"Repositioning time: {outputs[0].item():.2f}")
        print(f"Idle time: {outputs[1].item():.2f}")
        print(f"Phase transition delay: {outputs[2].item():.2f}")
        print(f"Motion intensity: {outputs[3].item():.2f}")
        print(f"Procedure phase logits: {outputs[4]}")