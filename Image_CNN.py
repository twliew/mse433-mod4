import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# -------------------------------
# LOAD MODELS
# -------------------------------

# YOLO (object detection)
yolo_model = YOLO("yolov8n.pt")  # lightweight, replace with custom if needed

# V-JEPA 2 (approximate via pretrained ViT backbone)
# NOTE: Official V-JEPA 2 isn't plug-and-play yet, so we simulate using ViT features
from transformers import ViTModel, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vjepa_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# -------------------------------
# CONFIG
# -------------------------------

IMAGE_FOLDER = "ep_lab_images"
MOTION_THRESHOLD = 2.0  # tune this
IDLE_THRESHOLD = 3      # number of frames

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def detect_objects(image):
    """YOLO detection"""
    results = yolo_model(image)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = yolo_model.names[cls]
            detections.append(label)
    
    return detections


def compute_optical_flow(prev_img, curr_img):
    """Motion estimation using Farneback optical flow"""
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)


def classify_phase(image):
    """V-JEPA-style phase classification using embeddings"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = vjepa_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    # Simple rule-based mapping (replace with trained classifier)
    value = embedding.detach().numpy().mean()

    if value < 0.2:
        return "Prep"
    elif value < 0.4:
        return "Mapping"
    elif value < 0.6:
        return "Ablation"
    else:
        return "Closure"


# -------------------------------
# MAIN PIPELINE
# -------------------------------

def process_images(folder_path):
    image_files = sorted(os.listdir(folder_path))

    prev_img = None
    idle_counter = 0
    last_phase = None

    results = []

    for i, file in enumerate(image_files):
        path = os.path.join(folder_path, file)
        image = cv2.imread(path)

        # -----------------------
        # YOLO: Object Detection
        # -----------------------
        detections = detect_objects(image)

        # -----------------------
        # OpenCV: Motion
        # -----------------------
        motion = 0
        if prev_img is not None:
            motion = compute_optical_flow(prev_img, image)

        # Motion intensity
        motion_intensity = motion

        # Idle detection
        if motion < MOTION_THRESHOLD:
            idle_counter += 1
        else:
            idle_counter = 0

        idle = idle_counter >= IDLE_THRESHOLD

        # -----------------------
        # V-JEPA: Phase Detection
        # -----------------------
        phase = classify_phase(image)

        # Phase transition delay
        phase_transition = False
        if last_phase is not None and phase != last_phase:
            phase_transition = True

        # -----------------------
        # STORE RESULTS
        # -----------------------
        results.append({
            "frame": file,
            "detections": detections,
            "motion_intensity": float(motion_intensity),
            "idle": idle,
            "phase": phase,
            "phase_transition": phase_transition
        })

        prev_img = image
        last_phase = phase

    return results


# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    results = process_images(IMAGE_FOLDER)

    for r in results:
        print(r)