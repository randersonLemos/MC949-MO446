import torch
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

import numpy as np
from ultralytics import YOLO
import cv2

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLO model
model_path = "yolo11n.pt"  # change to your model path
model = YOLO(model_path)
model.to(device)

# Load an image
img = np.empty((300, 300, 3))
if img is None:
    raise ValueError(f"Image not found: {image_path}")

# Run inference
results = model(img, device=device)  # ensure device is set

# Print results
print(results)
