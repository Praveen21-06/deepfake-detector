# adversarial_robustness.py

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from baseline_model import BaselineDeepfakeDetector
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import datetime

# Adversarial transforms
def add_gaussian_noise(img, std=0.1):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 1) * 255.0
    return Image.fromarray(noisy.astype(np.uint8))

def add_gaussian_blur(img, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius))

def jpeg_compress(img, quality=25):
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and weights from checkpoint path
model = BaselineDeepfakeDetector(pretrained=False).to(device)
checkpoint_path = 'models/checkpoints/baseline_best.pth'  # Updated checkpoint filename
checkpoint = torch.load(checkpoint_path, map_location=device)

# Inspect checkpoint keys to determine how to load state_dict
if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {checkpoint.keys()}")
    # Attempt to load model weights from common keys
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume entire checkpoint is a state_dict
        model.load_state_dict(checkpoint)
else:
    # Checkpoint is raw state dict
    model.load_state_dict(checkpoint)

model.eval()

# Load validation set WITHOUT transforms to get PIL images
raw_val_ds = datasets.ImageFolder('data/processed/val', transform=None)

# Base transform applied AFTER adversarial perturbation
base_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define perturbations
perturbations = {
    'none': lambda img: img,
    'gaussian_noise_std0.05': lambda img: add_gaussian_noise(img, std=0.05),
    'gaussian_noise_std0.1': lambda img: add_gaussian_noise(img, std=0.1),
    'gaussian_blur_r2': lambda img: add_gaussian_blur(img, radius=2),
    'gaussian_blur_r5': lambda img: add_gaussian_blur(img, radius=5),
    'jpeg_q50': lambda img: jpeg_compress(img, quality=50),
    'jpeg_q25': lambda img: jpeg_compress(img, quality=25),
}

results = {}

for name, perturb in perturbations.items():
    correct = total = 0
    for idx in tqdm(range(len(raw_val_ds)), desc=f"Testing {name}"):
        pil_img, label = raw_val_ds[idx]
        adv_img = perturb(pil_img)

        inp = base_tf(adv_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            pred = out.argmax(dim=1).item()

        correct += (pred == label)
        total += 1

    acc = 100 * correct / total
    results[name] = acc
    print(f"{name}: {acc:.2f}%")

# Save results
out_dict = {
    'timestamp': datetime.datetime.now().isoformat(),
    'results': results
}
os.makedirs('results/metrics', exist_ok=True)
with open('results/metrics/adversarial_robustness.json', 'w') as f:
    json.dump(out_dict, f, indent=2)

print("\nAdversarial robustness testing complete. Results saved to results/metrics/adversarial_robustness.json")
