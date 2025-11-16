# inference_benchmark.py

import time
import torch
from torchvision import datasets, transforms
from baseline_model import BaselineDeepfakeDetector
from tqdm import tqdm

device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model once on desired device
def load_model(device):
    model = BaselineDeepfakeDetector(pretrained=False).to(device)
    checkpoint_path = 'models/checkpoints/baseline_best.pth'  # Update if needed
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# Dataset and transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_ds = datasets.ImageFolder('data/processed/val', transform=transform)

def benchmark_inference(device, batch_size=1):
    model = load_model(device)
    times = []
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, label = val_ds[i]
            inp = img.unsqueeze(0).to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = model(inp)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            times.append(end - start)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    return avg_time, min_time, max_time

if __name__ == "__main__":
    print("Running inference benchmark on CPU...")
    cpu_results = benchmark_inference(device_cpu)
    print(f"CPU Inference Latency:\n Average: {cpu_results[0]:.4f}s, Min: {cpu_results[1]:.4f}s, Max: {cpu_results[2]:.4f}s")

    if torch.cuda.is_available():
        print("\nRunning inference benchmark on GPU...")
        gpu_results = benchmark_inference(device_gpu)
        print(f"GPU Inference Latency:\n Average: {gpu_results[0]:.4f}s, Min: {gpu_results[1]:.4f}s, Max: {gpu_results[2]:.4f}s")
    else:
        print("\nGPU not available; only CPU benchmark performed.")
