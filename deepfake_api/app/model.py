import io
import time
import base64
from PIL import Image
import torch
from torchvision import transforms
from baseline_model import BaselineDeepfakeDetector

# Labels
LABELS = ["real", "fake"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model using the same architecture as training
MODEL_PATH = "models/baseline_best.pth"
model = BaselineDeepfakeDetector(pretrained=False).to(device)

# Load the trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Handle different checkpoint formats (same as your training scripts)
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

# Preprocessing pipeline (same as your validation transform)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_image(base64_str: str):
    # Decode base64
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    # Inference
    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    latency = time.time() - start

    return {
        "label": LABELS[idx.item()],
        "confidence": confidence.item(),
        "latency": latency
    }
