import torch
from baseline_model import BaselineDeepfakeDetector

MODEL_PATH = "models/baseline_best.pth"

try:
    # Initialize model architecture (same as training)
    model = BaselineDeepfakeDetector(pretrained=False)
    
    # Load your trained weights
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"Failed to load model: {e}")
