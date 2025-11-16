from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import io
from PIL import Image
import numpy as np

# --- NEW IMPORTS for Your Model ---
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. --- IMPORT YOUR MODEL CLASS ---
# We copy-paste the class definition from baseline_model.py
class BaselineDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(BaselineDeepfakeDetector, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
# --- END OF MODEL CLASS ---


# 2. --- DEFINE MODEL/PREPROCESSING ---

# !! THIS IS THE LINE I UPDATED FOR YOU !!
MODEL_PATH = "models/checkpoints/baseline_best.pth"
IMAGE_SIZE = 224

# These are the standard normalization values for EfficientNet
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# !! IMPORTANT: Verify this list matches your training.
# (In 'preprocess_images.py', 'real' and 'fake' are listed.
# If you used ImageFolder, it sorts alphabetically, so 'fake'=0, 'real'=1)
CLASS_NAMES = ["FAKE", "REAL"] 


def load_my_model():
    """Loads and returns the trained PyTorch model."""
    try:
        # Use 'cpu' for inference, as it's more stable for APIs
        device = torch.device("cpu") 
        model = BaselineDeepfakeDetector(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_bytes):
    """
    Validates, resizes, and transforms image bytes into a model-ready tensor.
    This logic is adapted from your 'validate_and_resize_image' function.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Validation logic from your script
        if img.size[0] < 64 or img.size[1] < 64:
            raise ValueError("Image is too small (< 64x64)")
            
        arr = np.array(img)
        if np.std(arr) < 10:
            raise ValueError("Image has low variance (likely blank)")
        
        # Apply the transforms (resize, ToTensor, normalize)
        tensor = data_transform(img)
        
        # Add a batch dimension (model expects [batch_size, C, H, W])
        tensor = tensor.unsqueeze(0) 
        
        return tensor
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def run_prediction(model, processed_image_tensor):
    """Runs the model and returns (label, confidence)."""
    with torch.no_grad():
        outputs = model(processed_image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get the top probability and its index
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get the class name
        prediction_label = CLASS_NAMES[predicted_idx.item()]
        
        return (prediction_label, confidence.item())

# --- END OF MODEL/PREPROCESSING ---


# 3. --- INITIALIZE APP AND MODEL ---
app = FastAPI(title="Deepfake Detector API")
model = load_my_model()

# 4. --- API ENDPOINTS ---
@app.get("/")
def read_root():
    if model is None:
        return {"error": "Model failed to load. Please check API server logs."}
    return {"message": "Welcome! The Deepfake Detector API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an uploaded file, runs it through the model,
    and returns a prediction.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # 1. Read and preprocess the file
    image_bytes = await file.read()
    processed_image_tensor = preprocess_image(image_bytes)
    
    if processed_image_tensor is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    # 2. Run the prediction
    try:
        prediction, confidence = run_prediction(model, processed_image_tensor)
        
        # 3. Return the result
        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# 5. --- RUN THE APP ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)