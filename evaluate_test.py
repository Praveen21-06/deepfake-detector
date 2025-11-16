# evaluate_test.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from baseline_model import BaselineDeepfakeDetector
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load model
    model = BaselineDeepfakeDetector(pretrained=False)
    model.load_state_dict(torch.load('models/checkpoints/baseline_best.pth', map_location=device))
    model.to(device).eval()

    # Prepare test data
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_ds = datasets.ImageFolder('data/processed/test', transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print("\\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    main()
