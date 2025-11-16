# cross_validate.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import transforms, datasets
from baseline_model import BaselineDeepfakeDetector
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import datetime

def run_cross_validation():
    # Best hyperparameters
    best_config = {
        'lr': 0.001,
        'batch_size': 32,
        'weight_decay': 1e-05,
        'epochs': 5
    }

    print("Running 5-Fold Cross-Validation with best hyperparameters:")
    print(best_config)

    # Transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load train and val datasets separately
    train_ds = datasets.ImageFolder('data/processed/train', transform=transform)
    val_ds   = datasets.ImageFolder('data/processed/val', transform=transform)
    # Concatenate them
    full_ds = ConcatDataset([train_ds, val_ds])

    # Prepare targets for kfold
    targets = [label for _, label in train_ds.samples] + [label for _, label in val_ds.samples]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(targets)):
        print(f"\nFold {fold+1}/5")

        # Samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler   = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_ds, batch_size=best_config['batch_size'],
                                  sampler=train_sampler, num_workers=0)
        val_loader   = DataLoader(full_ds, batch_size=best_config['batch_size'],
                                  sampler=val_sampler,   num_workers=0)

        # Model
        model = BaselineDeepfakeDetector(pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=best_config['lr'],
                                weight_decay=best_config['weight_decay'])

        best_val_acc = 0.0

        for epoch in range(best_config['epochs']):
            # Train
            model.train()
            train_correct = train_total = 0
            for inputs, targets_batch in train_loader:
                inputs, targets_batch = inputs.to(device), targets_batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets_batch)
                loss.backward()
                optimizer.step()
                preds = outputs.argmax(dim=1)
                train_correct += (preds == targets_batch).sum().item()
                train_total += targets_batch.size(0)

            # Validate
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for inputs, targets_batch in val_loader:
                    inputs, targets_batch = inputs.to(device), targets_batch.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == targets_batch).sum().item()
                    val_total += targets_batch.size(0)

            train_acc = 100 * train_correct / train_total
            val_acc   = 100 * val_correct   / val_total
            best_val_acc = max(best_val_acc, val_acc)
            print(f"  Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")

        cv_results.append({'fold': fold+1, 'best_val_acc': best_val_acc})

    # Summarize
    accuracies = [r['best_val_acc'] for r in cv_results]
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'best_config': best_config,
        'cv_results': cv_results,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies))
    }

    results_file = 'results/metrics/cross_validation_results.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nCROSS-VALIDATION SUMMARY")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
    print(f"Min Accuracy: {summary['min_accuracy']:.2f}%")
    print(f"Max Accuracy: {summary['max_accuracy']:.2f}%")
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    run_cross_validation()
