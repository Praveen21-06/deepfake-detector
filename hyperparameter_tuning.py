# hyperparameter_tuning.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from baseline_model import BaselineDeepfakeDetector
from tqdm import tqdm
import itertools
import datetime

def run_hyperparam_tuning():
    # Hyperparameter grid
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'batch_size': [8, 16, 32],
        'weight_decay': [1e-3, 1e-4, 1e-5]
    }

    # Data transforms
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Dataset directories
    data_dir = 'data/processed'
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')

    # Results file
    results_file = 'results/metrics/hyperparam_tuning.jsonl'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running hyperparameter tuning on device: {device}")

    # Loop over all hyperparameter combinations
    for lr, batch_size, weight_decay in itertools.product(
        param_grid['lr'], param_grid['batch_size'], param_grid['weight_decay']
    ):
        config = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay}
        print(f"\nTesting config: {config}")

        # Prepare data loaders with single process
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize model, loss, optimizer
        model = BaselineDeepfakeDetector(pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Single training epoch
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total

        # Single validation epoch
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total

        # Record and save results
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Result: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}% saved.")

    print("\nHyperparameter tuning completed. Results saved to:", results_file)

if __name__ == '__main__':
    run_hyperparam_tuning()
