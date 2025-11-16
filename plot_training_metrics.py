# plot_training_metrics.py

import matplotlib.pyplot as plt

# Replace these lists with your actual values from training logs
epochs = list(range(1, 11))
train_loss = [0.4037,0.1849,0.1278,0.0868,0.0717,0.0691,0.0557,0.0376,0.0403,0.0391]
val_loss   = [0.1766,0.1467,0.1380,0.1628,0.1208,0.1314,0.1243,0.1147,0.1592,0.1135]
train_acc  = [81.45,92.35,95.15,96.35,97.10,97.45,98.05,98.50,98.50,98.55]
val_acc    = [93.25,94.30,94.30,94.30,94.95,95.20,95.55,95.50,94.95,95.20]

plt.figure(figsize=(12,5))
# Loss subplot
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)

# Accuracy subplot
plt.subplot(1,2,2)
plt.plot(epochs, train_acc, marker='o', label='Train Acc (%)')
plt.plot(epochs, val_acc, marker='o', label='Val Acc (%)')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(epochs)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/plots/training_metrics.png', dpi=300)
plt.show()
