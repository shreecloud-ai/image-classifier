import torch
from src.dataset import get_data_loaders
from src.resnet_model import ResNet18Transfer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

_, test_loader, classes = get_data_loaders()

model = ResNet18Transfer(freeze_backbone=False)
checkpoint = torch.load("checkpoints/best_resnet18_clean.pth", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

print("Running final evaluation on test set...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

print("\n" + "="*70)
print(f"FINAL TEST EVALUATION - ResNet-18")
print("="*70)
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Best Validation during training: 95.46%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Confusion Matrix - ResNet-18 (Test Accuracy: {accuracy:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\n✅ Evaluation complete!")
