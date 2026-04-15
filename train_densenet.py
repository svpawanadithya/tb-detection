import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import os

# -----------------------
# Config
# -----------------------
data_dir = "dataset_split/train"   # ✅ balanced dataset split
val_dir = "dataset_split/val"
batch_size = 32
num_epochs = 20
learning_rate = 0.001
save_path = "best_densenet_tb.pth"
metrics_dir = "metrics_out"

# Create metrics directory if not exists
os.makedirs(metrics_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚡ Training on CPU")

# -----------------------
# Data Transforms
# -----------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# Dataset + Loaders
# -----------------------
train_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Model
# -----------------------
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)
model = model.to(device)

# -----------------------
# Class Weights
# -----------------------
class_counts = Counter([label for _, label in train_dataset.samples])
print("Class Counts:", class_counts)

weights = 1.0 / torch.tensor([class_counts[0], class_counts[1]], dtype=torch.float)
weights = weights / weights.sum()
print("Class Weights:", weights)

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# Training Loop
# -----------------------
best_acc = 0.0
train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / total

    # Validate
    model.eval()
    val_correct, val_total, val_loss_total = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss_total / val_total

    # Save histories
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best model saved (Val Acc: {val_acc*100:.2f}%)")

print("Training complete. Best Val Accuracy: {:.2f}%".format(best_acc*100))

# -----------------------
# Plot Accuracy & Loss
# -----------------------
epochs_range = range(1, num_epochs+1)

plt.figure(figsize=(10, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_history, label='Train Acc')
plt.plot(epochs_range, val_acc_history, label='Val Acc')
plt.title('Accuracy : Training vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_history, label='Train Loss')
plt.plot(epochs_range, val_loss_history, label='Val Loss')
plt.title('Loss : Training vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(metrics_dir, "training_curves.png"))
plt.show()
