import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Transforms
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet mean
                         [0.229, 0.224, 0.225]) # Imagenet std
])

# =====================
# Dataset Path (update if needed)
# =====================
dataset_path = r"C:\Users\Asus\Desktop\archive\TB_Chest_Radiography_Database"

# =====================
# Load Dataset
# =====================
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = dataset.classes
print("✅ Dataset Loaded Successfully!")
print(f"Classes found: {class_names}")
print(f"Total Images: {len(dataset)}")

# =====================
# Train / Val / Test Split
# =====================
train_size = int(0.7 * len(dataset))   # 70%
val_size   = int(0.15 * len(dataset))  # 15%
test_size  = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =====================
# Show first sample
# =====================
sample_img, sample_label = next(iter(train_loader))
print(f"First image shape: {sample_img[0].shape}, Label: {class_names[sample_label[0]]}")
