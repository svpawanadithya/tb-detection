import os
import shutil
import random

# Path to your dataset
data_dir = "dataset/train"   # <- where Normal/ and Tuberculosis/ are present
output_dir = "dataset_split" # <- new folder where split data will be stored

# Train/Val/Test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Classes
classes = ["Normal", "Tuberculosis"]

# Make sure output folders exist
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split function
for cls in classes:
    cls_folder = os.path.join(data_dir, cls)
    files = os.listdir(cls_folder)
    random.shuffle(files)

    total = len(files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(cls_folder, f), os.path.join(output_dir, "train", cls, f))
    for f in val_files:
        shutil.copy(os.path.join(cls_folder, f), os.path.join(output_dir, "val", cls, f))
    for f in test_files:
        shutil.copy(os.path.join(cls_folder, f), os.path.join(output_dir, "test", cls, f))

    print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("✅ Dataset split completed! New folders created in:", output_dir)
