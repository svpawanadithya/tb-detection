import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv

# -------------------------------
# 1. Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. Model setup
# -------------------------------
from torchvision.models import DenseNet121_Weights
model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # 2 classes: Normal, Tuberculosis
model.load_state_dict(torch.load("best_densenet_tb.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ['Normal', 'Tuberculosis']

# -------------------------------
# 3. Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Test folder path
# -------------------------------
test_folder = r"C:\Users\Asus\Desktop\archive\dataset\test"
test_images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not test_images:
    print("No images found in", test_folder)
    exit()

# -------------------------------
# 5. CSV file path
# -------------------------------
csv_file = os.path.join(test_folder, "predictions.csv")

# -------------------------------
# 6. Predict each image
# -------------------------------
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Prediction"])

    for img_file in test_images:
        img_path = os.path.join(test_folder, img_file)
        image = Image.open(img_path).convert('RGB')
        input_img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]

        # Save result to CSV
        writer.writerow([img_file, predicted_class])

        # Show result in window
        plt.imshow(image)
        plt.title(f"{img_file} -> Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()

        print(f"{img_file}: Prediction -> {predicted_class}")

print(f"\nAll predictions saved to {csv_file}")
