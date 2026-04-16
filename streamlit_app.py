import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import DenseNet121_Weights
import numpy as np
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import openpyxl
import os

# -------------------
# Load DenseNet Model
# -------------------
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_densenet_tb.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------
# Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------
# Grad-CAM Function
# -------------------
def generate_gradcam(model, img_tensor, target_class):
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_backward_hook(save_gradient)

    output = model(img_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam

    handle1.remove()
    handle2.remove()
    return cam

# -------------------
# PDF Report
# -------------------
def generate_pdf_report(original_path, gradcam_path, prediction, confidence, stage, precautions):
    pdf_path = "TB_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, height - 50, "Tuberculosis Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Prediction: {prediction}")
    c.drawString(50, height - 120, f"Confidence: {confidence:.2f}%")
    c.drawString(50, height - 140, f"Stage: {stage}")

    # Add images
    c.drawString(50, height - 180, "Uploaded Chest X-ray:")
    c.drawImage(original_path, 50, height - 420, width=200, height=200)

    c.drawString(300, height - 180, "Grad-CAM / Reference Image:")
    c.drawImage(gradcam_path, 300, height - 420, width=200, height=200)

    # Precautions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 460, "Precautions & Responsibilities:")
    c.setFont("Helvetica", 11)
    y = height - 480
    for p in precautions:
        c.drawString(70, y, f"- {p}")
        y -= 20

    c.save()
    return pdf_path

# -------------------
# Excel Logging
# -------------------
def update_excel(file_path, study_id, gender, age, prediction, confidence, stage):
    if os.path.exists(file_path):
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Study_ID", "Gender", "Age", "Prediction", "Confidence", "Stage"])

    sheet.append([study_id, gender, age, prediction, f"{confidence:.2f}%", stage])
    workbook.save(file_path)

# -------------------
# Streamlit UI
# -------------------
st.title("🫁 Tuberculosis Detection with Grad-CAM")
st.write("Upload a Chest X-ray to check TB presence, severity stage, and precautions.")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = "Tuberculosis" if predicted.item() == 1 else "Normal (No TB)"
    confidence = probs[0][predicted.item()].item() * 100

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Stage + precautions
    if prediction == "Tuberculosis":
        if confidence < 65:
            stage = "Mild TB"
            precautions = [
                "Monitor symptoms closely",
                "Maintain a nutritious diet",
                "Consult doctor if symptoms persist"
            ]
        elif confidence < 80:
            stage = "Moderate TB"
            precautions = [
                "Consult a pulmonologist",
                "Adhere to prescribed medications",
                "Avoid close contact to prevent spread"
            ]
        else:
            stage = "Severe TB"
            precautions = [
                "Immediate medical attention required",
                "Strict medication adherence is crucial",
                "Family members should also get screened"
            ]
    else:
        stage = "Normal"
        precautions = [
            "Maintain healthy lifestyle and nutrition",
            "Avoid smoking and alcohol",
            "Regular exercise and adequate sleep",
            "Periodic health check-ups"
        ]

    st.subheader(f"Stage: {stage}")
    st.write("### Precautions & Responsibilities:")
    for p in precautions:
        st.write(f"- {p}")

    # -------------------
    # Grad-CAM handling
    # -------------------
    original_path = "uploaded_xray.jpg"
    image.resize((224, 224)).save(original_path)
    gradcam_path = "gradcam_output.jpg"

    if prediction == "Tuberculosis":
        cam = generate_gradcam(model, img_tensor, predicted.item())
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                                           0.6, heatmap, 0.4, 0)
        cv2.imwrite(gradcam_path, superimposed_img)
        grad_display = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    else:
        img_np = np.array(image.resize((224, 224)))
        blue_map = cv2.applyColorMap(np.uint8(np.zeros((224, 224))), cv2.COLORMAP_COOL)
        superimposed_img = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                                           0.85, blue_map, 0.15, 0)
        cv2.imwrite(gradcam_path, superimposed_img)
        grad_display = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # -------------------
    # Display side-by-side
    # -------------------
    col1, col2 = st.columns(2)
    equal_size = (300, 300)

    with col1:
        st.image(image.resize(equal_size), caption="Uploaded X-ray", use_container_width=False)
    with col2:
        grad_resized = cv2.resize(grad_display, equal_size)
        st.image(grad_resized, caption="Grad-CAM / Reference", use_container_width=False)

    # -------------------
    # Metadata inputs
    # -------------------
    study_id = uploaded_file.name
    gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
    age = st.number_input("Enter Age", min_value=1, max_value=120, value=30)

    # -------------------
    # Update Excel
    # -------------------
    update_excel("tb_results.xlsx", study_id, gender, age, prediction, confidence, stage)
    st.success("✅ Record saved to Excel (tb_results.xlsx)")

    # -------------------
    # PDF Report
    # -------------------
    pdf_path = generate_pdf_report(original_path, gradcam_path, prediction, confidence, stage, precautions)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 Download Report",
            f,
            file_name="TB_Report.pdf",
            mime="application/pdf"
        )
