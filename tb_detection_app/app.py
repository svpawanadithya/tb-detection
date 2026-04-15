import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page title and layout
st.set_page_config(page_title="TB Detection using CNN", layout="centered")

st.title("🩺 Tuberculosis Bacilli Detection")
st.write("Upload a chest X-ray image to detect whether it indicates Tuberculosis or not.")

# Load trained model
@st.cache_resource
def load_tb_model():
    model = load_model('tb_detector_model.h5')
    return model

model = load_tb_model()

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded X-ray", use_column_width=True)
    
    # Preprocess image
    img = image_data.resize((224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    label = "🦠 TB Positive" if prediction[0][0] > 0.5 else "✅ TB Negative"
    confidence = round(float(prediction[0][0])*100, 2) if prediction[0][0] > 0.5 else round(100 - float(prediction[0][0])*100, 2)

    st.subheader("Prediction Result:")
    st.write(f"**{label}** with {confidence}% confidence.")

    if "Positive" in label:
        st.warning("⚠️ Consult a medical professional for further examination.")
    else:
        st.success("✅ No Tuberculosis detected in the image.")

st.markdown("---")
st.caption("Developed by [Your Name] | Department of ECE | Using DenseNet CNN Architecture")
