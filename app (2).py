import streamlit as st
import numpy as np
import joblib
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# Model download from Google Drive
MODEL_PATH = "eurosat_fnn_model.keras"
MODEL_FILE_ID = "1RYTIwhBuLaXTXVWk-Vo7Fve5__MkJ-57"  # <-- REPLACE with your file ID
MODEL_URL = f"https://drive.google.com/file/d/1RYTIwhBuLaXTXVWk-Vo7Fve5__MkJ-57/view?usp=drive_link"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model and class labels
model = load_model(MODEL_PATH)
class_indices = joblib.load("class_indices.pkl")  # Must be in repo or loaded the same way
class_names = list(class_indices.keys())  # Or .values() depending on your pkl

# Streamlit UI
st.set_page_config(page_title="EuroSAT Classifier", layout="centered")
st.title("🌍 EuroSAT Satellite Image Classification")

uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((64, 64))
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
