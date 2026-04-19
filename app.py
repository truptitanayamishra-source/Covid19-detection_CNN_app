import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "best_model_deep_cnn.h5"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please add best_model_deep_cnn.h5 to repo.")
        st.stop()

    return load_model(model_path)

# Class labels
class_names = ["COVID-19", "Normal", "Viral Pneumonia"]

# -------------------------------
# UI
# -------------------------------
st.title("🩺 COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to get prediction")

# Load model
model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to array
    img = np.array(image)

    # Handle grayscale / RGBA
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Resize + normalize
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display results
    st.subheader("Prediction:")
    st.write(f"**{class_names[pred_class]}**")

    st.subheader("Confidence:")
    st.write(f"{confidence:.2%}")
