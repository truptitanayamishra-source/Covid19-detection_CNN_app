import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

class_names = ["Covid", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to get prediction")

# File uploader
uploaded_file = st.file_uploader("https://www.dreamstime.com/royalty-free-stock-image-chest-ray-image-image28672896", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Output
    st.subheader("Prediction:")
    st.write(f"**{class_names[pred_class]}**")

    st.subheader("Confidence:")
    st.write(f"{confidence:.2f}")

    # Show probabilities
    st.subheader("Class Probabilities:")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[0][i]:.4f}")
