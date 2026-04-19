import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("best_model_deep_cnn.h5")

classes = ["COVID-19", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    st.write("Prediction:", classes[class_idx])
    st.write("Confidence:", float(np.max(pred)))
