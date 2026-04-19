import streamlit as st
import numpy as np
import cv2
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
from PIL import Image
@st.cache_resource
def load_trained_model():
    return load_model("best_model_deep_cnn.h5")

# Class labels
class_names = ["Covid", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image OR use the sample/test image")

try:
    model = load_trained_model()

    # Upload option
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

    # ✅ Your GitHub image link used correctly here
    default_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0140673620303706-fx1_lrg.jpg"

    image_url = st.text_input("Or paste image URL here", value=default_url)

    image = None

    # Load from upload
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    # Load from URL
    elif image_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(image_url, headers=headers)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("Invalid image URL")

    # -------------------------------
    # Prediction
    # -------------------------------
    if image is not None:

        st.image(image, caption="Input Image", use_container_width=True)

        img = np.array(image)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        pred_class = np.argmax(predictions)
        confidence = np.max(predictions)

        st.subheader("Prediction:")
        st.write(f"**{class_names[pred_class]}**")

        st.subheader("Confidence:")
        st.write(f"{confidence:.2%}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
