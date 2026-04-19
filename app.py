import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
@st.cache_resource
def load_trained_model():
    return load_model("best_model_deep_cnn.h5")

class_names = ["Covid", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to get prediction")

try:
    model = load_trained_model()
    
    # File uploader - FIXED: Changed URL to proper label
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])
    image_url = st.text_input("https://upload.wikimedia.org/wikipedia/commons/8/8e/Chest_Xray_PA_3-8-2010.png")

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
        # Preprocess image
        img = np.array(image)
        
        # Handle both grayscale and color images
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
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
        st.write(f"{confidence:.2%}")
    
        # Show probabilities
        st.subheader("Class Probabilities:")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[0][i]:.4f}")

except FileNotFoundError:
    st.error("❌ Model file 'best_model_deep_cnn.h5' not found. Please train and save the model first.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
