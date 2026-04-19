# Note: Streamlit cannot be run directly in Jupyter notebooks
# To run the web app, use the following command in terminal:
# streamlit run app.py

# The complete Streamlit app code is saved in app.py file
# This cell contains the same code for reference

"""
Streamlit Web App Code (also saved in app.py):

import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
''', unsafe_allow_html=True)

def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

@st.cache_resource
def load_trained_model():
    model_path = "best_model_deep_cnn.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please train and save the model first.")
        st.stop()

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image):
    try:
        img = np.array(image)

        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🩺 COVID-19 Detection from Chest X-ray</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("ℹ️ About")
        st.write('''
        This application uses a deep learning model to detect COVID-19 from chest X-ray images.

        **Model Classes:**
        - COVID-19
        - Normal
        - Viral Pneumonia

        **Note:** This is for educational purposes only and should not replace professional medical diagnosis.
        ''')

        st.header("📋 Instructions")
        st.write('''
        1. Upload a chest X-ray image (JPG, PNG, JPEG)
        2. Wait for the model to process the image
        3. View the prediction and confidence scores
        ''')

    model = load_trained_model()
    class_names = ["COVID-19", "Normal", "Viral Pneumonia"]

    st.write("### 📤 Upload Chest X-ray Image")
    st.write("Please upload a clear chest X-ray image for analysis.")

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "png", "jpeg"],
        help="Upload a chest X-ray image in JPG, PNG, or JPEG format"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="📊 Uploaded X-ray Image", use_container_width=True)

            processed_img = preprocess_image(image)

            if processed_img is not None:
                with st.spinner("🔍 Analyzing image... Please wait."):
                    predictions = model.predict(processed_img, verbose=0)
                    pred_class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])

                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("📋 Diagnosis:")

                pred_class = class_names[pred_class_idx]
                confidence_color = get_confidence_color(confidence)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"### {pred_class}")
                with col2:
                    st.markdown(f'<p class="{confidence_color}">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                if confidence >= 0.8:
                    st.success("🟢 High confidence prediction")
                elif confidence >= 0.6:
                    st.warning("🟡 Moderate confidence - Consider professional medical opinion")
                else:
                    st.error("🔴 Low confidence - Please consult a medical professional")

                st.subheader("📊 Detailed Probabilities")
                st.write("Probability distribution across all classes:")

                prob_data = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
                st.bar_chart(prob_data)

                st.write("**Raw Probabilities:**")
                for i, cls in enumerate(class_names):
                    st.write(f"- {cls}: {predictions[0][i]:.4f}")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.write("Please try uploading a different image or contact support if the problem persists.")

    else:
        st.info("👆 Please upload a chest X-ray image to get started.")
        st.markdown('''
        ### 📸 Sample Image
        For testing purposes, you can use any chest X-ray image. The model has been trained to classify:
        - **COVID-19**: Characteristic patterns associated with COVID-19 infection
        - **Normal**: Healthy lung appearance
        - **Viral Pneumonia**: Other viral pneumonia patterns
        ''')

if __name__ == "__main__":
    main()
"""

print("✅ Streamlit web app code has been implemented!")
print("📁 The complete app is saved in 'app.py'")
print("🚀 To run the web app, execute in terminal:")
print("   streamlit run app.py")
print("")
print("🌐 The app will be available at: http://localhost:8501")
print("")
print("📋 Features implemented:")
print("   ✓ File upload for X-ray images")
print("   ✓ Image preprocessing and validation")
print("   ✓ Real-time prediction with confidence scores")
print("   ✓ Professional UI with medical theme")
print("   ✓ Error handling and user feedback")
print("   ✓ Probability distribution charts")
print("   ✓ Responsive design for mobile/desktop")
