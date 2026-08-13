import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Pizza Classifier", layout="centered")

# Title
st.title("🍕 Pizza Classification App")
st.write("Upload an image to classify if it's pizza or not pizza!")

# Load model
@st.cache_resource
def load_trained_model():
    model_path = "SRC/mlp_model.keras"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("Model file 'mlp_model.keras' not found!")
        return None

# Image preprocessing
def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to target size
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize pixel values (assuming model was trained with normalized data)
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Load model
model = load_trained_model()

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader("Prediction")
            
            # Preprocess and predict
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)
            confidence = prediction[0][0]
            
            # Determine class (assuming binary classification: pizza=1, not_pizza=0)
            if confidence > 0.5:
                predicted_class = "🍕 PIZZA"
                confidence_percent = confidence * 100
            else:
                predicted_class = "❌ NOT PIZZA"
                confidence_percent = (1 - confidence) * 100
            
            # Display results
            st.metric("Classification", predicted_class)
            st.metric("Confidence", f"{confidence_percent:.2f}%")
            
            # Progress bar
            st.progress(min(confidence, 1.0))
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Created with ❤️ using Streamlit and TensorFlow
        </div>
        """,
        unsafe_allow_html=True,
    )
