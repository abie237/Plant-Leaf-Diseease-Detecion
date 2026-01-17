import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_inceptionV3.h5")
    return model

model = load_model()

# ---------------------------
# Class names
# ---------------------------
CLASS_NAMES = ["Healthy", "RedRot", "RedRust"]

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_image(image):
    image = image.resize((224, 224))   # FIXED — matches model input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ---------------------------
# Simple leaf detection
# (Checks if image is too plain or not greenish)
# ---------------------------
def detect_leaf(image):
    img_np = np.array(image)

    # Criterion 1: Image must not be too plain
    std_dev = np.std(img_np)
    if std_dev < 10:
        return False  # too flat to contain a leaf

    # Criterion 2: must have some green-ish pixels
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    green_pixels = np.sum((g > r) & (g > b))

    if green_pixels < 500:
        return False

    return True


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌿 Plant Leaf Disease Detector (InceptionV3)")
st.write("Upload a leaf image and the model will classify it as Healthy, RedRot, or RedRust.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Analyzing image...")

    # 1️⃣ Check if image contains a leaf
    if not detect_leaf(image):
        st.error("❌ No leaf detected in the image. Please upload an image containing a leaf.")
    else:
        st.success("✔ Leaf detected!")

        # 2️⃣ Preprocess and classify
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        confidence = np.max(predictions)
        class_index = np.argmax(predictions)
        class_name = CLASS_NAMES[class_index]

        # 3️⃣ Results
        st.subheader("🌱 Prediction Results")
        st.write(f"*Class:* {class_name}")
        st.write(f"*Confidence:* {confidence * 100:.2f}%")