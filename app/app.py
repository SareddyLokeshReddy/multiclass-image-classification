import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "model/image_classifier.keras"
CONFIDENCE_THRESHOLD = 0.60

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Infer class count safely
MODEL_OUTPUTS = model.output_shape[-1]

# Define class names STRICTLY based on model
BASE_CLASSES = ["cats", "dogs", "horses"]

CLASS_NAMES = BASE_CLASSES[:MODEL_OUTPUTS]

# ---------------- UI ----------------
st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("🐾 Multi-Class Image Classification")
st.write("Cats | Dogs | Horses | Unknown")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    probs = predictions[0]
    max_prob = float(np.max(probs))
    predicted_index = int(np.argmax(probs))

    # ABSOLUTE SAFETY CHECK
    if predicted_index >= len(CLASS_NAMES) or max_prob < CONFIDENCE_THRESHOLD:
        predicted_class = "UNKNOWN"
    else:
        predicted_class = CLASS_NAMES[predicted_index]

    # ---------------- OUTPUT ----------------
    st.subheader("Prediction Result")
    st.success(f"Class: {predicted_class}")
    st.info(f"Confidence: {max_prob * 100:.2f}%")

    st.subheader("Raw Probabilities")
    for i, prob in enumerate(probs):
        label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        st.write(f"{label}: {prob * 100:.2f}%")
