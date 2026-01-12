import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses"]
CONFIDENCE_THRESHOLD = 0.60  # 60%

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("🐾 Multi-Class Image Classification")
st.markdown(
    """
    **Supported Classes:** Cats • Dogs • Horses  
    **Other objects → UNKNOWN**
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)[0]

    max_prob = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))

    # UNKNOWN logic
    if max_prob < CONFIDENCE_THRESHOLD:
        predicted_class = "UNKNOWN"
    else:
        predicted_class = CLASS_NAMES[predicted_index]

    # ---------------- OUTPUT ----------------
    st.subheader("🧠 Prediction Result")
    st.success(f"**Class:** {predicted_class}")
    st.info(f"**Confidence:** {max_prob * 100:.2f}%")

    st.subheader("📊 Raw Probabilities")
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{cls}: {prob * 100:.2f}%")

    if predicted_class == "UNKNOWN":
        st.warning("⚠️ Image does not belong to trained classes")
