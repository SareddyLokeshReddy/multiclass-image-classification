import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses"]
UNKNOWN_THRESHOLD = 0.60  # 60% confidence required

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- UI ----------------
st.set_page_config(page_title="Image Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Multi-Class Image Classification")
st.markdown(
    """
    **Classes Supported:**  
    🐱 Cats | 🐶 Dogs | 🐴 Horses  
    ❌ Any other object → **UNKNOWN**
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload an image (jpg / png / jpeg)",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)[0]
    max_prob = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))

    # Decision logic (IMPORTANT)
    if max_prob < UNKNOWN_THRESHOLD:
        predicted_class = "UNKNOWN"
    else:
        predicted_class = CLASS_NAMES[predicted_index]

    # ---------------- OUTPUT ----------------
    st.subheader("🔍 Prediction Result")

    if predicted_class == "UNKNOWN":
        st.error("❌ **UNKNOWN OBJECT DETECTED**")
        st.write(f"Highest confidence: **{max_prob*100:.2f}%** (Below threshold)")
    else:
        st.success(f"✅ **Predicted Class:** {predicted_class.upper()}")
        st.write(f"Confidence: **{max_prob*100:.2f}%**")

    # ---------------- PROBABILITIES ----------------
    st.subheader("📊 Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{cls}: {prob*100:.2f}%")

    st.caption("Model predicts UNKNOWN if confidence < 60%")
