import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses", "unknown"]

# Thresholds (VERY IMPORTANT)
PRIMARY_THRESHOLD = 0.55     # for known classes
FALLBACK_THRESHOLD = 0.40    # minimum acceptance

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Multi-Class Image Classification", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🐾 Multi-Class Image Classification</h1>
    <p style='text-align: center;'>Cats | Dogs | Horses | Unknown</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION LOGIC
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (MUST match training)
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Model Prediction
    predictions = model.predict(img)[0]
    max_prob = np.max(predictions)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]

    # ==============================
    # UNKNOWN HANDLING (FIXED)
    # ==============================
    KNOWN_CLASSES = ["cats", "dogs", "horses"]

    if predicted_class in KNOWN_CLASSES and max_prob >= FALLBACK_THRESHOLD:
        final_class = predicted_class
    else:
        final_class = "unknown"

    # ==============================
    # DISPLAY RESULT
    # ==============================
    st.markdown("## 🧠 Prediction Result")

    if final_class == "unknown":
        st.error("❌ **UNKNOWN OBJECT DETECTED**")
    else:
        st.success(f"✅ **{final_class.upper()}** detected")

    st.write(f"**Confidence:** {max_prob * 100:.2f}%")

    # ==============================
    # SHOW PROBABILITIES (DEBUG + INTERVIEW PROOF)
    # ==============================
    st.markdown("### 📊 Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.write(f"- **{cls}** : {prob * 100:.2f}%")

# ==============================
# FOOTER
# ==============================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built using TensorFlow, Keras & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
