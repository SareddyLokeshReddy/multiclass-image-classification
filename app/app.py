import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# CONFIG
# =============================
MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses", "unknown"]

ANIMAL_CONFIDENCE_THRESHOLD = 0.55
UNKNOWN_CONFIDENCE_THRESHOLD = 0.60

# =============================
# PAGE SETUP
# =============================
st.set_page_config(
    page_title="Animal Image Classification",
    page_icon="🐾",
    layout="centered"
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =============================
# UI
# =============================
st.markdown(
    """
    <h1 style='text-align:center;'>🐾 Animal Image Classification</h1>
    <p style='text-align:center; font-size:18px;'>
        Classes: <b>Cat</b>, <b>Dog</b>, <b>Horse</b>, <b>Unknown</b><br>
        Unknown is predicted only when the image does not match animals confidently.
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"]
)

# =============================
# PREDICTION
# =============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = preds[predicted_index]

    animal_probs = preds[:3]
    max_animal_prob = np.max(animal_probs)

    st.markdown("## 🔍 Prediction Result")

    # =============================
    # FINAL DECISION LOGIC
    # =============================
    if predicted_class == "unknown":
        if confidence >= UNKNOWN_CONFIDENCE_THRESHOLD:
            st.error("❌ **UNKNOWN OBJECT DETECTED**")
            st.write(f"Confidence: **{confidence*100:.2f}%**")
        else:
            st.warning("⚠️ **LOW CONFIDENCE – REVIEW IMAGE**")

    else:
        if max_animal_prob >= ANIMAL_CONFIDENCE_THRESHOLD:
            st.success(f"✅ **Predicted Class: {predicted_class.upper()}**")
            st.write(f"Confidence: **{max_animal_prob*100:.2f}%**")
        else:
            st.error("❌ **UNKNOWN OBJECT DETECTED**")
            st.write(f"Confidence: **{max_animal_prob*100:.2f}%**")

    # =============================
    # PROBABILITIES
    # =============================
    st.markdown("### 📊 Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, preds):
        st.write(f"- **{cls.capitalize()}**: {prob*100:.2f}%")

# =============================
# FOOTER
# =============================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
        AI/ML Project using CNN, TensorFlow, Keras & Streamlit 🚀
    </p>
    """,
    unsafe_allow_html=True
)
