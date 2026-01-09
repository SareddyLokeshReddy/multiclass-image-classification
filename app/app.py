import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Multi-Class Image Classifier",
    page_icon="🐾",
    layout="centered"
)

MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses", "unknown"]

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e0f7fa, #fff3e0);
}

.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
}

.sub-title {
    font-size: 18px;
    text-align: center;
    color: #555;
}

.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

.class-name {
    font-size: 26px;
    font-weight: bold;
    color: #27ae60;
}

.confidence {
    font-size: 20px;
    color: #2980b9;
}

.prob-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- UI ----------------
st.markdown('<div class="main-title">🐾 Multi-Class Image Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Cats | Dogs | Horses | Unknown</div>', unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="📷 Uploaded Image",
        use_column_width=True
    )

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100
    predicted_class = CLASS_NAMES[predicted_index]

    # Result Section
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")
    st.markdown(
        f'<div class="class-name">Class: {predicted_class.upper()}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Probability Section
    st.subheader("📊 Class Probabilities")
    st.markdown('<div class="prob-box">', unsafe_allow_html=True)
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.progress(float(prob))
        st.write(f"**{cls.capitalize()}** : {prob * 100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Please upload an image to start classification.")
