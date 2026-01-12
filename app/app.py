import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "model/image_classifier.keras"
CLASS_NAMES = ["cats", "dogs", "horses", "unknown"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🐾 Multi-Class Image Classification")
st.write("Classes: Cats | Dogs | Horses | Unknown")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    st.subheader("Prediction Result")
    st.write(f"**Class:** {predicted_class.upper()}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Probabilities")
    for cls, prob in zip(CLASS_NAMES, predictions):
        st.write(f"{cls}: {prob*100:.2f}%")
