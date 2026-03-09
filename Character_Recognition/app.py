import streamlit as st
import numpy as np
from PIL import Image
import pickle

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✏️",
    layout="centered"
)

# ----------------- UI STYLE -----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f9;
    font-family: 'Segoe UI', sans-serif;
}
.main-card {
    background-color: white;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}
.title {
    font-size: 34px;
    font-weight: 600;
    color: #2c3e50;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 25px;
}
.result-box {
    background-color: #e8f4ff;
    color: #1f4e79;
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: 500;
    text-align: center;
    margin-top: 20px;
}
.stButton>button {
    background-color: #4a90e2;
    color: white;
    border-radius: 8px;
    padding: 8px 25px;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background-color: #357abd;
}
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
try:
    mlp_model = pickle.load(open("mlp_emnist_model.pkl", "rb"))
except:
    st.error("❌ Model file not found! Train the model first.")
    st.stop()

# ----------------- UI -----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">✏️ Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a 28×28 handwritten letter image (A–Z)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if st.button("Predict Character"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        # -------- IMAGE PREPROCESSING --------
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))

        image_array = np.array(image)

        # Invert if background is white
        if np.mean(image_array) > 127:
            image_array = 255 - image_array

        # Normalize (same as training)
        image_array = image_array / 255.0

        # EMNIST rotation fix
        image_array = np.rot90(image_array, k=3)
        image_array = np.fliplr(image_array)

        image_array = image_array.reshape(1, -1)

        # -------- PREDICTION --------
        prediction = mlp_model.predict(image_array)[0]
        probabilities = mlp_model.predict_proba(image_array)[0]

        predicted_letter = chr(prediction + 64)

        # -------- DISPLAY RESULT --------
        st.markdown(
            f'<div class="result-box">Predicted Character: {predicted_letter}</div>',
            unsafe_allow_html=True
        )

        # -------- CONFIDENCE --------
        st.subheader("Prediction Confidence")
        for i, prob in enumerate(probabilities):
            letter = chr(i + 65)
            st.write(letter)
            st.progress(float(prob))

st.markdown('</div>', unsafe_allow_html=True)