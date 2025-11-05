# streamlit run app/gui_app.py

import streamlit as st # pyright: ignore[reportMissingImports]
import os
import tempfile
from PIL import Image
from predictor import predict_all # pyright: ignore[reportMissingImports]

# === Streamlit Page Config ===
st.set_page_config(page_title="Nationality & Emotion Detector", layout="centered")

st.title("🧠 Nationality-Based Emotion Analyzer")
st.markdown("Upload a face image to predict nationality, emotion, age, and dress color (based on rules).")

# === Upload Section ===
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        file_path = tmp_file.name
        tmp_file.write(uploaded_file.read())

    # Show preview
    img = Image.open(file_path)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("🔍 Predict"):
        with st.spinner("Processing..."):
            results = predict_all(file_path)

        if "error" in results:
            st.error(results["error"])
        else:
            st.subheader("🔎 Prediction Results")
            st.markdown(f"**Nationality**: {results['nationality']}")
            st.markdown(f"**Emotion**: {results['emotion']}")
            if results["age"] != "N/A":
                st.markdown(f"**Age**: {results['age']} years")
            if results["dress_color"] != "N/A":
                st.markdown(f"**Dress Color**: {results['dress_color'].title()}")

    # Clean up temp file
    os.remove(file_path)
