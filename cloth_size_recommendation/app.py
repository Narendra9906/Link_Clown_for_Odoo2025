import streamlit as st
from PIL import Image
from fit_predictor import predict_fit

st.set_page_config(page_title="ReWear Fit Predictor", layout="centered")

st.title("👕 ReWear – AI Fit Prediction")
st.markdown("Upload your photo and a garment photo – AI will check fit!")

st.header("1️⃣ Upload Garment Image")
garment_file = st.file_uploader("Garment Photo", type=["jpg", "jpeg", "png"])

st.header("2️⃣ Upload Your Full Body Photo")
user_file = st.file_uploader("Your Photo", type=["jpg", "jpeg", "png"])

st.header("3️⃣ Garment Chest Size (inches)")
garment_chest_inch = st.number_input("Enter Garment Chest Width (in)", min_value=30, max_value=60, value=38)

if st.button("🔍 Predict Fit"):
    if garment_file and user_file:
        garment_img = Image.open(garment_file).convert("RGB")
        user_img = Image.open(user_file).convert("RGB")

        result = predict_fit(user_img, garment_img, garment_chest_inch)
        st.success(f"{result['fit']}")
        st.info(f"Similarity Score: {result['similarity']:.4f} {result['info']}")
        st.image(garment_img, caption="👕 Garment Image", use_column_width=True)
        st.image(user_img, caption="🧍 Your Image", use_column_width=True)
    else:
        st.warning("Please upload both images.")