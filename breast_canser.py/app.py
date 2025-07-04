# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import warnings
import os


warnings.filterwarnings("ignore")

# Load model and scaler

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Page Setup ---
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide", initial_sidebar_state="expanded")

# --- CSS Styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #6a1b9a;
            text-align: center;
        }
        .subheading {
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
        .result-success {
            color: #2e7d32;
            font-weight: bold;
        }
        .result-danger {
            color: #c62828;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main Title ---
st.markdown('<div class="main-title">Breast Cancer Prediction Web App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheading">Enter the tumor characteristics below to predict whether it is Malignant (1) or Benign (0).</div>', unsafe_allow_html=True)

# --- Sidebar: App Info + Portfolio ---
with st.sidebar:
    st.title("App Info")
    st.markdown("""
    - **Developer:** Naima Khan  
    - **Model:** Random Forest Classifier  
    - **Input:** 30 tumor features  
    - **Output:** 0 = Benign, 1 = Malignant  
    """)

    if st.button("About"):
        st.info("This AI-powered web app helps in the early detection of breast cancer using machine learning. Built with Python, Scikit-learn, and Streamlit.")

    st.markdown("---")
    st.subheader(" Developer Portfolio")

    # Optional profile picture
    # image = Image.open("your_photo.jpg")
    # st.image(image, width=150, caption="Naima Khan")

    st.markdown("""
    **Naima Khan**  
    AI Engineer & Machine Learning Enthusiast  
    Based in Pakistan 🇵🇰

    **Skills:**  
    - Python, Streamlit, Scikit-learn  
    - Machine Learning, Deep Learning  
    - Data Visualization, Pandas, NumPy

    **Projects:**  
    - [Credit Card Fraud Detection](#)  
    - [Wine Quality Classifier](#)  
    - [Customer Churn Prediction](#)  
    """)

    st.markdown(" **Connect with me:**")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("GitHub", "https://github.com/naima-khan")
    with col2:
        st.link_button("Facebook", "https://facebook.com/ajk.khan")

# --- Feature Names ---
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- Input Form ---
with st.form("input_form"):
    st.subheader("Step 1: Input Tumor Feature Values")
    input_data = []
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            value = st.number_input(f"{feature}", min_value=0.0, value=0.0, format="%.4f")
            input_data.append(value)

    submitted = st.form_submit_button("Step 2: Predict Tumor Type")

# --- Prediction ---
if submitted:
    try:
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][prediction]

        result_text = "Malignant (1)" if prediction == 1 else "Benign (0)"
        result_color = "result-danger" if prediction == 1 else "result-success"

        st.markdown(f'<h3 class="{result_color}">Prediction: {result_text}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p><strong>Model Confidence:</strong> {proba * 100:.2f}%</p>', unsafe_allow_html=True)

        with st.expander("View Your Input Data"):
            df = pd.DataFrame([input_data], columns=feature_names)
            st.dataframe(df.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"An error occurred during prediction:\n\n{e}")



