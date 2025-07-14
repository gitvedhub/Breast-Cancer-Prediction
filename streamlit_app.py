import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Load the trained model
with open('breast_cancer.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and display header image
image = Image.open('R (1).jpeg')
st.image(image, use_container_width=True, caption="🎗️ Breast Cancer Awareness")

# Title and intro
st.title("🧬 Breast Cancer Risk Prediction")
st.markdown("""
Enter diagnostic parameters below to predict whether a tumor is **benign** or **malignant**.  
Each feature includes a reference range based on typical clinical data.
""")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    texture_mean = st.number_input("Texture Mean", format="%.3f", help="Typical range: 9.0 – 40.0")
    smoothness_mean = st.number_input("Smoothness Mean", format="%.5f", help="Typical range: 0.05 – 0.15")
    compactness_mean = st.number_input("Compactness Mean", format="%.5f", help="Typical range: 0.02 – 0.35")
    concavity_se = st.number_input("Concavity SE", format="%.5f", help="Typical range: 0.0 – 0.4")
    concave_points_mean = st.number_input("Concave Points Mean", format="%.5f", help="Typical range: 0.0 – 0.2")
    symmetry_mean = st.number_input("Symmetry Mean", format="%.5f", help="Typical range: 0.15 – 0.3")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", format="%.5f", help="Typical range: 0.05 – 0.1")
    texture_se = st.number_input("Texture SE", format="%.3f", help="Typical range: 0.3 – 5.0")
    smoothness_se = st.number_input("Smoothness SE", format="%.5f", help="Typical range: 0.002 – 0.02")
    compactness_se = st.number_input("Compactness SE", format="%.5f", help="Typical range: 0.01 – 0.15")
    concavity_worst = st.number_input("Concavity Worst", format="%.5f", help="Typical range: 0.0 – 1.5")

with col2:
    concave_points_worst = st.number_input("Concave Points Worst", format="%.5f", help="Typical range: 0.0 – 0.6")
    symmetry_worst = st.number_input("Symmetry Worst", format="%.5f", help="Typical range: 0.2 – 0.5")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", format="%.5f", help="Typical range: 0.05 – 0.25")
    area_se = st.number_input("Area SE", format="%.3f", help="Typical range: 5 – 550")
    smoothness_worst = st.number_input("Smoothness Worst", format="%.5f", help="Typical range: 0.1 – 0.22")
    compactness_worst = st.number_input("Compactness Worst", format="%.5f", help="Typical range: 0.02 – 1.6")
    symmetry_se = st.number_input("Symmetry SE", format="%.5f", help="Typical range: 0.005 – 0.08")
    fractal_dimension_se = st.number_input("Fractal Dimension SE", format="%.5f", help="Typical range: 0.001 – 0.03")
    concave_points_se = st.number_input("Concave Points SE", format="%.5f", help="Typical range: 0.005 – 0.07")
    area_worst = st.number_input("Area Worst", format="%.3f", help="Typical range: 150 – 2500")
    texture_worst = st.number_input("Texture Worst", format="%.3f", help="Typical range: 10.0 – 50.0")

# Prediction button
if st.button("🔍 Predict"):
    input_data = np.array([[texture_mean, smoothness_mean, compactness_mean, concavity_se,
                            concave_points_mean, symmetry_mean, fractal_dimension_mean,
                            texture_se, smoothness_se, compactness_se, concavity_worst,
                            concave_points_worst, symmetry_worst, fractal_dimension_worst,
                            area_se, smoothness_worst, compactness_worst, symmetry_se,
                            fractal_dimension_se, concave_points_se, area_worst, texture_worst]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Prediction: Malignant (High Risk of Breast Cancer)")
    else:
        st.success("✅ Prediction: Benign (Low Risk)")

# Footer
st.markdown("---")
st.caption("🔬 Developed by Vedika Shinde | Based on SVM Model | Streamlit UI with Reference Ranges")
