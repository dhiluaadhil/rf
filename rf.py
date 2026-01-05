import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Cancer Risk Predictor", page_icon="🎗️")
st.title("🎗️ Cancer Risk Prediction (Random Forest)")

# Load Model
model_path = 'cancer_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)

    st.write("Enter the details below to predict the likelihood of a cancer diagnosis.")

    # Form for inputs based on your CSV columns
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
            
        with col2:
            gen_risk = st.selectbox("Genetic Risk (0=Low, 1=Medium, 2=High)", [0, 1, 2])
            phys_act = st.number_input("Physical Activity (hours/week)", 0.0, 168.0, 3.0)
            alcohol = st.number_input("Alcohol Intake (units/week)", 0.0, 100.0, 2.0)
            history = st.selectbox("Personal History of Cancer (0=No, 1=Yes)", [0, 1])

        submit = st.form_submit_button("Predict Risk")

    if submit:
        # Arrange features in the exact order the model was trained on
        features = np.array([[age, gender, bmi, smoking, gen_risk, phys_act, alcohol, history]])
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.error("🚨 **High Risk:** The model predicts a positive diagnosis.")
        else:
            st.success("✅ **Low Risk:** The model predicts a negative diagnosis.")
else:
    st.warning("Please upload 'cancer_model.pkl' to your GitHub repo.")
