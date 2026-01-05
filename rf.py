import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Cancer Risk Predictor", layout="wide")

st.title("🎗️ Cancer Risk Prediction Dashboard")

model_path = 'cancer_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    with st.sidebar:
        st.header("Patient Data Input")
        age = st.slider("Age", 18, 100, 50)
        gender = st.radio("Gender", ["Female", "Male"])
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        smoking = st.selectbox("Smoker?", ["No", "Yes"])
        gen_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
        phys_act = st.slider("Physical Activity (hrs/week)", 0, 20, 5)
        alcohol = st.slider("Alcohol Units/week", 0, 30, 5)
        history = st.selectbox("Previous Cancer History?", ["No", "Yes"])

    # --- Convert inputs to numbers for the model ---
    gender_num = 0 if gender == "Female" else 1
    smoking_num = 1 if smoking == "Yes" else 0
    history_num = 1 if history == "Yes" else 0
    gen_risk_map = {"Low": 0, "Medium": 1, "High": 2}
    gen_num = gen_risk_map[gen_risk]

    # Create input array
    features = np.array([[age, gender_num, bmi, smoking_num, gen_num, phys_act, alcohol, history_num]])

    if st.button("Run Diagnostic"):
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1] * 100

        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                st.error(f"### Result: High Risk detected")
            else:
                st.success(f"### Result: Low Risk detected")
        
        with col2:
            st.metric("Confidence Level", f"{probability:.2f}%")
            st.progress(probability / 100)

else:
    st.error("Model file not found. Please upload 'cancer_model.pkl' to GitHub.")
