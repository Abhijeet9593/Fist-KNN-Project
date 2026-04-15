import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

st.set_page_config(page_title="Health Prediction App", layout="centered")

st.title("Diabetes Prediction System")
st.write("Enter the following details to check the health status:")

# Create input fields based on the model's features
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

# Prediction Logic
if st.button("Predict"):
    # Arrange inputs in the order the model expects 
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.error("The model predicts a high risk of Diabetes.")
    else:
        st.success("The model predicts a low risk of Diabetes.")
