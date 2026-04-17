import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ML Model Deployment", layout="centered")

st.title("🚀 Streamlit ML Model Deployment")
st.write("Enter the input values below and get prediction from the trained model.")

# Example input fields (change based on your dataset features)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Prediction: {prediction[0]}")
