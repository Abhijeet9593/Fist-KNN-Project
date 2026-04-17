import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Load the model with caching so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App header
st.title("🩺 Diabetes Prediction App")
st.write("""
This application uses a K-Nearest Neighbors machine learning model to predict the likelihood of diabetes based on clinical metrics. 
Please enter the patient details in the sidebar to get a prediction.
""")

# Sidebar inputs
st.sidebar.header("Patient Data Inputs")

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
    glucose = st.sidebar.slider('Glucose (mg/dL)', min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=140, value=70)
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.slider('Insulin (IU/mL)', min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input('BMI (Weight in kg/(Height in m)^2)', min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.000, max_value=3.000, value=0.500, format="%.3f")
    age = st.sidebar.slider('Age (years)', min_value=1, max_value=120, value=30)
    
    # Store a dictionary into a dataframe. The column names MUST match the model's feature names exactly.
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader("Patient Metrics Overview")
st.dataframe(input_df, hide_index=True)

# Predict action
st.subheader("Prediction")
if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(input_df)
    
    # K-Nearest Neighbors can also output prediction probabilities
    prediction_proba = model.predict_proba(input_df)
    
    # Output the results
    if prediction[0] == 1:
        st.error(f"**Result: Diabetic**")
        st.write(f"The model predicts a high likelihood of diabetes with a probability of **{prediction_proba[0][1] * 100:.2f}%**.")
    else:
        st.success(f"**Result: Not Diabetic**")
        st.write(f"The model predicts a low likelihood of diabetes. Confidence level: **{prediction_proba[0][0] * 100:.2f}%**.")
        
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")
