import streamlit as st
import pandas as pd
import pickle

# 1. Configure the page
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# 2. Load the trained model
# Using @st.cache_resource ensures the model is only loaded once, improving performance
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. Build the UI Header
st.title("🏥 Diabetes Prediction Dashboard")
st.markdown("""
Welcome to the predictive diagnostic tool. Please enter the patient's medical details below to determine the likelihood of diabetes. 
*This tool is for educational purposes and should not replace professional medical advice.*
""")
st.divider()

# 4. Create input columns for a clean interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics & History")
    age = st.number_input("Age (Years)", min_value=0, max_value=120, value=25, step=1)
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.00, max_value=3.00, value=0.50, step=0.01)

with col2:
    st.subheader("Clinical Measurements")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100, step=1)
    blood_pressure = st.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=150, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79, step=1)

st.divider()

# 5. Prediction Logic
if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    # Constructing a DataFrame ensures feature names match what the model was trained on
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, pedigree, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Optional: Get probabilities if your model supports it
    try:
        probability = model.predict_proba(input_data)[0]
        prob_positive = round(probability[1] * 100, 2)
    except:
        prob_positive = None

    # 6. Display Results
    st.subheader("Diagnostic Result:")
    
    if prediction == 1:
        if prob_positive:
            st.error(f"⚠️ **High Risk Detected:** The model indicates a positive result for diabetes with a {prob_positive}% probability.")
        else:
            st.error("⚠️ **High Risk Detected:** The model indicates a positive result for diabetes.")
        st.info("Recommendation: Please consult with a healthcare provider for a comprehensive evaluation.")
    else:
        if prob_positive:
            st.success(f"✅ **Low Risk:** The model indicates a negative result for diabetes (Probability: {prob_positive}%).")
        else:
            st.success("✅ **Low Risk:** The model indicates a negative result for diabetes.")
        st.balloons()
