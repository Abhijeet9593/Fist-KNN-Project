import streamlit as st
import pickle
import numpy as np

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="wide"
)

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #2C3E50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #566573;
    }
    .prediction-box {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- LOAD MODEL -------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------- HEADER -------------------
st.markdown('<div class="title">🤖 Machine Learning Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the details in sidebar and get prediction instantly 🚀</div>', unsafe_allow_html=True)
st.write("")

# ------------------- SIDEBAR INPUTS -------------------
st.sidebar.header("📌 Input Features")

feature1 = st.sidebar.number_input("Feature 1", value=0.0)
feature2 = st.sidebar.number_input("Feature 2", value=0.0)
feature3 = st.sidebar.number_input("Feature 3", value=0.0)
feature4 = st.sidebar.number_input("Feature 4", value=0.0)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict Now")

# ------------------- MAIN LAYOUT -------------------
col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("📊 Entered Feature Values")
    st.write("Here are the values you entered:")

    st.table({
        "Feature": ["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
        "Value": [feature1, feature2, feature3, feature4]
    })

with col2:
    st.subheader("🎯 Prediction Result")

    if predict_btn:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)[0]

        st.markdown(
            f"<div class='prediction-box'>Prediction: {prediction}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("👈 Enter values from sidebar and click **Predict Now**")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    "<center>✨ Built with Streamlit | Deployed ML Model App 🚀</center>",
    unsafe_allow_html=True
)
