"""
Heart Disease Prediction System - Streamlit app
Loads the SVC model trained in Heart_Disease_Prediction.ipynb and predicts
heart disease from 4 inputs: Age group, Max Heart Rate group, Sex, Exercise Angina.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_heart_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    # same category lists as the notebook, so the numeric codes match training
    return {
        "Age": LabelEncoder().fit(["Young", "Middle", "Senior"]),
        "MaxHeartRate": LabelEncoder().fit(["Low", "Medium", "High"]),
        "Sex": LabelEncoder().fit(["Male", "Female"]),
        "ExerciseAngina": LabelEncoder().fit(["Yes", "No"]),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Select the patient details below and the trained Support Vector Classifier "
    "will predict whether the patient has heart disease. Trained on the UCI Heart "
    "Disease (Cleveland) dataset using 4 input attributes."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.selectbox("Age group", ["Young", "Middle", "Senior"],
                       help="Young: under 45, Middle: 45 to 54, Senior: 55 and above")
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    heart_rate = st.selectbox("Max heart rate achieved", ["Low", "Medium", "High"],
                              help="Low: under 130, Medium: 130 to 159, High: 160 and above")
    angina = st.selectbox("Exercise induced angina", ["No", "Yes"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Age": [age],
        "MaxHeartRate": [heart_rate],
        "Sex": [sex],
        "ExerciseAngina": [angina],
    })

    encoded = pd.DataFrame({
        col: encoders[col].transform(user_input[col]) for col in user_input.columns
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### Heart Disease: YES")
    else:
        st.success("### Heart Disease: NO")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Made for a university assignment. Not a medical tool, do not use for real diagnosis.")
