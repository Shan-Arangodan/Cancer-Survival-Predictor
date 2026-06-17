import streamlit as st
import pandas as pd
import joblib
import os

st.title("Cancer Outcome Predictor")
st.write("Enter patient details to predict the outcome")

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "CancerRF.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------------------
# Category mappings (recovered from LabelEncoder on the raw dataset)
# ----------------------------
gender_map = {"Female": 0, "Male": 1}

state_map = {
    "Chandigarh": 0,
    "Delhi": 1,
    "Gujarat": 2,
    "Karnataka": 3,
    "Kerala": 4,
    "Maharashtra": 5,
    "Tamil Nadu": 6,
    "Telangana": 7,
    "West Bengal": 8,
}

cancer_type_map = {
    "Breast Cancer": 0,
    "Cervical Cancer": 1,
    "Colorectal Cancer": 2,
    "Leukemia": 3,
    "Lung Cancer": 4,
    "Oral Cancer": 5,
    "Ovarian Cancer": 6,
    "Prostate Cancer": 7,
    "Stomach Cancer": 8,
}

stage_map = {
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3,
}

treatment_type_map = {
    "Chemo + Radiation": 0,
    "Chemotherapy": 1,
    "Palliative Care": 2,
    "Radiation": 3,
    "Surgery": 4,
    "Surgery + Chemotherapy": 5,
    "Targeted Therapy": 6,
}

# Status is the target in training (0=Alive, 1=Deceased) — not collected as input

# ----------------------------
# Inputs
# ----------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=50)

gender_label = st.selectbox("Gender", options=list(gender_map.keys()))
gender = gender_map[gender_label]

state_label = st.selectbox("State", options=list(state_map.keys()))
state = state_map[state_label]

cancer_type_label = st.selectbox("Cancer Type", options=list(cancer_type_map.keys()))
cancer_type = cancer_type_map[cancer_type_label]

stage_label = st.selectbox("Stage", options=list(stage_map.keys()))
stage = stage_map[stage_label]

treatment_type_label = st.selectbox("Treatment Type", options=list(treatment_type_map.keys()))
treatment_type = treatment_type_map[treatment_type_label]

survival_months = st.number_input("Survival Months", min_value=0.0, value=12.0)

diagnosis_year = st.number_input("Diagnosis Year", min_value=2000, max_value=2026, value=2023)
diagnosis_month = st.number_input("Diagnosis Month", min_value=1, max_value=12, value=1)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([[
        age, gender, state, cancer_type, stage,
        treatment_type, survival_months, diagnosis_year, diagnosis_month
    ]], columns=[
        "Age", "Gender", "State", "Cancer_Type", "Stage",
        "Treatment_Type", "Survival_Months", "Diagnosis_Year", "Diagnosis_Month"
    ])

    try:
        prediction = model.predict(input_df)
        status_map = {
            0: ("Alive", "🟢 The model predicts this **patient** is likely to be alive based on the entered details."),
            1: ("Deceased", "🔴 The model predicts this **patient** is likely to be deceased based on the entered details."),
        }
        label, message = status_map.get(
            prediction[0],
            (str(prediction[0]), f"Predicted raw output: {prediction[0]}")
        )
        st.subheader(f"Predicted Patient Outcome: {label}")
        st.write(message)
        st.caption(
            "This predicts patient survival status, not cancer remission or tumor status."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")