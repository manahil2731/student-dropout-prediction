import streamlit as st
import pandas as pd
import joblib

# Load trained model, scaler, and feature column list
model = joblib.load("model/logistic_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
numeric_cols = joblib.load("model/numeric_cols.pkl")

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓")

st.title("🎓 Student Dropout Risk Predictor")
st.write(
    "Enter a student's academic and financial information below to estimate "
    "their dropout risk. This tool is intended to support early intervention, "
    "not to make final decisions about any individual student."
)

st.header("Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age at enrollment", 17, 60, 20)
    admission_grade = st.slider("Admission grade (0-200)", 0, 200, 120)
    sem1_grade = st.slider("1st semester grade (0-20)", 0.0, 20.0, 12.0)
    sem2_grade = st.slider("2nd semester grade (0-20)", 0.0, 20.0, 12.0)

with col2:
    scholarship = st.selectbox("Scholarship holder", ["No", "Yes"])
    tuition_paid = st.selectbox("Tuition fees up to date", ["No", "Yes"])
    debtor = st.selectbox("Debtor", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict Dropout Risk", type="primary"):
    student_data = {
        "Age at enrollment": age,
        "Admission grade": admission_grade,
        "Curricular units 1st sem (grade)": sem1_grade,
        "Curricular units 2nd sem (grade)": sem2_grade,
        "Scholarship holder": 1 if scholarship == "Yes" else 0,
        "Tuition fees up to date": 1 if tuition_paid == "Yes" else 0,
        "Debtor": 1 if debtor == "Yes" else 0,
        "Gender": 1 if gender == "Male" else 0,
    }

    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0
    for key, value in student_data.items():
        if key in input_df.columns:
            input_df.at[0, key] = value

    cols_to_scale = [c for c in numeric_cols if c in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    proba = model.predict_proba(input_df)[0]
    prob_dropout = proba[1]
    prediction = model.predict(input_df)[0]

    if prob_dropout >= 0.7:
        risk_level, color = "High Risk", "red"
    elif prob_dropout >= 0.4:
        risk_level, color = "Medium Risk", "orange"
    else:
        risk_level, color = "Low Risk", "green"

    st.header("Prediction Result")
    st.metric("Dropout Probability", f"{prob_dropout*100:.1f}%")
    st.markdown(f"**Predicted Class:** {'Dropout' if prediction == 1 else 'Not Dropout'}")
    st.markdown(f"**Risk Level:** :{color}[{risk_level}]")

    if risk_level == "High Risk":
        st.warning("Recommend academic advisor follow-up and financial aid review.")
    elif risk_level == "Medium Risk":
        st.info("Recommend monitoring and periodic check-ins.")
    else:
        st.success("Student currently shows low dropout risk.")

st.caption(
    "Model: Logistic Regression | Trained on the UCI Predict Students' Dropout "
    "and Academic Success dataset (4,424 records)."
)
