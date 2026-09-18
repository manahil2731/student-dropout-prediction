import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
numeric_cols = joblib.load("numeric_cols.pkl")

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter a student's information below to estimate dropout risk.")

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
        risk_level = "High Risk"
    elif prob_dropout >= 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    st.header("Prediction Result")
    st.metric("Dropout Probability", f"{prob_dropout*100:.1f}%")
    st.write(f"**Predicted Class:** {'Dropout' if prediction == 1 else 'Not Dropout'}")
    st.write(f"**Risk Level:** {risk_level}")
