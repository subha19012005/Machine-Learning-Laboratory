# streamlitapp.py
import streamlit as st
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

# ---------------------------
# Rename columns for consistency
# ---------------------------
data = data.rename(columns={
    'chest pain type': 'chest_pain_type',
    'resting bp s': 'resting_bp_s',
    'max heart rate': 'max_heart_rate'
})

# ---------------------------
# Bin numeric features for discrete BN
# ---------------------------
data['age_bin'] = pd.cut(data['age'], bins=[0,40,55,100], labels=['young','middle','old'])
data['resting_bp_bin'] = pd.cut(data['resting_bp_s'], bins=[0,120,140,300], labels=['low','normal','high'])
data['cholesterol_bin'] = pd.cut(data['cholesterol'], bins=[0,200,240,600], labels=['low','normal','high'])
data['max_hr_bin'] = pd.cut(data['max_heart_rate'], bins=[0,120,150,250], labels=['low','normal','high'])

features = ['age_bin','sex','chest_pain_type','resting_bp_bin','cholesterol_bin','max_hr_bin','target']
data = data[features]

# ---------------------------
# Define and train Discrete Bayesian Network
# ---------------------------
model = DiscreteBayesianNetwork([
    ('age_bin','target'),
    ('sex','target'),
    ('chest_pain_type','target'),
    ('resting_bp_bin','target'),
    ('cholesterol_bin','target'),
    ('max_hr_bin','target')
])
model.fit(data, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Heart Disease Prediction with Bayesian Network")
st.write("Enter patient medical details:")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=130)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=220)
hr = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=200, value=150)

# ---------------------------
# Predict Heart Disease
# ---------------------------
if st.button("Predict Heart Disease"):

    # Bin numeric inputs
    age_bin = 'young' if age <= 40 else 'middle' if age <= 55 else 'old'
    bp_bin = 'low' if bp <= 120 else 'normal' if bp <= 140 else 'high'
    chol_bin = 'low' if chol <= 200 else 'normal' if chol <= 240 else 'high'
    hr_bin = 'low' if hr <= 120 else 'normal' if hr <= 150 else 'high'

    # Prepare evidence for BN
    patient = {
        'age_bin': age_bin,
        'sex': 1 if sex=="Male" else 0,
        'chest_pain_type': int(cp),
        'resting_bp_bin': bp_bin,
        'cholesterol_bin': chol_bin,
        'max_hr_bin': hr_bin
    }

    # Query BN
    result = inference.query(variables=['target'], evidence=patient)
    prob = result.values[1]

    # ---------------------------
    # Show result
    # ---------------------------
    st.write(f"**Probability of Heart Disease:** {prob:.2f}")

    if prob > 0.5:
        st.error("⚠️ High risk of Heart Disease — consult a doctor immediately!")
    else:
        st.success("✅ Low risk of Heart Disease — maintain healthy lifestyle!")

    # Optional: show patient input for clarity
    st.write("### Patient Input Summary")
    st.json(patient)