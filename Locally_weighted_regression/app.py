import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv("Advertising.csv")

X = data['TV'].values
y = data['Sales'].values

def lwr(x_query, X, y, tau=50):
    m = len(X)

    weights = np.exp(-(X - x_query)**2 / (2 * tau**2))

    X_design = np.vstack([np.ones(m), X]).T
    W = np.diag(weights)

    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)

    return theta[0] + theta[1] * x_query

st.title("Sales Prediction")

st.write("Enter TV advertising budget")

tv_budget = st.number_input("TV Budget")

if st.button("Predict"):
    prediction = lwr(tv_budget, X, y)
    st.success(f"Predicted Sales: {prediction:.2f}")