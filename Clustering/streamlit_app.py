import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Features
X = df.iloc[:, [3, 4]].values

# Train model
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)

# UI
st.title("Customer Segmentation App")

st.write("Enter customer details:")

income = st.number_input("Annual Income")
score = st.number_input("Spending Score")

if st.button("Predict Cluster"):
    cluster = kmeans.predict([[income, score]])
    st.success(f"This customer belongs to Cluster {cluster[0] + 1}")