import streamlit as st
import numpy as np
import joblib

# Title
st.title("🌸 Iris Flower Classifier")

# Load the model
model = joblib.load("model.pkl")

# Get user input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5)

# Predict on button click
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Iris species: **{prediction}**")
