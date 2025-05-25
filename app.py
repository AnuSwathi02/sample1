import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Test Fraud Detection App")

st.title("🧪 Test Fraud Detection App (No Model)")

st.write("This is a test app to verify your Streamlit deployment works before loading a real model.")

# Create input fields for features
def dummy_input():
    features = {}
    for i in range(1, 5):  # Just a few inputs to test
        features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    features["Amount"] = st.number_input("Amount", value=0.0)
    return pd.DataFrame([features])

input_df = dummy_input()

if st.button("Run Dummy Prediction"):
    st.success("✅ Test run successful. (No model used)")
    st.write("Input values:")
    st.dataframe(input_df)
