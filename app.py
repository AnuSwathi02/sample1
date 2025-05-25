import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('xgb_fraud_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a CSV file or enter transaction details below to detect fraud.")

# --- Helper Function ---
def preprocess_input(df):
    # Scale Amount column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    return df

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload Credit Card Transactions CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Input Data")
    st.dataframe(data.head())

    # Preprocess
    processed = preprocess_input(data.copy())

    # Predict
    predictions = model.predict(processed)
    data['Prediction'] = np.where(predictions == 1, 'Fraud', 'Legit')

    st.subheader("Prediction Results")
    st.dataframe(data[['Prediction']])

# --- Manual Input Section ---
st.markdown("### Or enter a single transaction manually")

with st.form("manual_input"):
    input_features = {}
    for i in range(1, 29):  # V1 to V28
        input_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    amount = st.number_input("Amount", value=0.0)
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_df = pd.DataFrame([input_features])
    input_df['Amount'] = amount
    input_df = preprocess_input(input_df)

    pred = model.predict(input_df)[0]
    label = "Fraud ❗" if pred == 1 else "Legit ✅"
    st.success(f"Prediction: {label}")
