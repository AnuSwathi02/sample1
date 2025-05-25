import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# --- Access Control (Simple Password) ---
PASSWORD = "mysecret"  # 🔒 Change this to your desired password

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("login_form"):
        password = st.text_input("Enter password to access the app", type="password")
        login = st.form_submit_button("Login")
        if login:
            if password == PASSWORD:
                st.session_state.authenticated = True
            else:
                st.error("Incorrect password")

if st.session_state.authenticated:

    # --- Load Model ---
    model = joblib.load("xgb_fraud_model.pkl")

    st.title("💳 Credit Card Fraud Detection App")
    st.write("Upload a CSV file or enter transaction details below to detect fraud.")

    # --- Helper: Preprocessing ---
    def preprocess_input(df):
        df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        for col in ['Time', 'Class']:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    # --- File Upload Section ---
    uploaded_file = st.file_uploader("📤 Upload Credit Card Transactions CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Raw Input Data")
        st.dataframe(data.head())

        # Preprocess
        processed = preprocess_input(data.copy())

        # Predict
        predictions = model.predict(processed)
        data['Prediction'] = np.where(predictions == 1, 'Fraud', 'Legit')

        st.subheader("✅ Prediction Results")
        st.dataframe(data[['Prediction']])

        # --- Confusion Matrix ---
        if 'Class' in data.columns:
            y_true = data['Class']
            y_pred = np.where(predictions == 1, 1, 0)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
            disp.plot(ax=ax)
            st.subheader("📊 Confusion Matrix")
            st.pyplot(fig)

        # --- Download predictions ---
        csv_download = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction CSV", csv_download, file_name="fraud_predictions.csv", mime="text/csv")

        # --- Feature Importance ---
        st.subheader("📈 Feature Importances (XGBoost)")
        importances = model.feature_importances_
        features = processed.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax2.invert_yaxis()
        ax2.set_title("Feature Importances")
        st.pyplot(fig2)

    # --- Manual Input Section ---
    st.markdown("### ✍️ Or enter a single transaction manually")
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
        prob = model.predict_proba(input_df)[0][1]
        label = "Fraud ❗" if pred == 1 else "Legit ✅"
        st.success(f"Prediction: {label} (Probability of fraud: {prob:.2f})")
