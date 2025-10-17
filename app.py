import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io  # for creating CSV in memory

# Load model & preprocessor
@st.cache_resource
def load_model():
    preprocess = joblib.load("preprocess.pkl")
    model = joblib.load("model.pkl")
    X_columns = joblib.load("X_columns.pkl")  # feature names
    X_stats = joblib.load("X_stats.pkl")      # min, max, mean for sliders
    return preprocess, model, X_columns, X_stats

preprocess, model, X_columns, X_stats = load_model()

# App title
st.title("🩺 Breast Cancer Prediction App")
st.markdown(
    "Enter the tumor measurements below and the system will predict whether "
    "the case is **Malignant** or **Benign**."
)

# User input sliders
st.subheader("📋 Input Tumor Features:")

user_data = {}
for col in X_columns:
    stats = X_stats[col]
    user_data[col] = st.slider(
        f"{col}",
        float(stats["min"]),
        float(stats["max"]),
        float(stats["mean"]),
        format="%.5f"
    )

# Convert user input into DataFrame
user_df = pd.DataFrame([user_data])[X_columns]

# Display entered data
st.subheader("📊 Entered Data:")
st.dataframe(user_df)

# Prediction
if st.button("🔍 Predict"):
    try:
        # Apply preprocessing
        X_trans = preprocess.transform(user_df)

        # Make prediction
        pred = model.predict(X_trans)[0]
        prob = model.predict_proba(X_trans)[0]

        # Display result
        st.subheader("🔎 Prediction Result:")
        if pred == 1:
            st.error(f"⚠️ Diagnosis: **Malignant**\n\nConfidence: {prob[1]*100:.2f}%")
        else:
            st.success(f"✅ Diagnosis: **Benign**\n\nConfidence: {prob[0]*100:.2f}%")

        # Add prediction and confidence to DataFrame
        user_df["Prediction"] = "Malignant" if pred == 1 else "Benign"
        user_df["Confidence"] = f"{prob[pred]*100:.2f}%"

        # Create CSV in memory
        csv_buffer = io.StringIO()
        user_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        # Download button for CSV file
        st.download_button(
            label="💾 Download as CSV",
            data=csv_bytes,
            file_name="user_input_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")