import streamlit as st
import pandas as pd
import joblib

# ====== 1. Load model & preprocessor ======
@st.cache_resource
def load_model():
    preprocess = joblib.load("preprocess.pkl")
    model = joblib.load("model.pkl")
    X_columns = joblib.load("X_columns.pkl")  # feature names
    X_stats = joblib.load("X_stats.pkl")      # min, max, mean for sliders
    return preprocess, model, X_columns, X_stats

preprocess, model, X_columns, X_stats = load_model()

# ====== 2. Streamlit App ======
def main():

    # ====== Page Config ======
    st.set_page_config(
        page_title="Breast Cancer Prediction 🧬",
        page_icon="🧬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # ====== Title & Description ======
    st.title("🧬 Breast Cancer Prediction App")
    st.markdown("""
    Welcome to the **Breast Cancer Prediction Tool**.  
    This app uses an **SVM (with preprocessing pipeline)** trained on the **Breast Cancer Wisconsin dataset**  
    to predict whether a tumor is **Benign** or **Malignant**.

    💡 Adjust values or use the default values to simulate predictions.
    """)

    st.markdown("---")  # horizontal line for separation

    # ====== 3. Input Features in Columns ======
    st.subheader("📋 Input Tumor Features:")
    col1, col2, col3 = st.columns(3)

    user_data = {}
    cols = [col1, col2, col3]

    # Loop through features and assign them to 3 columns evenly
    for idx, col_name in enumerate(X_columns):
        stats = X_stats[col_name]
        current_col = cols[idx % 3]  # Distribute sliders evenly across 3 columns

        with current_col:
            user_data[col_name] = st.number_input(
                f"{col_name}",
                float(stats["min"]),
                float(stats["max"]),
                float(stats["mean"]),
                format="%.5f"
            )

    # ====== 4. Create Input DataFrame ======
    user_df = pd.DataFrame([user_data])[X_columns]

    # ====== 6. Predict Button ======
    if st.button("Predict"):

        # Apply preprocessing
        X_trans = preprocess.transform(user_df)

        # Make prediction
        pred = model.predict(X_trans)[0]
        prob = model.predict_proba(X_trans)[0]

        st.subheader("🧠 Prediction Result")
        if pred == 1:
            st.error(f"🚨 **Malignant Tumor Detected!**")
            st.info(f"Prediction Confidence: **{prob[1] * 100:.2f}%** malignant")
        else:
            st.success(f"✅ **Benign Tumor Detected!**")
            st.info(f"Prediction Confidence: **{prob[0] * 100:.2f}%** benign")
            st.balloons()

        # ====== Save and Display Feature Summary ======
        st.subheader("📊 Feature Summary")

        # Add prediction result to user_df before saving
        user_df["Prediction"] = "Malignant" if pred == 1 else "Benign"
        user_df["Confidence"] = f"{prob[pred] * 100:.2f}%"

        # Initialize session state DataFrame if not exists
        if "all_inputs" not in st.session_state:
            st.session_state.all_inputs = pd.DataFrame(columns=user_df.columns)

        # Append new row to session state DataFrame
        st.session_state.all_inputs = pd.concat(
            [st.session_state.all_inputs, user_df], ignore_index=True
        )

        # Display stored data
        st.dataframe(st.session_state.all_inputs.style.background_gradient(cmap='coolwarm'))


# ====== 7. Run the App ======
if __name__ == "__main__":
    main()
