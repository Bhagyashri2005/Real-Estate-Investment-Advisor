import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4f7fb 0%, #eaf1f8 100%);
    }
    .main-header {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #123047, #176b68);
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #e4f3f2;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(20, 40, 60, 0.05);
    }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load saved files
# --------------------------------------------------
MODEL_DIR = "models"


@st.cache_resource
def load_objects():
    def load_pickle(filename):
        with open(os.path.join(MODEL_DIR, filename), "rb") as file:
            return pickle.load(file)

    classification_scaler = load_pickle("classification_scaler.pkl")
    regression_scaler = load_pickle("regression_scaler.pkl")
    label_encoders = load_pickle("label_encoders.pkl")
    classification_model = load_pickle("best_classification_model.pkl")
    regression_model = load_pickle("best_regression_model.pkl")
    feature_names = load_pickle("feature_names.pkl")
    processed_data = pd.read_csv(os.path.join(MODEL_DIR, "processed_data.csv"))

    return (
        classification_scaler,
        regression_scaler,
        label_encoders,
        classification_model,
        regression_model,
        feature_names,
        processed_data,
    )


try:
    (
        classification_scaler,
        regression_scaler,
        label_encoders,
        classification_model,
        regression_model,
        feature_names,
        processed_data,
    ) = load_objects()
except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏡 Real Estate Investment Advisor</h1>
        <p>Explore property investment potential and estimate its value after five years.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Adjust the sample property details below, then select **Analyze Property** "
    "to generate predictions."
)


# --------------------------------------------------
# Default values
# --------------------------------------------------
def get_default(feature):
    """Use the median of the training data as the default numeric value."""
    if feature in processed_data.columns:
        value = processed_data[feature].median()
        if pd.notna(value) and np.isfinite(value):
            return float(value)
    return 0.0


# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("📋 Property Details")

with st.form("property_input_form"):
    input_values = {}

    # Split inputs into 3 columns for a cleaner layout
    columns = st.columns(3)

    for index, feature in enumerate(feature_names):
        default_value = get_default(feature)

        with columns[index % 3]:
            input_values[feature] = st.number_input(
                label=feature.replace("_", " "),
                value=default_value,
                step=1.0,
                format="%.2f",
                key=f"field_{feature}",
            )

    submitted = st.form_submit_button(
        "🔍 Analyze Property",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame(
            [[input_values[feature] for feature in feature_names]],
            columns=feature_names,
        )

        input_df = input_df.apply(pd.to_numeric, errors="coerce")

        if input_df.isnull().any().any():
            st.error("Please enter valid numeric values for all fields.")
            st.stop()

        classification_input = classification_scaler.transform(input_df)
        regression_input = regression_scaler.transform(input_df)

        investment_prediction = classification_model.predict(
            classification_input
        )[0]

        future_price = float(
            regression_model.predict(regression_input)[0]
        )

        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if int(investment_prediction) == 1:
                st.success("### ✅ Good Investment")
            else:
                st.warning("### ⚠️ Not a Good Investment")

            if hasattr(classification_model, "predict_proba"):
                probabilities = classification_model.predict_proba(
                    classification_input
                )[0]
                confidence = float(np.max(probabilities)) * 100
                st.metric("Prediction Confidence", f"{confidence:.1f}%")

        with result_col2:
            if future_price >= 0:
                st.metric(
                    "Estimated Price After 5 Years",
                    f"₹{future_price:,.2f} Lakhs",
                )
            else:
                st.error(
                    "The model predicted a negative price. "
                    "This is not a valid property-price estimate. "
                    "Review the model and input preprocessing."
                )

        st.caption(
            "Predictions are estimates based on the trained model and should "
            "not be treated as financial advice."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Real Estate Investment Advisor | Machine Learning + Streamlit")