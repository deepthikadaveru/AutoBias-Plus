import streamlit as st
import pandas as pd

from utils import basic_preprocessing, detect_dataset_type, suggest_target_columns
from bias_detection import exploratory_bias, classification_bias, regression_bias
from metrics import (
    bias_score_exploratory,
    bias_score_classification,
    bias_score_regression,
    interpret_bias_score
)
from bias_mitigation import mitigate_classification_bias, mitigate_regression_bias

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AutoBias+", layout="wide")

st.title("AutoBias+")
st.subheader("Automated Bias Detection & Mitigation Tool for ML Datasets")
st.markdown("---")

# Session state
if "score" not in st.session_state:
    st.session_state.score = None

# -------------------------------------------------
# DATASET UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = basic_preprocessing(df)

    st.success("Dataset uploaded and preprocessed successfully")
    st.dataframe(df.head())

    # -------------------------------------------------
    # COLUMN SELECTION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Column Selection")

    cols = df.columns.tolist()
    target_choice = st.selectbox(
        "Select Target Column",
        ["-- Select Target --"] + cols
    )
    target_col = None if target_choice == "-- Select Target --" else target_choice

    sensitive_cols = st.multiselect(
        "Select Sensitive Attribute(s)",
        [c for c in cols if c != target_col]
    )

    if target_col is None:
        suggestions = suggest_target_columns(df)
        if suggestions:
            st.info(f"💡 Suggested target columns: {suggestions}")

    dataset_type = detect_dataset_type(df, target_col)

    st.markdown("---")
    st.subheader("Dataset Type Detected")
    st.write(dataset_type.capitalize())

    # -------------------------------------------------
    # BIAS DETECTION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Bias Detection")

    if st.button("Detect Bias"):
        if dataset_type == "exploratory":
            results = exploratory_bias(df, sensitive_cols)
            score, reasons = bias_score_exploratory(results)
        elif dataset_type == "classification":
            results = classification_bias(df, target_col, sensitive_cols)
            score, reasons = bias_score_classification(results)
        else:
            results = regression_bias(df, target_col, sensitive_cols)
            score, reasons = bias_score_regression(results)

        st.session_state.score = score
        label = interpret_bias_score(score)

        st.subheader("Bias Score")
        st.write(f"{score:.2f} — {label}")

        st.write("### Reasons")
        if reasons:
            for r in reasons:
                st.write("- ", r)
        else:
            st.write("No significant bias indicators detected.")

    # -------------------------------------------------
    # BIAS MITIGATION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Bias Mitigation (Eliminator)")

    if st.session_state.score is None:
        st.info("Run bias detection first.")
    elif dataset_type == "exploratory":
        st.info("Mitigation not applicable for exploratory datasets.")
    else:
        ignored = [c for c in sensitive_cols if df[c].nunique() > 10]
        if ignored:
            st.warning(
                f"⚠️ Ignored high-cardinality attributes: {ignored}. "
                "These are not suitable for safe bias mitigation."
            )

        if st.button("Apply Bias Mitigation"):
            if dataset_type == "classification":
                mitigated_df = mitigate_classification_bias(
                    df, target_col, sensitive_cols
                )
                new_results = classification_bias(
                    mitigated_df, target_col, sensitive_cols
                )
                new_score, _ = bias_score_classification(new_results)

            else:
                mitigated_df = mitigate_regression_bias(
                    df, target_col, sensitive_cols
                )
                new_results = regression_bias(
                    mitigated_df, target_col, sensitive_cols
                )
                new_score, _ = bias_score_regression(new_results)

            # Explain structural bias clearly
            if abs(new_score - st.session_state.score) < 0.05:
                st.info(
                    "ℹ️ Bias remains largely unchanged because the disparity is structural "
                    "(e.g., rent genuinely varies by city). AutoBias+ avoids unsafe mitigation "
                    "that would distort real-world patterns."
                )

            st.subheader("Bias Score Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Mitigation**")
                st.write(f"{st.session_state.score:.2f}")
            with col2:
                st.write("**After Mitigation**")
                st.write(f"{new_score:.2f}")

            st.write("### Sample of Mitigated Dataset")
            st.dataframe(mitigated_df.head())
