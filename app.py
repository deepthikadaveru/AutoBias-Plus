import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
st.set_page_config(
    page_title="AutoBias+",
    layout="wide"
)

st.title("AutoBias+")
st.subheader("Automated Bias Detection & Mitigation Tool for ML Datasets")
st.caption(
    "AutoBias+ audits datasets for hidden bias, distinguishes structural vs removable bias, "
    "and applies safe data-level mitigation where possible."
)
st.markdown("---")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "score" not in st.session_state:
    st.session_state.score = None

# -------------------------------------------------
# DATASET UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

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
            st.info(
                f"💡 Suggested target columns (based on semantics & distribution): {suggestions}"
            )

    dataset_type = detect_dataset_type(df, target_col)

    st.markdown("---")
    st.subheader("Dataset Type Detected")
    st.write(f"**{dataset_type.capitalize()} dataset**")

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
        st.write(f"**{score:.2f} — {label}**")

        st.write("### Why this score?")
        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No significant bias indicators detected.")

    # -------------------------------------------------
    # BIAS MITIGATION
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Bias Mitigation (Eliminator)")

    if st.session_state.score is None:
        st.info("Please run bias detection first.")

    elif dataset_type == "exploratory":
        st.info(
            "Exploratory datasets do not have a target variable. "
            "Bias mitigation is not applicable."
        )

    else:
        # Warn about invalid sensitive attributes
        ignored = [
            c for c in sensitive_cols
            if df[c].nunique() > 10 and df[c].dtype != "object"
        ]
        if ignored:
            st.warning(
                f"⚠️ These features are predictive, not sensitive attributes: {ignored}. "
                "They are ignored to avoid incorrect mitigation."
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

            # Structural bias explanation
            if abs(new_score - st.session_state.score) < 0.05:
                st.info(
                    "ℹ️ Bias remains largely unchanged because it is **structural** "
                    "(e.g., genuine outcome differences across groups). "
                    "AutoBias+ avoids unsafe mitigation that would distort reality."
                )

            # -------------------------------------------------
            # VISUALS
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("Bias Score Comparison")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Mitigation**")
                st.write(f"{st.session_state.score:.2f}")
            with col2:
                st.write("**After Mitigation**")
                st.write(f"{new_score:.2f}")

            # Bias score bar chart
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(
                ["Before", "After"],
                [st.session_state.score, new_score],
                width=0.4
            )
            ax.set_ylabel("Bias Score")
            ax.set_ylim(0, 1)
            ax.set_title("Bias Score Change", fontsize=10)
            st.pyplot(fig, use_container_width=False)


            # -------------------------------------------------
            # GROUP-LEVEL VISUAL
            # -------------------------------------------------
            if dataset_type == "classification":
                st.subheader("Class Distribution After Mitigation")
                st.bar_chart(
                    mitigated_df[target_col].value_counts()
                )

            else:
                if sensitive_cols:
                    st.subheader("Mean Target Value by Sensitive Group")
                    group_means = mitigated_df.groupby(
                        sensitive_cols[0]
                    )[target_col].mean()
                    st.bar_chart(group_means)

            # -------------------------------------------------
            # DATA PREVIEW
            # -------------------------------------------------
            st.subheader("Sample of Mitigated Dataset")
            st.dataframe(mitigated_df.head())

            st.caption(
                "Note: Mitigation operates only at the data level and does not "
                "guarantee full fairness if bias is structural."
            )
