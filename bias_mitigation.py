import pandas as pd

# =========================
# CLASSIFICATION MITIGATION
# =========================
def mitigate_classification_bias(df, target_col, sensitive_cols):
    mitigated_df = df.copy()

    class_counts = mitigated_df[target_col].value_counts()
    max_count = class_counts.max()

    balanced_frames = []
    for cls, count in class_counts.items():
        subset = mitigated_df[mitigated_df[target_col] == cls]
        if count < max_count:
            subset = subset.sample(max_count, replace=True, random_state=42)
        balanced_frames.append(subset)

    mitigated_df = pd.concat(balanced_frames)
    return mitigated_df.reset_index(drop=True)


# =========================
# REGRESSION MITIGATION
# =========================
def mitigate_regression_bias(df, target_col, sensitive_cols):
    mitigated_df = df.copy()

    # Outlier clipping (safe)
    lower = mitigated_df[target_col].quantile(0.10)
    upper = mitigated_df[target_col].quantile(0.90)
    mitigated_df[target_col] = mitigated_df[target_col].clip(lower, upper)

    overall_mean = mitigated_df[target_col].mean()

    # Apply only to low-cardinality sensitive attributes
    for col in sensitive_cols:
        if mitigated_df[col].nunique() > 10:
            continue

        group_means = mitigated_df.groupby(col)[target_col].mean()
        for grp, grp_mean in group_means.items():
            adjustment = overall_mean - grp_mean
            mitigated_df.loc[
                mitigated_df[col] == grp, target_col
            ] += adjustment * 0.6

    return mitigated_df
