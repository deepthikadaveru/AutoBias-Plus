import pandas as pd

# =========================
# EXPLORATORY BIAS
# =========================
def exploratory_bias(df, sensitive_cols):
    results = {}

    representation_bias = {}
    for col in sensitive_cols:
        representation_bias[col] = df[col].value_counts(normalize=True)

    results["representation_bias"] = representation_bias

    skewed_features = {}
    for col in df.select_dtypes(include="number").columns:
        skew = df[col].skew()
        if abs(skew) > 1:
            skewed_features[col] = skew

    results["skewed_features"] = skewed_features
    return results


# =========================
# CLASSIFICATION BIAS
# =========================
def classification_bias(df, target_col, sensitive_cols):
    results = {}

    class_distribution = df[target_col].value_counts(normalize=True)
    imbalance_ratio = class_distribution.max()

    if imbalance_ratio > 0.75:
        severity = "High"
    elif imbalance_ratio > 0.6:
        severity = "Moderate"
    else:
        severity = "Low"

    results["class_distribution"] = class_distribution
    results["imbalance_severity"] = severity

    representation_bias = {}
    for col in sensitive_cols:
        representation_bias[col] = df[col].value_counts(normalize=True)

    results["representation_bias"] = representation_bias
    return results


# =========================
# REGRESSION BIAS
# =========================
def regression_bias(df, target_col, sensitive_cols):
    results = {}

    results["mean"] = df[target_col].mean()
    results["median"] = df[target_col].median()
    results["skewness"] = df[target_col].skew()

    group_outcomes = {}
    group_sizes = {}

    for col in sensitive_cols:
        group_outcomes[col] = df.groupby(col)[target_col].mean()
        group_sizes[col] = df[col].nunique()

    results["group_outcomes"] = group_outcomes
    results["group_sizes"] = group_sizes

    correlations = {}
    for col in sensitive_cols:
        try:
            correlations[col] = df[target_col].corr(
                df[col].astype("category").cat.codes
            )
        except:
            correlations[col] = None

    results["correlation"] = correlations
    return results
