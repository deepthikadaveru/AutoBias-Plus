# =========================
# EXPLORATORY BIAS SCORE
# =========================
def bias_score_exploratory(results):
    score = 0
    reasons = []

    # Representation bias
    for col, dist in results["representation_bias"].items():
        if not dist.empty and dist.min() < 0.2:
            score += 0.3
            reasons.append(f"Under-representation detected in {col}")

    # Skewed numerical features
    if len(results["skewed_features"]) > 0:
        score += 0.3
        reasons.append("Highly skewed numerical features present")

    return min(score, 1.0), reasons


# =========================
# CLASSIFICATION BIAS SCORE
# =========================
def bias_score_classification(results):
    score = 0
    reasons = []

    # Class imbalance
    severity = results["imbalance_severity"]
    if severity == "High":
        score += 0.6
        reasons.append("Severe class imbalance")
    elif severity == "Moderate":
        score += 0.4
        reasons.append("Moderate class imbalance")

    # Representation bias
    for col, dist in results["representation_bias"].items():
        if not dist.empty and dist.min() < 0.2:
            score += 0.3
            reasons.append(f"Representation bias in {col}")

    return min(score, 1.0), reasons


# =========================
# REGRESSION BIAS SCORE
# =========================
def bias_score_regression(results):
    score = 0
    reasons = []

    # Target skewness
    if abs(results["skewness"]) > 1:
        score += 0.3
        reasons.append("Highly skewed target variable")

    # Group outcome disparity (LOW-CARDINALITY ONLY)
    for col, stats in results["group_outcomes"].items():
        if results["group_sizes"].get(col, 0) > 10:
            continue  # ignore high-cardinality sensitive attributes

        if not stats.empty and (stats.max() - stats.min()) > stats.mean() * 0.3:
            score += 0.4
            reasons.append(f"Outcome disparity across groups in {col}")

    # Correlation with sensitive attributes (LOW-CARDINALITY ONLY)
    for col, corr in results["correlation"].items():
        if results["group_sizes"].get(col, 0) > 10:
            continue  # ignore high-cardinality sensitive attributes

        if corr is not None and abs(corr) > 0.3:
            score += 0.3
            reasons.append(f"Strong correlation with sensitive attribute {col}")

    return min(score, 1.0), reasons


# =========================
# SCORE INTERPRETATION
# =========================
def interpret_bias_score(score):
    if score < 0.3:
        return "Low Bias Risk"
    elif score < 0.6:
        return "Moderate Bias Risk"
    else:
        return "High Bias Risk"
