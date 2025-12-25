import pandas as pd

# -----------------------------
# BASIC PREPROCESSING
# -----------------------------
def basic_preprocessing(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


# -----------------------------
# DATASET TYPE DETECTION
# -----------------------------
def detect_dataset_type(df, target_col):
    if target_col is None or target_col not in df.columns:
        return "exploratory"

    unique_vals = df[target_col].nunique()

    if df[target_col].dtype == "object" or unique_vals <= 10:
        return "classification"
    else:
        return "regression"


# -----------------------------
# RANKED TARGET SUGGESTIONS
# -----------------------------
def suggest_target_columns(df):
    suggestions = []

    for col in df.columns:
        unique_vals = df[col].nunique()
        col_lower = col.lower()
        score = 0

        # Semantic hints
        if any(k in col_lower for k in ["income", "status", "result", "label", "target", "outcome"]):
            score += 3

        # Classification-like
        if df[col].dtype == "object" and unique_vals <= 10:
            score += 2

        # Regression-like
        if df[col].dtype != "object" and unique_vals > 20:
            score += 1

        if score > 0:
            suggestions.append((col, score))

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in suggestions]
