import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


def detect_ml_anomalies(
    df,
    contamination=0.05,
    random_state=42
):
    """
    Detect potential anomalous rows using Isolation Forest.

    ML is applied only to numerical columns.

    Missing numerical values are temporarily filled with
    the column median for ML processing.

    The original dataset is not modified.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # No numerical data
    if not numeric_columns:
        return pd.DataFrame()

    # Isolation Forest needs enough observations
    if len(df) < 10:
        return pd.DataFrame()

    # Create ML input
    X = df[numeric_columns].copy()

    # Temporary missing-value handling
    for column in numeric_columns:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

    # Remove constant columns
    variable_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) > 1
    ]

    if not variable_columns:
        return pd.DataFrame()

    X = X[variable_columns]

    # Keep contamination within valid range
    contamination = min(
        max(float(contamination), 0.01),
        0.20
    )

    # Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    predictions = model.fit_predict(X)

    # sklearn:
    #  1  = normal
    # -1  = anomaly

    decision_scores = model.decision_function(X)

    # Higher value = more normal
    # Convert so higher value means more anomalous
    anomaly_scores = -decision_scores

    result = df.copy()

    result["Anomaly Score"] = np.round(
        anomaly_scores,
        5
    )

    result["Anomaly Prediction"] = predictions

    result["Potential Anomaly"] = (
        predictions == -1
    )

    # Sort anomalies first
    result = result.sort_values(
        by="Anomaly Score",
        ascending=False
    )

    result = result.reset_index(
        drop=True
    )

    return result