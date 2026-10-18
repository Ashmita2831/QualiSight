import pandas as pd


def get_dataset_overview(df):
    """
    Returns basic information about the dataset.
    """

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage": round(
            df.memory_usage(deep=True).sum() / (1024 * 1024), 2
        )
    }

    return overview


def get_column_profile(df):
    """
    Creates a detailed profile for every column.
    """

    profile = []

    for column in df.columns:

        missing = int(df[column].isnull().sum())
        total = len(df)

        missing_percentage = (
            (missing / total) * 100
            if total > 0
            else 0
        )

        profile.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing": missing,
            "Missing %": round(missing_percentage, 2),
            "Unique": int(df[column].nunique(dropna=True)),
            "Non-Null": int(df[column].notnull().sum())
        })

    return pd.DataFrame(profile)


def get_statistical_summary(df):
    """
    Returns statistical summary for numerical columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["median"] = numeric_df.median()

    return summary.round(2)


def get_categorical_summary(df):
    """
    Returns common values for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    results = []

    for column in categorical_columns:

        value_counts = (
            df[column]
            .value_counts(dropna=False)
            .head(10)
        )

        for value, count in value_counts.items():

            results.append({
                "Column": column,
                "Value": str(value),
                "Count": int(count)
            })

    return pd.DataFrame(results)