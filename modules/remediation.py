import pandas as pd
import numpy as np


def fill_missing_values(
    df,
    column,
    method
):

    result = df.copy()

    if column not in result.columns:
        return result

    if method == "Median":

        if pd.api.types.is_numeric_dtype(
            result[column]
        ):

            result[column] = (
                result[column]
                .fillna(
                    result[column].median()
                )
            )

    elif method == "Mean":

        if pd.api.types.is_numeric_dtype(
            result[column]
        ):

            result[column] = (
                result[column]
                .fillna(
                    result[column].mean()
                )
            )

    elif method == "Mode":

        mode = result[column].mode()

        if not mode.empty:

            result[column] = (
                result[column]
                .fillna(mode.iloc[0])
            )

    elif method == "Forward Fill":

        result[column] = (
            result[column]
            .ffill()
        )

    return result


def remove_duplicates(df):

    result = df.copy()

    return result.drop_duplicates().reset_index(
        drop=True
    )


def standardize_category_column(
    df,
    column
):

    result = df.copy()

    if column not in result.columns:
        return result

    if (
        result[column].dtype == "object"
        or str(
            result[column].dtype
        ) == "category"
    ):

        result[column] = (
            result[column]
            .astype(str)
            .str.strip()
        )

    return result


def handle_outliers(
    df,
    column,
    method="cap"
):

    result = df.copy()

    if column not in result.columns:
        return result

    if not pd.api.types.is_numeric_dtype(
        result[column]
    ):

        return result

    series = result[column]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return result

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    if method == "cap":

        result[column] = (
            result[column]
            .clip(
                lower=lower_bound,
                upper=upper_bound
            )
        )

    elif method == "remove":

        result = result[
            (
                result[column] >=
                lower_bound
            )
            &
            (
                result[column] <=
                upper_bound
            )
        ].reset_index(
            drop=True
        )

    return result


def replace_invalid_values(
    df,
    column,
    lower=None,
    upper=None
):

    result = df.copy()

    if column not in result.columns:
        return result

    if not pd.api.types.is_numeric_dtype(
        result[column]
    ):

        return result

    mask = pd.Series(
        False,
        index=result.index
    )

    if lower is not None:

        mask |= (
            result[column] < lower
        )

    if upper is not None:

        mask |= (
            result[column] > upper
        )

    result.loc[
        mask,
        column
    ] = np.nan

    return result