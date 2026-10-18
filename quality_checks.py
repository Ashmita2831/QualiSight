import pandas as pd
import numpy as np


def calculate_severity(percentage):

    if percentage >= 30:
        return "Critical"

    elif percentage >= 15:
        return "High"

    elif percentage >= 5:
        return "Medium"

    else:
        return "Low"


def severity_weight(severity):

    weights = {
        "Low": 1,
        "Medium": 2,
        "High": 4,
        "Critical": 6
    }

    return weights.get(severity, 1)


def check_missing_values(df):

    issues = []

    total_rows = len(df)

    if total_rows == 0:
        return issues

    for column in df.columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count > 0:

            percentage = (
                missing_count /
                total_rows
            ) * 100

            severity = calculate_severity(
                percentage
            )

            issues.append({

                "Type": "Missing Values",

                "Column": column,

                "Count": missing_count,

                "Percentage": round(
                    percentage,
                    2
                ),

                "Severity": severity,

                "Description":
                    f"{missing_count} missing values "
                    f"were detected in '{column}'.",

                "Example":
                    "Missing values can affect "
                    "analysis and model performance."
            })

    return issues


def check_duplicate_rows(df):

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:
        return []

    percentage = (
        duplicate_count /
        len(df)
    ) * 100 if len(df) > 0 else 0

    severity = calculate_severity(
        percentage
    )

    return [{

        "Type": "Duplicate Rows",

        "Column": "Entire Dataset",

        "Count": duplicate_count,

        "Percentage": round(
            percentage,
            2
        ),

        "Severity": severity,

        "Description":
            f"{duplicate_count} duplicate rows "
            f"were detected.",

        "Example":
            "Duplicate records can distort "
            "aggregations and analysis."
    }]


def check_categorical_inconsistencies(df):

    issues = []

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        series = df[column].dropna().astype(str)

        if series.empty:
            continue

        normalized = (
            series
            .str.strip()
            .str.lower()
        )

        grouped = {}

        for original, norm in zip(
            series,
            normalized
        ):

            grouped.setdefault(
                norm,
                set()
            ).add(original)

        for normalized_value, variants in grouped.items():

            if len(variants) > 1:

                count = sum(
                    normalized == normalized_value
                )

                percentage = (
                    count /
                    len(df)
                ) * 100

                severity = calculate_severity(
                    percentage
                )

                issues.append({

                    "Type":
                        "Categorical Inconsistency",

                    "Column":
                        column,

                    "Count":
                        len(variants),

                    "Percentage":
                        round(percentage, 2),

                    "Severity":
                        severity,

                    "Description":
                        f"Multiple representations "
                        f"of '{normalized_value}' "
                        f"were found.",

                    "Example":
                        ", ".join(
                            list(variants)[:5]
                        )
                })

    return issues


def check_outliers(df):

    issues = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_mask = (
            (series < lower_bound) |
            (series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        if outlier_count > 0:

            percentage = (
                outlier_count /
                len(df)
            ) * 100

            severity = calculate_severity(
                percentage
            )

            issues.append({

                "Type":
                    "Statistical Outlier",

                "Column":
                    column,

                "Count":
                    outlier_count,

                "Percentage":
                    round(percentage, 2),

                "Severity":
                    severity,

                "Description":
                    f"{outlier_count} statistical "
                    f"outliers detected in "
                    f"'{column}'.",

                "Example":
                    f"IQR range: "
                    f"{lower_bound:.2f} "
                    f"to "
                    f"{upper_bound:.2f}"
            })

    return issues


def check_context_invalid_values(
    df,
    context
):

    issues = []

    rules = {

        "Academic": {
            "marks": (0, 100),
            "score": (0, 100),
            "percentage": (0, 100),
            "attendance": (0, 100),
            "cgpa": (0, 10)
        },

        "HR": {
            "age": (0, 120),
            "salary": (0, None),
            "experience": (0, 60)
        },

        "Healthcare": {
            "age": (0, 120),
            "bmi": (0, 100)
        },

        "Sales": {
            "discount": (0, 100),
            "quantity": (0, None),
            "price": (0, None)
        }
    }

    context_rules = rules.get(
        context,
        {}
    )

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        for keyword, limits in context_rules.items():

            if keyword not in column_lower:
                continue

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                continue

            lower, upper = limits

            invalid_mask = pd.Series(
                False,
                index=df.index
            )

            if lower is not None:
                invalid_mask |= (
                    df[column] < lower
                )

            if upper is not None:
                invalid_mask |= (
                    df[column] > upper
                )

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count == 0:
                continue

            percentage = (
                invalid_count /
                len(df)
            ) * 100

            severity = calculate_severity(
                percentage
            )

            issues.append({

                "Type":
                    "Context Invalid Value",

                "Column":
                    column,

                "Count":
                    invalid_count,

                "Percentage":
                    round(percentage, 2),

                "Severity":
                    severity,

                "Description":
                    f"{invalid_count} values in "
                    f"'{column}' appear to violate "
                    f"the expected {context} range.",

                "Example":
                    f"Expected range: "
                    f"{lower if lower is not None else '-∞'} "
                    f"to "
                    f"{upper if upper is not None else '∞'}"
            })

    return issues


def run_quality_checks(
    df,
    context
):

    issues = []

    issues.extend(
        check_missing_values(df)
    )

    issues.extend(
        check_duplicate_rows(df)
    )

    issues.extend(
        check_categorical_inconsistencies(df)
    )

    issues.extend(
        check_outliers(df)
    )

    issues.extend(
        check_context_invalid_values(
            df,
            context
        )
    )

    severity_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    issues.sort(
        key=lambda x:
        severity_order.get(
            x["Severity"],
            4
        )
    )

    return issues