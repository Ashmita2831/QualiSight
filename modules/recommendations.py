def generate_recommendation(issue):

    issue_type = issue.get(
        "Type",
        ""
    )

    if issue_type == "Missing Values":

        return (
            "Review the missing records and "
            "choose an appropriate treatment such "
            "as median, mean, mode, or another "
            "domain-specific method."
        )

    elif issue_type == "Duplicate Rows":

        return (
            "Review duplicate records and remove "
            "them if they represent repeated "
            "instances of the same observation."
        )

    elif issue_type == "Categorical Inconsistency":

        return (
            "Standardize variations in categorical "
            "values, such as differences in "
            "capitalization or extra spaces."
        )

    elif issue_type == "Statistical Outlier":

        return (
            "Investigate the unusual values before "
            "removing them. They may represent "
            "legitimate extreme observations."
        )

    elif issue_type == "Context Invalid Value":

        return (
            "Review values that fall outside the "
            "expected domain range and correct "
            "or replace them where appropriate."
        )

    return (
        "Review the detected issue and determine "
        "the appropriate data-quality treatment."
    )


def generate_insights(
    df,
    context_result,
    issues,
    anomaly_result
):

    insights = []

    context = context_result.get(
        "context",
        "General"
    )

    confidence = context_result.get(
        "confidence",
        0
    )

    insights.append(
        f"Detected dataset context: "
        f"{context} with approximately "
        f"{confidence:.1f}% confidence."
    )

    if len(df) > 0:

        missing_total = int(
            df.isnull().sum().sum()
        )

        if missing_total > 0:

            insights.append(
                f"The dataset contains "
                f"{missing_total} missing values "
                f"that may require treatment."
            )

        else:

            insights.append(
                "No missing values were detected."
            )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        insights.append(
            f"{duplicate_count} duplicate rows "
            f"were detected."
        )

    anomaly_count = anomaly_result.get(
        "anomaly_count",
        0
    )

    if anomaly_count > 0:

        insights.append(
            f"Isolation Forest identified "
            f"{anomaly_count} potential anomalous "
            f"records. These should be reviewed "
            f"before treating them as errors."
        )

    else:

        insights.append(
            "The ML anomaly detector did not "
            "identify potential anomalies under "
            "the selected settings."
        )

    if issues:

        high_priority = [
            issue
            for issue in issues
            if issue["Severity"]
            in ["Critical", "High"]
        ]

        if high_priority:

            insights.append(
                f"{len(high_priority)} high-priority "
                f"quality issues require attention."
            )

        else:

            insights.append(
                "The detected rule-based issues "
                "are currently below the high-priority "
                "threshold."
            )

    return insights