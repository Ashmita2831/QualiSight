def calculate_quality_score(
    df,
    issues,
    anomaly_result=None
):

    if len(df) == 0:
        return 0

    severity_weight = {

        "Low": 1,

        "Medium": 2,

        "High": 4,

        "Critical": 6
    }

    penalty = 0

    for issue in issues:

        percentage = issue.get(
            "Percentage",
            0
        )

        severity = issue.get(
            "Severity",
            "Low"
        )

        weight = severity_weight.get(
            severity,
            1
        )

        penalty += (
            percentage *
            weight *
            0.15
        )

    # ML anomalies are treated as a
    # separate signal and receive a
    # smaller penalty to avoid
    # over-penalizing uncertain anomalies.

    if anomaly_result:

        anomaly_percentage = (
            anomaly_result.get(
                "anomaly_percentage",
                0
            )
        )

        penalty += (
            anomaly_percentage *
            0.10
        )

    score = max(
        0,
        100 - penalty
    )

    return round(
        min(score, 100),
        2
    )


def get_score_label(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 60:
        return "Needs Attention"

    else:
        return "Poor"