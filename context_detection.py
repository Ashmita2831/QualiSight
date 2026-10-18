import re


CONTEXT_KEYWORDS = {

    "Academic": [
        "marks",
        "score",
        "grade",
        "student",
        "attendance",
        "percentage",
        "subject",
        "semester",
        "cgpa",
        "gpa",
        "course"
    ],

    "HR": [
        "employee",
        "salary",
        "department",
        "experience",
        "designation",
        "age",
        "joining",
        "job",
        "performance",
        "income"
    ],

    "Sales": [
        "sales",
        "revenue",
        "price",
        "quantity",
        "product",
        "customer",
        "discount",
        "order",
        "profit",
        "amount"
    ],

    "Healthcare": [
        "patient",
        "glucose",
        "blood",
        "pressure",
        "bmi",
        "diagnosis",
        "heart",
        "age",
        "insulin",
        "cholesterol"
    ]
}


def normalize_column_name(column):
    """
    Converts column names into a standard format.
    """

    column = str(column).lower()

    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    return column.strip("_")


def detect_context(df):
    """
    Detects the probable dataset context based on
    column names.
    """

    scores = {
        context: 0
        for context in CONTEXT_KEYWORDS
    }

    normalized_columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    for context, keywords in CONTEXT_KEYWORDS.items():

        for column in normalized_columns:

            for keyword in keywords:

                normalized_keyword = normalize_column_name(
                    keyword
                )

                if normalized_keyword in column:
                    scores[context] += 1

    total_score = sum(scores.values())

    if total_score == 0:

        return {
            "context": "General",
            "confidence": 0,
            "scores": scores
        }

    detected_context = max(
        scores,
        key=scores.get
    )

    confidence = (
        scores[detected_context] /
        total_score
    ) * 100

    return {
        "context": detected_context,
        "confidence": round(confidence, 2),
        "scores": scores
    }