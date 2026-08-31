import os

import pandas as pd
from dotenv import load_dotenv

from wtest.met import classification_metrics


load_dotenv()


def load_tickets(path=None):
    if path is None:
        path = os.getenv("TICKETS_DATA_PATH")

    if not path:
        raise ValueError(
            "TICKETS_DATA_PATH is not configured"
        )

    return pd.read_csv(path)


def evaluate_predictions(
    predictions,
    expected_category,
    expected_sentiment,
    expected_urgency,
):
    predicted_category = [
        prediction["category"]
        for prediction in predictions
    ]

    predicted_sentiment = [
        prediction["sentiment"]
        for prediction in predictions
    ]

    predicted_urgency = [
        prediction["urgency"]
        for prediction in predictions
    ]

    return {
        "category": classification_metrics(
            expected_category,
            predicted_category,
        ),
        "sentiment": classification_metrics(
            expected_sentiment,
            predicted_sentiment,
        ),
        "urgency": classification_metrics(
            expected_urgency,
            predicted_urgency,
        ),
    }


def evaluate_dataset(
    predictions,
    path=None,
):
    df = load_tickets(path)

    if len(predictions) != len(df):
        raise ValueError(
            "Number of predictions must match "
            "number of tickets"
        )

    expected_category = df["category"].tolist()
    expected_sentiment = df["sentiment"].tolist()
    expected_urgency = df["urgency"].tolist()

    return evaluate_predictions(
        predictions=predictions,
        expected_category=expected_category,
        expected_sentiment=expected_sentiment,
        expected_urgency=expected_urgency,
    )