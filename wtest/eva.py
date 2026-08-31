import os
import sys
import pandas as pd
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.insert(0, PROJECT_ROOT)
from src.infrence.model import load_model
from structured.generator import StructuredGeneration
from structured.schemas import TicketOutput
from prompt.classifer import build_ticket_classifier
from structured.parser import parse_and_validate_ticket

from wtest.feval import (
    load_tickets,
    evaluate_dataset,
)


def generate_predictions(df, generator):
    predictions = []
    detailed_results = []

    total = len(df)

    for index, row in df.iterrows():

        ticket_id = row["ticket_id"]
        message = row["message"]

        print(
            f"Processing "
            f"{index + 1}/{total} "
            f"(Ticket ID: {ticket_id})"
        )

        try:
            # Build prompt
            prompt = build_ticket_classifier(
                message
            )

            # Run model
            result = generator.generate(
                prompt,
                TicketOutput,
            )
            result = parse_and_validate_ticket(result)

            # Store prediction
            prediction = {
                "category": result.category,
                "sentiment": result.sentiment,
                "urgency": result.urgency,
            }

            predictions.append(prediction)

            # Store detailed result
            detailed_results.append(
                {
                    "ticket_id": ticket_id,
                    "message": message,

                    "expected_category":
                        row["category"],

                    "predicted_category":
                        result.category,

                    "expected_sentiment":
                        row["sentiment"],

                    "predicted_sentiment":
                        result.sentiment,

                    "expected_urgency":
                        row["urgency"],

                    "predicted_urgency":
                        result.urgency,

                    "summary":
                        result.summary,

                    "category_correct":
                        row["category"]
                        == result.category,

                    "sentiment_correct":
                        row["sentiment"]
                        == result.sentiment,

                    "urgency_correct":
                        row["urgency"]
                        == result.urgency,
                }
            )

        except Exception as e:

            print(
                f"ERROR - Ticket {ticket_id}: {e}"
            )

            # Keep list lengths equal
            predictions.append(
                {
                    "category": "ERROR",
                    "sentiment": "ERROR",
                    "urgency": "ERROR",
                }
            )

            detailed_results.append(
                {
                    "ticket_id": ticket_id,
                    "message": message,

                    "expected_category":
                        row["category"],

                    "predicted_category":
                        "ERROR",

                    "expected_sentiment":
                        row["sentiment"],

                    "predicted_sentiment":
                        "ERROR",

                    "expected_urgency":
                        row["urgency"],

                    "predicted_urgency":
                        "ERROR",

                    "summary": "",

                    "category_correct": False,
                    "sentiment_correct": False,
                    "urgency_correct": False,
                }
            )

    return predictions, detailed_results


def print_report(report):

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    for task, metrics in report.items():

        print(f"\n{task.upper()}")
        print("-" * 30)

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.2%}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.2%}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.2%}"
        )

        print(
            f"F1 Score : "
            f"{metrics['f1']:.2%}"
        )

    print("\n" + "=" * 60)


def main():

    print("Loading model...")

    tokenizer, model = load_model()

    generator = StructuredGeneration(
        model,
        tokenizer,
    )

    print("Loading test dataset...")

    df = load_tickets()

    print(
        f"Found {len(df)} tickets."
    )

    print("\nStarting evaluation...\n")

    predictions, detailed_results = (
        generate_predictions(
            df,
            generator,
        )
    )

    report = evaluate_dataset(
        predictions=predictions
    )

    print_report(report)

    # Save detailed results
    results_dir = os.path.join(
        PROJECT_ROOT,
        "Wtest",
        "results",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        detailed_results
    )

    results_path = os.path.join(
        results_dir,
        "ticket_predictions.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\nDetailed results saved to:"
        f"\n{results_path}"
    )


if __name__ == "__main__":
    main()