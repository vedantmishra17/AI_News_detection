"""
News-Melt-AI: Topic Classification Evaluation

Dataset: AG News test split
Model: facebook/bart-large-mnli
Evaluation: Accuracy, macro F1, weighted F1, classification report,
and confusion matrix.

Outputs:
    evaluation/results/classifier_metrics.json
    evaluation/results/classifier_predictions.csv
"""

import csv
import json
import sys
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.classifier import NEWS_CATEGORIES, _load_classifier


RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
METRICS_FILE = RESULTS_DIR / "classifier_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "classifier_predictions.csv"

# Set to 500 for a manageable initial evaluation.
# Set to None to evaluate the entire test dataset.
SAMPLE_SIZE = 500

# AG News labels:
# 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech
AG_NEWS_MAPPING = {
    0: "World News",
    1: "Sports",
    2: "Business & Finance",
    3: "Technology",
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEWS-MELT-AI: TOPIC CLASSIFICATION EVALUATION")
    print("=" * 60)

    # Require the intended zero-shot model.
    print("\nLoading zero-shot classification model...")

    classifier = _load_classifier()

    if classifier is None:
        raise RuntimeError(
            "The BART-MNLI classifier could not be loaded. "
            "Evaluation stopped to avoid measuring keyword fallback "
            "instead of the intended model."
        )

    model_name = getattr(
        classifier.model.config,
        "_name_or_path",
        "facebook/bart-large-mnli",
    )

    print(f"Model: {model_name}")

    # Load AG News test data.
    print("\nLoading AG News test dataset...")

    dataset = load_dataset(
        "fancyzhx/ag_news",
        split="test",
    )

    if SAMPLE_SIZE is not None:
        dataset = dataset.select(
            range(min(SAMPLE_SIZE, len(dataset)))
        )

    texts = [
        f"{row['text']}"
        for row in dataset
    ]

    y_true = [
        AG_NEWS_MAPPING[int(row["label"])]
        for row in dataset
    ]

    y_pred = []
    rows = []

    # Use only categories represented by AG News.
    candidate_labels = list(AG_NEWS_MAPPING.values())

    print(f"\nEvaluating {len(texts)} articles...")
    print("This may take several minutes.")

    for index, (text, actual_label) in enumerate(
        zip(texts, y_true),
        start=1,
    ):
        # Match the application's input truncation.
        result = classifier(
            text[:512],
            candidate_labels=candidate_labels,
            multi_label=False,
        )

        predicted_label = result["labels"][0]
        confidence = float(result["scores"][0])

        y_pred.append(predicted_label)

        rows.append(
            {
                "text": text,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "correct": int(actual_label == predicted_label),
                "confidence": round(confidence, 4),
            }
        )

        if index % 25 == 0 or index == len(texts):
            print(f"Processed {index}/{len(texts)} articles")

    # Calculate metrics.
    labels = candidate_labels

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    metrics = {
        "project": "News-melt-AI",
        "task": "Zero-shot topic classification",
        "dataset": "AG News test split",
        "model": model_name,
        "sample_size": len(texts),
        "candidate_categories": labels,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": {
            "labels": labels,
            "values": matrix.tolist(),
        },
        "classification_report": report,
        "limitations": [
            (
                "AG News has four broad categories; results do not "
                "validate all ten application categories."
            ),
            (
                "The evaluation uses the first 512 characters of "
                "each article, matching the application's truncation."
            ),
            (
                "The application accepts ten categories, while this "
                "evaluation restricts candidates to the four AG News "
                "categories for a compatible comparison."
            ),
        ],
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "text",
                "actual_label",
                "predicted_label",
                "correct",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    print(f"Dataset     : AG News test")
    print(f"Model       : {model_name}")
    print(f"Sample size : {len(texts)}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print("Category order:", labels)
    print(matrix)

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()