"""
Sentiment Analysis Evaluation
Dataset: SST-2 validation split
Model: DistilBERT SST-2, loaded through the project's sentiment module.

Outputs:
    evaluation/results/sentiment_metrics.json
    evaluation/results/sentiment_predictions.csv
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
    precision_recall_fscore_support,
)

# Resolve project paths and make project imports reliable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.sentiment import _load_pipeline


RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
METRICS_FILE = RESULTS_DIR / "sentiment_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "sentiment_predictions.csv"

# Initial evaluation size. Set to None to evaluate the full validation split.
SAMPLE_SIZE = 500

LABEL_TO_ID = {
    "negative": 0,
    "positive": 1,
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEWS-MELT-AI: SENTIMENT EVALUATION")
    print("=" * 60)

    # Require the intended transformer model.
    # Do not silently evaluate using the TextBlob fallback.
    print("\nLoading the configured sentiment transformer...")

    try:
        pipeline = _load_pipeline()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the sentiment transformer. "
            "Check model dependencies and internet access."
        ) from exc

    if pipeline is None:
        raise RuntimeError(
            "The transformer pipeline is unavailable. "
            "Evaluation stopped to prevent reporting TextBlob fallback "
            "results as DistilBERT results."
        )

    model_name = getattr(pipeline.model.config, "_name_or_path", "unknown")

    print(f"Model: {model_name}")

    # Load the labeled validation data.
    print("\nLoading SST-2 validation dataset...")

    try:
        dataset = load_dataset(
            "glue",
            "sst2",
            split="validation",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load SST-2. Check internet access and the "
            "Hugging Face datasets installation."
        ) from exc

    if SAMPLE_SIZE is not None:
        sample_size = min(SAMPLE_SIZE, len(dataset))
        dataset = dataset.select(range(sample_size))

    texts = dataset["sentence"]
    y_true = dataset["label"]

    y_pred = []
    y_true_scored = []
    rows = []

    print(f"\nEvaluating {len(texts)} examples...")

    for index, (text, actual_label) in enumerate(
        zip(texts, y_true),
        start=1,
    ):
        try:
            # Call the loaded transformer directly so fallback predictions cannot
            # be attributed to DistilBERT.
            output = pipeline(text, truncation=True, max_length=512)[0]
            predicted_label = str(output.get("label", "")).strip().lower()
            if predicted_label not in LABEL_TO_ID:
                raise ValueError(f"Unexpected transformer label: {predicted_label}")
            predicted_id = LABEL_TO_ID[predicted_label]
            confidence = float(output.get("score", 0.0))
            y_pred.append(predicted_id)
            y_true_scored.append(actual_label)
            error = ""
        except Exception as exc:
            predicted_label, predicted_id, confidence = "", None, None
            error = type(exc).__name__

        rows.append(
            {
                "text": text,
                "actual_label": (
                    "Positive" if actual_label == 1 else "Negative"
                ),
                "predicted_label": predicted_label.title(),
                "correct": int(actual_label == predicted_id) if predicted_id is not None else "",
                "confidence": confidence,
                "backend": model_name,
                "error": error,
            }
        )

        if index % 50 == 0 or index == len(texts):
            print(f"Processed {index}/{len(texts)} examples")

    # Calculate evaluation metrics.
    if not y_pred:
        raise RuntimeError("All transformer inference calls failed; no metrics were computed.")
    accuracy = accuracy_score(y_true_scored, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_scored,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    report = classification_report(
        y_true_scored,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_scored,
        y_pred,
        labels=[0, 1],
    )

    metrics = {
        "project": "News-melt-AI",
        "task": "Binary sentiment classification",
        "dataset": "GLUE SST-2 validation",
        "dataset_id": "glue/sst2",
        "sampling": f"First {len(texts)} records of validation split; deterministic, no shuffle.",
        "label_mapping": {"0": "Negative", "1": "Positive"},
        "model": model_name,
        "evaluation_method": "Direct transformer pipeline inference; failures excluded from metrics and recorded in predictions.",
        "requested_sample_size": len(texts),
        "sample_size": len(y_pred),
        "inference_failures": len(texts) - len(y_pred),
        "accuracy": float(accuracy),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(f1),
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "values": matrix.tolist(),
        },
        "classification_report": report,
    }

    # Save metrics.
    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save individual predictions for inspection.
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
                "backend",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Print a readable report.
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Dataset       : GLUE SST-2 validation")
    print(f"Model         : {model_name}")
    print(f"Sample size   : {len(texts)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Positive prec.: {precision:.4f}")
    print(f"Positive recall: {recall:.4f}")
    print(f"Positive F1   : {f1:.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Label order: Negative, Positive")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_true_scored,
            y_pred,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
