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
import argparse
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
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
DEFAULT_SAMPLE_SIZE = 500

# AG News labels:
# 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech
AG_NEWS_MAPPING = {
    0: "World News",
    1: "Sports",
    2: "Business & Finance",
    3: "Technology",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of initial AG News test examples (default: 500)")
    parser.add_argument("--dataset-arrow", help="Read a cached split Arrow file directly, without the global datasets cache lock.")
    parser.add_argument("--output-suffix", default="", help="Write a separate metrics/predictions pair with this filename suffix.")
    args = parser.parse_args()
    if args.output_suffix and not all(ch.isalnum() or ch in "-_" for ch in args.output_suffix):
        parser.error("--output-suffix may contain only letters, numbers, hyphens, and underscores")
    metrics_file = RESULTS_DIR / f"classifier_metrics{('_' + args.output_suffix) if args.output_suffix else ''}.json"
    predictions_file = RESULTS_DIR / f"classifier_predictions{('_' + args.output_suffix) if args.output_suffix else ''}.csv"
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

    dataset = Dataset.from_file(args.dataset_arrow) if args.dataset_arrow else load_dataset(
        "fancyzhx/ag_news", split="test")

    if args.sample_size is not None:
        dataset = dataset.select(
            range(min(max(1, args.sample_size), len(dataset)))
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

    # Batch to reduce Python/pipeline overhead while keeping memory use bounded.
    batch_size = 8
    inference_failures = 0
    for start in range(0, len(texts), batch_size):
        chunk_texts = [text[:512] for text in texts[start:start + batch_size]]
        chunk_gold = y_true[start:start + batch_size]
        try:
            outputs = classifier(
                chunk_texts,
                candidate_labels=candidate_labels,
                multi_label=False,
                batch_size=16,
            )
        except Exception:
            outputs = []
            # Isolate failures so successful examples remain scoreable.
            for text in chunk_texts:
                try:
                    outputs.append(classifier(text, candidate_labels=candidate_labels, multi_label=False))
                except Exception:
                    outputs.append(None)
        for text, actual_label, result in zip(texts[start:start + batch_size], chunk_gold, outputs):
            if not result:
                inference_failures += 1
                y_pred.append("__INFERENCE_ERROR__")
                rows.append({"text": text, "actual_label": actual_label, "predicted_label": "",
                             "correct": "", "confidence": "", "error": "inference failure"})
                continue
            predicted_label = result["labels"][0]
            confidence = float(result["scores"][0])
            y_pred.append(predicted_label)
            rows.append({"text": text, "actual_label": actual_label, "predicted_label": predicted_label,
                         "correct": int(actual_label == predicted_label), "confidence": round(confidence, 4), "error": ""})
        done = min(start + batch_size, len(texts))
        print(f"Processed {done}/{len(texts)} articles", flush=True)

    # Calculate metrics.
    labels = candidate_labels

    scored = [(gold, pred) for gold, pred in zip(y_true, y_pred) if pred != "__INFERENCE_ERROR__"]
    if not scored:
        raise RuntimeError("All classifier inference calls failed; no metrics were computed.")
    y_true_scored, y_pred_scored = zip(*scored)
    accuracy = accuracy_score(y_true_scored, y_pred_scored)
    macro_f1 = f1_score(
        y_true_scored,
        y_pred_scored,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true_scored,
        y_pred_scored,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true_scored,
        y_pred_scored,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_scored,
        y_pred_scored,
        labels=labels,
    )

    metrics = {
        "project": "News-melt-AI",
        "task": "Zero-shot topic classification",
        "dataset": "AG News test split",
        "dataset_id": "fancyzhx/ag_news",
        "sampling": f"First {len(texts)} records of the test split; deterministic, no shuffle.",
        "label_mapping": {str(key): value for key, value in AG_NEWS_MAPPING.items()},
        "model": model_name,
        "sample_size": len(texts),
        "scored_sample_size": len(y_pred_scored),
        "inference_failures": inference_failures,
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

    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with predictions_file.open(
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
                "error",
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
            y_true_scored,
            y_pred_scored,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print("Category order:", labels)
    print(matrix)

    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Predictions saved to: {predictions_file}")


if __name__ == "__main__":
    main()
