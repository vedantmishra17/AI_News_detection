"""
News-melt-AI: Summarization Evaluation

Dataset: CNN/DailyMail test split
Model: facebook/bart-large-cnn
Metrics: ROUGE-1, ROUGE-2, ROUGE-L

Outputs:
    evaluation/results/summarization_metrics.json
    evaluation/results/summarization_predictions.csv
"""

import csv
import json
import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.summarizer import _load_summarizer


RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
METRICS_FILE = RESULTS_DIR / "summarization_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "summarization_predictions.csv"

DEFAULT_SAMPLE_SIZE = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of initial CNN/DailyMail test records (default: 100)")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEWS-MELT-AI: SUMMARIZATION EVALUATION")
    print("=" * 60)

    print("\nLoading summarization model...")

    summarizer = _load_summarizer()

    if summarizer is None:
        raise RuntimeError(
            "The pretrained summarization model could not be loaded. "
            "Evaluation stopped rather than substituting another method."
        )

    model_name = getattr(
        summarizer.model.config,
        "_name_or_path",
        "facebook/bart-large-cnn",
    )

    print(f"Model: {model_name}")

    print("\nLoading CNN/DailyMail test dataset...")

    dataset = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="test",
    )

    dataset = dataset.select(range(min(max(1, args.sample_size), len(dataset))))

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    results = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    print(f"\nEvaluating {len(dataset)} articles...")

    for index, article in enumerate(dataset, start=1):
        text = article["article"]
        reference = article["highlights"]

        source_text = ""
        try:
            encoded = summarizer.tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
            source_text = summarizer.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            input_tokens = int(encoded["input_ids"].shape[-1])
            max_length = max(8, min(60, input_tokens - 1))
            min_length = min(15, max_length - 1, max(4, input_tokens // 2))
            generated = summarizer(
                source_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )[0]["summary_text"].strip()
        except Exception as exc:
            results.append({"article_id": article.get("id", ""), "article": source_text,
                            "reference_summary": reference, "generated_summary": "",
                            "rouge1_f1": "", "rouge2_f1": "", "rougeL_f1": "",
                            "error": type(exc).__name__})
            continue

        scores = scorer.score(reference, generated)

        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rougeL = scores["rougeL"].fmeasure

        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)

        results.append(
            {
                "article": source_text,
                "article_id": article.get("id", ""),
                "reference_summary": reference,
                "generated_summary": generated,
                "rouge1_f1": round(rouge1, 4),
                "rouge2_f1": round(rouge2, 4),
                "rougeL_f1": round(rougeL, 4),
                "error": "",
            }
        )

        if index % 10 == 0 or index == len(dataset):
            print(f"Processed {index}/{len(dataset)} articles", flush=True)

    metrics = {
        "project": "News-melt-AI",
        "task": "Abstractive summarization",
        "dataset": "CNN/DailyMail 3.0.0 test split",
        "dataset_id": "abisee/cnn_dailymail",
        "dataset_configuration": "3.0.0",
        "sampling": f"First {len(dataset)} records of test split; deterministic, no shuffle.",
        "model": model_name,
        "requested_sample_size": len(dataset),
        "sample_size": len(rouge1_scores),
        "inference_failures": len(dataset) - len(rouge1_scores),
        "input_token_limit": 1024,
        "rouge1_f1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else None,
        "rouge2_f1": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else None,
        "rougeL_f1": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else None,
        "limitations": [
            "ROUGE measures text overlap, not factual correctness.",
            "Only a sample of the test split is evaluated.",
            "Input is truncated to match the project's current approach.",
        ],
    }

    if not rouge1_scores:
        raise RuntimeError("Summarization inference failed for every requested article; no ROUGE metrics were computed.")

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
                "article",
                "article_id",
                "reference_summary",
                "generated_summary",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("SUMMARIZATION RESULTS")
    print("=" * 60)
    print(f"Dataset : CNN/DailyMail 3.0.0 test")
    print(f"Model   : {model_name}")
    print(f"Samples : {len(results)}")
    print(f"ROUGE-1 : {metrics['rouge1_f1']:.4f}")
    print(f"ROUGE-2 : {metrics['rouge2_f1']:.4f}")
    print(f"ROUGE-L : {metrics['rougeL_f1']:.4f}")

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
