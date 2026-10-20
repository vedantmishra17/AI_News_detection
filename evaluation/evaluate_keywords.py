"""Evaluate the existing frequency extractor on the Inspec test set."""
import csv
import json
import re
import sys
import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nlp.keywords import extract_keywords

OUT = ROOT / "evaluation" / "results"
K = 5
SAMPLE_SIZE = 500


def norm(term):
    return " ".join(re.findall(r"[^\W_]+", str(term).casefold(), flags=re.UNICODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--dataset-arrow", help="Read a cached split Arrow file directly, without acquiring the global datasets cache lock.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = Dataset.from_file(args.dataset_arrow) if args.dataset_arrow else load_dataset("midas/inspec", "generation", split="test")
    # Stable sampling from a public, fixed test split; the first 500 are explicit.
    sample_size = min(max(1, args.sample_size), len(data))
    data = data.select(range(sample_size))
    rows = []
    p_sum = r_sum = f_sum = rr_sum = 0.0
    gold_phrase_lengths = []
    for row in data:
        document = row["document"]
        text = " ".join(document) if isinstance(document, list) else str(document)
        gold_raw = row["extractive_keyphrases"]
        gold = {norm(x) for x in gold_raw if norm(x)}
        gold_phrase_lengths.extend(len(term.split()) for term in gold)
        pred = extract_keywords(text, top_n=K)
        pred_norm = [norm(term) for term, _ in pred[:K]]
        matched = [term for term in pred_norm if term in gold]
        precision = len(set(matched)) / K
        recall = len(set(matched)) / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reciprocal_rank = next((1 / (i + 1) for i, term in enumerate(pred_norm) if term in gold), 0.0)
        p_sum += precision; r_sum += recall; f_sum += f1; rr_sum += reciprocal_rank
        rows.append({"id": row["id"], "document": text, "gold_extractive_keyphrases": json.dumps(gold_raw, ensure_ascii=False),
                     "predicted_keywords": json.dumps([x for x, _ in pred[:K]], ensure_ascii=False),
                     "precision_at_k": precision, "recall_at_k": recall, "f1_at_k": f1,
                     "reciprocal_rank": reciprocal_rank})
    n = len(rows)
    metrics = {"task": "keyword extraction", "dataset": "midas/inspec", "configuration": "generation",
               "split": "test", "samples": n, "sampling": "First N records of test split; deterministic, no shuffle.",
               "dataset_license": "Hugging Face dataset metadata lists Apache-2.0; verify source dataset terms before redistribution.",
               "model": "src.nlp.keywords.extract_keywords (frequency baseline)",
               "k": K, "gold_field": "extractive_keyphrases", "input_field": "document",
               "normalization": "Unicode casefold; retain Unicode letters/numbers; collapse whitespace; exact phrase match.",
               "prediction_granularity": "Frequency-ranked single-token unigrams of at least three ASCII letters.",
               "gold_phrase_diagnostics": {
                   "gold_phrase_count": len(gold_phrase_lengths),
                   "mean_gold_phrase_tokens": sum(gold_phrase_lengths) / len(gold_phrase_lengths) if gold_phrase_lengths else None,
                   "multiword_gold_fraction": sum(length > 1 for length in gold_phrase_lengths) / len(gold_phrase_lengths) if gold_phrase_lengths else None,
               },
               "precision_at_k": p_sum / n if n else None, "recall_at_k": r_sum / n if n else None,
               "f1_at_k": f_sum / n if n else None, "mean_reciprocal_rank": rr_sum / n if n else None,
               "limitations": ["Exact phrase match penalizes valid paraphrases and alternate tokenization.",
                               "Inspec contains academic abstracts, not general news.",
                               "Abstractive, absent keyphrases are not used as gold for this extractive production baseline."]}
    (OUT / "keyword_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (OUT / "keyword_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
