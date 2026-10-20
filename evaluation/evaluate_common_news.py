"""Prepare a stable CNN/DailyMail news subset; score only supplied annotations.

Run with --prepare to download the test split and write article/annotation CSVs.
No labels are generated here. Manual labels remain blank until a human supplies them.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "results"
SUBSET = OUT / "common_news_subset.csv"
TEMPLATE = OUT / "common_news_annotations.csv"
METRICS = OUT / "common_news_metrics.json"
PREDICTIONS = OUT / "common_news_predictions.csv"
ANNOTATION_PREDICTIONS = OUT / "common_news_annotation_predictions.csv"
FIELDS = ["article_id", "source_dataset", "source_split", "title", "article_text", "reference_summary",
          "gold_sentiment", "gold_topic", "gold_entities", "gold_keywords", "annotation_status", "annotator"]
SENTIMENT_LABELS = {"Positive", "Negative", "Neutral"}
TOPIC_LABELS = {"Politics", "Technology", "Business & Finance", "Health & Medicine", "Science", "Sports",
                "Entertainment", "World News", "Environment", "Crime & Law"}


def prepare(sample_size=150):
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    # ID hash gives stable selection independent of dataset iteration order.
    chosen = sorted(ds, key=lambda row: hashlib.sha256(row["id"].encode("utf-8")).hexdigest())[:sample_size]
    OUT.mkdir(parents=True, exist_ok=True)
    existing = {}
    if TEMPLATE.exists():
        with TEMPLATE.open(encoding="utf-8-sig", newline="") as old_file:
            existing = {row.get("article_id"): row for row in csv.DictReader(old_file)}
    with SUBSET.open("w", encoding="utf-8-sig", newline="") as sf, TEMPLATE.open("w", encoding="utf-8-sig", newline="") as tf:
        sw, tw = csv.DictWriter(sf, fieldnames=FIELDS), csv.DictWriter(tf, fieldnames=FIELDS)
        sw.writeheader(); tw.writeheader()
        for row in chosen:
            record = {"article_id": row["id"], "source_dataset": "abisee/cnn_dailymail 3.0.0",
                      "source_split": "test", "title": "", "article_text": row["article"],
                      "reference_summary": row["highlights"], "gold_sentiment": "", "gold_topic": "",
                      "gold_entities": "", "gold_keywords": "", "annotation_status": "unannotated", "annotator": ""}
            old = existing.get(record["article_id"], {})
            for field in ("gold_sentiment", "gold_topic", "gold_entities", "gold_keywords", "annotation_status", "annotator"):
                if old.get(field):
                    record[field] = old[field]
            sw.writerow(record); tw.writerow(record)
    manifest = {"dataset": "abisee/cnn_dailymail", "configuration": "3.0.0", "split": "test",
                "target_sample_size": sample_size, "sample_size": len(chosen),
                "sampling": "Sort all test records by SHA-256(article id); select first N.",
                "article_ids": [row["id"] for row in chosen],
                "fields": {"id": "id", "article_text": "article", "reference_summary": "highlights"},
                "license_note": "Hugging Face card metadata lists Apache-2.0; its card licensing text specifically describes v1.0.0. Verify rights for this non-anonymized 3.0.0 subset before redistribution.",
                "dataset_labels": [], "annotation_status": "unannotated; human annotation required",
                "models": {"summarization": "facebook/bart-large-cnn", "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
                           "topic": "facebook/bart-large-mnli", "ner": "en_core_web_sm", "keywords": "frequency baseline"},
                "metric_definitions": {"summary": "ROUGE-1/2/L F1 with stemming", "sentiment_topic": "accuracy, precision, recall, F1, confusion matrix when annotated",
                                       "ner": "exact entity text span and type match; LOC→GPE approximate", "keywords": "precision/recall/F1 at K with normalized exact phrase match"},
                "files": [str(SUBSET.relative_to(ROOT)), str(TEMPLATE.relative_to(ROOT))]}
    (OUT / "common_news_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(chosen)} records. No task labels were created.")


def evaluate_annotations(rows):
    """Score only valid, non-empty human labels in the annotation worksheet."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from src.nlp.classifier import classify_topic
    from src.nlp.keywords import extract_keywords
    from src.nlp.ner import extract_entities
    from src.nlp.sentiment import analyze_sentiment

    gold = {task: [] for task in ("sentiment", "topic", "ner", "keywords")}
    predicted = {task: [] for task in gold}
    predictions = []

    for row in rows:
        text = row.get("article_text", "") or ""
        article_id = row.get("article_id", "")
        if row.get("gold_sentiment", "") in SENTIMENT_LABELS and text:
            label = row["gold_sentiment"]
            guess = analyze_sentiment(text).get("label")
            if guess in SENTIMENT_LABELS:
                gold["sentiment"].append(label); predicted["sentiment"].append(guess)
                predictions.append({"article_id": article_id, "task": "sentiment", "gold": label, "prediction": guess})
        if row.get("gold_topic", "") in TOPIC_LABELS and text:
            label = row["gold_topic"]
            guess = classify_topic(text).get("category")
            if guess in TOPIC_LABELS:
                gold["topic"].append(label); predicted["topic"].append(guess)
                predictions.append({"article_id": article_id, "task": "topic", "gold": label, "prediction": guess})

        if row.get("gold_entities", "").strip() and text:
            try:
                entity_gold = json.loads(row["gold_entities"])
            except (TypeError, json.JSONDecodeError):
                entity_gold = None
            if isinstance(entity_gold, list) and all(isinstance(e, dict) and isinstance(e.get("text"), str)
                                                     and isinstance(e.get("label"), str)
                                                     and e.get("label") in {"PERSON", "ORG", "GPE"} for e in entity_gold):
                entity_guess = extract_entities(text).get("all_entities", [])
                norm = lambda e: (" ".join(e["text"].casefold().split()), e["label"])
                gset = {norm(e) for e in entity_gold}
                pset = {norm(e) for e in entity_guess if isinstance(e, dict) and e.get("label") in {"PERSON", "ORG", "GPE"}}
                gold["ner"].append(gset); predicted["ner"].append(pset)
                predictions.append({"article_id": article_id, "task": "ner", "gold": json.dumps(sorted(gset)),
                                    "prediction": json.dumps(sorted(pset))})

        if row.get("gold_keywords", "").strip() and text:
            try:
                keyword_gold = json.loads(row["gold_keywords"])
            except (TypeError, json.JSONDecodeError):
                keyword_gold = None
            if isinstance(keyword_gold, list) and all(isinstance(k, str) for k in keyword_gold):
                keyword_guess = [k for k, _ in extract_keywords(text, top_n=5)]
                normalize = lambda phrase: " ".join("".join(ch if ch.isalnum() else " " for ch in phrase.casefold()).split())
                gset = {normalize(k) for k in keyword_gold if normalize(k)}
                ranked = [normalize(k) for k in keyword_guess if normalize(k)]
                gold["keywords"].append(gset); predicted["keywords"].append(ranked)
                predictions.append({"article_id": article_id, "task": "keywords", "gold": json.dumps(sorted(gset)),
                                    "prediction": json.dumps(ranked)})

    metrics = {}
    for task in ("sentiment", "topic"):
        labels = sorted(SENTIMENT_LABELS if task == "sentiment" else TOPIC_LABELS)
        if gold[task]:
            metrics[task] = {"samples": len(gold[task]), "accuracy": accuracy_score(gold[task], predicted[task]),
                             "macro_f1": f1_score(gold[task], predicted[task], labels=labels, average="macro", zero_division=0),
                             "classification_report": classification_report(gold[task], predicted[task], labels=labels,
                                                                             output_dict=True, zero_division=0),
                             "confusion_matrix": {"labels": labels, "values": confusion_matrix(gold[task], predicted[task], labels=labels).tolist()}}
        else:
            metrics[task] = {"status": "annotation_pending", "samples": 0}

    if gold["ner"]:
        tp = sum(len(g & p) for g, p in zip(gold["ner"], predicted["ner"]))
        fp = sum(len(p - g) for g, p in zip(gold["ner"], predicted["ner"]))
        fn = sum(len(g - p) for g, p in zip(gold["ner"], predicted["ner"]))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics["ner"] = {"samples": len(gold["ner"]), "metric": "micro exact normalized entity text and label",
                          "precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    else:
        metrics["ner"] = {"status": "annotation_pending", "samples": 0}

    if gold["keywords"]:
        precision_values, recall_values, reciprocal_ranks = [], [], []
        for expected, ranked in zip(gold["keywords"], predicted["keywords"]):
            top = ranked[:5]
            hits = set(top) & expected
            precision_values.append(len(hits) / 5)
            recall_values.append(len(hits) / len(expected) if expected else 0.0)
            reciprocal_ranks.append(next((1 / i for i, item in enumerate(ranked, 1) if item in expected), 0.0))
        precision = sum(precision_values) / len(precision_values)
        recall = sum(recall_values) / len(recall_values)
        metrics["keywords"] = {"samples": len(gold["keywords"]), "k": 5,
                                "precision_at_k": precision, "recall_at_k": recall,
                                "f1_at_k": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                                "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
                                "matching": "Unicode casefold, non-alphanumeric to spaces, normalized exact phrase"}
    else:
        metrics["keywords"] = {"status": "annotation_pending", "samples": 0}

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Download and prepare deterministic subset")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--evaluate-summary", action="store_true", help="Evaluate BART on prepared reference summaries")
    parser.add_argument("--evaluate-annotations", action="store_true", help="Score valid human labels in the annotation worksheet")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.sample_size)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    annotated = []
    # The template is the source of human edits. Prefer it when present so
    # scoring cannot silently ignore annotations in favor of the source copy.
    with (TEMPLATE if TEMPLATE.exists() else SUBSET).open(encoding="utf-8-sig", newline="") as f:
        annotated = list(csv.DictReader(f))
    summary_result = None
    if args.evaluate_summary:
        from rouge_score import rouge_scorer
        from src.nlp.summarizer import _load_summarizer
        model = _load_summarizer()
        if model is None:
            raise RuntimeError("BART summarizer could not be loaded; common-news ROUGE was not computed.")
        selected = [r for r in annotated if r.get("article_text") and r.get("reference_summary")][:args.sample_size]
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        sums = {key: [] for key in ("rouge1", "rouge2", "rougeL")}
        pred_rows = []
        for index, row in enumerate(selected, 1):
            tok = model.tokenizer(row["article_text"], truncation=True, max_length=1024, return_tensors="pt")
            source = model.tokenizer.decode(tok["input_ids"][0], skip_special_tokens=True)
            max_len = max(8, min(60, int(tok["input_ids"].shape[-1]) - 1))
            generated = model(source, max_length=max_len, min_length=min(15, max_len - 1),
                              do_sample=False, truncation=True)[0]["summary_text"].strip()
            score = scorer.score(row["reference_summary"], generated)
            for key in sums:
                sums[key].append(score[key].fmeasure)
            pred_rows.append({"article_id": row["article_id"], "task": "summarization",
                              "gold": row["reference_summary"], "prediction": generated,
                              "rouge1_f1": score["rouge1"].fmeasure, "rouge2_f1": score["rouge2"].fmeasure,
                              "rougeL_f1": score["rougeL"].fmeasure})
            print(f"Summarized {index}/{len(selected)} common-news articles", flush=True)
        if not pred_rows:
            raise RuntimeError("No common-news article/reference pairs are available.")
        summary_result = {f"{key}_f1": sum(values) / len(values) for key, values in sums.items()}
        summary_result.update({"samples": len(pred_rows), "model": "facebook/bart-large-cnn",
                               "metric": "ROUGE F1 with stemmer; article tokenizer limit 1024 tokens"})
        with PREDICTIONS.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
            writer.writeheader(); writer.writerows(pred_rows)
    annotation_metrics = None
    annotation_predictions = []
    previous_summary_metric = None
    if METRICS.exists() and not args.evaluate_summary:
        try:
            with METRICS.open(encoding="utf-8") as previous_file:
                previous_summary_metric = json.load(previous_file).get("metrics", {}).get("summarization")
        except (OSError, json.JSONDecodeError, AttributeError):
            previous_summary_metric = None
    if args.evaluate_annotations:
        annotation_metrics, annotation_predictions = evaluate_annotations(annotated)
        with ANNOTATION_PREDICTIONS.open("w", encoding="utf-8-sig", newline="") as f:
            fields = ["article_id", "task", "gold", "prediction"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(annotation_predictions)
    task_counts = {task: (annotation_metrics or {}).get(task, {}).get("samples", 0)
                   for task in ("sentiment", "topic", "ner", "keywords")}
    summary_metric = summary_result or previous_summary_metric
    status_name = "partial_task_labels_pending" if summary_metric or any(task_counts.values()) else "annotation_pending"
    status = {"status": status_name,
              "dataset": "abisee/cnn_dailymail 3.0.0 test",
              "sample_size": len(annotated), "task_sample_counts": {"summary_reference_available": sum(bool(r.get("reference_summary")) for r in annotated),
                  **task_counts},
              "metrics": ({**({"summarization": summary_metric} if summary_metric else {}),
                           **(annotation_metrics or {})}),
              "note": ("Metrics use only non-empty valid human labels; unannotated tasks remain pending."
                       if any(task_counts.values()) else
                       "No sentiment/topic/NER/keyword ground-truth annotations were supplied; those task scores are not reported.")}
    (OUT / "common_news_metrics.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not args.evaluate_summary and not PREDICTIONS.exists():
        with PREDICTIONS.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["article_id", "task", "gold", "prediction", "correct"])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
