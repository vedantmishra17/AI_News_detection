"""Exact entity-span evaluation of spaCy against CoNLL-2003 test."""
import csv
import json
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "results"
SAMPLE_SIZE = 500
LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}  # LOC→GPE is approximate.


def bio_spans(tokens, tag_ids, tag_names):
    """Return (start_char, end_char, mapped_type) gold spans; MISC is omitted."""
    offsets = []
    text_parts = []
    cursor = 0
    for token in tokens:
        if text_parts:
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    text = " ".join(tokens)
    spans = []
    active_label = None
    active_start = active_end = None

    def flush():
        if active_label:
            spans.append((active_start, active_end, active_label))

    for (start, end), tag_id in zip(offsets + [(len(text), len(text))], list(tag_ids) + [0]):
        tag = tag_names[tag_id]
        if tag == "O":
            flush(); active_label = None
            continue
        prefix, _, raw_label = tag.partition("-")
        label = LABEL_MAP.get(raw_label)
        if label is None:  # MISC is deliberately excluded from scoring.
            flush(); active_label = None
            continue
        if prefix == "B" or label != active_label or active_label is None:
            flush()
            active_start, active_end, active_label = start, end, label
        else:
            active_end = end
    return text, spans


def score_counts(counts):
    tp = sum(v["tp"] for v in counts.values())
    fp = sum(v["fp"] for v in counts.values())
    fn = sum(v["fn"] for v in counts.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--dataset-arrow", help="Read a cached split Arrow file directly, without acquiring the global datasets cache lock.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading spaCy model...", flush=True)
    nlp = spacy.load("en_core_web_sm")
    print("Loading CoNLL-2003 test split...", flush=True)
    ds = Dataset.from_file(args.dataset_arrow) if args.dataset_arrow else load_dataset(
        "eriktks/conll2003", split="test", trust_remote_code=True)
    sample_size = min(max(1, args.sample_size), len(ds))
    ds = ds.select(range(sample_size))
    tag_names = ds.features["ner_tags"].feature.names
    counts = defaultdict(Counter)
    rows = []
    for i, row in enumerate(ds, 1):
        text, gold = bio_spans(row["tokens"], row["ner_tags"], tag_names)
        doc = nlp(text)
        pred = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ in LABEL_MAP.values()]
        gold_set, pred_set = set(gold), set(pred)
        for label in LABEL_MAP.values():
            gt = {s for s in gold_set if s[2] == label}
            pt = {s for s in pred_set if s[2] == label}
            counts[label]["tp"] += len(gt & pt)
            counts[label]["fp"] += len(pt - gt)
            counts[label]["fn"] += len(gt - pt)
        rows.append({"tokens": json.dumps(row["tokens"], ensure_ascii=False),
                     "gold_entities": json.dumps(gold, ensure_ascii=False),
                     "predicted_entities": json.dumps(pred, ensure_ascii=False)})
        if i % 100 == 0 or i == len(ds):
            print(f"Processed {i}/{len(ds)} sentences", flush=True)
    per_label = {}
    for label, c in counts.items():
        p, r, f = score_counts({label: c})
        per_label[label] = {**dict(c), "precision": p, "recall": r, "f1": f}
    p, r, f = score_counts(counts)
    entity_f1s = [values["f1"] for values in per_label.values()]
    entity_supports = [counts[label]["tp"] + counts[label]["fn"] for label in per_label]
    support_total = sum(entity_supports)
    macro_f1 = sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0
    weighted_f1 = (sum(f1 * support for f1, support in zip(entity_f1s, entity_supports)) / support_total
                   if support_total else 0.0)
    metrics = {"dataset": "CoNLL-2003 test", "dataset_id": "eriktks/conll2003", "dataset_config": "eriktks/conll2003",
               "model": "en_core_web_sm", "requested_sample_size": args.sample_size, "samples": len(rows),
               "sampling": "First N records of test split; deterministic, no shuffle.",
               "metric": "Exact entity span and type match; micro P/R/F1 and classwise F1 averages",
               "precision": p, "recall": r, "micro_f1": f, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
               "classwise_average_definition": "Macro is the unweighted mean F1 over PERSON/ORG/GPE; weighted uses gold support per class.",
               "per_entity": per_label, "label_mapping": LABEL_MAP,
               "misc_policy": "MISC gold entities are omitted from scoring; predicted spaCy labels outside PERSON/ORG/GPE are omitted.",
               "limitations": ["LOC→GPE is an explicit approximation; these ontologies are not identical.",
                               "CoNLL newswire is not the common-news annotation set."]}
    (OUT / "ner_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (OUT / "ner_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["tokens", "gold_entities", "predicted_entities"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
