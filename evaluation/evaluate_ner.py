from collections import defaultdict
import json
from pathlib import Path

import spacy
from datasets import load_dataset
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

OUTPUT_DIR = Path("evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 500

# CoNLL-2003 label -> spaCy label
LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "GPE",
}


def get_spacy_tags(doc):
    """Convert spaCy character spans to token-level entity tags."""
    tags = ["O"] * len(doc)

    for ent in doc.ents:
        if ent.label_ not in LABEL_MAP.values():
            continue

        label = ent.label_
        for token in doc:
            if token.idx >= ent.start_char and token.idx < ent.end_char:
                prefix = "B-" if token.idx == ent.start_char else "I-"
                tags[token.i] = f"{prefix}{label}"

    return tags


def convert_gold_tags(tokens, ner_tags, tag_names):
    """Convert CoNLL-2003 tags into the common entity-label scheme."""
    result = []

    for tag_id in ner_tags:
        tag = tag_names[tag_id]

        if tag == "O":
            result.append("O")
            continue

        prefix, _, label = tag.partition("-")
        mapped_label = LABEL_MAP.get(label)

        if mapped_label is None:
            result.append("O")
        else:
            result.append(f"{prefix}-{mapped_label}")

    return result


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading CoNLL-2003 test dataset...")
    dataset = load_dataset(
        "eriktks/conll2003",
        split="test",
        trust_remote_code=True,
    )

    dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    tag_names = dataset.features["ner_tags"].feature.names

    gold_sequences = []
    predicted_sequences = []

    for row in dataset:
        tokens = row["tokens"]
        gold_tags = convert_gold_tags(
            tokens,
            row["ner_tags"],
            tag_names,
        )

        text = " ".join(tokens)
        doc = nlp(text)

        # Align spaCy predictions to the original dataset tokens.
        predicted_tags = ["O"] * len(tokens)

        token_positions = []
        cursor = 0
        for token in tokens:
            start = text.find(token, cursor)
            if start < 0:
                token_positions.append(None)
                continue

            token_positions.append((start, start + len(token)))
            cursor = start + len(token)

        for ent in doc.ents:
            if ent.label_ not in LABEL_MAP.values():
                continue

            label = ent.label_
            matched_indices = []

            for i, position in enumerate(token_positions):
                if position is None:
                    continue

                start, end = position
                if start < ent.end_char and end > ent.start_char:
                    matched_indices.append(i)

            for j, idx in enumerate(matched_indices):
                prefix = "B-" if j == 0 else "I-"
                predicted_tags[idx] = f"{prefix}{label}"

        gold_sequences.append(gold_tags)
        predicted_sequences.append(predicted_tags)

    results = {
        "dataset": "CoNLL-2003 test",
        "model": "en_core_web_sm",
        "samples": len(gold_sequences),
        "entity_types": ["PERSON", "ORG", "GPE"],
        "precision": precision_score(gold_sequences, predicted_sequences),
        "recall": recall_score(gold_sequences, predicted_sequences),
        "f1": f1_score(gold_sequences, predicted_sequences),
        "classification_report": classification_report(
            gold_sequences,
            predicted_sequences,
            digits=4,
        ),
    }

    output_path = OUTPUT_DIR / "ner_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("NER RESULTS")
    print("=" * 55)
    print(f"Dataset   : {results['dataset']}")
    print(f"Model     : {results['model']}")
    print(f"Samples   : {results['samples']}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1']:.4f}")
    print("\nPer-entity report:")
    print(results["classification_report"])
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()