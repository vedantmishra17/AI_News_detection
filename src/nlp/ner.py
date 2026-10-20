"""
Named Entity Recognition (NER)
Extracts people, organizations, and locations from news articles using spaCy.
"""

from functools import lru_cache
from collections import defaultdict


@lru_cache(maxsize=1)
def _load_nlp():
    """Load and cache the spaCy model."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[NER] Could not load spaCy model: {e}")
        return None


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text.

    Args:
        text: Article text to analyze

    Returns:
        dict with keys: PERSON, ORG, GPE (places), DATE, all_entities
    """
    if not text or not text.strip():
        return {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "all_entities": []}

    nlp = _load_nlp()
    if nlp is None:
        return {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "all_entities": []}

    try:
        doc = nlp(text[:5000])  # spaCy limit
        entity_map = defaultdict(list)
        all_entities = []

        for ent in doc.ents:
            label = ent.label_
            name = ent.text.strip()
            if name and len(name) > 1:
                entity_map[label].append(name)
                all_entities.append({"text": name, "label": label})

        # Deduplicate per category
        return {
            "PERSON": list(dict.fromkeys(entity_map.get("PERSON", []))),
            "ORG": list(dict.fromkeys(entity_map.get("ORG", []))),
            "GPE": list(dict.fromkeys(entity_map.get("GPE", []))),  # Geo-political entities (countries, cities)
            "DATE": list(dict.fromkeys(entity_map.get("DATE", []))),
            "all_entities": all_entities,
        }
    except Exception as e:
        print(f"[NER] Extraction error: {e}")
        return {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "all_entities": []}


def aggregate_entities(articles: list[dict]) -> dict:
    """
    Aggregate entities across multiple articles and count frequency.

    Args:
        articles: List of article dicts, each expected to have 'entities' key.

    Returns:
        dict with PERSON, ORG, GPE tallies (name -> count)
    """
    from collections import Counter
    counts = {"PERSON": Counter(), "ORG": Counter(), "GPE": Counter()}

    for article in articles:
        if not isinstance(article, dict):
            continue
        entities = article.get("entities") or {}
        if not isinstance(entities, dict):
            continue
        for category in ["PERSON", "ORG", "GPE"]:
            for name in entities.get(category, []):
                counts[category][name] += 1

    return {
        cat: dict(counter.most_common(20))
        for cat, counter in counts.items()
    }
