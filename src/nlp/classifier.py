"""
Topic Classifier
Classifies news articles into standard news categories using zero-shot classification.
"""

from functools import lru_cache

NEWS_CATEGORIES = [
    "Politics",
    "Technology",
    "Business & Finance",
    "Health & Medicine",
    "Science",
    "Sports",
    "Entertainment",
    "World News",
    "Environment",
    "Crime & Law",
]


@lru_cache(maxsize=1)
def _load_classifier():
    """Load and cache the zero-shot classification pipeline."""
    try:
        from transformers import pipeline
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    except Exception as e:
        print(f"[Classifier] Could not load zero-shot model: {e}. Using keyword fallback.")
        return None


# Keyword-based fallback mapping
_KEYWORD_MAP = {
    "Politics": ["election", "president", "government", "senate", "congress", "minister", "parliament", "vote", "democrat", "republican", "policy"],
    "Technology": ["ai", "artificial intelligence", "software", "hardware", "tech", "startup", "algorithm", "robot", "data", "cyber", "app", "smartphone"],
    "Business & Finance": ["market", "stock", "economy", "trade", "investment", "bank", "finance", "gdp", "inflation", "startup", "revenue", "profit"],
    "Health & Medicine": ["health", "disease", "vaccine", "hospital", "medicine", "drug", "cancer", "mental health", "covid", "fda", "treatment", "virus"],
    "Science": ["space", "nasa", "research", "study", "discovery", "planet", "climate", "biology", "chemistry", "physics", "experiment"],
    "Sports": ["football", "basketball", "soccer", "tennis", "olympics", "nba", "nfl", "match", "tournament", "athlete", "championship", "cricket"],
    "Entertainment": ["movie", "music", "celebrity", "film", "award", "netflix", "box office", "actor", "actress", "album", "concert", "tv show"],
    "World News": ["war", "conflict", "ukraine", "russia", "china", "india", "europe", "middle east", "united nations", "nato", "sanctions"],
    "Environment": ["climate change", "global warming", "pollution", "carbon", "renewable", "forest", "emission", "environment", "wildlife", "biodiversity"],
    "Crime & Law": ["crime", "police", "arrest", "court", "trial", "fraud", "murder", "lawsuit", "prison", "judge", "investigation"],
}


def classify_topic(text: str) -> dict:
    """
    Classify the topic of a news article.

    Args:
        text: Title + description of the article

    Returns:
        dict with keys: category (str), confidence (float), all_scores (dict)
    """
    if not text or not text.strip():
        return {"category": "World News", "confidence": 0.0, "all_scores": {}}

    pipe = _load_classifier()

    if pipe is not None:
        try:
            result = pipe(text[:512], candidate_labels=NEWS_CATEGORIES, multi_label=False)
            top_label = result["labels"][0]
            top_score = round(result["scores"][0], 4)
            all_scores = {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}
            return {"category": top_label, "confidence": top_score, "all_scores": all_scores}
        except Exception as e:
            print(f"[Classifier] Inference error: {e}")

    # Fallback: keyword matching
    return _keyword_classify(text)


def _keyword_classify(text: str) -> dict:
    """Keyword-based topic classification fallback."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in _KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    if best_score == 0:
        best_category = "World News"
    return {
        "category": best_category,
        "confidence": min(best_score / 5, 1.0),
        "all_scores": {k: v / max(sum(scores.values()), 1) for k, v in scores.items()},
    }
