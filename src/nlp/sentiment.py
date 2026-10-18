"""
Sentiment Analysis Module
Uses a HuggingFace pipeline for fast, accurate sentiment scoring.
Falls back to TextBlob if the model is not yet downloaded.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load and cache the sentiment analysis pipeline."""
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        print(f"[Sentiment] Could not load transformer model: {e}. Using TextBlob fallback.")
        return None


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a given text.

    Returns:
        dict with keys: label (Positive/Negative/Neutral), score (0.0–1.0), compound (-1 to 1)
    """
    if not text or not text.strip():
        return {"label": "Neutral", "score": 0.5, "compound": 0.0}

    pipe = _load_pipeline()

    if pipe is not None:
        try:
            result = pipe(text[:512])[0]
            label = result["label"]
            score = round(result["score"], 4)
            # Map HuggingFace labels to our labels
            mapped_label = "Positive" if label == "POSITIVE" else "Negative"
            compound = score if mapped_label == "Positive" else -score
            return {"label": mapped_label, "score": score, "compound": round(compound, 4)}
        except Exception as e:
            print(f"[Sentiment] Transformer inference error: {e}")

    # Fallback: TextBlob
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.05:
            label, score = "Positive", round(polarity, 4)
        elif polarity < -0.05:
            label, score = "Negative", round(-polarity, 4)
        else:
            label, score = "Neutral", 0.5
        return {"label": label, "score": score, "compound": round(polarity, 4)}
    except Exception:
        return {"label": "Neutral", "score": 0.5, "compound": 0.0}


def batch_analyze_sentiment(texts: list[str]) -> list[dict]:
    """Analyze sentiment for a list of texts."""
    return [analyze_sentiment(t) for t in texts]
