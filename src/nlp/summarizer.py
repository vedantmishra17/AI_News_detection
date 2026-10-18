"""
Article Summarizer
Uses HuggingFace facebook/bart-large-cnn for extractive/abstractive summarization.
Falls back to simple extractive summarization if the model is unavailable.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_summarizer():
    """Load and cache the summarization pipeline."""
    try:
        from transformers import pipeline
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            truncation=True,
        )
    except Exception as e:
        print(f"[Summarizer] Could not load BART model: {e}. Using extractive fallback.")
        return None


def summarize(text: str, max_length: int = 80, min_length: int = 20) -> str:
    """
    Summarize a news article.

    Args:
        text: Full article text
        max_length: Maximum token length for the summary
        min_length: Minimum token length for the summary

    Returns:
        A concise summary string
    """
    if not text or len(text.strip()) < 50:
        return text.strip() if text else "No content available."

    pipe = _load_summarizer()

    if pipe is not None:
        try:
            # BART works best with <= 1024 tokens, truncate input text
            input_text = text[:1024]
            result = pipe(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            return result[0]["summary_text"].strip()
        except Exception as e:
            print(f"[Summarizer] Inference error: {e}")

    # Fallback: simple extractive summarization (first 2 sentences)
    return _extractive_summary(text, num_sentences=2)


def _extractive_summary(text: str, num_sentences: int = 2) -> str:
    """Simple extractive summary — returns the first N sentences."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary_sentences = sentences[:num_sentences]
    return " ".join(summary_sentences)
