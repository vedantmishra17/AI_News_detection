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
            # Use the model tokenizer: 1024 is a token limit, not a character limit.
            tokenizer = pipe.tokenizer
            encoded = tokenizer(text, truncation=True, max_length=min(1024, tokenizer.model_max_length), return_tensors="pt")
            input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
            input_tokens = int(encoded["input_ids"].shape[-1])
            # Keep summaries proportionate to short inputs and valid for generation.
            max_length = max(8, min(int(max_length), input_tokens - 1, 120))
            min_length = max(1, min(int(min_length), max_length - 1, max(4, input_tokens // 2)))
            result = pipe(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            return result[0]["summary_text"].strip()
        except Exception as e:
            print(f"[Summarizer] Inference error: {type(e).__name__}")

    # Fallback: simple extractive summarization (first 2 sentences)
    return _extractive_summary(text, num_sentences=2)


def _extractive_summary(text: str, num_sentences: int = 2) -> str:
    """Simple extractive summary — returns the first N sentences."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary_sentences = sentences[:num_sentences]
    return " ".join(summary_sentences)
