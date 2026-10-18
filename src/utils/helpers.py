"""
Helper Utilities
Shared utility functions for data normalization, deduplication, and formatting.
"""

import re
import hashlib
from datetime import datetime


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """
    Remove duplicate articles based on title similarity.

    Args:
        articles: List of article dicts

    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []
    for art in articles:
        title = art.get("title", "").strip().lower()
        # Create a hash of the normalized title
        title_hash = hashlib.md5(re.sub(r"\W+", "", title).encode()).hexdigest()
        if title_hash not in seen:
            seen.add(title_hash)
            unique.append(art)
    return unique


def normalize_date(date_str: str) -> datetime | None:
    """
    Parse various date string formats into a datetime object.

    Args:
        date_str: Date string in various formats

    Returns:
        datetime object or None
    """
    if not date_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try dateutil as last resort
    try:
        from dateutil import parser
        return parser.parse(date_str)
    except Exception:
        return None


def format_date(date_str: str, fmt: str = "%b %d, %Y %H:%M") -> str:
    """Format a date string for display."""
    dt = normalize_date(date_str)
    if dt:
        return dt.strftime(fmt)
    return date_str or "Unknown Date"


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to a maximum number of characters."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and non-printable characters."""
    if not text:
        return ""
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentiment_color(label: str) -> str:
    """Return a color hex code for a sentiment label."""
    colors = {
        "Positive": "#22c55e",
        "Negative": "#ef4444",
        "Neutral": "#94a3b8",
    }
    return colors.get(label, "#94a3b8")


def sentiment_emoji(label: str) -> str:
    """Return an emoji for a sentiment label."""
    emojis = {
        "Positive": "😊",
        "Negative": "😟",
        "Neutral": "😐",
    }
    return emojis.get(label, "😐")
