"""
Helper Utilities
Shared utility functions for data normalization, deduplication, and formatting.
"""

import re
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit


ARTICLE_FIELDS = (
    "title", "description", "content", "url", "source", "published_at",
    "image_url", "collector", "sentiment", "topic", "entities", "summary",
    "keywords",
)


def normalize_article(article: dict, collector: str = "") -> dict:
    """Return a complete, JSON-friendly article record without inventing data."""
    if not isinstance(article, dict):
        article = {}
    title = clean_text(article.get("title") or "")
    description = clean_text(article.get("description") or "")
    content = clean_text(article.get("content") or "")
    published = normalize_date(article.get("published_at") or article.get("publishedAt"))
    record = {
        "title": title, "description": description,
        # RSS snippets stay snippets; only use explicit content when supplied.
        "content": content, "url": clean_text(article.get("url") or ""),
        "source": clean_text(article.get("source") or "Unknown"),
        "published_at": published.isoformat() if published else None,
        "image_url": article.get("image_url") or None,
        "collector": clean_text(article.get("collector") or collector),
        "sentiment": article.get("sentiment"), "topic": article.get("topic"),
        "entities": article.get("entities"), "summary": article.get("summary"),
        "keywords": article.get("keywords"),
    }
    return record


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """
    Remove duplicate articles using normalized URLs and titles.
    Preserve the original order of articles.
    """
    seen_urls = set()
    seen_titles = set()
    unique = []

    for art in articles:
        if not isinstance(art, dict):
            continue
        url = clean_text(art.get("url") or "").strip().rstrip("/")
        title = clean_text(art.get("title") or "").strip().lower()

        normalized_title = re.sub(r"\W+", "", title, flags=re.UNICODE)

        # Prefer URL-based deduplication when a URL is available.
        if url:
            try:
                parts = urlsplit(url)
                if parts.scheme and parts.netloc:
                    path = parts.path.rstrip("/")
                    normalized_url = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment))
                else:
                    normalized_url = url.rstrip("/")
            except ValueError:
                normalized_url = url.rstrip("/")

            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

        # Use title as a fallback when a URL is unavailable.
        elif normalized_title:
            if normalized_title in seen_titles:
                continue

            seen_titles.add(normalized_title)

        # Keep articles that have neither a URL nor a title.
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
    if isinstance(date_str, datetime):
        return date_str
    if not date_str or not isinstance(date_str, str):
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
    """
    Clean text by removing unwanted control characters
    and normalizing whitespace while preserving Unicode.
    """
    if not text:
        return ""

    # Convert the input to a string to handle unexpected values safely.
    text = unicodedata.normalize("NFC", str(text))

    # Remove Unicode control characters except whitespace separators.
    text = "".join(" " if unicodedata.category(char) == "Cc" and not char.isspace() else char for char in text)

    # Normalize repeated spaces, tabs, and newlines.
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
