"""Fetch news from Google News RSS feeds and other RSS/Atom sources."""

import logging
from urllib.parse import quote_plus

import feedparser
import requests

from src.utils.helpers import clean_text, normalize_article

logger = logging.getLogger(__name__)

GOOGLE_NEWS_TOPICS = {
    "World": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB",
    "Technology": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
    "Business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    "Science": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNR1p0Y1RjU0FtVnVHZ0pWVXlnQVAB",
    "Health": "https://news.google.com/rss/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ",
    "Sports": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
    "Entertainment": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB",
}
GOOGLE_NEWS_SEARCH_BASE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


class RSSCollector:
    """Collect RSS records with a bounded network timeout and item count."""

    def __init__(self):
        self.last_error = None

    def get_topic_feed(self, topic: str = "World", max_items: int = 30) -> list[dict]:
        url = GOOGLE_NEWS_TOPICS.get(topic, GOOGLE_NEWS_TOPICS["World"])
        return self._fetch_rss(url, source=f"Google News ({topic})", max_items=max_items)

    def search_feed(self, query: str, max_items: int = 30) -> list[dict]:
        url = GOOGLE_NEWS_SEARCH_BASE.format(query=quote_plus(query or ""))
        return self._fetch_rss(url, source="Google News Search", max_items=max_items)

    def fetch_custom_feed(self, url: str, source_name: str = "RSS", max_items: int = 30) -> list[dict]:
        return self._fetch_rss(url, source=source_name, max_items=max_items)

    def _fetch_rss(self, url: str, source: str, max_items: int) -> list[dict]:
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            self.last_error = "invalid_url"
            logger.warning("RSS feed URL must use HTTP or HTTPS.")
            return []
        try:
            response = requests.get(url, timeout=(5, 15), headers={"User-Agent": "News-melt-AI/1.0"})
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            if getattr(feed, "bozo", False):
                self.last_error = "malformed_feed"
                logger.warning("RSS parser reported malformed feed content for %s.", source)
            articles = []
            for entry in feed.entries[:max(0, min(int(max_items), 100))]:
                if not isinstance(entry, dict):
                    continue
                title = clean_text(entry.get("title") or "")
                if not title:
                    continue
                description = self._strip_html(entry.get("summary") or "")
                articles.append(normalize_article({
                    "title": title,
                    "description": description,
                    # Feeds generally provide snippets, not full article text.
                    "content": "",
                    "url": entry.get("link"),
                    "source": source,
                    "published_at": entry.get("published") or entry.get("updated"),
                    "image_url": self._extract_image(entry),
                    "collector": "RSS",
                }, collector="RSS"))
            if not getattr(feed, "bozo", False):
                self.last_error = None
            return articles
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.warning("RSS request failed for source %s (%s).", source, type(exc).__name__)
            return []

    @staticmethod
    def _strip_html(text: str) -> str:
        from bs4 import BeautifulSoup
        try:
            if not isinstance(text, str):
                return clean_text(text)
            return clean_text(BeautifulSoup(text, "html.parser").get_text(separator=" "))
        except Exception:
            return text

    @staticmethod
    def _extract_image(entry) -> str | None:
        for key in ("media_thumbnail", "media_content"):
            values = entry.get(key) or []
            if values and isinstance(values[0], dict):
                image = values[0].get("url")
                if image and image.startswith(("https://", "http://")):
                    return image
        return None
