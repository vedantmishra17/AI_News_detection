"""
RSS Collector
Fetches news from Google News RSS feeds and other RSS sources.
"""

import feedparser
from datetime import datetime
from email.utils import parsedate_to_datetime


# Google News RSS topic feeds
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
    """Collects news from Google News RSS and other RSS feeds."""

    def get_topic_feed(self, topic: str = "World", max_items: int = 30) -> list[dict]:
        """Fetch articles from a Google News topic feed."""
        url = GOOGLE_NEWS_TOPICS.get(topic, GOOGLE_NEWS_TOPICS["World"])
        return self._fetch_rss(url, source=f"Google News ({topic})", max_items=max_items)

    def search_feed(self, query: str, max_items: int = 30) -> list[dict]:
        """Search Google News RSS by keyword."""
        url = GOOGLE_NEWS_SEARCH_BASE.format(query=query.replace(" ", "+"))
        return self._fetch_rss(url, source="Google News Search", max_items=max_items)

    def fetch_custom_feed(self, url: str, source_name: str = "RSS", max_items: int = 30) -> list[dict]:
        """Fetch any arbitrary RSS/Atom feed."""
        return self._fetch_rss(url, source=source_name, max_items=max_items)

    def _fetch_rss(self, url: str, source: str, max_items: int) -> list[dict]:
        """Internal method to parse an RSS feed."""
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_items]:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Parse published date
                published_at = ""
                if hasattr(entry, "published"):
                    try:
                        dt = parsedate_to_datetime(entry.published)
                        published_at = dt.isoformat()
                    except Exception:
                        published_at = entry.get("published", "")

                # Extract description — strip HTML tags from summary
                summary = entry.get("summary", "") or ""
                description = self._strip_html(summary)

                articles.append(
                    {
                        "title": title,
                        "description": description,
                        "content": description,
                        "url": entry.get("link", ""),
                        "source": source,
                        "published_at": published_at,
                        "image_url": self._extract_image(entry),
                        "collector": "RSS",
                    }
                )
            return articles
        except Exception as e:
            print(f"[RSS] Error fetching feed ({url}): {e}")
            return []

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from a string."""
        from bs4 import BeautifulSoup
        try:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
        except Exception:
            return text

    def _extract_image(self, entry) -> str:
        """Try to extract a thumbnail image URL from the RSS entry."""
        # Check media_thumbnail
        if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            return entry.media_thumbnail[0].get("url", "")
        # Check media_content
        if hasattr(entry, "media_content") and entry.media_content:
            return entry.media_content[0].get("url", "")
        return ""
