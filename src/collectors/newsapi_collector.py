"""
NewsAPI Collector
Fetches top headlines and everything endpoint from NewsAPI.
"""

import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()


class NewsAPICollector:
    """Collects news articles from NewsAPI."""

    def __init__(self):
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables.")
        self.client = NewsApiClient(api_key=api_key)

    def get_top_headlines(
        self,
        query: str = None,
        category: str = None,
        country: str = "us",
        page_size: int = 50,
    ) -> list[dict]:
        """Fetch top headlines."""
        params = {"country": country, "page_size": page_size}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        try:
            response = self.client.get_top_headlines(**params)
            return self._parse_articles(response.get("articles", []))
        except Exception as e:
            print(f"[NewsAPI] Error fetching top headlines: {e}")
            return []

    def get_everything(
        self,
        query: str,
        from_date: str = None,
        to_date: str = None,
        sort_by: str = "publishedAt",
        page_size: int = 50,
        language: str = "en",
    ) -> list[dict]:
        """Fetch articles matching a search query."""
        if not from_date:
            from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            response = self.client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                sort_by=sort_by,
                page_size=page_size,
                language=language,
            )
            return self._parse_articles(response.get("articles", []))
        except Exception as e:
            print(f"[NewsAPI] Error fetching everything: {e}")
            return []

    def _parse_articles(self, raw_articles: list) -> list[dict]:
        """Normalize raw articles into a consistent schema."""
        parsed = []
        for art in raw_articles:
            if not art.get("title") or art["title"] == "[Removed]":
                continue
            parsed.append(
                {
                    "title": art.get("title", ""),
                    "description": art.get("description", "") or "",
                    "content": art.get("content", "") or "",
                    "url": art.get("url", ""),
                    "source": art.get("source", {}).get("name", "Unknown"),
                    "published_at": art.get("publishedAt", ""),
                    "image_url": art.get("urlToImage", ""),
                    "collector": "NewsAPI",
                }
            )
        return parsed
