"""
NewsAPI Collector
Fetches top headlines and everything endpoint from NewsAPI.
"""

import os
from datetime import datetime, timedelta, timezone
import logging
from newsapi import NewsApiClient
from dotenv import load_dotenv
from src.utils.helpers import normalize_article

logger = logging.getLogger(__name__)

load_dotenv()


class NewsAPICollector:
    """Collects news articles from NewsAPI."""

    def __init__(self):
        self.last_error = None
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            self.client = None
            self.last_error = "missing_api_key"
            return
        self.client = NewsApiClient(api_key=api_key)

    def get_top_headlines(
        self,
        query: str = None,
        category: str = None,
        country: str = "us",
        page_size: int = 50,
    ) -> list[dict]:
        """Fetch top headlines."""
        if self.client is None:
            logger.warning("NewsAPI is disabled because NEWS_API_KEY is not configured.")
            return []
        params = {"country": country, "page_size": max(1, min(int(page_size), 100))}
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        try:
            response = self.client.get_top_headlines(**params)
            self._check_response(response)
            self.last_error = None
            return self._parse_articles(response.get("articles", []))
        except Exception as e:
            self.last_error = type(e).__name__
            logger.warning("NewsAPI top headlines request failed: %s", type(e).__name__)
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
        if self.client is None:
            logger.warning("NewsAPI is disabled because NEWS_API_KEY is not configured.")
            return []
        if not from_date:
            from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            response = self.client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                sort_by=sort_by,
                page_size=max(1, min(int(page_size), 100)),
                language=language,
            )
            self._check_response(response)
            self.last_error = None
            return self._parse_articles(response.get("articles", []))
        except Exception as e:
            self.last_error = type(e).__name__
            logger.warning("NewsAPI search request failed: %s", type(e).__name__)
            return []

    @staticmethod
    def _check_response(response):
        if not isinstance(response, dict) or response.get("status") != "ok":
            code = response.get("code", "unknown") if isinstance(response, dict) else "malformed response"
            raise ValueError(f"NewsAPI returned an unsuccessful response ({code}).")
        if not isinstance(response.get("articles"), list):
            raise ValueError("NewsAPI returned a malformed articles field.")

    def _parse_articles(self, raw_articles: list) -> list[dict]:
        """Normalize raw articles into a consistent schema."""
        parsed = []
        for art in raw_articles if isinstance(raw_articles, list) else []:
            if not isinstance(art, dict) or not art.get("title") or art["title"] == "[Removed]":
                continue
            source = art.get("source") if isinstance(art.get("source"), dict) else {}
            parsed.append(normalize_article({
                "title": art.get("title"), "description": art.get("description"),
                "content": art.get("content"), "url": art.get("url"),
                "source": source.get("name"), "published_at": art.get("publishedAt"),
                "image_url": art.get("urlToImage"), "collector": "NewsAPI",
            }, collector="NewsAPI"))
        return parsed
