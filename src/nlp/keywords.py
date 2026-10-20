"""
Keyword Extraction
Ranks single-document keywords by word frequency and provides corpus-level TF-IDF terms.
Used to generate word clouds and trending topic lists.
"""

import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Common English stopwords (extended for news)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "that", "this", "these", "those", "it", "its", "he",
    "she", "they", "we", "you", "i", "my", "our", "your", "his", "her",
    "their", "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "both", "each", "more", "said", "says", "also", "new",
    "one", "two", "three", "first", "last", "after", "before", "since",
    "report", "according", "reuters", "ap", "percent", "year", "years",
    "month", "week", "day", "time", "people", "s", "re", "ve", "ll", "d",
}


def extract_keywords(text: str, top_n: int = 15) -> list[tuple[str, float]]:
    """
    Rank single-token keywords by frequency within one document.

    Args:
        text: Article text
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, score) tuples sorted by score descending
    """
    if not text or len(text.strip()) < 20:
        return []

    # Tokenize and clean
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]

    if not filtered:
        return []

    # Count word frequency (simple approach for single doc)
    counts = Counter(filtered)
    total = sum(counts.values())
    scored = [(word, round(count / total, 6)) for word, count in counts.most_common(top_n)]
    return scored


def extract_keywords_tfidf(corpus: list[str], top_n: int = 20) -> dict[str, float]:
    """
    Extract top keywords across a corpus using TF-IDF.

    Args:
        corpus: List of article texts
        top_n: Number of top keywords to return

    Returns:
        Dict of keyword -> tfidf score
    """
    if not corpus:
        return {}

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all docs
        scores = tfidf_matrix.sum(axis=0).A1
        word_scores = dict(zip(feature_names, scores))

        # Filter stopwords and very short tokens
        filtered = {
            w: s for w, s in word_scores.items()
            if w not in _STOPWORDS and len(w) > 2
        }

        # Sort and return top N
        sorted_keywords = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_keywords[:top_n])

    except Exception as e:
        print(f"[Keywords] TF-IDF error: {e}")
        return {}


def get_trending_topics(articles: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    """
    Detect trending topics by counting keyword frequency across article titles.

    Args:
        articles: List of article dicts with 'title' key
        top_n: Number of trending topics to return

    Returns:
        List of (topic, count) tuples
    """
    all_words = []
    for art in articles:
        title = art.get("title", "")
        words = re.findall(r"\b[a-zA-Z]{4,}\b", title.lower())
        filtered = [w for w in words if w not in _STOPWORDS]
        all_words.extend(filtered)

    counter = Counter(all_words)
    return counter.most_common(top_n)


def build_wordcloud_data(articles: list[dict]) -> dict[str, float]:
    """
    Build a word frequency map suitable for WordCloud generation.

    Args:
        articles: List of articles with 'title' and 'description'

    Returns:
        Dict of word -> frequency score
    """
    corpus = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}"
        corpus.append(text)

    return extract_keywords_tfidf(corpus, top_n=100)
