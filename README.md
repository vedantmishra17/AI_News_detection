# News-melt-AI

An AI-powered news intelligence dashboard that collects articles from multiple sources, performs NLP analysis, and visualizes insights in real-time.

## Features

- 📰 **Multi-source news collection** — NewsAPI + Google News RSS
- 😊 **Sentiment Analysis** — Positive / Negative / Neutral scoring
- 📝 **Article Summarization** — AI-generated TL;DR for each article
- 🏷️ **Topic Classification** — Automatic category tagging
- 🔍 **Named Entity Recognition (NER)** — People, Organizations, Locations
- 📈 **Trending Topics** — Real-time hot topic detection
- 🔎 **Search & Filter** — Filter by keyword, topic, sentiment, source
- ☁️ **Word Cloud** — Visual keyword frequency map

## Project Structure

```
News-melt-AI/
├── app.py                  # Streamlit main app entry point
├── requirements.txt        # All Python dependencies
├── .env                    # API keys (not committed)
├── src/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── newsapi_collector.py   # NewsAPI integration
│   │   └── rss_collector.py       # Google News RSS integration
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── sentiment.py           # Sentiment analysis
│   │   ├── summarizer.py          # Article summarization
│   │   ├── classifier.py          # Topic classification
│   │   ├── ner.py                 # Named Entity Recognition
│   │   └── keywords.py            # Keyword extraction
│   └── utils/
│       ├── __init__.py
│       ├── cache.py               # Caching utilities
│       └── helpers.py             # Shared helper functions
├── data/                   # Cached data / AG News dataset
└── assets/                 # Static assets (logo, CSS)
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Add your API key** — already saved in `.env`

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Tech Stack

- **Frontend**: Streamlit
- **NLP**: HuggingFace Transformers, spaCy, scikit-learn
- **Data**: NewsAPI, Google News RSS, AG News
- **Visualization**: Plotly, WordCloud, Matplotlib
