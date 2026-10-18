"""
News-melt-AI — Main Streamlit Application
AI-powered news intelligence dashboard.
"""

import sys
import os
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
import time

from dotenv import load_dotenv
load_dotenv()

from src.collectors.newsapi_collector import NewsAPICollector
from src.collectors.rss_collector import RSSCollector, GOOGLE_NEWS_TOPICS
from src.nlp.sentiment import analyze_sentiment
from src.nlp.classifier import classify_topic
from src.nlp.ner import extract_entities, aggregate_entities
from src.nlp.keywords import get_trending_topics, build_wordcloud_data
from src.nlp.summarizer import summarize
from src.utils.helpers import (
    deduplicate_articles, format_date, truncate_text,
    clean_text, sentiment_color, sentiment_emoji,
)
from src.utils.cache import get_cached, set_cache, clear_cache, cache_stats

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="News-melt-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background: #0d1117; }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2030 0%, #252840 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.2);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Article cards */
    .article-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .article-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #6366f1, #ec4899);
        border-radius: 4px 0 0 4px;
    }
    .article-card:hover {
        border-color: rgba(99,102,241,0.4);
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(99,102,241,0.15);
    }
    .article-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.4rem;
        line-height: 1.4;
    }
    .article-meta {
        font-size: 0.78rem;
        color: #64748b;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }
    .article-description {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-positive { background: rgba(34,197,94,0.15); color: #22c55e; }
    .badge-negative { background: rgba(239,68,68,0.15); color: #ef4444; }
    .badge-neutral  { background: rgba(148,163,184,0.15); color: #94a3b8; }
    .badge-category { background: rgba(99,102,241,0.15); color: #818cf8; }

    /* Section headings */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #6366f1;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(99,102,241,0.15);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────
if "articles" not in st.session_state:
    st.session_state.articles = []
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = None


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 News-melt-AI")
    st.markdown("---")

    st.markdown("### 📡 Data Sources")
    use_newsapi = st.checkbox("NewsAPI", value=True)
    use_rss = st.checkbox("Google News RSS", value=True)

    st.markdown("### 🔍 Filters")
    search_query = st.text_input("Search keyword", placeholder="e.g., climate, AI, Tesla")

    rss_topic = st.selectbox(
        "RSS Topic",
        options=["All"] + list(GOOGLE_NEWS_TOPICS.keys()),
        index=0,
    )

    st.markdown("### ⚙️ Settings")
    max_articles = st.slider("Max articles to fetch", 10, 100, 50, step=10)
    country = st.selectbox("Country (NewsAPI)", ["us", "gb", "in", "au", "ca"], index=0)

    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch & Analyze", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Cache", use_container_width=True)

    if clear_btn:
        n = clear_cache()
        st.success(f"Cleared {n} cached entries.")

    st.markdown("---")
    stats = cache_stats()
    st.caption(f"Cache: {stats['entries']} entries · {stats['total_size_kb']} KB")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 News-melt-AI</h1>
    <p>Multi-source · AI-powered · Real-time news intelligence dashboard</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FETCH & ANALYZE ARTICLES
# ──────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_articles(query: str, use_newsapi: bool, use_rss: bool, rss_topic: str, max_articles: int, country: str) -> list[dict]:
    """Fetch and deduplicate articles from all sources."""
    articles = []

    if use_newsapi:
        collector = NewsAPICollector()
        if query:
            articles += collector.get_everything(query=query, page_size=max_articles)
        else:
            articles += collector.get_top_headlines(country=country, page_size=max_articles)

    if use_rss:
        rss = RSSCollector()
        if query:
            articles += rss.search_feed(query=query, max_items=max_articles // 2)
        elif rss_topic != "All":
            articles += rss.get_topic_feed(topic=rss_topic, max_items=max_articles // 2)
        else:
            for topic in list(GOOGLE_NEWS_TOPICS.keys())[:3]:
                articles += rss.get_topic_feed(topic=topic, max_items=15)

    return deduplicate_articles(articles)


def run_nlp_analysis(articles: list[dict]) -> list[dict]:
    """Run NLP pipeline on all articles."""
    analyzed = []
    progress = st.progress(0, text="Running NLP analysis…")

    for i, art in enumerate(articles):
        text = clean_text(f"{art.get('title', '')} {art.get('description', '')}")
        title = art.get("title", "")

        # Sentiment
        art["sentiment"] = analyze_sentiment(text)

        # Topic classification
        art["topic"] = classify_topic(text)

        # Named entities
        art["entities"] = extract_entities(text)

        # Summary (only if content is long enough)
        content = clean_text(art.get("content", "") or art.get("description", ""))
        if len(content) > 100:
            art["summary"] = summarize(content, max_length=60, min_length=15)
        else:
            art["summary"] = art.get("description", "")

        analyzed.append(art)
        progress.progress((i + 1) / len(articles), text=f"Analyzing {i+1}/{len(articles)} articles…")

    progress.empty()
    return analyzed


if fetch_btn:
    with st.spinner("Fetching news articles…"):
        raw_articles = fetch_articles(
            query=search_query,
            use_newsapi=use_newsapi,
            use_rss=use_rss,
            rss_topic=rss_topic,
            max_articles=max_articles,
            country=country,
        )

    if not raw_articles:
        st.error("No articles found. Try different filters or check your API key.")
    else:
        st.session_state.articles = run_nlp_analysis(raw_articles)
        st.session_state.analyzed = True
        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")
        st.success(f"✅ Loaded and analyzed {len(st.session_state.articles)} articles.")


# ──────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────
articles = st.session_state.articles

if not articles:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📰</div>
        <h2 style="color: #64748b; font-weight: 500;">No articles loaded yet</h2>
        <p style="color: #475569;">Configure your sources in the sidebar and click <strong>Fetch & Analyze</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────
# METRICS ROW
# ──────────────────────────────────────────────
df = pd.DataFrame(articles)
sentiment_counts = df["sentiment"].apply(lambda x: x.get("label", "Neutral")).value_counts()
topic_counts = df["topic"].apply(lambda x: x.get("category", "Other")).value_counts()

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, "📰", len(articles), "Total Articles"),
    (col2, "😊", sentiment_counts.get("Positive", 0), "Positive"),
    (col3, "😟", sentiment_counts.get("Negative", 0), "Negative"),
    (col4, "😐", sentiment_counts.get("Neutral", 0), "Neutral"),
    (col5, "🏷️", len(topic_counts), "Topics Found"),
]
for col, icon, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.8rem">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.last_fetch_time:
    st.caption(f"Last updated: {st.session_state.last_fetch_time}")


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📰 Articles", "📊 Analytics", "🔥 Trending", "🏷️ Entities", "☁️ Word Cloud"
])


# ──────── TAB 1: ARTICLES ────────
with tab1:
    st.markdown('<div class="section-title">News Feed</div>', unsafe_allow_html=True)

    # Filters
    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            ["Positive", "Negative", "Neutral"],
            default=[],
        )
    with fcol2:
        topic_filter = st.multiselect(
            "Filter by Topic",
            sorted(topic_counts.index.tolist()),
            default=[],
        )
    with fcol3:
        source_names = sorted(df["source"].unique().tolist())
        source_filter = st.multiselect("Filter by Source", source_names, default=[])

    # Apply filters
    filtered = articles.copy()
    if sentiment_filter:
        filtered = [a for a in filtered if a.get("sentiment", {}).get("label") in sentiment_filter]
    if topic_filter:
        filtered = [a for a in filtered if a.get("topic", {}).get("category") in topic_filter]
    if source_filter:
        filtered = [a for a in filtered if a.get("source") in source_filter]

    st.caption(f"Showing {len(filtered)} of {len(articles)} articles")

    # Render article cards
    for art in filtered:
        sentiment = art.get("sentiment", {})
        s_label = sentiment.get("label", "Neutral")
        s_badge_class = f"badge-{s_label.lower()}"
        topic_label = art.get("topic", {}).get("category", "General")
        description = truncate_text(art.get("description", ""), 200)
        published = format_date(art.get("published_at", ""))
        source = art.get("source", "Unknown")
        url = art.get("url", "#")
        summary = art.get("summary", "")

        st.markdown(f"""
        <div class="article-card">
            <div class="article-title">
                <a href="{url}" target="_blank" style="color:#e2e8f0; text-decoration:none;">
                    {art.get('title', 'No title')}
                </a>
            </div>
            <div class="article-meta">
                <span>📰 {source}</span>
                <span>🕒 {published}</span>
                <span class="badge {s_badge_class}">{sentiment_emoji(s_label)} {s_label}</span>
                <span class="badge badge-category">🏷️ {topic_label}</span>
            </div>
            <div class="article-description">{description}</div>
            {"<div style='margin-top:0.5rem; padding:0.5rem; background:rgba(99,102,241,0.08); border-radius:8px; font-size:0.82rem; color:#818cf8;'>💡 <strong>Summary:</strong> " + summary + "</div>" if summary and summary != description else ""}
        </div>
        """, unsafe_allow_html=True)


# ──────── TAB 2: ANALYTICS ────────
with tab2:
    st.markdown('<div class="section-title">Sentiment & Topic Analytics</div>', unsafe_allow_html=True)

    acol1, acol2 = st.columns(2)

    with acol1:
        # Sentiment donut chart
        fig_sentiment = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index.tolist(),
            values=sentiment_counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=["#22c55e", "#ef4444", "#94a3b8"]),
            textinfo="label+percent",
            textfont=dict(color="white", size=13),
        )])
        fig_sentiment.update_layout(
            title=dict(text="Sentiment Distribution", font=dict(color="#e2e8f0", size=16)),
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font=dict(color="#e2e8f0"),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8")),
            height=350,
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with acol2:
        # Topic bar chart
        fig_topics = px.bar(
            x=topic_counts.values[:10],
            y=topic_counts.index[:10],
            orientation="h",
            color=topic_counts.values[:10],
            color_continuous_scale=[[0, "#4338ca"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
            labels={"x": "Article Count", "y": "Topic"},
        )
        fig_topics.update_layout(
            title=dict(text="Articles by Topic", font=dict(color="#e2e8f0", size=16)),
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font=dict(color="#e2e8f0"),
            coloraxis_showscale=False,
            height=350,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_topics, use_container_width=True)

    # Sentiment over time
    st.markdown('<div class="section-title">Sentiment Over Time</div>', unsafe_allow_html=True)
    df_time = df.copy()
    df_time["sentiment_label"] = df_time["sentiment"].apply(lambda x: x.get("label", "Neutral"))
    df_time["published_at"] = pd.to_datetime(df_time["published_at"], errors="coerce", utc=True)
    df_time = df_time.dropna(subset=["published_at"]).sort_values("published_at")

    if not df_time.empty:
        df_time["date"] = df_time["published_at"].dt.date
        timeline = df_time.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
        fig_timeline = px.line(
            timeline,
            x="date",
            y="count",
            color="sentiment_label",
            color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"},
            markers=True,
        )
        fig_timeline.update_layout(
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font=dict(color="#e2e8f0"),
            legend_title_text="Sentiment",
            height=300,
            xaxis=dict(gridcolor="#1e2235"),
            yaxis=dict(gridcolor="#1e2235"),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Not enough date information to plot timeline.")

    # Source distribution
    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
    source_counts = df["source"].value_counts().head(15)
    fig_sources = px.bar(
        x=source_counts.index,
        y=source_counts.values,
        color=source_counts.values,
        color_continuous_scale=[[0, "#4338ca"], [1, "#ec4899"]],
        labels={"x": "Source", "y": "Count"},
    )
    fig_sources.update_layout(
        paper_bgcolor="#1a1d2e",
        plot_bgcolor="#1a1d2e",
        font=dict(color="#e2e8f0"),
        coloraxis_showscale=False,
        height=300,
        xaxis=dict(tickangle=-30, gridcolor="#1e2235"),
        yaxis=dict(gridcolor="#1e2235"),
    )
    st.plotly_chart(fig_sources, use_container_width=True)


# ──────── TAB 3: TRENDING ────────
with tab3:
    st.markdown('<div class="section-title">🔥 Trending Topics</div>', unsafe_allow_html=True)
    trending = get_trending_topics(articles, top_n=20)

    if trending:
        tcol1, tcol2 = st.columns([2, 3])

        with tcol1:
            st.markdown("#### Top Keywords in Headlines")
            for i, (word, count) in enumerate(trending, 1):
                bar_pct = int((count / trending[0][1]) * 100)
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
                    <span style="color:#64748b; width:1.5rem; font-size:0.8rem;">#{i}</span>
                    <span style="color:#e2e8f0; font-weight:500; min-width:120px;">{word.title()}</span>
                    <div style="flex:1; background:#1e2235; border-radius:4px; height:8px;">
                        <div style="width:{bar_pct}%; background:linear-gradient(90deg,#6366f1,#ec4899); height:8px; border-radius:4px;"></div>
                    </div>
                    <span style="color:#818cf8; font-size:0.85rem;">{count}</span>
                </div>
                """, unsafe_allow_html=True)

        with tcol2:
            words, counts = zip(*trending)
            fig_trend = go.Figure(go.Bar(
                x=list(words),
                y=list(counts),
                marker=dict(
                    color=list(counts),
                    colorscale=[[0, "#4338ca"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
                    showscale=False,
                ),
            ))
            fig_trend.update_layout(
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font=dict(color="#e2e8f0"),
                xaxis=dict(tickangle=-30, gridcolor="#1e2235"),
                yaxis=dict(gridcolor="#1e2235"),
                height=380,
                title=dict(text="Trending Keywords Frequency", font=dict(color="#e2e8f0", size=15)),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No trending topics found.")


# ──────── TAB 4: ENTITIES (NER) ────────
with tab4:
    st.markdown('<div class="section-title">🏷️ Named Entity Recognition</div>', unsafe_allow_html=True)
    entity_agg = aggregate_entities(articles)

    ecol1, ecol2, ecol3 = st.columns(3)
    entity_configs = [
        (ecol1, "PERSON", "👤 People Mentioned", "#6366f1"),
        (ecol2, "ORG", "🏢 Organizations", "#8b5cf6"),
        (ecol3, "GPE", "🌍 Locations", "#ec4899"),
    ]

    for col, key, title, color in entity_configs:
        with col:
            st.markdown(f"#### {title}")
            data = entity_agg.get(key, {})
            if data:
                fig = go.Figure(go.Bar(
                    x=list(data.values())[:10],
                    y=list(data.keys())[:10],
                    orientation="h",
                    marker=dict(color=color),
                ))
                fig.update_layout(
                    paper_bgcolor="#1a1d2e",
                    plot_bgcolor="#1a1d2e",
                    font=dict(color="#e2e8f0", size=11),
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(gridcolor="#1e2235"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No entities found.")


# ──────── TAB 5: WORD CLOUD ────────
with tab5:
    st.markdown('<div class="section-title">☁️ Word Cloud</div>', unsafe_allow_html=True)

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        word_freq = build_wordcloud_data(articles)

        if word_freq:
            wc = WordCloud(
                width=1200,
                height=500,
                background_color="#0d1117",
                colormap="cool",
                max_words=80,
                prefer_horizontal=0.8,
                collocations=False,
                relative_scaling=0.5,
                min_font_size=10,
            ).generate_from_frequencies(word_freq)

            fig_wc, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")
            st.pyplot(fig_wc, use_container_width=True)

            # Also show top words as a table
            st.markdown("#### Top Keywords by TF-IDF Score")
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            kw_df = pd.DataFrame(top_words, columns=["Keyword", "Score"])
            kw_df["Score"] = kw_df["Score"].round(4)
            st.dataframe(
                kw_df,
                use_container_width=True,
                hide_index=True,
                column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=float(kw_df["Score"].max()))},
            )
        else:
            st.info("Not enough data to generate a word cloud.")

    except ImportError:
        st.warning("WordCloud library not installed. Run: `pip install wordcloud`")
    except Exception as e:
        st.error(f"Word cloud error: {e}")
