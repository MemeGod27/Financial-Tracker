from __future__ import annotations

import html
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai


_LOGGER = logging.getLogger("market_sentiment_tracker.performance")
_LOGGER.setLevel(logging.INFO)

_PAGE_STARTED = time.perf_counter()


def log_timing(operation: str, started: float) -> None:
    """Log elapsed time for performance diagnostics without exposing user data or keys."""
    duration_ms = (time.perf_counter() - started) * 1000
    _LOGGER.info("perf operation=%s duration_ms=%.1f", operation, duration_ms)


st.set_page_config(
    page_title="Market Sentiment Tracker",
    page_icon="📈",
    layout="wide",
)

CATALYSTS = [
    "Earnings",
    "Guidance",
    "Management",
    "Product",
    "Competition",
    "Regulation",
    "Partnership",
    "M&A",
    "Analyst",
    "Legal",
    "Macro",
    "Dividend",
    "Buyback",
    "Other",
]
IMPACTS = ["High", "Medium", "Low"]
IMPACT_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
PERIODS = {
    "1 month": ("1mo", 30),
    "3 months": ("3mo", 90),
    "6 months": ("6mo", 180),
    "1 year": ("1y", 365),
}
COMPANIES = {
    "apple": ("Apple Inc.", "AAPL", "NASDAQ"),
    "nvidia": ("NVIDIA Corporation", "NVDA", "NASDAQ"),
    "tesla": ("Tesla, Inc.", "TSLA", "NASDAQ"),
    "microsoft": ("Microsoft Corporation", "MSFT", "NASDAQ"),
    "amazon": ("Amazon.com, Inc.", "AMZN", "NASDAQ"),
    "alphabet": ("Alphabet Inc.", "GOOGL", "NASDAQ"),
    "google": ("Alphabet Inc.", "GOOGL", "NASDAQ"),
    "meta": ("Meta Platforms, Inc.", "META", "NASDAQ"),
    "facebook": ("Meta Platforms, Inc.", "META", "NASDAQ"),
}
POSITIVE_WORDS = {
    "beat", "beats", "beating", "growth", "grows", "strong", "record",
    "surge", "surges", "profit", "profits", "raises", "raised", "upgrade",
    "upgraded", "buy", "bullish", "expansion", "outperform", "positive",
    "partnership", "approval", "approved", "dividend", "buyback",
}
NEGATIVE_WORDS = {
    "miss", "misses", "missed", "decline", "declines", "weak", "loss",
    "losses", "cuts", "cut", "downgrade", "downgraded", "sell", "bearish",
    "lawsuit", "probe", "investigation", "recall", "warning", "layoffs",
    "negative", "debt", "default", "resigns", "resignation",
}
CATALYST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Earnings": ("earnings", "revenue", "profit", "quarter", "eps", "results"),
    "Guidance": ("guidance", "outlook", "forecast", "expects", "projection"),
    "Management": ("ceo", "cfo", "executive", "management", "resigns", "appoints"),
    "Product": ("product", "launch", "release", "technology", "patent"),
    "Competition": ("competitor", "market share", "competition", "rival"),
    "Regulation": ("regulatory", "regulator", "approval", "approved", "fda", "ftc"),
    "Partnership": ("partnership", "partner", "collaboration", "alliance"),
    "M&A": ("acquisition", "acquires", "merger", "buyout", "takeover"),
    "Analyst": ("analyst", "price target", "rating", "upgrade", "downgrade"),
    "Legal": ("lawsuit", "court", "legal", "settlement", "investigation", "probe"),
    "Macro": ("inflation", "interest rate", "fed ", "economy", "tariff", "recession"),
    "Dividend": ("dividend", "payout"),
    "Buyback": ("buyback", "repurchase", "share repurchase"),
}


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    .summary-card {
        border: 1px solid rgba(128, 128, 128, .25);
        border-radius: 14px;
        padding: 18px 22px;
        margin: 10px 0 22px 0;
        background: linear-gradient(110deg, rgba(48, 105, 150, .12), rgba(48, 150, 112, .08));
    }
    .summary-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    .summary-label { font-size: .82rem; opacity: .75; margin-bottom: 4px; }
    .summary-value { font-size: 1.15rem; font-weight: 650; }
    .news {
        border: 1px solid rgba(128, 128, 128, .23);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 9px;
        margin-right: 6px;
        font-size: .78rem;
        font-weight: 600;
        background: rgba(128, 128, 128, .14);
    }
    .pill.positive { color: #16834b; }
    .pill.negative { color: #c23b3b; }
    .pill.neutral { opacity: .8; }
    .meta { font-size: .82rem; opacity: .7; margin-top: 5px; }
    .brief {
        border-left: 4px solid #3478c8;
        border-radius: 6px;
        padding: 12px 20px;
        background: rgba(52, 120, 200, .07);
    }
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.1rem 0.9rem;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: .01em;
    }
    .sidebar-section-label {
        color: #94a3b8;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .09em;
        margin: .25rem 0 .45rem;
        text-transform: uppercase;
    }
    .sidebar-asset-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        margin: .45rem 0 1.1rem;
        padding: .8rem .9rem;
    }
    .sidebar-asset-name {
        color: #f8fafc;
        font-size: 1.03rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .sidebar-asset-meta {
        color: #94a3b8;
        font-size: .79rem;
        font-weight: 600;
        letter-spacing: .035em;
        margin-top: .24rem;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input) {
        background: #020817;
        border: 1px solid #1e293b;
        border-radius: 12px;
        position: relative;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input)::before {
        color: #94a3b8;
        content: "⌕";
        font-size: 1.25rem;
        left: .8rem;
        line-height: 1;
        pointer-events: none;
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: transparent;
        border: 0;
        border-radius: 12px;
        color: #f8fafc;
        min-height: 2.75rem;
        padding-left: 2.35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
        color: #64748b;
        opacity: 1;
    }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input):focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #020817;
        border: 1px solid #1e293b;
        border-radius: 12px;
        color: #f8fafc;
        min-height: 2.75rem;
    }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        color: #f8fafc;
        font-weight: 650;
        min-height: 2.65rem;
        transition: background-color .15s ease, border-color .15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: #172338;
        border-color: #334155;
        color: #f8fafc;
    }
    .sidebar-ai-status {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 9px;
        color: #22c55e;
        font-size: .79rem;
        font-weight: 650;
        margin-top: .75rem;
        padding: .55rem .7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def as_utc(value: Any) -> datetime | None:
    """Convert common Yahoo Finance timestamp representations to UTC."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(parsed) else parsed.to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None


def extract_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize Yahoo Finance's current and legacy news item structures."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title") or item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    published = (
        as_utc(content.get("pubDate"))
        or as_utc(content.get("providerPublishTime"))
        or as_utc(item.get("providerPublishTime"))
        or utc_now()
    )

    provider = content.get("provider") or item.get("publisher") or {}
    source = (
        provider.get("displayName")
        if isinstance(provider, dict)
        else str(provider)
    ) or item.get("publisher") or "Unknown source"

    canonical = content.get("canonicalUrl") or {}
    click_url = content.get("clickThroughUrl") or {}
    url = (
        canonical.get("url") if isinstance(canonical, dict) else None
    ) or (
        click_url.get("url") if isinstance(click_url, dict) else None
    ) or item.get("link")

    return {
        "headline": title.strip(),
        "published": published,
        "source": str(source),
        "url": url if isinstance(url, str) else "",
    }


def safe_external_url(value: Any) -> str:
    """Accept only well-formed HTTPS URLs suitable for external links."""
    if not isinstance(value, str):
        return ""
    url = value.strip()
    if not url or any(ord(char) < 32 or char.isspace() for char in url):
        return ""
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ""
        parsed.port
    except ValueError:
        return ""
    return url


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str, period: str) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Fetch price history and reuse cached ticker metadata and news."""
    started = time.perf_counter()
    instrument = yf.Ticker(ticker)
    prices = instrument.history(period=period, auto_adjust=False)
    log_timing("yahoo_price_retrieval", started)

    currency, news = fetch_company_metadata(ticker)
    return prices, news, currency


@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_metadata(ticker: str) -> tuple[str, list[dict[str, Any]]]:
    """Cache currency and news independently of the selected price-history period."""
    started = time.perf_counter()
    instrument = yf.Ticker(ticker)
    try:
        currency = instrument.fast_info.get("currency") or "USD"
    except Exception:
        currency = "USD"

    try:
        raw_news = instrument.news or []
    except Exception:
        raw_news = []

    news = [
        normalized
        for item in raw_news
        if isinstance(item, dict)
        for normalized in [extract_news_item(item)]
        if normalized
    ]
    log_timing("yahoo_news_retrieval_and_normalization", started)
    return currency, news


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str) -> Any:
    """Cache the Gemini client instance so each page rerun does not recreate it."""
    return genai.Client(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_company(query: str) -> tuple[str, str, str] | None:
    """Cache the small company metadata lookup for a faster sidebar render."""
    return COMPANIES.get(query.casefold())


@st.cache_data(ttl=3600, show_spinner=False)
def keyword_classification(headline: str) -> dict[str, Any]:
    """Fallback classification for headlines when Gemini is unavailable."""
    text = headline.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in text)
    negative = sum(1 for word in NEGATIVE_WORDS if word in text)
    score = (
        0.0 if positive == negative else (positive - negative) / max(positive + negative, 1)
    )
    score = max(-1.0, min(1.0, score))

    catalyst = "Other"
    for category, keywords in CATALYST_KEYWORDS.items():
        if any(keyword.strip() in text for keyword in keywords):
            catalyst = category
            break

    if abs(score) >= 0.6:
        impact = "High"
    elif abs(score) >= 0.25:
        impact = "Medium"
    else:
        impact = "Low"

    return {"score": score, "catalyst": catalyst, "impact": impact}


def relevance_score(headline: str) -> float:
    """Prioritize financial-material headlines for the AI model when the list is large."""
    text = headline.casefold()
    topic_hits = 0
    for terms in CATALYST_KEYWORDS.values():
        for term in terms:
            if term.strip() in text:
                topic_hits += 1
    fallback = keyword_classification(headline)
    return topic_hits * 2.0 + IMPACT_WEIGHTS.get(fallback["impact"], 1) + abs(fallback["score"])


def parse_json_response(text: str) -> Any:
    """Parse JSON from a model response, including fenced JSON."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if not match:
            raise
        return json.loads(match.group(1))


def normalize_classification(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Gemini classification record."""
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(-1.0, min(1.0, score))

    raw_catalyst = str(data.get("catalyst", "Other")).strip()
    catalyst_lookup = {item.casefold(): item for item in CATALYSTS}
    catalyst = catalyst_lookup.get(raw_catalyst.casefold(), "Other")

    raw_impact = str(data.get("impact", "Low")).strip().title()
    impact = raw_impact if raw_impact in IMPACTS else "Low"
    return {"score": score, "catalyst": catalyst, "impact": impact}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_gemini_classification(api_key: str, headline_batch: tuple[str, ...], model_name: str) -> dict[int, dict[str, Any]]:
    """Run Gemini once per unique batch of selected headlines to avoid duplicate calls."""
    if not api_key or not headline_batch:
        return {}

    prompt = (
        "Classify these financial headlines. Return a JSON array with one object per headline in the same order. "
        "Each object must have: {'id': int, 'score': float, 'catalyst': string, 'impact': 'High'|'Medium'|'Low'}. "
        f"Allowed catalyst values: {', '.join(CATALYSTS)}. "
        "Use 0 for ambiguous neutral stories. Do not invent facts. "
        f"Headlines: {json.dumps([{'id': i, 'headline': headline} for i, headline in enumerate(headline_batch)], ensure_ascii=False, separators=(',', ':'))}"
    )

    for attempt in range(3):
        try:
            response = get_gemini_client(api_key).models.generate_content(
                model=model_name,
                contents=prompt,
            )
            parsed = parse_json_response(response.text or "")
            records = (
                parsed
                if isinstance(parsed, list)
                else parsed.get("items", [])
                if isinstance(parsed, dict)
                else []
            )
            result: dict[int, dict[str, Any]] = {}
            for record in records:
                if isinstance(record, dict) and "id" in record:
                    try:
                        record_id = int(record["id"])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= record_id < len(headline_batch):
                        result[record_id] = normalize_classification(record)
            return result
        except Exception:
            if attempt < 2:
                time.sleep(0.35 * (attempt + 1))
    return {}


def gemini_classify(
    api_key: str | None,
    headlines: list[dict[str, Any]],
    model_name: str = "gemini-2.0-flash",
) -> dict[int, dict[str, Any]]:
    """Classify only the newest 15 headlines when there are many to keep the cost and latency low."""
    if not api_key or not headlines:
        return {}

    selected_indices = sorted(
        range(len(headlines)),
        key=lambda idx: headlines[idx]["published"],
        reverse=True,
    )[:15]
    selected_headlines = tuple(headlines[i]["headline"] for i in selected_indices)
    if not selected_headlines:
        return {}

    started = time.perf_counter()
    batch = cached_gemini_classification(api_key, selected_headlines, model_name)
    log_timing("gemini_sentiment_classification", started)

    mapped: dict[int, dict[str, Any]] = {}
    for local_index, classification in batch.items():
        if local_index < len(selected_indices):
            mapped[selected_indices[local_index]] = classification
    return mapped


def score_headlines(
    headlines: list[dict[str, Any]],
    api_key: str | None,
) -> tuple[pd.DataFrame, str]:
    """Score the headlines, falling back to deterministic keyword scoring when AI is unavailable."""
    columns = [
        "published", "date", "headline", "source", "url",
        "score", "sentiment", "catalyst", "impact", "recency_weight",
    ]
    if not headlines:
        return pd.DataFrame(columns=columns), "fallback"

    started = time.perf_counter()
    classifications = gemini_classify(api_key, headlines) if api_key else {}
    log_timing("sentiment_pipeline", started)

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(headlines):
        classification = classifications.get(i) or keyword_classification(item["headline"])
        score = float(classification["score"])
        rows.append(
            {
                **item,
                **classification,
                "date": item["published"].date(),
                "sentiment": (
                    "positive" if score > 0.10
                    else "negative" if score < -0.10
                    else "neutral"
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns), "fallback"

    now = pd.Timestamp.now(tz="UTC")
    ages_days = ((now - pd.to_datetime(df["published"], utc=True)).dt.total_seconds() / 86400).clip(lower=0)
    df["recency_weight"] = (0.5 ** (ages_days / 7.0)).clip(lower=0.01)
    return df, "gemini" if classifications else "fallback"


@st.cache_data(show_spinner=False)
def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested impact-weighted ranking, now cached for quick rerenders."""
    ranked = df.copy()
    ranked["rank_score"] = (1 + ranked["score"].abs()) * ranked["impact"].map(IMPACT_WEIGHTS).fillna(1)
    return ranked.sort_values(["rank_score", "published"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def prepare_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare grouped daily sentiment once and reuse it in multiple charts."""
    if df.empty:
        return pd.DataFrame(columns=["date", "score"])
    return df.groupby("date", as_index=False)["score"].mean().sort_values("date")


@st.cache_data(show_spinner=False)
def summarize_catalysts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute catalyst counts and mean scores once for KPIs and charts."""
    columns = ["catalyst", "Headlines", "MeanSentiment"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    return (
        df.groupby("catalyst", as_index=False)
        .agg(Headlines=("headline", "count"), MeanSentiment=("score", "mean"))
        .sort_values("Headlines", ascending=True)
    )


def weighted_average_sentiment(df: pd.DataFrame) -> float | None:
    """Calculate recency-weighted average sentiment."""
    if df.empty:
        return None
    weights = df["recency_weight"].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return float(df["score"].mean())
    return float((df["score"] * weights).sum() / total)


def calculate_confidence(df: pd.DataFrame) -> int:
    """Estimate confidence using sample count, agreement, and recency."""
    if df.empty:
        return 0

    count_factor = 1 - math.exp(-len(df) / 5.0)
    deviation = float(df["score"].std(ddof=0)) if len(df) > 1 else 0.0
    agreement_factor = 1 - min(deviation, 0.75) / 0.75
    recency_factor = float(df["recency_weight"].mean())
    confidence = 100 * (
        0.35 * count_factor
        + 0.40 * agreement_factor
        + 0.25 * recency_factor
    )
    return int(round(max(0, min(100, confidence))))


def top_catalyst(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "—"
    return str(stats.loc[stats["Headlines"].idxmax(), "catalyst"])


def catalyst_extreme(stats: pd.DataFrame, bullish: bool) -> str:
    """Return the catalyst with the highest/lowest mean sentiment."""
    if stats.empty:
        return "—"
    index = stats["MeanSentiment"].idxmax() if bullish else stats["MeanSentiment"].idxmin()
    return str(stats.loc[index, "catalyst"])


@st.cache_data(ttl=3600, show_spinner=False)
def write_briefing(
    api_key: str,
    ticker: str,
    day_move: float,
    ranked_headlines: pd.DataFrame,
    average: float,
    confidence: int,
) -> str:
    """Generate an evidence-led analyst briefing with a compact prompt."""
    started = time.perf_counter()
    records = [
        {
            "published_utc": row.published.strftime("%Y-%m-%d %H:%M UTC"),
            "headline": row.headline,
            "sentiment_score": round(float(row.score), 2),
            "catalyst": row.catalyst,
            "impact": row.impact,
        }
        for row in ranked_headlines.head(5).itertuples()
    ]
    prompt = (
        f"Write a concise, neutral equity research briefing for {ticker}. "
        "Separate facts from interpretation, do not recommend a trade, and do not invent facts. "
        f"Weighted average sentiment: {average:+.2f}; latest session move: {day_move:+.2f}%; confidence: {confidence}/100. "
        f"Key headlines: {json.dumps(records, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = get_gemini_client(api_key).models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    log_timing("analyst_briefing_generation", started)
    if not text:
        raise ValueError("The briefing model returned an empty response.")
    return text


def classify_error(exc: Exception) -> str:
    """Return a user-friendly message without exposing credentials."""
    message = str(exc)
    if "api key" in message.lower() or "permission" in message.lower():
        return "AI services temporarily unavailable."
    if "quota" in message.lower() or "rate" in message.lower():
        return "AI services temporarily unavailable."
    return "AI services temporarily unavailable."


st.title("📈 Market Sentiment Tracker")
st.caption(
    "A news and market context dashboard. Sentiment and impact labels are "
    "analytical aids, not price forecasts."
)

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None
if not isinstance(api_key, str) or not api_key.strip():
    api_key = None

with st.sidebar:
    st.markdown("### Market research")
    st.markdown('<div class="sidebar-section-label">Company search</div>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Company search",
        value="Apple",
        placeholder="Try Apple, NVIDIA, Tesla",
        label_visibility="collapsed",
    ).strip()

    if search_query:
        match = resolve_company(search_query)
        if match:
            company_name, ticker_symbol, exchange = match
        else:
            company_name = search_query
            ticker_symbol = search_query.upper()
            exchange = "Exchange varies"
    else:
        company_name = "Apple Inc."
        ticker_symbol = "AAPL"
        exchange = "NASDAQ"

    st.markdown(
        f"""
        <div class="sidebar-asset-card">
            <div class="sidebar-asset-name">{html.escape(company_name)}</div>
            <div class="sidebar-asset-meta">{html.escape(ticker_symbol)} &bull; {html.escape(exchange)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Analysis period</div>', unsafe_allow_html=True)
    period_label = st.selectbox(
        "Analysis period",
        list(PERIODS),
        index=1,
        label_visibility="collapsed",
    )
    period_code, period_days = PERIODS[period_label]

    if api_key:
        st.markdown(
            '<div class="sidebar-ai-status">🤖 AI Analysis Enabled</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("↻ Refresh market data", use_container_width=True):
        fetch_market_data.clear()

    if st.button("Clear cached analysis", use_container_width=True):
        fetch_market_data.clear()
        fetch_company_metadata.clear()
        cached_gemini_classification.clear()
        keyword_classification.clear()
        prepare_daily_sentiment.clear()
        add_ranking.clear()
        summarize_catalysts.clear()

if not ticker_symbol:
    st.warning("Enter a ticker symbol to continue.")
    st.stop()

# End-user ticker variable for the rest of the app.
ticker = ticker_symbol

try:
    started = time.perf_counter()
    with st.spinner(f"Loading {ticker} market data and news…"):
        prices, all_news, currency = fetch_market_data(ticker, period_code)
    log_timing("data_load_total", started)
except Exception as exc:
    st.error(f"Could not load market data for {ticker}: {exc}")
    st.stop()

cutoff = utc_now() - timedelta(days=period_days)
news = [item for item in all_news if item["published"] >= cutoff]
scored, method = score_headlines(news, api_key)
ranked = add_ranking(scored) if not scored.empty else scored.copy()
daily = prepare_daily_sentiment(scored)
catalyst_stats = summarize_catalysts(scored)

if prices is not None and not prices.empty and "Close" in prices.columns:
    close_prices = prices["Close"].dropna()
    latest_close = float(close_prices.iloc[-1]) if not close_prices.empty else float("nan")
    if len(close_prices) > 1:
        previous_close = float(close_prices.iloc[-2])
        day_move = (
            (latest_close - previous_close) / previous_close * 100
            if previous_close
            else 0.0
        )
    else:
        day_move = 0.0
else:
    latest_close = float("nan")
    day_move = 0.0

avg = weighted_average_sentiment(scored)
confidence = calculate_confidence(scored)
outlook = (
    "Bullish" if avg is not None and avg > 0.15
    else "Bearish" if avg is not None and avg < -0.15
    else "Neutral"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest close", f"{currency} {latest_close:,.2f}" if not math.isnan(latest_close) else "—")
k2.metric("Latest session", f"{day_move:+.2f}%")
k3.metric("Avg. sentiment", f"{avg:+.2f}" if avg is not None else "—")
k4.metric("Confidence", f"{confidence}%")
k5.metric("Headlines analysed", f"{len(scored)}")

summary_catalyst = top_catalyst(catalyst_stats)
st.markdown(
    f"""
    <div class="summary-card">
      <div class="summary-title">Market summary · {html.escape(ticker)}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:16px">
        <div><div class="summary-label">Overall outlook</div>
          <div class="summary-value">{outlook}</div></div>
        <div><div class="summary-label">Average sentiment</div>
          <div class="summary-value">{f'{avg:+.2f}' if avg is not None else '—'}</div></div>
        <div><div class="summary-label">Confidence score</div>
          <div class="summary-value">{confidence}%</div></div>
        <div><div class="summary-label">Headlines analysed</div>
          <div class="summary-value">{len(scored)}</div></div>
        <div><div class="summary-label">Top catalyst</div>
          <div class="summary-value">{html.escape(summary_catalyst)}</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if method != "gemini":
    st.info("AI analysis unavailable.")

if avg is not None:
    if avg > 0.50:
        st.success(f"Very bullish news sentiment detected ({avg:+.2f}).")
    elif avg < -0.50:
        st.error(f"Very bearish news sentiment detected ({avg:+.2f}).")

if not scored.empty and confidence < 40:
    st.warning(
        f"Low confidence ({confidence}%). This estimate may be less reliable "
        "because of headline count, sentiment disagreement, or headline age."
    )

divergence = (
    avg is not None
    and ((avg > 0.15 and day_move < -0.5) or (avg < -0.15 and day_move > 0.5))
)
if divergence:
    st.warning(
        "Potential sentiment/price divergence: headline tone and the latest "
        "session move point in opposite directions. This is a flag to investigate, "
        "not proof of mispricing or causation."
    )


tab_overview, tab_news, tab_catalysts, tab_brief = st.tabs(
    ["Overview", "Headlines", "Catalysts", "Analyst briefing"]
)

with tab_overview:
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("**Price and news sentiment**")
        if prices is None or prices.empty:
            st.info("No price history was returned for this ticker.")
        else:
            chart = go.Figure()
            chart.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices["Close"],
                    name="Close",
                    mode="lines",
                    line=dict(width=2.5, color="#3478c8"),
                )
            )
            if not daily.empty:
                chart.add_trace(
                    go.Bar(
                        x=daily["date"],
                        y=daily["score"],
                        name="Daily news sentiment",
                        opacity=0.42,
                        yaxis="y2",
                    )
                )
            chart.update_layout(
                height=430,
                margin=dict(t=20, b=10, l=10, r=10),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12),
                yaxis=dict(title=f"Price ({currency})", showgrid=True),
                yaxis2=dict(
                    title="Sentiment (-1 to +1)",
                    overlaying="y",
                    side="right",
                    range=[-1, 1],
                    showgrid=False,
                ),
            )
            st.plotly_chart(chart, use_container_width=True)
            st.caption(
                "Sentiment is shown only on dates with returned headlines. "
                "Yahoo Finance may provide a limited news history."
            )

    with right:
        st.markdown("**Sentiment distribution**")
        if scored.empty:
            st.info("No period-matched headlines were returned.")
        else:
            counts = scored["sentiment"].value_counts()
            pie = go.Figure(
                go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[
                        int(counts.get("positive", 0)),
                        int(counts.get("neutral", 0)),
                        int(counts.get("negative", 0)),
                    ],
                    hole=0.62,
                    marker=dict(colors=["#23a26d", "#8a96a3", "#d94f4f"]),
                )
            )
            pie.update_layout(
                height=265,
                margin=dict(t=10, b=10, l=5, r=5),
                showlegend=True,
            )
            st.plotly_chart(pie, use_container_width=True)

            st.markdown("**Daily sentiment trend**")
            trend = go.Figure()
            trend.add_hrect(
                y0=0.15, y1=1,
                fillcolor="rgba(35, 162, 109, 0.10)",
                line_width=0,
                layer="below",
            )
            trend.add_hrect(
                y0=-1, y1=-0.15,
                fillcolor="rgba(217, 79, 79, 0.10)",
                line_width=0,
                layer="below",
            )
            trend.add_trace(
                go.Scatter(
                    x=daily["date"],
                    y=daily["score"],
                    mode="lines+markers",
                    name="Daily average",
                    line=dict(color="#3478c8", width=2.5),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(52, 120, 200, 0.16)",
                    hovertemplate="%{x}<br>Sentiment: %{y:.2f}<extra></extra>",
                )
            )
            trend.add_hline(y=0, line_dash="dot", line_color="#7d8790")
            trend.add_hline(y=0.15, line_dash="dash", line_color="rgba(35, 162, 109, .55)")
            trend.add_hline(y=-0.15, line_dash="dash", line_color="rgba(217, 79, 79, .55)")
            trend.update_yaxes(range=[-1, 1], title="Average score")
            trend.update_layout(
                height=260,
                margin=dict(t=10, b=5, l=5, r=5),
                showlegend=False,
                hovermode="x",
            )
            st.plotly_chart(trend, use_container_width=True)

with tab_news:
    if scored.empty:
        st.info("No recent headlines were returned for this ticker in the selected period.")
    else:
        filt = st.radio(
            "Filter headlines",
            ["All", "Positive", "Neutral", "Negative"],
            horizontal=True,
        )
        view = scored if filt == "All" else scored[scored["sentiment"] == filt.lower()]
        view = view.sort_values(["recency_weight", "score"], ascending=[False, False])

        for row in view.itertuples():
            title = html.escape(str(row.headline))
            safe_url = safe_external_url(row.url)
            link = (
                f'<a href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if safe_url
                else f"<b>{title}</b>"
            )
            st.markdown(
                f'<div class="news">'
                f'<span class="pill {html.escape(str(row.sentiment))}">'
                f'{html.escape(str(row.sentiment).title())} {row.score:+.2f}</span>'
                f'<span class="pill neutral">{html.escape(str(row.impact))} impact</span>'
                f'<span class="meta">{html.escape(str(row.catalyst))}</span>'
                f'<div style="margin-top:7px">{link}</div>'
                f'<div class="meta">{html.escape(str(row.source))} · '
                f'{row.published:%d %b %Y, %H:%M UTC}</div></div>',
                unsafe_allow_html=True,
            )

        csv_data = scored[
            [
                "published", "headline", "sentiment", "score", "catalyst",
                "impact", "source", "url",
            ]
        ].to_csv(index=False)
        st.download_button(
            "⬇️ Download analysed headlines (CSV)",
            csv_data,
            f"{ticker}_sentiment.csv",
            "text/csv",
        )

with tab_catalysts:
    if scored.empty:
        st.info("Catalyst analysis needs headlines in the selected period.")
    else:
        if method != "gemini":
            st.info(
                "Catalyst categories and impact levels use basic keyword fallback "
                "defaults because AI analysis is unavailable."
            )

        high_impact_count = int((scored["impact"] == "High").sum())
        catalyst_kpis = st.columns(4)
        catalyst_kpis[0].metric("High impact headlines", high_impact_count)
        catalyst_kpis[1].metric("Top catalyst", summary_catalyst)
        catalyst_kpis[2].metric("Most bullish catalyst", catalyst_extreme(catalyst_stats, True))
        catalyst_kpis[3].metric("Most bearish catalyst", catalyst_extreme(catalyst_stats, False))

        left, right = st.columns(2)
        with left:
            st.markdown("**Headline volume by catalyst**")
            catalyst_chart = go.Figure(
                go.Bar(
                    x=catalyst_stats["Headlines"],
                    y=catalyst_stats["catalyst"],
                    orientation="h",
                    marker=dict(
                        color=catalyst_stats["MeanSentiment"],
                        colorscale=[[0, "#d94f4f"], [0.5, "#a0a8b0"], [1, "#23a26d"]],
                        cmin=-1,
                        cmax=1,
                        colorbar=dict(title="Avg. score"),
                    ),
                    hovertemplate=(
                        "%{y}<br>Headlines: %{x}<br>"
                        "Average sentiment: %{marker.color:.2f}<extra></extra>"
                    ),
                )
            )
            catalyst_chart.update_layout(
                height=360,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Headline count",
                yaxis_title="",
            )
            st.plotly_chart(catalyst_chart, use_container_width=True)

        with right:
            st.markdown("**Potential news impact mix**")
            impact_counts = scored["impact"].value_counts()
            impact_chart = go.Figure(
                go.Pie(
                    labels=IMPACTS,
                    values=[int(impact_counts.get(level, 0)) for level in IMPACTS],
                    hole=0.55,
                    marker=dict(colors=["#d94f4f", "#e7a943", "#7d9bb8"]),
                )
            )
            impact_chart.update_layout(
                height=360,
                margin=dict(t=10, b=10, l=5, r=5),
                showlegend=True,
            )
            st.plotly_chart(impact_chart, use_container_width=True)

        st.markdown("**Highest-ranked headlines**")
        display = ranked.head(10)[
            [
                "published", "headline", "sentiment", "score",
                "catalyst", "impact", "rank_score", "source",
            ]
        ].copy()
        display["published"] = pd.to_datetime(display["published"], utc=True).dt.strftime(
            "%d %b %Y %H:%M UTC"
        )
        display["score"] = display["score"].map(lambda value: f"{value:+.2f}")
        display["rank_score"] = display["rank_score"].map(lambda value: f"{value:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Ranking = (1 + absolute sentiment score) × impact weight "
            "(High=3, Medium=2, Low=1). It is a prioritisation heuristic, "
            "not a forecast of price movement."
        )

with tab_brief:
    if scored.empty:
        st.info("The briefing needs at least one headline in the selected period.")
    elif not api_key:
        st.info("AI services temporarily unavailable.")
    else:
        st.caption(
            "On-demand, evidence-led summary of up to five highest-ranked headlines."
        )
        top = ranked.head(5)
        state_key = f"brief_{ticker}_{period_label}"
        if st.button("✨ Generate analyst briefing", type="primary"):
            started = time.perf_counter()
            try:
                with st.spinner("Preparing briefing…"):
                    st.session_state[state_key] = write_briefing(
                        api_key,
                        ticker,
                        day_move,
                        top,
                        avg if avg is not None else 0.0,
                        confidence,
                    )
            except Exception:
                st.error("AI services temporarily unavailable.")
            finally:
                log_timing("analyst_briefing_generation_total", started)

        if state_key in st.session_state:
            st.markdown('<div class="brief">', unsafe_allow_html=True)
            st.markdown(st.session_state[state_key])
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("See the ranked headlines used"):
                for i, row in enumerate(top.itertuples(), 1):
                    st.write(
                        f"{i}. [{row.impact} impact · {row.catalyst} · "
                        f"{row.score:+.2f}] {row.headline}"
                    )

st.divider()
st.caption(
    "Important: Sentiment scores, confidence estimates, and impact rankings are "
    "analytical aids only. They do not measure the probability of a price move "
    "and should not be used as standalone buy/sell signals."
)

log_timing("total_page_load", _PAGE_STARTED)
