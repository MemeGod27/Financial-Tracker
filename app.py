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
MODEL_NAME = "gemini-2.0-flash"


def log_timing(operation: str, started: float) -> None:
    _LOGGER.info("perf operation=%s duration_ms=%.1f", operation, (time.perf_counter() - started) * 1000)


st.set_page_config(page_title="Market Sentiment Tracker", page_icon="📈", layout="wide")

CATALYSTS = ["Earnings", "Guidance", "Management", "Product", "Competition", "Regulation", "Partnership", "M&A", "Analyst", "Legal", "Macro", "Dividend", "Buyback", "Other"]
IMPACTS = ["High", "Medium", "Low"]
IMPACT_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
PERIODS = {"1 month": ("1mo", 30), "3 months": ("3mo", 90), "6 months": ("6mo", 180), "1 year": ("1y", 365)}
# A curated picker universe; match aliases, company names, and symbols by partial input.
COMPANIES = [
    {"name": "Apple Inc.", "ticker": "AAPL", "exchange": "NASDAQ", "aliases": ("apple",)},
    {"name": "NVIDIA Corporation", "ticker": "NVDA", "exchange": "NASDAQ", "aliases": ("nvidia",)},
    {"name": "Tesla, Inc.", "ticker": "TSLA", "exchange": "NASDAQ", "aliases": ("tesla",)},
    {"name": "Microsoft Corporation", "ticker": "MSFT", "exchange": "NASDAQ", "aliases": ("microsoft",)},
    {"name": "Amazon.com, Inc.", "ticker": "AMZN", "exchange": "NASDAQ", "aliases": ("amazon",)},
    {"name": "Alphabet Inc.", "ticker": "GOOGL", "exchange": "NASDAQ", "aliases": ("alphabet", "google")},
    {"name": "Meta Platforms, Inc.", "ticker": "META", "exchange": "NASDAQ", "aliases": ("meta", "facebook")},
]
POSITIVE_WORDS = {"beat", "beats", "beating", "growth", "grows", "strong", "record", "surge", "surges", "profit", "profits", "raises", "raised", "upgrade", "upgraded", "buy", "bullish", "expansion", "outperform", "positive", "partnership", "approval", "approved", "dividend", "buyback"}
NEGATIVE_WORDS = {"miss", "misses", "missed", "decline", "declines", "weak", "loss", "losses", "cuts", "cut", "downgrade", "downgraded", "sell", "bearish", "lawsuit", "probe", "investigation", "recall", "warning", "layoffs", "negative", "debt", "default", "resigns", "resignation"}
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
    "Dividend": ("dividend", "payout"), "Buyback": ("buyback", "repurchase", "share repurchase"),
}

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
[data-testid="stHeader"] { background: #0f172a; }
.block-container { padding-top: 1.4rem; padding-bottom: 1.6rem; max-width: 1600px; }
section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1rem .9rem; }
.sidebar-section-label { color: #94a3b8; font-size: .72rem; font-weight: 700; letter-spacing: .09em; margin: .35rem 0 .45rem; text-transform: uppercase; }
.sidebar-asset-card, .summary-card, .news, .trust-card { background: #111827; border: 1px solid #1e293b; border-radius: 8px; }
.sidebar-asset-card { margin: .55rem 0 1.15rem; padding: .85rem .95rem; }
.sidebar-asset-name { color: #f8fafc; font-size: 1rem; font-weight: 700; line-height: 1.35; }
.sidebar-asset-meta { color: #94a3b8; font-size: .8rem; font-weight: 600; margin-top: .25rem; }
.summary-card { padding: 1rem 1.15rem; margin: .8rem 0 1.2rem; }
.summary-heading { border-bottom: 1px solid #1e293b; color: #f8fafc; font-size: .84rem; font-weight: 700; letter-spacing: .06em; margin-bottom: .9rem; padding-bottom: .7rem; text-transform: uppercase; }
.summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; }
.summary-label { color: #94a3b8; font-size: .75rem; margin-bottom: .35rem; }
.summary-value { color: #f8fafc; font-size: 1.18rem; font-weight: 700; line-height: 1.3; }
.summary-value.positive { color: #22c55e; } .summary-value.negative { color: #ef4444; } .summary-value.neutral { color: #94a3b8; }
.news { padding: 12px 14px; margin: 8px 0; }
.pill { display: inline-block; border: 1px solid #1e293b; border-radius: 4px; padding: 2px 7px; margin-right: 6px; font-size: .75rem; font-weight: 650; background: #0f172a; }
.pill.positive { color: #22c55e; } .pill.negative { color: #ef4444; } .pill.neutral { color: #94a3b8; }
.meta { color: #94a3b8; font-size: .8rem; margin-top: 5px; }
.trust-card { display: flex; flex-wrap: wrap; gap: .55rem 2rem; margin-top: .8rem; padding: .75rem .9rem; }
.trust-item { color: #94a3b8; font-size: .78rem; } .trust-item strong { color: #e2e8f0; font-weight: 650; }
.sidebar-ai-status { background: #111827; border: 1px solid #1e293b; border-radius: 6px; color: #22c55e; font-size: .79rem; font-weight: 650; margin-top: .7rem; padding: .55rem .7rem; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #111827; border-color: #1e293b; color: #f8fafc; }
section[data-testid="stSidebar"] [data-testid="stButton"] > button { background: #111827; border: 1px solid #1e293b; color: #f8fafc; }
section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover { border-color: #3b82f6; color: #f8fafc; }
[data-testid="stMetric"] { background: #111827; border: 1px solid #1e293b; border-radius: 7px; padding: .65rem .8rem; }
a { color: #60a5fa; }
@media (max-width: 700px) { .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
""", unsafe_allow_html=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: Any) -> datetime | None:
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
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = content.get("title") or item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    published = as_utc(content.get("pubDate")) or as_utc(content.get("providerPublishTime")) or as_utc(item.get("providerPublishTime")) or utc_now()
    provider = content.get("provider") or item.get("publisher") or {}
    source = (provider.get("displayName") if isinstance(provider, dict) else str(provider)) or item.get("publisher") or "Unknown source"
    canonical, click_url = content.get("canonicalUrl") or {}, content.get("clickThroughUrl") or {}
    url = (canonical.get("url") if isinstance(canonical, dict) else None) or (click_url.get("url") if isinstance(click_url, dict) else None) or item.get("link")
    return {"headline": title.strip(), "published": published, "source": str(source), "url": url if isinstance(url, str) else ""}


def safe_external_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    url = value.strip()
    if not url or any(ord(char) < 32 or char.isspace() for char in url):
        return ""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.casefold() != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return ""
        parsed.port
    except ValueError:
        return ""
    return url


@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_metadata(ticker: str) -> tuple[str, list[dict[str, Any]]]:
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
    news = [normalized for item in raw_news if isinstance(item, dict) for normalized in [extract_news_item(item)] if normalized]
    log_timing("yahoo_news_retrieval_and_normalization", started)
    return currency, news


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str, period: str) -> tuple[pd.DataFrame, list[dict[str, Any]], str, datetime]:
    started = time.perf_counter()
    prices = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    log_timing("yahoo_price_retrieval", started)
    currency, news = fetch_company_metadata(ticker)
    return prices, news, currency, utc_now()


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str) -> Any:
    return genai.Client(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def keyword_classification(headline: str) -> dict[str, Any]:
    text = headline.lower()
    positive = sum(word in text for word in POSITIVE_WORDS)
    negative = sum(word in text for word in NEGATIVE_WORDS)
    score = 0.0 if positive == negative else (positive - negative) / max(positive + negative, 1)
    score = max(-1.0, min(1.0, score))
    catalyst = "Other"
    for category, keywords in CATALYST_KEYWORDS.items():
        if any(keyword.strip() in text for keyword in keywords):
            catalyst = category
            break
    impact = "High" if abs(score) >= .6 else "Medium" if abs(score) >= .25 else "Low"
    return {"score": score, "catalyst": catalyst, "impact": impact}


def parse_json_response(text: str) -> Any:
    cleaned = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if not match:
            raise
        return json.loads(match.group(1))


def normalize_classification(data: dict[str, Any]) -> dict[str, Any]:
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(-1.0, min(1.0, score))
    catalyst_lookup = {item.casefold(): item for item in CATALYSTS}
    catalyst = catalyst_lookup.get(str(data.get("catalyst", "Other")).strip().casefold(), "Other")
    impact = str(data.get("impact", "Low")).strip().title()
    return {"score": score, "catalyst": catalyst, "impact": impact if impact in IMPACTS else "Low"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_gemini_classification(api_key: str, headlines: tuple[str, ...], model_name: str) -> dict[int, dict[str, Any]]:
    if not api_key or not headlines:
        return {}
    prompt = ("Classify these financial headlines. Return a JSON array with one object per headline in the same order. "
              "Each object must have id, score (-1 to 1), catalyst, and impact (High, Medium, Low). "
              f"Allowed catalyst values: {', '.join(CATALYSTS)}. Use 0 for ambiguous neutral stories; do not invent facts. "
              f"Headlines: {json.dumps([{'id': i, 'headline': item} for i, item in enumerate(headlines)], ensure_ascii=False, separators=(',', ':'))}")
    for attempt in range(3):
        try:
            response = get_gemini_client(api_key).models.generate_content(model=model_name, contents=prompt)
            parsed = parse_json_response(response.text or "")
            records = parsed if isinstance(parsed, list) else parsed.get("items", []) if isinstance(parsed, dict) else []
            result = {}
            for record in records:
                if isinstance(record, dict) and "id" in record:
                    try:
                        index = int(record["id"])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(headlines):
                        result[index] = normalize_classification(record)
            return result
        except Exception:
            if attempt < 2:
                time.sleep(.35 * (attempt + 1))
    return {}


def gemini_classify(api_key: str | None, headlines: list[dict[str, Any]], model_name: str = MODEL_NAME) -> dict[int, dict[str, Any]]:
    if not api_key or not headlines:
        return {}
    selected = sorted(range(len(headlines)), key=lambda idx: headlines[idx]["published"], reverse=True)[:15]
    started = time.perf_counter()
    try:
        batch = cached_gemini_classification(api_key, tuple(headlines[i]["headline"] for i in selected), model_name)
    except Exception:
        _LOGGER.warning("Gemini sentiment classification unavailable; using fallback.")
        batch = {}
    log_timing("gemini_sentiment_classification", started)
    return {selected[i]: value for i, value in batch.items() if i < len(selected)}


def score_headlines(headlines: list[dict[str, Any]], api_key: str | None) -> tuple[pd.DataFrame, str]:
    columns = ["published", "date", "headline", "source", "url", "score", "sentiment", "catalyst", "impact", "recency_weight"]
    if not headlines:
        return pd.DataFrame(columns=columns), "fallback"
    started = time.perf_counter()
    classifications = gemini_classify(api_key, headlines) if api_key else {}
    log_timing("sentiment_pipeline", started)
    rows = []
    for i, item in enumerate(headlines):
        classification = classifications.get(i) or keyword_classification(item["headline"])
        score = float(classification["score"])
        rows.append({**item, **classification, "date": item["published"].date(), "sentiment": "positive" if score > .1 else "negative" if score < -.1 else "neutral"})
    df = pd.DataFrame(rows)
    now = pd.Timestamp.now(tz="UTC")
    ages = ((now - pd.to_datetime(df["published"], utc=True)).dt.total_seconds() / 86400).clip(lower=0)
    df["recency_weight"] = (0.5 ** (ages / 7.0)).clip(lower=.01)
    return df, "gemini" if classifications else "fallback"


@st.cache_data(show_spinner=False)
def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_score"] = (1 + ranked["score"].abs()) * ranked["impact"].map(IMPACT_WEIGHTS).fillna(1)
    return ranked.sort_values(["rank_score", "published"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def prepare_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "score"]) if df.empty else df.groupby("date", as_index=False)["score"].mean().sort_values("date")


@st.cache_data(show_spinner=False)
def summarize_catalysts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["catalyst", "Headlines", "MeanSentiment"])
    return df.groupby("catalyst", as_index=False).agg(Headlines=("headline", "count"), MeanSentiment=("score", "mean")).sort_values("Headlines", ascending=True)


def weighted_average_sentiment(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    weights = df["recency_weight"].astype(float)
    return float((df["score"] * weights).sum() / weights.sum()) if float(weights.sum()) > 0 else float(df["score"].mean())


def calculate_confidence(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    count_factor = 1 - math.exp(-len(df) / 5.0)
    deviation = float(df["score"].std(ddof=0)) if len(df) > 1 else 0.0
    agreement_factor = 1 - min(deviation, .75) / .75
    confidence = 100 * (.35 * count_factor + .4 * agreement_factor + .25 * float(df["recency_weight"].mean()))
    return int(round(max(0, min(100, confidence))))


def top_catalyst(stats: pd.DataFrame) -> str:
    return "—" if stats.empty else str(stats.loc[stats["Headlines"].idxmax(), "catalyst"])


def catalyst_extreme(stats: pd.DataFrame, bullish: bool) -> str:
    if stats.empty:
        return "—"
    idx = stats["MeanSentiment"].idxmax() if bullish else stats["MeanSentiment"].idxmin()
    return str(stats.loc[idx, "catalyst"])


@st.cache_data(ttl=3600, show_spinner=False)
def write_briefing(api_key: str, ticker: str, day_move: float, ranked: pd.DataFrame, average: float, confidence: int) -> str:
    rows = [{"published_utc": row.published.strftime("%Y-%m-%d %H:%M UTC"), "headline": row.headline, "sentiment_score": round(float(row.score), 2), "catalyst": row.catalyst, "impact": row.impact} for row in ranked.head(5).itertuples()]
    prompt = (f"Write a concise, neutral equity research briefing for {ticker}. Separate facts from interpretation, "
              "do not recommend a trade, and do not invent facts. "
              f"Weighted average sentiment: {average:+.2f}; latest session move: {day_move:+.2f}%; confidence: {confidence}/100. "
              f"Key headlines: {json.dumps(rows, ensure_ascii=False, separators=(',', ':'))}")
    response = get_gemini_client(api_key).models.generate_content(model=MODEL_NAME, contents=prompt)
    text = (response.text or "").strip()
    if not text:
        raise ValueError("The briefing model returned an empty response.")
    return text


def company_matches(query: str) -> list[dict[str, str]]:
    needle = query.strip().casefold()
    if not needle:
        return COMPANIES.copy()
    def matches(company: dict[str, Any]) -> bool:
        fields = (company["name"], company["ticker"], company["exchange"], *company["aliases"])
        return any(needle in field.casefold() for field in fields)
    return [company for company in COMPANIES if matches(company)]


st.title("📈 Market Sentiment Tracker")
st.caption("Financial news and market context. Sentiment labels are analytical aids, not price forecasts.")
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None
if not isinstance(api_key, str) or not api_key.strip():
    api_key = None

with st.sidebar:
    st.markdown("### Market research")
    st.markdown('<div class="sidebar-section-label">Company search</div>', unsafe_allow_html=True)
    search_query = st.text_input("Company search", value="Apple", placeholder="Search name or ticker", label_visibility="collapsed", key="company_search").strip()
    matches = company_matches(search_query)
    if matches:
        selected = st.selectbox("Select a company", matches, format_func=lambda company: f"{company['name']}  ·  {company['ticker']} • {company['exchange']}", label_visibility="collapsed", key="company_picker")
        company_name, ticker_symbol, exchange = selected["name"], selected["ticker"], selected["exchange"]
    else:
        ticker_symbol = search_query.upper().strip()
        company_name, exchange = (ticker_symbol, "Exchange varies") if ticker_symbol else ("Apple Inc.", "NASDAQ")
        st.caption("No curated match. Enter a ticker symbol to search market data.")
    st.markdown(f'<div class="sidebar-asset-card"><div class="sidebar-asset-name">{html.escape(company_name)}</div><div class="sidebar-asset-meta">{html.escape(ticker_symbol)} &bull; {html.escape(exchange)}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Analysis period</div>', unsafe_allow_html=True)
    period_label = st.selectbox("Analysis period", list(PERIODS), index=1, label_visibility="collapsed")
    period_code, period_days = PERIODS[period_label]
    if api_key:
        st.markdown('<div class="sidebar-ai-status">AI sentiment analysis enabled</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("↻ Refresh market data", use_container_width=True):
        fetch_market_data.clear()
        fetch_company_metadata.clear()
        st.rerun()
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
ticker = ticker_symbol

try:
    started = time.perf_counter()
    with st.spinner(f"Loading {ticker} market data and news…"):
        prices, all_news, currency, last_refresh = fetch_market_data(ticker, period_code)
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
        day_move = (latest_close - previous_close) / previous_close * 100 if previous_close else 0.0
    else:
        day_move = 0.0
else:
    latest_close, day_move = float("nan"), 0.0
avg = weighted_average_sentiment(scored)
confidence = calculate_confidence(scored)
outlook = "Bullish" if avg is not None and avg > .15 else "Bearish" if avg is not None and avg < -.15 else "Neutral"
outlook_class = "positive" if outlook == "Bullish" else "negative" if outlook == "Bearish" else "neutral"
summary_catalyst = top_catalyst(catalyst_stats)

price_cols = st.columns(3)
price_cols[0].metric("Latest close", f"{currency} {latest_close:,.2f}" if not math.isnan(latest_close) else "—")
price_cols[1].metric("Latest session", f"{day_move:+.2f}%")
price_cols[2].metric("Headlines analysed", f"{len(scored)}")
st.markdown(f"""
<div class="summary-card">
  <div class="summary-heading">Market intelligence &nbsp;·&nbsp; {html.escape(company_name)} ({html.escape(ticker)})</div>
  <div class="summary-grid">
    <div><div class="summary-label">Overall outlook</div><div class="summary-value {outlook_class}">{outlook}</div></div>
    <div><div class="summary-label">Sentiment score</div><div class="summary-value">{f'{avg:+.2f}' if avg is not None else '—'}</div></div>
    <div><div class="summary-label">Confidence estimate</div><div class="summary-value">{confidence}%</div></div>
    <div><div class="summary-label">Top catalyst</div><div class="summary-value">{html.escape(summary_catalyst)}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("Confidence is a heuristic based on headline count, score agreement, and recency; it is not a statistical probability.")
if method != "gemini":
    st.info("AI analysis unavailable; sentiment and catalyst labels use basic keyword-based estimates.")
if not scored.empty and confidence < 40:
    st.warning(f"Low confidence ({confidence}%). This estimate may be less reliable because of headline count, sentiment disagreement, or headline age.")
divergence = avg is not None and ((avg > .15 and day_move < -.5) or (avg < -.15 and day_move > .5))
if divergence:
    st.warning("Potential sentiment/price divergence: headline tone and the latest session move differ. This is a flag to investigate, not proof of mispricing or causation.")

tab_overview, tab_news, tab_catalysts, tab_brief = st.tabs(["Overview", "Headlines", "Catalysts", "Analyst briefing"])
with tab_overview:
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("**Price and news sentiment**")
        if prices is None or prices.empty:
            st.info("No price history was returned for this ticker.")
        else:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close", mode="lines", line=dict(width=2, color="#3b82f6")))
            if not daily.empty:
                chart.add_trace(go.Bar(x=daily["date"], y=daily["score"], name="Daily news sentiment", opacity=.42, yaxis="y2", marker_color="#94a3b8"))
            chart.update_layout(height=430, margin=dict(t=20, b=10, l=10, r=10), hovermode="x unified", legend=dict(orientation="h", y=1.12), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), yaxis=dict(title=f"Price ({currency})", showgrid=True, gridcolor="#1e293b"), yaxis2=dict(title="Sentiment (-1 to +1)", overlaying="y", side="right", range=[-1, 1], showgrid=False))
            st.plotly_chart(chart, use_container_width=True)
            st.caption("Sentiment is shown only on dates with returned headlines. Yahoo Finance may provide a limited news history.")
    with right:
        st.markdown("**Sentiment distribution**")
        if scored.empty:
            st.info("No period-matched headlines were returned.")
        else:
            counts = scored["sentiment"].value_counts()
            pie = go.Figure(go.Pie(labels=["Positive", "Neutral", "Negative"], values=[int(counts.get(k, 0)) for k in ("positive", "neutral", "negative")], hole=.62, marker=dict(colors=["#22c55e", "#94a3b8", "#ef4444"])))
            pie.update_layout(height=265, margin=dict(t=10, b=10, l=5, r=5), showlegend=True, paper_bgcolor="#0f172a", font=dict(color="#cbd5e1"))
            st.plotly_chart(pie, use_container_width=True)
            st.markdown("**Daily sentiment trend**")
            trend = go.Figure()
            trend.add_hrect(y0=.15, y1=1, fillcolor="rgba(34,197,94,.08)", line_width=0, layer="below")
            trend.add_hrect(y0=-1, y1=-.15, fillcolor="rgba(239,68,68,.08)", line_width=0, layer="below")
            trend.add_trace(go.Scatter(x=daily["date"], y=daily["score"], mode="lines+markers", name="Daily average", line=dict(color="#3b82f6", width=2), marker=dict(size=6), hovertemplate="%{x}<br>Sentiment: %{y:.2f}<extra></extra>"))
            trend.add_hline(y=0, line_dash="dot", line_color="#64748b")
            trend.add_hline(y=.15, line_dash="dash", line_color="#22c55e")
            trend.add_hline(y=-.15, line_dash="dash", line_color="#ef4444")
            trend.update_yaxes(range=[-1, 1], title="Average score")
            trend.update_layout(height=260, margin=dict(t=10, b=5, l=5, r=5), showlegend=False, hovermode="x", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), yaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(trend, use_container_width=True)

with tab_news:
    if scored.empty:
        st.info("No recent headlines were returned for this ticker in the selected period.")
    else:
        filt = st.radio("Filter headlines", ["All", "Positive", "Neutral", "Negative"], horizontal=True)
        view = scored if filt == "All" else scored[scored["sentiment"] == filt.lower()]
        view = view.sort_values(["recency_weight", "score"], ascending=[False, False])
        for row in view.itertuples():
            title = html.escape(str(row.headline))
            safe_url = safe_external_url(row.url)
            link = f'<a href="{html.escape(safe_url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>' if safe_url else f"<b>{title}</b>"
            st.markdown(f'<div class="news"><span class="pill {html.escape(str(row.sentiment))}">{html.escape(str(row.sentiment).title())} {row.score:+.2f}</span><span class="pill neutral">{html.escape(str(row.impact))} impact</span><span class="meta">{html.escape(str(row.catalyst))}</span><div style="margin-top:7px">{link}</div><div class="meta">{html.escape(str(row.source))} · {row.published:%d %b %Y, %H:%M UTC}</div></div>', unsafe_allow_html=True)
        csv_data = scored[["published", "headline", "sentiment", "score", "catalyst", "impact", "source", "url"]].to_csv(index=False)
        st.download_button("⬇️ Download analysed headlines (CSV)", csv_data, f"{ticker}_sentiment.csv", "text/csv")

with tab_catalysts:
    if scored.empty:
        st.info("Catalyst analysis needs headlines in the selected period.")
    else:
        high_count = int((scored["impact"] == "High").sum())
        kpis = st.columns(4)
        kpis[0].metric("High impact headlines", high_count)
        kpis[1].metric("Top catalyst", summary_catalyst)
        kpis[2].metric("Most bullish catalyst", catalyst_extreme(catalyst_stats, True))
        kpis[3].metric("Most bearish catalyst", catalyst_extreme(catalyst_stats, False))
        left, right = st.columns(2)
        with left:
            st.markdown("**Headline volume by catalyst**")
            figure = go.Figure(go.Bar(x=catalyst_stats["Headlines"], y=catalyst_stats["catalyst"], orientation="h", marker=dict(color=catalyst_stats["MeanSentiment"], colorscale=[[0, "#ef4444"], [.5, "#94a3b8"], [1, "#22c55e"]], cmin=-1, cmax=1, colorbar=dict(title="Avg. score"))))
            figure.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Headline count", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(figure, use_container_width=True)
        with right:
            st.markdown("**Potential news impact mix**")
            impact_counts = scored["impact"].value_counts()
            figure = go.Figure(go.Pie(labels=IMPACTS, values=[int(impact_counts.get(level, 0)) for level in IMPACTS], hole=.55, marker=dict(colors=["#ef4444", "#e7a943", "#94a3b8"])))
            figure.update_layout(height=360, margin=dict(t=10, b=10, l=5, r=5), showlegend=True, paper_bgcolor="#0f172a", font=dict(color="#cbd5e1"))
            st.plotly_chart(figure, use_container_width=True)
        st.markdown("**Highest-ranked headlines**")
        display = ranked.head(10)[["published", "headline", "sentiment", "score", "catalyst", "impact", "rank_score", "source"]].copy()
        display["published"] = pd.to_datetime(display["published"], utc=True).dt.strftime("%d %b %Y %H:%M UTC")
        display["score"] = display["score"].map(lambda value: f"{value:+.2f}")
        display["rank_score"] = display["rank_score"].map(lambda value: f"{value:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Ranking = (1 + absolute sentiment score) × impact weight (High=3, Medium=2, Low=1). This prioritises review; it is not a price forecast.")

with tab_brief:
    if scored.empty:
        st.info("The briefing needs at least one headline in the selected period.")
    elif not api_key:
        st.info("AI services temporarily unavailable.")
    else:
        st.caption("On-demand, evidence-led summary of up to five highest-ranked headlines.")
        top = ranked.head(5)
        state_key = f"brief_{ticker}_{period_label}"
        if st.button("✨ Generate analyst briefing", type="primary"):
            started = time.perf_counter()
            try:
                with st.spinner("Preparing briefing…"):
                    st.session_state[state_key] = write_briefing(api_key, ticker, day_move, top, avg if avg is not None else 0.0, confidence)
            except Exception:
                st.error("AI services temporarily unavailable.")
            finally:
                log_timing("analyst_briefing_generation_total", started)
        if state_key in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state[state_key])
            with st.expander("See the ranked headlines used"):
                for i, row in enumerate(top.itertuples(), 1):
                    st.write(f"{i}. [{row.impact} impact · {row.catalyst} · {row.score:+.2f}] {row.headline}")

st.divider()
st.caption("Sentiment scores, confidence estimates, and impact rankings are analytical aids only. They do not measure the probability of a price move and should not be used as standalone buy/sell signals.")
st.markdown(f'<div class="trust-card"><div class="trust-item"><strong>Data sources</strong> &nbsp; Yahoo Finance · Google Gemini</div><div class="trust-item"><strong>Last refresh</strong> &nbsp; {last_refresh.strftime("%Y-%m-%d %H:%M:%S UTC")}</div><div class="trust-item"><strong>Model</strong> &nbsp; {MODEL_NAME.replace("-", " ").replace("flash", "Flash")}</div></div>', unsafe_allow_html=True)
log_timing("total_page_load", _PAGE_STARTED)
