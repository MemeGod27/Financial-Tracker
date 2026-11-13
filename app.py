from __future__ import annotations

import html
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai


_PAGE_STARTED = time.perf_counter()
_LOGGER = logging.getLogger("market_sentiment_tracker.performance")
_LOGGER.setLevel(logging.INFO)

st.set_page_config(page_title="Market Sentiment Tracker", page_icon="📈", layout="wide")

CATALYSTS = [
    "Earnings", "Guidance", "Management", "Product", "Competition",
    "Regulation", "Partnership", "M&A", "Analyst", "Legal", "Macro",
    "Dividend", "Buyback", "Other",
]
IMPACTS = ["High", "Medium", "Low"]
IMPACT_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
PERIODS = {
    "1 month": ("1mo", 30), "3 months": ("3mo", 90),
    "6 months": ("6mo", 180), "1 year": ("1y", 365),
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
    .summary-card { border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 18px 22px; margin: 10px 0 22px; background: linear-gradient(110deg, rgba(48,105,150,.12), rgba(48,150,112,.08)); }
    .summary-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    .summary-label { font-size: .82rem; opacity: .75; margin-bottom: 4px; }
    .summary-value { font-size: 1.15rem; font-weight: 650; }
    .news { border: 1px solid rgba(128,128,128,.23); border-radius: 12px; padding: 14px 16px; margin: 10px 0; }
    .pill { display: inline-block; border-radius: 999px; padding: 3px 9px; margin-right: 6px; font-size: .78rem; font-weight: 600; background: rgba(128,128,128,.14); }
    .pill.positive { color: #16834b; } .pill.negative { color: #c23b3b; } .pill.neutral { opacity: .8; }
    .meta { font-size: .82rem; opacity: .7; margin-top: 5px; }
    .brief { border-left: 4px solid #3478c8; border-radius: 6px; padding: 12px 20px; background: rgba(52,120,200,.07); }
    section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.1rem .9rem; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f8fafc; font-weight: 700; letter-spacing: .01em; }
    .sidebar-section-label { color: #94a3b8; font-size: .73rem; font-weight: 700; letter-spacing: .09em; margin: .25rem 0 .45rem; text-transform: uppercase; }
    .sidebar-asset-card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; margin: .45rem 0 1.1rem; padding: .8rem .9rem; }
    .sidebar-asset-name { color: #f8fafc; font-size: 1.03rem; font-weight: 700; line-height: 1.3; }
    .sidebar-asset-meta { color: #94a3b8; font-size: .79rem; font-weight: 600; letter-spacing: .035em; margin-top: .24rem; }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input) { background: #020817; border: 1px solid #1e293b; border-radius: 12px; position: relative; }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input)::before { color: #94a3b8; content: "⌕"; font-size: 1.25rem; left: .8rem; line-height: 1; pointer-events: none; position: absolute; top: 50%; transform: translateY(-50%); }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input { background: transparent; border: 0; border-radius: 12px; color: #f8fafc; min-height: 2.75rem; padding-left: 2.35rem; }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder { color: #64748b; opacity: 1; }
    section[data-testid="stSidebar"] [data-testid="stTextInput"] div:has(> input):focus-within { border-color: #3b82f6; box-shadow: 0 0 0 1px #3b82f6; }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #020817; border: 1px solid #1e293b; border-radius: 12px; color: #f8fafc; min-height: 2.75rem; }
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within { border-color: #3b82f6; }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button { background: #111827; border: 1px solid #1e293b; border-radius: 10px; color: #f8fafc; font-weight: 650; min-height: 2.65rem; transition: background-color .15s ease, border-color .15s ease; }
    section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover { background: #172338; border-color: #334155; color: #f8fafc; }
    .sidebar-ai-status { background: #111827; border: 1px solid #1e293b; border-radius: 9px; color: #22c55e; font-size: .79rem; font-weight: 650; margin-top: .75rem; padding: .55rem .7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def log_timing(operation: str, started: float) -> None:
    """Log elapsed time only; never include credentials or user-provided content."""
    _LOGGER.info("perf operation=%s duration_ms=%.1f", operation, (time.perf_counter() - started) * 1000)


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
    published = (
        as_utc(content.get("pubDate"))
        or as_utc(content.get("providerPublishTime"))
        or as_utc(item.get("providerPublishTime"))
        or utc_now()
    )
    provider = content.get("provider") or item.get("publisher") or {}
    source = (provider.get("displayName") if isinstance(provider, dict) else str(provider)) or item.get("publisher") or "Unknown source"
    canonical, click_url = content.get("canonicalUrl") or {}, content.get("clickThroughUrl") or {}
    url = (canonical.get("url") if isinstance(canonical, dict) else None) or (click_url.get("url") if isinstance(click_url, dict) else None) or item.get("link")
    return {"headline": title.strip(), "published": published, "source": str(source), "url": url if isinstance(url, str) else ""}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str, period: str) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Fetch both Yahoo datasets from one Ticker instance; cache the combined result."""
    instrument = yf.Ticker(ticker)
    started = time.perf_counter()
    prices = instrument.history(period=period, auto_adjust=False)
    log_timing("yahoo_price_retrieval", started)

    started = time.perf_counter()
    try:
        currency = instrument.fast_info.get("currency") or "USD"
    except Exception:
        currency = "USD"
    try:
        raw_news = instrument.news or []
    except Exception:
        raw_news = []
    news = [normalized for item in raw_news if isinstance(item, dict) if (normalized := extract_news_item(item))]
    log_timing("yahoo_news_retrieval_and_normalization", started)
    return prices, news, currency


def keyword_classification(headline: str) -> dict[str, Any]:
    text = headline.lower()
    positive = sum(word in text for word in POSITIVE_WORDS)
    negative = sum(word in text for word in NEGATIVE_WORDS)
    score = 0.0 if positive == negative else (positive - negative) / max(positive + negative, 1)
    score = max(-1.0, min(1.0, score))
    catalyst = "Other"
    for category, keywords in CATALYST_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            catalyst = category
            break
    impact = "High" if abs(score) >= .6 else "Medium" if abs(score) >= .25 else "Low"
    return {"score": score, "catalyst": catalyst, "impact": impact}


def relevance_score(headline: str) -> float:
    """Prioritize catalyst-rich and financially material headlines for AI processing."""
    text = headline.casefold()
    topic_hits = sum(1 for terms in CATALYST_KEYWORDS.values() for term in terms if term in text)
    fallback = keyword_classification(headline)
    return topic_hits * 2.0 + IMPACT_WEIGHTS[fallback["impact"]] + abs(fallback["score"])


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


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str) -> Any:
    """Reuse a Gemini client for this server-side credential."""
    return genai.Client(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_gemini_classifications(api_key: str, headlines: tuple[str, ...], model_name: str) -> dict[int, dict[str, Any]]:
    if not api_key or not headlines:
        return {}
    records = [{"id": i, "headline": headline} for i, headline in enumerate(headlines)]
    prompt = (
        "Classify these financial headlines. Return only a JSON array with one object per id: "
        '{"id": integer, "score": -1 to 1, "catalyst": allowed category, "impact": High|Medium|Low}. '
        "Score news sentiment, not expected stock movement. Use 0 if ambiguous; do not infer facts. "
        f"Categories: {', '.join(CATALYSTS)}. Data: {json.dumps(records, ensure_ascii=False, separators=(',', ':'))}"
    )
    for attempt in range(3):
        try:
            response = get_gemini_client(api_key).models.generate_content(model=model_name, contents=prompt)
            parsed = parse_json_response(response.text or "")
            result: dict[int, dict[str, Any]] = {}
            for record in parsed if isinstance(parsed, list) else parsed.get("items", []):
                if isinstance(record, dict) and "id" in record:
                    result[int(record["id"])] = normalize_classification(record)
            return result
        except Exception:
            if attempt < 2:
                time.sleep(.25 * (attempt + 1))
    return {}


def gemini_classify(api_key: str | None, headlines: list[dict[str, Any]], model_name: str = "gemini-2.0-flash") -> dict[int, dict[str, Any]]:
    if not api_key or not headlines:
        return {}
    selected = sorted(range(len(headlines)), key=lambda i: (relevance_score(headlines[i]["headline"]), headlines[i]["published"]), reverse=True)[:20]
    started = time.perf_counter()
    subset = _cached_gemini_classifications(api_key, tuple(headlines[i]["headline"] for i in selected), model_name)
    log_timing("gemini_sentiment_classification", started)
    # Remap batch-local IDs back to source headline indices.
    return {selected[local_id]: classification for local_id, classification in subset.items() if local_id < len(selected)}


@st.cache_data(ttl=900, show_spinner=False)
def score_headlines(headlines: list[dict[str, Any]], api_key: str | None) -> tuple[pd.DataFrame, str]:
    columns = ["published", "date", "headline", "source", "url", "score", "sentiment", "catalyst", "impact", "recency_weight"]
    if not headlines:
        return pd.DataFrame(columns=columns), "fallback"
    classifications = gemini_classify(api_key, headlines) if api_key else {}
    rows = []
    for i, item in enumerate(headlines):
        classification = classifications.get(i) or keyword_classification(item["headline"])
        published = item["published"]
        score = classification["score"]
        rows.append({**item, **classification, "date": published.date(), "sentiment": "positive" if score > .10 else "negative" if score < -.10 else "neutral"})
    df = pd.DataFrame(rows)
    age_days = ((pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["published"], utc=True)).dt.total_seconds() / 86400).clip(lower=0)
    df["recency_weight"] = (0.5 ** (age_days / 7.0)).clip(lower=.01)
    return df, "gemini" if classifications else "fallback"


@st.cache_data(show_spinner=False)
def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_score"] = (1 + ranked["score"].abs()) * ranked["impact"].map(IMPACT_WEIGHTS).fillna(1)
    return ranked.sort_values(["rank_score", "published"], ascending=[False, False])


@st.cache_data(show_spinner=False)
def daily_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "score"])
    return df.groupby("date", as_index=False)["score"].mean().sort_values("date")


def weighted_average_sentiment(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    weights = df["recency_weight"].astype(float)
    total = float(weights.sum())
    return float(df["score"].mean()) if total <= 0 else float((df["score"] * weights).sum() / total)


def calculate_confidence(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    count_factor = 1 - math.exp(-len(df) / 5.0)
    deviation = float(df["score"].std(ddof=0)) if len(df) > 1 else 0.0
    agreement_factor = 1 - min(deviation, .75) / .75
    recency_factor = float(df["recency_weight"].mean())
    value = 100 * (.35 * count_factor + .40 * agreement_factor + .25 * recency_factor)
    return int(round(max(0, min(100, value))))


def top_catalyst(df: pd.DataFrame) -> str:
    if df.empty or df["catalyst"].dropna().empty:
        return "—"
    return str(df["catalyst"].value_counts().index[0])


def catalyst_extreme(df: pd.DataFrame, bullish: bool) -> str:
    if df.empty:
        return "—"
    means = df.groupby("catalyst")["score"].mean()
    return str(means.idxmax() if bullish else means.idxmin()) if not means.empty else "—"


@st.cache_data(ttl=3600, show_spinner=False)
def write_briefing(api_key: str, ticker: str, day_move: float, ranked_headlines: pd.DataFrame, average: float, confidence: int) -> str:
    records = [{
        "published_utc": row.published.strftime("%Y-%m-%d %H:%M UTC"),
        "headline": row.headline, "sentiment_score": round(float(row.score), 2),
        "catalyst": row.catalyst, "impact": row.impact,
    } for row in ranked_headlines.head(5).itertuples()]
    prompt = (
        f"Write a concise, neutral, evidence-led equity news briefing for {ticker}. "
        "Summarize news flow, material catalysts, risks/uncertainty, latest session context "
        "without claiming causation, then conclude with a caveat. No trade recommendation or invented facts. "
        f"Weighted sentiment {average:+.2f}; latest move {day_move:+.2f}%; heuristic confidence {confidence}/100. "
        f"Ranked evidence: {json.dumps(records, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = get_gemini_client(api_key).models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    return text


@st.cache_data(show_spinner=False)
def resolve_company(query: str) -> tuple[str, str, str] | None:
    return COMPANIES.get(query.casefold())


st.title("📈 Market Sentiment Tracker")
st.caption("A news and market context dashboard. Sentiment and impact labels are analytical aids, not price forecasts.")

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None
if not isinstance(api_key, str) or not api_key.strip():
    api_key = None

with st.sidebar:
    st.markdown("### Market research")
    st.markdown('<div class="sidebar-section-label">Company search</div>', unsafe_allow_html=True)
    search_query = st.text_input("Company search", value="Apple", placeholder="Try Apple, NVIDIA, Tesla", label_visibility="collapsed").strip()
    match = resolve_company(search_query) if search_query else None
    if match:
        company_name, ticker, exchange = match
    else:
        company_name, ticker, exchange = (search_query, search_query.upper(), "Exchange varies") if search_query else ("Apple Inc.", "AAPL", "NASDAQ")
    st.markdown(
        f'<div class="sidebar-asset-card"><div class="sidebar-asset-name">{html.escape(company_name)}</div>'
        f'<div class="sidebar-asset-meta">{html.escape(ticker)} &bull; {html.escape(exchange)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-label">Analysis period</div>', unsafe_allow_html=True)
    period_label = st.selectbox("Analysis period", list(PERIODS), index=1, label_visibility="collapsed")
    period_code, period_days = PERIODS[period_label]
    if api_key:
        st.markdown('<div class="sidebar-ai-status">🤖 AI Analysis Enabled</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("↻ Refresh market data", use_container_width=True):
        fetch_market_data.clear()
        score_headlines.clear()
        add_ranking.clear()
        daily_sentiment_data.clear()
    if st.button("Clear cached analysis", use_container_width=True):
        fetch_market_data.clear()
        score_headlines.clear()
        add_ranking.clear()
        daily_sentiment_data.clear()

if not ticker:
    st.warning("Enter a ticker symbol to continue.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker} market data and news…"):
        prices, all_news, currency = fetch_market_data(ticker, period_code)
except Exception:
    _LOGGER.warning("Market data retrieval failed")
    st.error("Market data is temporarily unavailable.")
    st.stop()

cutoff = utc_now() - timedelta(days=period_days)
news = [item for item in all_news if item["published"] >= cutoff]
classify_started = time.perf_counter()
scored, method = score_headlines(news, api_key)
log_timing("sentiment_pipeline_including_cache", classify_started)
ranked = add_ranking(scored) if not scored.empty else scored.copy()
daily = daily_sentiment_data(scored)

if prices is not None and not prices.empty and "Close" in prices.columns:
    close_prices = prices["Close"].dropna()
    latest_close = float(close_prices.iloc[-1]) if not close_prices.empty else float("nan")
    previous_close = float(close_prices.iloc[-2]) if len(close_prices) > 1 else latest_close
    day_move = (latest_close - previous_close) / previous_close * 100 if previous_close else 0.0
else:
    latest_close, day_move = float("nan"), 0.0

avg = weighted_average_sentiment(scored)
confidence = calculate_confidence(scored)
outlook = "Bullish" if avg is not None and avg > .15 else "Bearish" if avg is not None and avg < -.15 else "Neutral"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Latest close", f"{currency} {latest_close:,.2f}" if not math.isnan(latest_close) else "—")
k2.metric("Latest session", f"{day_move:+.2f}%")
k3.metric("Avg. sentiment", f"{avg:+.2f}" if avg is not None else "—")
k4.metric("Confidence", f"{confidence}%")
k5.metric("Headlines analysed", f"{len(scored)}")

summary_catalyst = top_catalyst(scored)
st.markdown(
    f'<div class="summary-card"><div class="summary-title">Market summary · {html.escape(ticker)}</div>'
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:16px">'
    f'<div><div class="summary-label">Overall outlook</div><div class="summary-value">{outlook}</div></div>'
    f'<div><div class="summary-label">Average sentiment</div><div class="summary-value">{f"{avg:+.2f}" if avg is not None else "—"}</div></div>'
    f'<div><div class="summary-label">Confidence score</div><div class="summary-value">{confidence}%</div></div>'
    f'<div><div class="summary-label">Headlines analysed</div><div class="summary-value">{len(scored)}</div></div>'
    f'<div><div class="summary-label">Top catalyst</div><div class="summary-value">{html.escape(summary_catalyst)}</div></div></div></div>',
    unsafe_allow_html=True,
)
if method != "gemini":
    st.info("AI analysis unavailable.")
if avg is not None:
    if avg > .50:
        st.success(f"Very bullish news sentiment detected ({avg:+.2f}).")
    elif avg < -.50:
        st.error(f"Very bearish news sentiment detected ({avg:+.2f}).")
if not scored.empty and confidence < 40:
    st.warning(f"Low confidence ({confidence}%). This estimate may be less reliable because of headline count, sentiment disagreement, or headline age.")
if avg is not None and ((avg > .15 and day_move < -.5) or (avg < -.15 and day_move > .5)):
    st.warning("Potential sentiment/price divergence: headline tone and the latest session move point in opposite directions. This is a flag to investigate, not proof of mispricing or causation.")

tab_overview, tab_news, tab_catalysts, tab_brief = st.tabs(["Overview", "Headlines", "Catalysts", "Analyst briefing"])
with tab_overview:
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("**Price and news sentiment**")
        if prices is None or prices.empty:
            st.info("No price history was returned for this ticker.")
        else:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close", mode="lines", line=dict(width=2.5, color="#3478c8")))
            if not daily.empty:
                chart.add_trace(go.Bar(x=daily["date"], y=daily["score"], name="Daily news sentiment", opacity=.42, yaxis="y2"))
            chart.update_layout(height=430, margin=dict(t=20, b=10, l=10, r=10), hovermode="x unified", legend=dict(orientation="h", y=1.12), yaxis=dict(title=f"Price ({currency})", showgrid=True), yaxis2=dict(title="Sentiment (-1 to +1)", overlaying="y", side="right", range=[-1, 1], showgrid=False))
            st.plotly_chart(chart, use_container_width=True)
            st.caption("Sentiment is shown only on dates with returned headlines. Yahoo Finance may provide a limited news history.")
    with right:
        st.markdown("**Sentiment distribution**")
        if scored.empty:
            st.info("No period-matched headlines were returned.")
        else:
            counts = scored["sentiment"].value_counts()
            pie = go.Figure(go.Pie(labels=["Positive", "Neutral", "Negative"], values=[int(counts.get("positive", 0)), int(counts.get("neutral", 0)), int(counts.get("negative", 0))], hole=.62, marker=dict(colors=["#23a26d", "#8a96a3", "#d94f4f"])))
            pie.update_layout(height=265, margin=dict(t=10, b=10, l=5, r=5), showlegend=True)
            st.plotly_chart(pie, use_container_width=True)
            st.markdown("**Daily sentiment trend**")
            trend = go.Figure()
            trend.add_hrect(y0=.15, y1=1, fillcolor="rgba(35,162,109,.10)", line_width=0, layer="below")
            trend.add_hrect(y0=-1, y1=-.15, fillcolor="rgba(217,79,79,.10)", line_width=0, layer="below")
            trend.add_trace(go.Scatter(x=daily["date"], y=daily["score"], mode="lines+markers", name="Daily average", line=dict(color="#3478c8", width=2.5), marker=dict(size=6), fill="tozeroy", fillcolor="rgba(52,120,200,.16)", hovertemplate="%{x}<br>Sentiment: %{y:.2f}<extra></extra>"))
            trend.add_hline(y=0, line_dash="dot", line_color="#7d8790")
            trend.add_hline(y=.15, line_dash="dash", line_color="rgba(35,162,109,.55)")
            trend.add_hline(y=-.15, line_dash="dash", line_color="rgba(217,79,79,.55)")
            trend.update_yaxes(range=[-1, 1], title="Average score")
            trend.update_layout(height=260, margin=dict(t=10, b=5, l=5, r=5), showlegend=False, hovermode="x")
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
            safe_url = html.escape(str(row.url), quote=True)
            link = f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{title}</a>' if isinstance(row.url, str) and row.url.startswith("https://") else f"<b>{title}</b>"
            st.markdown(f'<div class="news"><span class="pill {html.escape(str(row.sentiment))}">{html.escape(str(row.sentiment).title())} {row.score:+.2f}</span><span class="pill neutral">{html.escape(str(row.impact))} impact</span><span class="meta">{html.escape(str(row.catalyst))}</span><div style="margin-top:7px">{link}</div><div class="meta">{html.escape(str(row.source))} · {row.published:%d %b %Y, %H:%M UTC}</div></div>', unsafe_allow_html=True)
        csv_data = scored[["published", "headline", "sentiment", "score", "catalyst", "impact", "source", "url"]].to_csv(index=False)
        st.download_button("⬇️ Download analysed headlines (CSV)", csv_data, f"{ticker}_sentiment.csv", "text/csv")

with tab_catalysts:
    if scored.empty:
        st.info("Catalyst analysis needs headlines in the selected period.")
    else:
        if method != "gemini":
            st.info("Catalyst categories and impact levels use basic keyword fallback defaults because AI analysis is unavailable.")
        high_impact_count = int((scored["impact"] == "High").sum())
        catalyst_kpis = st.columns(4)
        catalyst_kpis[0].metric("High impact headlines", high_impact_count)
        catalyst_kpis[1].metric("Top catalyst", summary_catalyst)
        catalyst_kpis[2].metric("Most bullish catalyst", catalyst_extreme(scored, True))
        catalyst_kpis[3].metric("Most bearish catalyst", catalyst_extreme(scored, False))
        left, right = st.columns(2)
        with left:
            st.markdown("**Headline volume by catalyst**")
            catalyst_stats = scored.groupby("catalyst").agg(Headlines=("headline", "count"), MeanSentiment=("score", "mean")).reset_index().sort_values("Headlines", ascending=True)
            catalyst_chart = go.Figure(go.Bar(x=catalyst_stats["Headlines"], y=catalyst_stats["catalyst"], orientation="h", marker=dict(color=catalyst_stats["MeanSentiment"], colorscale=[[0, "#d94f4f"], [.5, "#a0a8b0"], [1, "#23a26d"]], cmin=-1, cmax=1, colorbar=dict(title="Avg. score")), hovertemplate="%{y}<br>Headlines: %{x}<br>Average sentiment: %{marker.color:.2f}<extra></extra>"))
            catalyst_chart.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Headline count", yaxis_title="")
            st.plotly_chart(catalyst_chart, use_container_width=True)
        with right:
            st.markdown("**Potential news impact mix**")
            impact_counts = scored["impact"].value_counts()
            impact_chart = go.Figure(go.Pie(labels=IMPACTS, values=[int(impact_counts.get(level, 0)) for level in IMPACTS], hole=.55, marker=dict(colors=["#d94f4f", "#e7a943", "#7d9bb8"])))
            impact_chart.update_layout(height=360, margin=dict(t=10, b=10, l=5, r=5), showlegend=True)
            st.plotly_chart(impact_chart, use_container_width=True)
        st.markdown("**Highest-ranked headlines**")
        display = ranked.head(10)[["published", "headline", "sentiment", "score", "catalyst", "impact", "rank_score", "source"]].copy()
        display["published"] = pd.to_datetime(display["published"], utc=True).dt.strftime("%d %b %Y %H:%M UTC")
        display["score"] = display["score"].map(lambda value: f"{value:+.2f}")
        display["rank_score"] = display["rank_score"].map(lambda value: f"{value:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Ranking = (1 + absolute sentiment score) × impact weight (High=3, Medium=2, Low=1). It is a prioritisation heuristic, not a forecast of price movement.")

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
                log_timing("analyst_briefing_generation_including_cache", started)
        if state_key in st.session_state:
            st.markdown('<div class="brief">', unsafe_allow_html=True)
            st.markdown(st.session_state[state_key])
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("See the ranked headlines used"):
                for i, row in enumerate(top.itertuples(), 1):
                    st.write(f"{i}. [{row.impact} impact · {row.catalyst} · {row.score:+.2f}] {row.headline}")

st.divider()
st.caption("Important: Sentiment scores, confidence estimates, and impact rankings are analytical aids only. They do not measure the probability of a price move and should not be used as standalone buy/sell signals.")
log_timing("total_page_load", _PAGE_STARTED)
