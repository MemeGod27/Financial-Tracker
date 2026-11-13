from __future__ import annotations

import html
import json
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Market Sentiment Tracker",
    page_icon="📈",
    layout="wide",
)

BULL = "#22c55e"
BEAR = "#ef4444"
NEUTRAL = "#64748b"
PRIMARY = "#3b82f6"

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
    "Product": ("product", "launch", "release", "technology", " patent"),
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


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
      --bull: #22c55e;
      --bear: #ef4444;
      --neutral: #64748b;
      --primary: #3b82f6;
    }
    .stApp {
      background: #f3f6fb;
      color: #172033;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
      max-width: 1540px;
      padding: 1.6rem 2.4rem 3rem;
    }
    section[data-testid="stSidebar"] {
      background: #101a2b;
    }
    section[data-testid="stSidebar"] * {
      color: #e7edf7;
    }
    section[data-testid="stSidebar"] input {
      color: #172033 !important;
    }

    .hero {
      position: relative;
      overflow: hidden;
      padding: 26px 30px 24px;
      margin: 4px 0 24px;
      color: #f8fafc;
      border-radius: 20px;
      background: linear-gradient(120deg, #101a2b 0%, #172b48 58%, #18385c 100%);
      box-shadow: 0 12px 30px rgba(15, 30, 55, 0.16);
    }
    .hero:after {
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -65px;
      top: -180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(59,130,246,.3), transparent 68%);
    }
    .hero-top {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      position: relative;
      z-index: 1;
    }
    .hero-company {
      font-size: clamp(1.6rem, 3vw, 2.4rem);
      font-weight: 750;
      letter-spacing: -0.04em;
      line-height: 1.1;
    }
    .ticker-badge {
      border: 1px solid rgba(255,255,255,.25);
      background: rgba(255,255,255,.1);
      border-radius: 8px;
      padding: 5px 9px;
      font-size: .78rem;
      font-weight: 700;
      letter-spacing: .08em;
    }
    .hero-sub {
      color: #aab8ca;
      font-size: .88rem;
      margin-top: 8px;
      position: relative;
      z-index: 1;
    }
    .hero-stats {
      display: flex;
      gap: 34px;
      flex-wrap: wrap;
      margin-top: 22px;
      position: relative;
      z-index: 1;
    }
    .hero-stat-label {
      color: #aab8ca;
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .1em;
      margin-bottom: 4px;
    }
    .hero-stat-value {
      color: #f8fafc;
      font-size: 1.25rem;
      font-weight: 700;
    }
    .hero-stat-value.bull { color: #4ade80; }
    .hero-stat-value.bear { color: #fb7185; }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 0 0 18px;
    }
    .kpi-card {
      background: #fff;
      border: 1px solid #e5eaf1;
      border-radius: 14px;
      padding: 17px 18px;
      min-height: 96px;
      box-shadow: 0 3px 12px rgba(28,45,75,.045);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .kpi-card:hover,
    .news-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 24px rgba(28,45,75,.11);
    }
    .kpi-label {
      color: #64748b;
      font-size: .73rem;
      font-weight: 650;
      letter-spacing: .07em;
      text-transform: uppercase;
    }
    .kpi-value {
      color: #172033;
      font-size: 1.55rem;
      font-weight: 750;
      letter-spacing: -0.035em;
      margin-top: 7px;
      line-height: 1.1;
    }
    .kpi-value.bull { color: var(--bull); }
    .kpi-value.bear { color: var(--bear); }
    .kpi-value.neutral { color: var(--neutral); }

    .summary-card,
    .panel,
    .research-note {
      background: #fff;
      border: 1px solid #e5eaf1;
      border-radius: 16px;
      padding: 21px 23px;
      box-shadow: 0 4px 15px rgba(28,45,75,.045);
      margin-bottom: 18px;
    }
    .panel-title {
      color: #172033;
      font-size: 1.02rem;
      font-weight: 720;
      letter-spacing: -0.01em;
      margin: 0 0 14px;
    }
    .summary-title {
      font-size: 1.06rem;
      font-weight: 750;
      margin-bottom: 16px;
      color: #172033;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 18px;
    }
    .summary-label {
      color: #64748b;
      font-size: .73rem;
      text-transform: uppercase;
      letter-spacing: .07em;
      margin-bottom: 5px;
    }
    .summary-value {
      font-size: 1.02rem;
      font-weight: 700;
      color: #172033;
    }

    .news-card {
      background: #fff;
      border: 1px solid #e5eaf1;
      border-radius: 14px;
      padding: 17px 19px;
      margin: 11px 0;
      box-shadow: 0 3px 11px rgba(28,45,75,.04);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .news-title {
      font-size: 1rem;
      line-height: 1.48;
      font-weight: 650;
      color: #1c2b42;
      margin: 10px 0 8px;
    }
    .news-title a { color: inherit; text-decoration: none; }
    .news-title a:hover { color: var(--primary); }

    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      margin-right: 6px;
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .025em;
    }
    .badge-positive { background: #dcfce7; color: #15803d; }
    .badge-negative { background: #fee2e2; color: #b91c1c; }
    .badge-neutral { background: #f1f5f9; color: #475569; }
    .badge-primary { background: #dbeafe; color: #1d4ed8; }
    .news-meta {
      font-size: .77rem;
      color: #8290a3;
      margin-top: 8px;
    }

    .research-note {
      padding: 28px 34px;
      border-top: 4px solid var(--primary);
      box-shadow: 0 9px 28px rgba(28,45,75,.08);
    }
    .research-note h1,
    .research-note h2,
    .research-note h3 {
      color: #172033;
      letter-spacing: -0.025em;
    }
    .research-note p,
    .research-note li {
      font-size: .96rem;
      line-height: 1.78;
      color: #334155;
    }

    .stTabs [data-baseweb="tab-list"] {
      gap: 7px;
      border-bottom: 1px solid #e2e8f0;
      background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
      height: 45px;
      padding: 0 16px;
      border-radius: 9px 9px 0 0;
      color: #64748b;
      font-weight: 650;
    }
    .stTabs [aria-selected="true"] {
      color: #1d4ed8 !important;
      background: #fff;
      border: 1px solid #e2e8f0;
      border-bottom-color: #fff;
    }
    .stPlotlyChart {
      background: #fff;
      border: 1px solid #e5eaf1;
      border-radius: 14px;
      padding: 8px;
      box-shadow: 0 3px 12px rgba(28,45,75,.04);
    }
    div[data-testid="stDataFrame"] {
      background: #fff;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #e5eaf1;
    }
    div[data-testid="stAlert"] {
      border-radius: 12px;
    }

    @media (max-width:900px) {
      .block-container { padding: 1rem 1rem 2rem; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero-stats { gap: 20px; }
    }
    @media (max-width:520px) {
      .kpi-grid { grid-template-columns: 1fr 1fr; }
      .hero { padding: 22px 19px; }
      .summary-card, .panel { padding: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def as_utc(value: Any) -> datetime | None:
    """Convert common Yahoo Finance timestamp values to UTC."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
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
    )
    if published is None:
        published = utc_now()

    provider = content.get("provider") or item.get("publisher") or {}
    source = (
        provider.get("displayName") if isinstance(provider, dict) else str(provider)
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(ticker: str, period: str) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    instrument = yf.Ticker(ticker)
    prices = instrument.history(period=period, auto_adjust=False)

    currency = "USD"
    try:
        currency = instrument.fast_info.get("currency") or currency
    except Exception:
        pass

    news: list[dict[str, Any]] = []
    for item in instrument.news or []:
        if not isinstance(item, dict):
            continue
        normalized = extract_news_item(item)
        if normalized:
            news.append(normalized)

    return prices, news, currency


def keyword_classification(headline: str) -> dict[str, Any]:
    text = headline.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in text)
    negative = sum(1 for word in NEGATIVE_WORDS if word in text)
    score = 0.0 if positive == negative else (positive - negative) / max(positive + negative, 1)
    score = max(-1.0, min(1.0, score))

    catalyst = "Other"
    for category, keywords in CATALYST_KEYWORDS.items():
        if any(keyword.strip() in text for keyword in keywords):
            catalyst = category
            break

    impact = "High" if abs(score) >= 0.6 else "Medium" if abs(score) >= 0.25 else "Low"
    return {"score": score, "catalyst": catalyst, "impact": impact}


def parse_json_response(text: str) -> Any:
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
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(-1.0, min(1.0, score))

    raw_catalyst = str(data.get("catalyst", "Other")).strip()
    catalyst_lookup = {item.lower(): item for item in CATALYSTS}
    catalyst = catalyst_lookup.get(raw_catalyst.lower(), "Other")

    raw_impact = str(data.get("impact", "Low")).strip().title()
    impact = raw_impact if raw_impact in IMPACTS else "Low"
    return {"score": score, "catalyst": catalyst, "impact": impact}


def gemini_classify(
    api_key: str,
    headlines: list[dict[str, Any]],
    model_name: str = "gemini-2.0-flash",
) -> dict[int, dict[str, Any]]:
    if not api_key or not headlines:
        return {}

    payload = [{"id": i, "headline": item["headline"]} for i, item in enumerate(headlines)]
    prompt = f"""
You are an institutional equity research analyst classifying financial news.
Assess each headline in context. Consider earnings quality, guidance revisions,
margin impact, management changes, regulatory developments, competitive position,
capital allocation, partnerships and acquisitions, product development, analyst
actions, legal issues and macro conditions.

Return only a valid JSON array. For every headline return exactly:
{{"id": integer, "score": number, "catalyst": string, "impact": string}}
Rules:
- score must be between -1 and +1.
- catalyst must be one of: {', '.join(CATALYSTS)}.
- impact must be High, Medium, or Low.
- Include one output for every input id.

Headlines:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model_name, contents=prompt)
            parsed = parse_json_response(response.text or "")
            records = parsed if isinstance(parsed, list) else parsed.get("items", [])
            result: dict[int, dict[str, Any]] = {}
            for record in records:
                if isinstance(record, dict) and "id" in record:
                    result[int(record["id"])] = normalize_classification(record)
            return result
        except Exception:
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
    return {}


def score_headlines(
    headlines: list[dict[str, Any]],
    api_key: str,
) -> tuple[pd.DataFrame, str]:
    if not headlines:
        return pd.DataFrame(columns=["published", "date", "headline", "source", "url", "score", "sentiment", "catalyst", "impact", "recency_weight"]), "fallback"

    classifications = gemini_classify(api_key, headlines) if api_key else {}
    used_gemini = bool(classifications)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(headlines):
        classification = classifications.get(i) or keyword_classification(item["headline"])
        published = item["published"]
        rows.append({
            **item,
            **classification,
            "date": published.date(),
            "sentiment": (
                "positive" if classification["score"] > 0.10
                else "negative" if classification["score"] < -0.10
                else "neutral"
            ),
        })

    df = pd.DataFrame(rows)
    now = pd.Timestamp.now(tz="UTC")
    ages_days = (now - pd.to_datetime(df["published"], utc=True)).dt.total_seconds() / 86400
    ages_days = ages_days.clip(lower=0)
    df["recency_weight"] = (0.5 ** (ages_days / 7.0)).clip(lower=0.01)
    return df, "gemini" if used_gemini else "fallback"


def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_score"] = (
        (1 + ranked["score"].abs())
        * ranked["impact"].map(IMPACT_WEIGHTS).fillna(1)
    )
    return ranked.sort_values(["rank_score", "published"], ascending=[False, False])


def weighted_average_sentiment(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    weights = df["recency_weight"].astype(float)
    if float(weights.sum()) <= 0:
        return float(df["score"].mean())
    return float((df["score"] * weights).sum() / weights.sum())


def calculate_confidence(df: pd.DataFrame) -> int:
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


def top_catalyst(df: pd.DataFrame) -> str:
    if df.empty or df["catalyst"].dropna().empty:
        return "—"
    return str(df["catalyst"].value_counts().index[0])


def catalyst_extreme(df: pd.DataFrame, bullish: bool) -> str:
    if df.empty:
        return "—"
    means = df.groupby("catalyst")["score"].mean()
    if means.empty:
        return "—"
    return str(means.idxmax() if bullish else means.idxmin())


def write_briefing(
    api_key: str,
    ticker: str,
    day_move: float,
    ranked_headlines: pd.DataFrame,
    average: float,
    confidence: int,
) -> str:
    records = []
    for row in ranked_headlines.head(5).itertuples():
        records.append({
            "published_utc": row.published.strftime("%Y-%m-%d %H:%M UTC"),
            "headline": row.headline,
            "sentiment_score": round(float(row.score), 2),
            "catalyst": row.catalyst,
            "impact": row.impact,
        })

    prompt = f"""
Write a concise institutional equity research news briefing for {ticker}.
Use a neutral evidence-led tone and clearly distinguish facts from interpretation.
Do not make a buy/sell recommendation or invent facts.

Include:
1. Clear news-flow assessment.
2. Material catalysts and their fundamental relevance.
3. Risks, offsets, and uncertainty.
4. Context from the latest session price move without claiming causation.
5. A short conclusion with a caveat.

The inputs below are ranked by (1 + absolute sentiment score) × impact weight.
Include catalyst, impact level and sentiment score for material headlines.

Period weighted sentiment: {average:+.2f}
Latest session price move: {day_move:+.2f}%
Headline confidence heuristic: {confidence}/100

Ranked headlines:
{json.dumps(records, ensure_ascii=False, indent=2)}
""".strip()

    response = genai.Client(api_key=api_key).models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("The briefing model returned an empty response.")
    return text


def classify_error(exc: Exception) -> str:
    message = str(exc)
    low = message.lower()
    if "api key" in low or "permission" in low:
        return "Gemini could not authorize the request. Check that the API key is valid."
    if "quota" in low or "rate" in low:
        return "Gemini rate limit or quota reached. Please try again later."
    return f"Could not complete the Gemini request: {message}"


# ---------------------------------------------------------------------------
# UI and data flow
# ---------------------------------------------------------------------------

st.title("📈 Market Sentiment Tracker")
st.caption("A news and market context dashboard. Sentiment and impact labels are analytical aids, not price forecasts.")

with st.sidebar:
    st.header("Tracker settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    period_label = st.selectbox("Analysis period", list(PERIODS), index=1)
    period_code, period_days = PERIODS[period_label]
    api_key = st.text_input(
        "Gemini API key",
        value=st.secrets.get("GEMINI_API_KEY", ""),
        type="password",
        help="Optional. Without a key, keyword-based sentiment scoring is used.",
    )
    if st.button("Refresh market data"):
        fetch_market_data.clear()

if not ticker:
    st.warning("Enter a ticker symbol to continue.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker} market data and news…"):
        prices, all_news, currency = fetch_market_data(ticker, period_code)
except Exception as exc:
    st.error(f"Could not load market data for {ticker}: {exc}")
    st.stop()

cutoff = utc_now() - timedelta(days=period_days)
news = [item for item in all_news if item["published"] >= cutoff]
scored, method = score_headlines(news, api_key)
ranked = add_ranking(scored) if not scored.empty else scored.copy()

if prices is not None and not prices.empty and "Close" in prices.columns:
    latest_close = float(prices["Close"].dropna().iloc[-1])
    if len(prices["Close"].dropna()) > 1:
        previous_close = float(prices["Close"].dropna().iloc[-2])
        day_move = ((latest_close - previous_close) / previous_close * 100) if previous_close else 0.0
    else:
        day_move = 0.0
else:
    latest_close = float("nan")
    day_move = 0.0

avg = weighted_average_sentiment(scored)
confidence = calculate_confidence(scored)
outlook = "Bullish" if avg is not None and avg > 0.15 else "Bearish" if avg is not None and avg < -0.15 else "Neutral"

# Hero header and row
company_name = ticker
st.markdown(
    f'''<div class="hero"><div class="hero-top"><div class="hero-company">{html.escape(company_name)}</div><span class="ticker-badge">{html.escape(ticker)}</span></div><div class="hero-sub">Market Intelligence · {html.escape(period_label)} research window · {"Gemini-assisted analysis" if method == "gemini" else "Fundamental news monitoring"}</div><div class="hero-stats"><div><div class="hero-stat-label">Last price</div><div class="hero-stat-value">{currency} {latest_close:,.2f}</div></div><div><div class="hero-stat-label">Daily change</div><div class="hero-stat-value {'bull' if day_move > 0 else 'bear' if day_move < 0 else ''}">{day_move:+.2f}%</div></div><div><div class="hero-stat-label">Market sentiment</div><div class="hero-stat-value {'bull' if avg is not None and avg > 0.15 else 'bear' if avg is not None and avg < -0.15 else ''}">{avg:+.2f}</div></div><div><div class="hero-stat-label">Outlook</div><div class="hero-stat-value">{outlook}</div></div></div></div>''',
    unsafe_allow_html=True,
)

kpis = [
    ("Last price", f"{currency} {latest_close:,.2f}", "neutral" if not math.isnan(latest_close) else "neutral"),
    ("Daily change", f"{day_move:+.2f}%", "bull" if day_move > 0 else "bear" if day_move < 0 else "neutral"),
    ("Avg. sentiment", f"{avg:+.2f}" if avg is not None else "—", "bull" if avg is not None and avg > 0.15 else "bear" if avg is not None and avg < -0.15 else "neutral"),
    ("Confidence", f"{confidence}%", "bull" if confidence >= 65 else "neutral" if confidence >= 40 else "bear"),
    ("Headlines analysed", str(len(scored)), "neutral"),
]

st.markdown(
    '<div class="kpi-grid">' + ''.join(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value {tone}">{value}</div></div>' for label, value, tone in kpis) + '</div>',
    unsafe_allow_html=True,
)

summary_catalyst = top_catalyst(scored)
st.markdown(
    f'''
    <div class="summary-card">
      <div class="summary-title">Research snapshot</div>
      <div class="summary-grid">
        <div><div class="summary-label">Overall outlook</div><div class="summary-value">{outlook}</div></div>
        <div><div class="summary-label">Weighted sentiment</div><div class="summary-value">{avg:+.2f}</div></div>
        <div><div class="summary-label">Confidence score</div><div class="summary-value">{confidence}%</div></div>
        <div><div class="summary-label">Headlines reviewed</div><div class="summary-value">{len(scored)}</div></div>
        <div><div class="summary-label">Leading catalyst</div><div class="summary-value">{html.escape(summary_catalyst)}</div></div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

if avg is not None:
    if avg > 0.50:
        st.success(f"Very bullish news sentiment detected ({avg:+.2f}).")
    elif avg < -0.50:
        st.error(f"Very bearish news sentiment detected ({avg:+.2f}).")

if not scored.empty and confidence < 40:
    st.warning(f"Low confidence ({confidence}%). This estimate may be less reliable because of headline count, disagreement, or recency.")

divergence = avg is not None and ((avg > 0.15 and day_move < -0.5) or (avg < -0.15 and day_move > 0.5))
if divergence:
    st.warning("Potential sentiment/price divergence: current headline tone and the latest session move are moving in opposite directions. This is a signal to investigate, not a price forecast.")

if method != "gemini":
    st.caption("Keyword fallback scoring is active. Add a Gemini API key in the sidebar for model-based sentiment, catalyst, and impact analysis.")

# Tabs renamed to product-friendly names.
t_overview, t_news, t_catalysts, t_research = st.tabs(["Market Overview", "News Intelligence", "Catalyst Analytics", "Research Brief"])

with t_overview:
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("**Price action & market sentiment**")
        if prices is None or prices.empty:
            st.info("No price history is available for this symbol.")
        else:
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close", mode="lines", line=dict(color="#26364f", width=2.5)))
            if not scored.empty:
                daily = scored.groupby("date")["score"].mean().reset_index()
                chart.add_trace(go.Bar(x=daily["date"], y=daily["score"], name="Daily sentiment", opacity=0.45, yaxis="y2"))
            chart.update_layout(
                height=430,
                margin=dict(t=20, b=10, l=10, r=10),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12),
                yaxis=dict(title=f"Price ({currency})", showgrid=True),
                yaxis2=dict(title="Sentiment (-1 to +1)", overlaying="y", side="right", range=[-1, 1], showgrid=False),
            )
            st.plotly_chart(chart, use_container_width=True)
            st.caption("Sentiment is shown only on dates with returned headlines. Yahoo Finance may provide a limited news history.")

    with right:
        st.markdown("**Sentiment profile**")
        if scored.empty:
            st.info("No period-matched headlines were returned.")
        else:
            counts = scored["sentiment"].value_counts()
            pie = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[int(counts.get(x, 0)) for x in ("positive", "neutral", "negative")],
                hole=0.62,
                marker=dict(colors=[BULL, NEUTRAL, BEAR])
            ))
            pie.update_layout(height=265, margin=dict(t=10, b=10, l=5, r=5), showlegend=True)
            st.plotly_chart(pie, use_container_width=True)

            st.markdown("**Daily sentiment trend**")
            daily = scored.groupby("date")["score"].mean().reset_index().sort_values("date")
            trend = go.Figure()
            trend.add_hrect(y0=0.15, y1=1, fillcolor="rgba(34,197,94,.10)", line_width=0, layer="below")
            trend.add_hrect(y0=-1, y1=-0.15, fillcolor="rgba(239,68,68,.10)", line_width=0, layer="below")
            trend.add_trace(go.Scatter(x=daily["date"], y=daily["score"], mode="lines+markers", name="Daily average", line=dict(color="#26364f", width=2.7), marker=dict(size=6), fill="tozeroy", fillcolor="rgba(59,130,246,.12)"))
            trend.add_hline(y=0, line_dash="dot", line_color="#64748b")
            trend.add_hline(y=0.15, line_dash="dash", line_color="rgba(34,197,94,.7)")
            trend.add_hline(y=-0.15, line_dash="dash", line_color="rgba(239,68,68,.7)")
            trend.update_yaxes(range=[-1, 1], title="Average score")
            trend.update_layout(height=260, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(trend, use_container_width=True)

with t_news:
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
            tone = "positive" if row.sentiment == "positive" else "negative" if row.sentiment == "negative" else "neutral"
            st.markdown(
                f'<div class="news-card"><div><span class="badge badge-{tone}">{row.sentiment.title()} · {row.score:+.2f}</span><span class="badge badge-primary">{html.escape(str(row.catalyst))}</span><span class="badge badge-neutral">{html.escape(str(row.impact))} impact</span></div><div class="news-title">{link}</div><div class="news-meta">{html.escape(str(row.source))} · {row.published:%d %b %Y, %H:%M UTC}</div></div>',
                unsafe_allow_html=True,
            )
        csv_data = scored[["published", "headline", "sentiment", "score", "catalyst", "impact", "source", "url"]].to_csv(index=False)
        st.download_button("⬇️ Download analysed headlines (CSV)", csv_data, f"{ticker}_sentiment.csv", "text/csv")

with t_catalysts:
    if scored.empty:
        st.info("Catalyst analysis needs headlines in the selected period.")
    else:
        if method != "gemini":
            st.info("Catalyst categories and impact levels use basic fallback defaults because Gemini analysis is unavailable.")

        cols = st.columns(4)
        metrics = [
            ("High impact headlines", str(int((scored["impact"] == "High").sum()))),
            ("Top catalyst", top_catalyst(scored)),
            ("Most bullish catalyst", catalyst_extreme(scored, True)),
            ("Most bearish catalyst", catalyst_extreme(scored, False)),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value neutral" style="font-size:1.2rem">{value}</div></div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Catalyst distribution**")
            catalyst_counts = scored["catalyst"].value_counts().rename_axis("Catalyst").reset_index(name="Headlines")
            catalyst_chart = go.Figure(go.Bar(
                x=catalyst_counts["Headlines"],
                y=catalyst_counts["Catalyst"],
                orientation="h",
                marker=dict(color=catalyst_counts["Headlines"], colorscale=[[0, BEAR], [0.5, NEUTRAL], [1, BULL]]),
            ))
            catalyst_chart.update_layout(height=330, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Headline count")
            st.plotly_chart(catalyst_chart, use_container_width=True)

        with right:
            st.markdown("**Impact mix**")
            impact_counts = scored["impact"].value_counts()
            impact_chart = go.Figure(go.Pie(
                labels=IMPACTS,
                values=[int(impact_counts.get(level, 0)) for level in IMPACTS],
                hole=.55,
                marker=dict(colors=[BEAR, "#f59e0b", "#94a3b8"]),
            ))
            impact_chart.update_layout(height=330, margin=dict(t=10, b=10, l=5, r=5), showlegend=True)
            st.plotly_chart(impact_chart, use_container_width=True)

        st.markdown("**Highest-impact headlines**")
        ranked_display = ranked.head(10).copy()
        display = ranked_display[["published", "headline", "sentiment", "score", "catalyst", "impact", "source"]].copy()
        display["published"] = display["published"].dt.strftime("%d %b %Y %H:%M UTC")
        display["score"] = display["score"].map(lambda value: f"{value:+.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Impact ranking is a prioritisation heuristic based on Gemini labels and sentiment magnitude. It is not a forecast of price movement.")

with t_research:
    if scored.empty:
        st.info("The briefing needs at least one headline in the selected period.")
    elif not api_key:
        st.info("Add a Gemini API key in the sidebar to generate the analyst briefing.")
    else:
        st.caption("On-demand summary of up to five most directionally significant headlines.")
        top = ranked.head(5)
        state_key = f"brief_{ticker}_{period_label}"
        if st.button("✨ Generate research brief", type="primary"):
            try:
                with st.spinner("Preparing briefing…"):
                    st.session_state[state_key] = write_briefing(api_key, ticker, day_move, top, avg if avg is not None else 0.0, confidence)
            except Exception as exc:
                st.error(classify_error(exc))
        if state_key in st.session_state:
            st.markdown('<div class="research-note">', unsafe_allow_html=True)
            st.markdown(f"### {html.escape(ticker)} — Research Brief")
            st.markdown(st.session_state[state_key])
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("See the headlines used"):
                for i, row in enumerate(top.itertuples(), 1):
                    st.write(f"[{i}] ({row.sentiment}, {row.score:+.2f}) {row.headline}")

st.divider()
st.caption("Important: Sentiment scores, confidence estimates, and impact rankings are analytical aids only. They do not measure the probability of a price move and should not be used as standalone buy/sell signals.")
