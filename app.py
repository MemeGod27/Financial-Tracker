"""Automated Financial News and Market Sentiment Tracker (cloud version).

Sentiment scoring + briefing both use Gemini, so there is no torch/transformers
and the app fits on Streamlit Community Cloud's free tier.

Local:  streamlit run app.py
Cloud:  add GEMINI_API_KEY under App settings -> Secrets
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Sentiment Tracker", layout="wide")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# ------------------------------------------------------------------ API key
def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file locally
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.session_state.get("key")


# ---------------------------------------------------------------- Step 2: data
@st.cache_data(ttl=900, show_spinner=False)
def get_prices(ticker: str, period: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return hist
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    return hist[["Close", "Volume"]]


@st.cache_data(ttl=900, show_spinner=False)
def get_news(ticker: str) -> pd.DataFrame:
    """Headlines with timestamp + URL kept for auditability (handles both yfinance formats)."""
    rows = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)
        title = c.get("title")
        if not title:
            continue
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
        )
        ts = c.get("pubDate") or item.get("providerPublishTime")
        if ts is None:
            continue
        ts = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
        publisher = (c.get("provider") or {}).get("displayName") or item.get("publisher")
        rows.append({"published": ts, "headline": title, "source": publisher, "url": url})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published", ascending=False).reset_index(drop=True)
    return df


# ------------------------------------------- Step 3 (Option B): Gemini sentiment
SCORE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "INTEGER"},
            "score": {"type": "NUMBER"},
            "catalyst": {"type": "STRING"},
        },
        "required": ["id", "score", "catalyst"],
    },
}


@st.cache_data(ttl=3600, show_spinner=False)
def score_headlines(headlines: tuple[str, ...], api_key: str) -> pd.DataFrame:
    """Score each headline from -1.0 (bearish) to +1.0 (bullish) plus a catalyst tag."""
    numbered = "\n".join(f"{i}: {h}" for i, h in enumerate(headlines))
    prompt = f"""You are a financial news sentiment classifier.
For each numbered headline, return:
- id: the headline number
- score: float from -1.0 (very bearish) to +1.0 (very bullish) for the company's stock; 0 if neutral.
  Read financial jargon correctly (e.g. "slashes debt" is bullish).
- catalyst: ONE word from: Earnings, Regulation, M&A, Product, Macro, Legal, Management, Analyst, Other.

Headlines:
{numbered}"""
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCORE_SCHEMA,
            temperature=0,
        ),
    )
    data = {int(d["id"]): d for d in json.loads(resp.text)}
    rows = []
    for i in range(len(headlines)):
        d = data.get(i, {"score": 0.0, "catalyst": "Other"})
        s = max(-1.0, min(1.0, float(d["score"])))
        label = "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
        rows.append({"sentiment": label, "score": s, "catalyst": d["catalyst"]})
    return pd.DataFrame(rows)


BADGE = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"}


# ------------------------------------------------------ Step 4: analyst briefing
@st.cache_data(ttl=3600, show_spinner=False)
def analyst_briefing(ticker: str, pct_move: float, lines: tuple[str, ...], api_key: str) -> str:
    numbered = "\n".join(f"[{i}] {l}" for i, l in enumerate(lines, 1))
    prompt = f"""You are a senior equity research analyst writing a morning briefing.
Ticker: {ticker}. Latest daily price move: {pct_move:+.2f}%.

Top headlines (with sentiment score):
{numbered}

Write exactly three bullet points:
- Key Bullish Drivers
- Key Risk Factors
- Price-Sentiment Divergence (flag if news tone and the price move disagree)

Rules: cite headlines by number like [1], [2]. Use ONLY facts in the headlines
above. Do NOT invent any financial figures, targets or dates. Be concise."""
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text


# ------------------------------------------------------------ Step 5: dashboard
st.title("📈 Financial News & Market Sentiment Tracker")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL", help="e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS").strip().upper()
    period = st.selectbox("Price history", ["7d", "1mo", "3mo"], index=1)
    if not get_api_key():
        st.text_input("Gemini API key", type="password", key="key", help="Free key from Google AI Studio")
    st.caption("Data: Yahoo Finance via yfinance. Sentiment: Gemini. Not investment advice.")

api_key = get_api_key()
if not ticker:
    st.stop()
if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start.")
    st.stop()

prices = get_prices(ticker, period)
news = get_news(ticker)

if prices.empty:
    st.error(f"No price data found for '{ticker}'. Check the symbol (Indian stocks need .NS or .BO).")
    st.stop()

scored = pd.DataFrame()
if news.empty:
    st.warning("No recent headlines returned for this ticker.")
else:
    try:
        with st.spinner("Scoring headlines with Gemini..."):
            sc = score_headlines(tuple(news["headline"]), api_key)
        scored = pd.concat([news, sc], axis=1)
        scored["date"] = scored["published"].dt.tz_convert(None).dt.normalize()
    except Exception as e:
        st.error(f"Sentiment scoring failed: {e}")

last_close = float(prices["Close"].iloc[-1])
prev_close = float(prices["Close"].iloc[-2]) if len(prices) > 1 else last_close
pct_move = (last_close / prev_close - 1) * 100 if prev_close else 0.0
avg_sent = float(scored["score"].mean()) if not scored.empty else None

k1, k2, k3 = st.columns(3)
k1.metric("Current price", f"{last_close:,.2f}")
k2.metric("Latest daily change", f"{pct_move:+.2f}%")
k3.metric("Avg sentiment", f"{avg_sent:+.2f}" if avg_sent is not None else "n/a")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close", mode="lines+markers"), secondary_y=False)
if not scored.empty:
    daily = scored.groupby("date")["score"].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["score"],
            name="Avg daily sentiment",
            opacity=0.45,
            marker_color=["#2e9e5b" if s >= 0 else "#d64545" for s in daily["score"]],
        ),
        secondary_y=True,
    )
fig.update_yaxes(title_text="Close", secondary_y=False)
fig.update_yaxes(title_text="Sentiment (-1 to +1)", range=[-1, 1], secondary_y=True)
fig.update_layout(height=420, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Yahoo only exposes recent headlines, so sentiment bars cover just the days "
    "those articles fall on."
)

if not scored.empty:
    st.subheader("Morning Analyst Briefing")
    top = scored.reindex(scored["score"].abs().sort_values(ascending=False).index).head(5)
    lines = tuple(f"({r.sentiment}, {r.score:+.2f}, {r.catalyst}) {r.headline}" for r in top.itertuples())
    try:
        with st.spinner("Writing briefing..."):
            st.markdown(analyst_briefing(ticker, pct_move, lines, api_key))
        with st.expander("Headlines referenced as [1]-[5]"):
            for i, l in enumerate(lines, 1):
                st.write(f"[{i}] {l}")
    except Exception as e:
        st.warning(f"Briefing unavailable: {e}")

    st.subheader("Headline audit table")
    table = scored.assign(sentiment=scored["sentiment"].map(BADGE))[
        ["published", "headline", "sentiment", "score", "catalyst", "source", "url"]
    ]
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "published": st.column_config.DatetimeColumn("Published", format="DD MMM, HH:mm"),
            "score": st.column_config.NumberColumn("Score", format="%+.2f"),
            "url": st.column_config.LinkColumn("Source link", display_text="Open"),
        },
    )
