"""Financial News & Market Sentiment Tracker (cloud version, v2).

- Gemini scores headlines (with retries, model fallback, and a keyword fallback
  so the dashboard never goes blank when Gemini is busy).
- Analyst briefing is generated on demand to keep API usage low.

Local:  streamlit run app.py
Cloud:  add GEMINI_API_KEY under App settings -> Secrets
"""
import html
import json
import os
import re
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from google import genai
from google.genai import types
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Market Sentiment Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

POPULAR = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Bitcoin (BTC-USD)": "BTC-USD",
}
PERIODS = {"1W": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo"}

# ------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}
.hero {display:flex; align-items:center; gap:.75rem; flex-wrap:wrap;}
.hero h1 {margin:0; padding:0; font-size:2.1rem; font-weight:750; letter-spacing:-.02em;}
.hero .sym {font-size:.9rem; padding:.18rem .65rem; border-radius:999px;
            background:rgba(79,140,255,.16); color:#4f8cff; font-weight:650;}
.sub {opacity:.65; margin:.15rem 0 1.1rem 0;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:14px;
       padding:14px 18px; background:rgba(128,128,128,.07);}
.kpi-label {font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; opacity:.65;}
.kpi-value {font-size:1.85rem; font-weight:700; line-height:1.3;}
.kpi-sub {font-size:.85rem; opacity:.75;}
.up {color:#2e9e5b; opacity:1;}
.down {color:#d64545; opacity:1;}
.news {border:1px solid rgba(128,128,128,.22); border-radius:12px;
       padding:12px 16px; margin-bottom:10px; background:rgba(128,128,128,.05);}
.news a {text-decoration:none; font-weight:600; font-size:1.02rem;}
.news a:hover {text-decoration:underline;}
.meta {font-size:.8rem; opacity:.6; margin-top:5px;}
.badge {font-size:.74rem; font-weight:700; padding:.14rem .55rem;
        border-radius:999px; margin-right:.45rem; display:inline-block;}
.badge.positive {background:rgba(46,158,91,.18); color:#2e9e5b;}
.badge.negative {background:rgba(214,69,69,.18); color:#d64545;}
.badge.neutral {background:rgba(128,128,128,.22);}
.tag {font-size:.74rem; opacity:.7; border:1px solid rgba(128,128,128,.35);
      padding:.1rem .5rem; border-radius:999px;}
.brief {border-left:4px solid #4f8cff; padding:.4rem 1.1rem; border-radius:6px;
        background:rgba(79,140,255,.07);}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- API key
def server_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file when running locally
        key = None
    return key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------------------- market data
@st.cache_data(ttl=900, show_spinner=False)
def get_prices(ticker: str, period: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:  # raising means an empty result is NOT cached
        raise LookupError("no price data")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    return hist[["Close", "Volume"]]


@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker: str) -> pd.DataFrame:
    """Headlines with timestamp + URL (handles both yfinance news formats)."""
    rows = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)
        title = c.get("title")
        ts = c.get("pubDate") or item.get("providerPublishTime")
        if not title or ts is None:
            continue
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
        )
        ts = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
        publisher = (c.get("provider") or {}).get("displayName") or item.get("publisher") or "—"
        rows.append({"published": ts, "headline": title, "source": publisher, "url": url})
    if not rows:
        raise LookupError("no headlines")
    return pd.DataFrame(rows).sort_values("published", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


# ------------------------------------------------------------- Gemini helpers
RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "504", "DEADLINE")


@st.cache_data(ttl=3600, show_spinner=False)
def list_flash_models(api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)
    skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer")
    names = []
    for m in client.models.list():
        n = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "flash" in n and "generateContent" in actions and not any(s in n for s in skip):
            names.append(n)
    names.sort(reverse=True)  # newer versions first
    names.sort(key=lambda n: "lite" in n)  # lite models last (stable sort)
    return names


def gemini_generate(api_key: str, prompt: str, config=None):
    """Retry on busy errors, then fall back to other available Flash models."""
    client = genai.Client(api_key=api_key)
    queue, i, discovered, last = [PREFERRED_MODEL], 0, False, None
    while i < len(queue):
        for attempt in range(3):
            try:
                return client.models.generate_content(model=queue[i], contents=prompt, config=config)
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < 2 and any(t in str(e) for t in RETRYABLE):
                    time.sleep(1.5 * 2**attempt)
                    continue
                break
        i += 1
        if i == len(queue) and not discovered:
            discovered = True
            try:
                queue += [n for n in list_flash_models(api_key) if n not in queue][:3]
            except Exception:  # noqa: BLE001
                pass
    raise last


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


def label_for(score: float) -> str:
    return "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"


@st.cache_data(ttl=3600, show_spinner=False)
def gemini_scores(headlines: tuple[str, ...], api_key: str) -> pd.DataFrame:
    numbered = "\n".join(f"{i}: {h}" for i, h in enumerate(headlines))
    prompt = f"""You are a financial news sentiment classifier.
For each numbered headline return:
- id: the headline number
- score: float from -1.0 (very bearish) to +1.0 (very bullish) for the company's stock; 0 if neutral.
  Read financial jargon correctly (e.g. "slashes debt" is bullish).
- catalyst: ONE word from: Earnings, Regulation, M&A, Product, Macro, Legal, Management, Analyst, Other.

Headlines:
{numbered}"""
    resp = gemini_generate(
        api_key,
        prompt,
        types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=SCORE_SCHEMA, temperature=0
        ),
    )
    data = {int(d["id"]): d for d in json.loads(resp.text)}
    rows = []
    for i in range(len(headlines)):
        d = data.get(i, {"score": 0.0, "catalyst": "Other"})
        s = max(-1.0, min(1.0, float(d["score"])))
        rows.append({"sentiment": label_for(s), "score": s, "catalyst": d["catalyst"]})
    return pd.DataFrame(rows)


POS_WORDS = set(
    "beat beats surge surges soar soars jump jumps rally rallies record growth upgrade upgrades raises "
    "profit profits gains gain rise rises strong bullish outperform buy boost wins approval expands tops "
    "higher climb climbs launch partnership".split()
)
NEG_WORDS = set(
    "miss misses fall falls drop drops plunge plunges slump cut cuts downgrade downgrades lawsuit probe "
    "investigation fine fraud recall weak bearish loss losses layoffs warns warning decline declines "
    "tumble tumbles ban risk risks concern concerns lower sell sinks crash".split()
)


def keyword_scores(headlines: list[str]) -> pd.DataFrame:
    rows = []
    for h in headlines:
        words = re.findall(r"[a-z']+", h.lower())
        p, n = sum(w in POS_WORDS for w in words), sum(w in NEG_WORDS for w in words)
        s = (p - n) / max(1, p + n)
        rows.append({"sentiment": label_for(s), "score": s, "catalyst": "—"})
    return pd.DataFrame(rows)


def score_news(news: pd.DataFrame, api_key: str | None):
    """Returns (scores, method, error). Falls back to keywords if Gemini fails."""
    if api_key:
        try:
            return gemini_scores(tuple(news["headline"]), api_key), "gemini", None
        except Exception as e:  # noqa: BLE001
            return keyword_scores(news["headline"].tolist()), "keyword", str(e)
    return keyword_scores(news["headline"].tolist()), "keyword", "no key"


def write_briefing(api_key: str, ticker: str, move: float, lines: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {l}" for i, l in enumerate(lines, 1))
    prompt = f"""You are a senior equity research analyst writing a morning briefing.
Ticker: {ticker}. Latest daily price move: {move:+.2f}%.

Top headlines (with sentiment score):
{numbered}

Write exactly three bullet points:
- **Key Bullish Drivers**
- **Key Risk Factors**
- **Price-Sentiment Divergence** (flag if news tone and the price move disagree)

Rules: cite headlines by number like [1], [2]. Use ONLY facts in the headlines above.
Do NOT invent any financial figures, targets or dates. Be concise."""
    return gemini_generate(api_key, prompt).text


# ------------------------------------------------------------------ UI helpers
def kpi(label: str, value: str, sub: str = "", tone: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub {tone}">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def tone_of(x: float) -> str:
    return "up" if x > 0 else "down" if x < 0 else ""


def mood(avg: float) -> str:
    return "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📈 Sentiment Tracker")
    choice = st.selectbox("Stock", list(POPULAR) + ["Other…"])
    if choice == "Other…":
        ticker = st.text_input(
            "Ticker symbol", "AAPL", help="US: AAPL · India NSE: RELIANCE.NS · BSE: RELIANCE.BO"
        ).strip().upper()
    else:
        ticker = POPULAR[choice]
    period_label = st.radio("Price range", list(PERIODS), index=1, horizontal=True)

    api_key = server_key()
    if not api_key:
        st.text_input("Gemini API key", type="password", key="user_key", help="Free key from Google AI Studio")
        api_key = st.session_state.get("user_key") or None

    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Prices & news: Yahoo Finance. Sentiment: Gemini AI. Not investment advice.")

if not ticker:
    st.stop()

# ------------------------------------------------------------------- load data
try:
    prices = get_prices(ticker, PERIODS[period_label])
except Exception:  # noqa: BLE001
    st.error(
        f"Couldn't load prices for **{ticker}**. Check the symbol (Indian stocks need .NS or .BO), "
        "or click Refresh in a minute if Yahoo is rate-limiting."
    )
    st.stop()

try:
    news = get_news(ticker)
except Exception:  # noqa: BLE001
    news = pd.DataFrame()

scored, method = pd.DataFrame(), None
if not news.empty:
    with st.spinner("Scoring headlines..."):
        sc, method, err = score_news(news, api_key)
    scored = pd.concat([news, sc], axis=1)
    scored["date"] = scored["published"].dt.tz_convert(None).dt.normalize()

cur = "₹" if ticker.endswith((".NS", ".BO")) else "$"
last, prev, first = float(prices["Close"].iloc[-1]), float(prices["Close"].iloc[-2]) if len(prices) > 1 else float(prices["Close"].iloc[-1]), float(prices["Close"].iloc[0])
day_move = (last / prev - 1) * 100 if prev else 0.0
range_move = (last / first - 1) * 100 if first else 0.0
avg = float(scored["score"].mean()) if not scored.empty else None

# ---------------------------------------------------------------------- header
name = get_name(ticker)
st.markdown(
    f'<div class="hero"><h1>{html.escape(name)}</h1><span class="sym">{html.escape(ticker)}</span></div>'
    f'<div class="sub">Price action and news sentiment for the last {period_label}</div>',
    unsafe_allow_html=True,
)

if method == "keyword":
    if err == "no key":
        st.info("Add a Gemini API key in the sidebar for AI sentiment. Showing basic keyword scoring for now.")
    else:
        st.warning(
            "Gemini is busy right now, so sentiment is shown using basic keyword scoring. "
            "Click **Refresh data** in a minute to retry with AI."
        )

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Price", f"{cur}{last:,.2f}", f"as of {prices.index[-1]:%d %b}")
with c2:
    kpi("Latest day", f"{day_move:+.2f}%", "vs previous close", tone_of(day_move))
with c3:
    kpi(f"{period_label} change", f"{range_move:+.2f}%", f"from {cur}{first:,.2f}", tone_of(range_move))
with c4:
    if avg is None:
        kpi("News sentiment", "n/a", "no headlines found")
    else:
        kpi("News sentiment", f"{avg:+.2f}", f"{mood(avg)} · {len(scored)} headlines", tone_of(avg))

st.write("")
tab_overview, tab_news, tab_brief = st.tabs(["📊 Overview", "📰 Headlines", "🧠 Analyst briefing"])

# --------------------------------------------------------------------- overview
with tab_overview:
    left, right = st.columns([2.2, 1])
    with left:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not scored.empty:
            daily = scored.groupby("date")["score"].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=daily["date"],
                    y=daily["score"],
                    name="Avg sentiment",
                    opacity=0.5,
                    marker_color=["#2e9e5b" if s >= 0 else "#d64545" for s in daily["score"]],
                ),
                secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices["Close"], name="Close", mode="lines",
                line=dict(width=3, color="#4f8cff"),
            ),
            secondary_y=False,
        )
        fig.update_yaxes(title_text=f"Close ({cur})", secondary_y=False, showgrid=True)
        fig.update_yaxes(title_text="Sentiment", range=[-1, 1], secondary_y=True, showgrid=False)
        fig.update_xaxes(showgrid=False)
        fig.update_layout(
            height=430, margin=dict(t=10, b=10, l=10, r=10), hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Sentiment bars appear only on days that have headlines (Yahoo exposes recent news only).")

        if avg is not None:
            if avg > 0.15 and day_move < -0.5:
                st.warning("⚠️ **Divergence:** news tone is bullish, but the price fell on the latest day.")
            elif avg < -0.15 and day_move > 0.5:
                st.warning("⚠️ **Divergence:** news tone is bearish, but the price rose on the latest day.")
            else:
                st.success("✅ News tone and the latest price move are broadly aligned.")

    with right:
        st.markdown("**Sentiment mix**")
        if scored.empty:
            st.info("No headlines to analyse.")
        else:
            counts = scored["sentiment"].value_counts()
            order = ["positive", "neutral", "negative"]
            colors = {"positive": "#2e9e5b", "neutral": "#8a8f98", "negative": "#d64545"}
            pie = go.Figure(
                go.Pie(
                    labels=[o.title() for o in order],
                    values=[int(counts.get(o, 0)) for o in order],
                    hole=0.65, sort=False,
                    marker=dict(colors=[colors[o] for o in order]),
                    textinfo="value",
                )
            )
            pie.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(pie, width="stretch")
            if method == "gemini":
                cats = scored["catalyst"].value_counts().head(4)
                st.markdown("**Top catalysts**")
                st.markdown("  \n".join(f"• {k} — {v}" for k, v in cats.items()))

# ---------------------------------------------------------------------- headlines
with tab_news:
    if scored.empty:
        st.info("No recent headlines were returned for this ticker.")
    else:
        flt = st.radio("Show", ["All", "Positive", "Neutral", "Negative"], horizontal=True)
        view = scored if flt == "All" else scored[scored["sentiment"] == flt.lower()]
        if view.empty:
            st.caption("Nothing in this category.")
        for r in view.itertuples():
            title = html.escape(r.headline)
            link = (
                f'<a href="{html.escape(r.url, quote=True)}" target="_blank" rel="noopener">{title}</a>'
                if isinstance(r.url, str) and r.url.startswith("http")
                else f"<b>{title}</b>"
            )
            tag = f'<span class="tag">{html.escape(str(r.catalyst))}</span>' if r.catalyst != "—" else ""
            st.markdown(
                f'<div class="news"><span class="badge {r.sentiment}">{r.sentiment.title()} {r.score:+.2f}</span>{tag}'
                f'<div style="margin-top:6px">{link}</div>'
                f'<div class="meta">{html.escape(str(r.source))} · {r.published:%d %b %Y, %H:%M} UTC</div></div>',
                unsafe_allow_html=True,
            )
        csv = scored[["published", "headline", "sentiment", "score", "catalyst", "source", "url"]].to_csv(index=False)
        st.download_button("⬇️ Download as CSV", csv, f"{ticker}_headlines_sentiment.csv", "text/csv")

# ------------------------------------------------------------------------ briefing
with tab_brief:
    if scored.empty:
        st.info("Briefing needs at least one headline.")
    elif not api_key:
        st.info("Add a Gemini API key in the sidebar to generate the briefing.")
    else:
        st.caption("An AI analyst summary of the five highest-impact headlines. Generated on demand.")
        top = scored.reindex(scored["score"].abs().sort_values(ascending=False).index).head(5)
        lines = [f"({r.sentiment}, {r.score:+.2f}) {r.headline}" for r in top.itertuples()]
        state_key = f"brief_{ticker}"
        if st.button("✨ Generate briefing", type="primary"):
            with st.spinner("Writing briefing..."):
                try:
                    st.session_state[state_key] = write_briefing(api_key, ticker, day_move, lines)
                except Exception:  # noqa: BLE001
                    st.session_state.pop(state_key, None)
                    st.error("Gemini is busy right now. Wait a minute and click Generate again.")
        if state_key in st.session_state:
            st.markdown('<div class="brief">', unsafe_allow_html=True)
            st.markdown(st.session_state[state_key])
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("Headlines referenced as [1]–[5]"):
                for i, l in enumerate(lines, 1):
                    st.write(f"[{i}] {l}")
