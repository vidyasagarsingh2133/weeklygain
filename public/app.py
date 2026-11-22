from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from urllib.parse import urlencode

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

st.set_page_config(page_title="Nifty Trend Scanner", page_icon="📊", layout="wide")

UNIVERSES = {
    "📈 Nifty 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "🚀 Nifty Next 50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "💯 Nifty 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "🏢 Nifty Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "🌱 Nifty Smallcap 50": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap50list.csv",
    "🌿 Nifty Smallcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
    "🌳 Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}


def normalize_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    symbol = str(value).strip().upper()
    if not symbol or symbol in {"SYMBOL", "TICKER", "NAN", "-"}:
        return None
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_universe_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def symbols_from_csv(data: pd.DataFrame) -> list[str]:
    columns = {str(c).strip().lower(): c for c in data.columns}
    symbol_col = next(
        (columns[name] for name in ("symbol", "ticker", "tradingsymbol") if name in columns),
        data.columns[0] if len(data.columns) else None,
    )
    if symbol_col is None:
        return []
    values = [normalize_symbol(v) for v in data[symbol_col]]
    return sorted(set(v for v in values if v))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_daily_prices(tickers: tuple[str, ...], start_date: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers=list(tickers), start=start_date.isoformat(), auto_adjust=True,
        group_by="ticker", threads=True, progress=False, actions=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        frames = {}
        for ticker in tickers:
            if ticker in level0:
                frames[ticker] = raw[ticker].copy()
            elif ticker in level1:
                frames[ticker] = raw.xs(ticker, axis=1, level=1).copy()
        return pd.concat(frames, names=["Ticker", "Date"]) if frames else pd.DataFrame()
    single = raw.copy()
    single["Ticker"] = tickers[0]
    single["Date"] = single.index
    return single.set_index(["Ticker", "Date"])


def ohlc_for_ticker(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    try:
        frame = download.xs(ticker, level="Ticker").copy()
    except (KeyError, ValueError):
        return pd.DataFrame()
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def to_timeframe(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "Daily":
        return daily
    return daily.resample("W-FRI", label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()


def add_indicators(bars: pd.DataFrame, sma_length: int, atr_length: int, factor: float, pivot_bars: int) -> pd.DataFrame:
    df = bars.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    df["sma_low"] = low.rolling(sma_length, min_periods=sma_length).mean()
    df["sma_high"] = high.rolling(sma_length, min_periods=sma_length).mean()
    above_band = (close > df["sma_low"]) & (close > df["sma_high"])
    df["green_arrow"] = above_band & ~above_band.shift(1, fill_value=False)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_length, adjust=False, min_periods=atr_length).mean()
    midpoint = (high + low) / 2
    basic_upper, basic_lower = midpoint + factor * atr, midpoint - factor * atr

    count = len(df)
    final_upper = np.full(count, np.nan)
    final_lower = np.full(count, np.nan)
    supertrend = np.full(count, np.nan)
    is_green = np.zeros(count, dtype=bool)
    upper_values, lower_values, close_values = basic_upper.to_numpy(), basic_lower.to_numpy(), close.to_numpy()

    for i in range(count):
        if not np.isfinite(upper_values[i]) or not np.isfinite(lower_values[i]):
            continue
        if i == 0 or not np.isfinite(final_upper[i - 1]):
            final_upper[i], final_lower[i] = upper_values[i], lower_values[i]
            supertrend[i] = final_upper[i]
            continue
        prior_close = close_values[i - 1]
        final_upper[i] = upper_values[i] if upper_values[i] < final_upper[i - 1] or prior_close > final_upper[i - 1] else final_upper[i - 1]
        final_lower[i] = lower_values[i] if lower_values[i] > final_lower[i - 1] or prior_close < final_lower[i - 1] else final_lower[i - 1]
        if supertrend[i - 1] == final_upper[i - 1]:
            if close_values[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i], is_green[i] = final_lower[i], True
        elif close_values[i] >= final_lower[i]:
            supertrend[i], is_green[i] = final_lower[i], True
        else:
            supertrend[i] = final_upper[i]

    df["supertrend"], df["st_green"] = supertrend, is_green

    window = pivot_bars * 2 + 1
    rolling_high = high.rolling(window, center=True, min_periods=window).max()
    confirmed_pivot = high.where(high.eq(rolling_high)).shift(pivot_bars).ffill()
    previous_level = confirmed_pivot.shift(1)
    cross_up = (close > previous_level) & (close.shift(1) <= previous_level.shift(1))
    bull_state = False
    bos_bull, mss_bull = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    for i, crossed in enumerate(cross_up.fillna(False).to_numpy()):
        if crossed:
            if bull_state:
                bos_bull[i] = True
            else:
                mss_bull[i] = True
            bull_state = True
    df["bos_bull"], df["mss_bull"] = bos_bull, mss_bull
    return df


def confirmation_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    tc_buy = df["bos_bull"] | df["mss_bull"]
    if mode == "Green arrow (SMA band)":
        return df["green_arrow"]
    if mode == "TC buy (BOS/MSS proxy)":
        return tc_buy
    return tc_buy | df["green_arrow"]


def run_backtest(df: pd.DataFrame, ticker: str, timeframe: str, confirm_mode: str) -> tuple[list[dict], pd.DataFrame]:
    out = df.copy()
    if out.empty:
        return [], out
    out["confirmation"] = confirmation_mask(out, confirm_mode)
    out["setup"] = out["st_green"] & out["confirmation"]
    out["breakout"] = False
    trades, position = [], None

    for i in range(1, len(out)):
        row, previous = out.iloc[i], out.iloc[i - 1]
        if position is not None and not bool(row["st_green"]):
            exit_price, exit_date = float(row["Close"]), out.index[i]
            trades.append({
                **position, "exit_date": exit_date, "exit_price": exit_price, "status": "CLOSED",
                "holding_bars": i - position["entry_bar"],
                "holding_days": (exit_date.date() - position["entry_date"].date()).days,
                "pnl_per_share": exit_price - position["entry_price"],
                "return_pct": (exit_price / position["entry_price"] - 1) * 100,
            })
            position = None
            continue
        trigger_level = float(previous["High"])
        triggered = position is None and bool(previous["setup"]) and bool(row["st_green"]) and float(row["High"]) > trigger_level
        out.iat[i, out.columns.get_loc("breakout")] = triggered
        if triggered:
            position = {
                "ticker": ticker, "timeframe": timeframe, "entry_date": out.index[i],
                "entry_price": max(float(row["Open"]), trigger_level),
                "trigger_level": trigger_level, "entry_bar": i,
            }

    if position is not None:
        last, mark = out.iloc[-1], float(out.iloc[-1]["Close"])
        trades.append({
            **position, "exit_date": pd.NaT, "exit_price": np.nan, "status": "OPEN",
            "holding_bars": len(out) - 1 - position["entry_bar"],
            "holding_days": (out.index[-1].date() - position["entry_date"].date()).days,
            "pnl_per_share": mark - position["entry_price"],
            "return_pct": (mark / position["entry_price"] - 1) * 100, "mark_price": mark,
        })
    return trades, out


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_stock_news(ticker: str, limit: int = 8) -> list[dict]:
    symbol = ticker.removesuffix(".NS")
    params = urlencode({"q": f'"{symbol}" NSE stock', "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    try:
        response = requests.get(
            f"https://news.google.com/rss/search?{params}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception:
        return []
    news = []
    for item in feed.entries[:limit]:
        try:
            published = pd.to_datetime(getattr(item, "published", ""), utc=True).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            published = "Recently"
        summary = BeautifulSoup(getattr(item, "summary", ""), "html.parser").get_text(" ", strip=True)
        source = getattr(item, "source", {})
        news.append({
            "headline": getattr(item, "title", "News headline"),
            "source": source.get("title", "Google News") if isinstance(source, dict) else "Google News",
            "published": published, "summary": summary, "url": getattr(item, "link", ""),
        })
    return news


st.title("📊 Nifty Daily & Weekly Trend Scanner")
st.caption("After-market scanner, backtest, and stock news. Uses third-party data; not investment advice.")

with st.sidebar:
    st.header("⚙️ Settings")
    selected_universe = st.selectbox("Choose index", list(UNIVERSES))
    timeframes = st.multiselect("Choose timeframe", ["Daily", "Weekly"], default=["Daily", "Weekly"])
    confirm_mode = st.selectbox("TC confirmation", ["TC buy OR green arrow", "Green arrow (SMA band)", "TC buy (BOS/MSS proxy)"])
    sma_length = st.number_input("SMA band length", min_value=1, value=65)
    atr_length = st.number_input("Supertrend ATR length", min_value=1, value=15)
    factor = st.number_input("Supertrend factor", min_value=0.01, value=3.0, step=0.1)
    pivot_bars = st.number_input("Structure pivot bars", min_value=1, value=2)
    years = st.selectbox("Backtest history (years)", [1, 2, 3, 5], index=3)
    uploaded = st.file_uploader("Optional constituents CSV/XLSX", type=["csv", "xlsx"])

if not timeframes:
    st.info("Select Daily, Weekly, or both in the sidebar.")
    st.stop()

try:
    if uploaded is not None:
        universe_frame = pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
    else:
        universe_frame = fetch_universe_csv(UNIVERSES[selected_universe])
    symbols = symbols_from_csv(universe_frame)
except Exception as exc:
    st.error(f"Could not load constituents ({exc}). Upload a current CSV/XLSX in the sidebar.")
    st.stop()

if not symbols:
    st.error("No symbols found. The uploaded file needs a Symbol or Ticker column.")
    st.stop()

symbols = symbols[:500]
tickers = tuple(symbols)
download_start = date.today() - timedelta(days=365 * int(years) + 60)
with st.spinner(f"Downloading end-of-day data for {len(tickers)} stocks…"):
    prices = download_daily_prices(tickers, download_start)
if prices.empty:
    st.error("No price history was returned. Try again later or check symbols and internet connection.")
    st.stop()

scan_rows, all_trades, tested_by_ticker_timeframe = [], [], {}
progress = st.progress(0)
for symbol_index, ticker in enumerate(tickers):
    daily = ohlc_for_ticker(prices, ticker)
    if not daily.empty:
        for timeframe in timeframes:
            bars = to_timeframe(daily, timeframe)
            if len(bars) < max(int(sma_length), int(atr_length)) + int(pivot_bars) * 2 + 5:
                continue
            indicators = add_indicators(bars, int(sma_length), int(atr_length), float(factor), int(pivot_bars))
            trades, tested = run_backtest(indicators, ticker, timeframe, confirm_mode)
            all_trades.extend(trades)
            tested_by_ticker_timeframe[(ticker, timeframe)] = tested
            latest = tested.iloc[-1]
            previous = tested.iloc[-2] if len(tested) > 1 else latest
            triggered = bool(previous["setup"]) and bool(latest["st_green"]) and float(latest["High"]) > float(previous["High"])
            status = "🚀 ENTRY TRIGGERED" if triggered else "🟡 ARMED" if bool(latest["setup"]) else "—"
            scan_rows.append({
                "Stock": ticker.removesuffix(".NS"), "Timeframe": timeframe,
                "Date": tested.index[-1], "Status": status,
                "Supertrend": "🟢 Green" if bool(latest["st_green"]) else "🔴 Red",
                "TC confirmation": "✅ Yes" if bool(latest["confirmation"]) else "—",
                "Close": float(latest["Close"]), "Next breakout level": float(latest["High"]),
                "Supertrend level": float(latest["supertrend"]) if pd.notna(latest["supertrend"]) else np.nan,
            })
    progress.progress((symbol_index + 1) / len(tickers))

scan, trades = pd.DataFrame(scan_rows), pd.DataFrame(all_trades)
if scan.empty:
    st.warning("No usable price histories were returned for this selection.")
    st.stop()
scan["Date"] = pd.to_datetime(scan["Date"]).dt.strftime("%Y-%m-%d")
setups = scan[scan["Status"] != "—"].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Constituents loaded", len(symbols))
m2.metric("Stocks with a setup", int((scan["Status"] != "—").sum()))
if not trades.empty:
    closed = trades[trades["status"] == "CLOSED"]
    m3.metric("Closed backtest trades", len(closed))
    m4.metric("Win rate", f"{(closed['return_pct'] > 0).mean() * 100:.1f}%" if len(closed) else "—")
else:
    m3.metric("Closed backtest trades", 0)
    m4.metric("Win rate", "—")

st.subheader("🔎 Current scan")
st.caption("ARMED means green Supertrend plus confirmation. ENTRY TRIGGERED means price also broke the prior setup bar's high.")
st.dataframe(setups.sort_values(["Timeframe", "Status", "Stock"]), use_container_width=True, hide_index=True)
with st.expander("Show all scanned stocks"):
    st.dataframe(scan.sort_values(["Timeframe", "Stock"]), use_container_width=True, hide_index=True)

st.subheader("🧪 Backtest trades")
if trades.empty:
    st.info("No trades were generated for the selected settings.")
else:
    display_trades = trades.copy()
    for column in ("entry_date", "exit_date"):
        display_trades[column] = pd.to_datetime(display_trades[column], errors="coerce").dt.strftime("%Y-%m-%d")
    shown = [c for c in ["ticker", "timeframe", "entry_date", "entry_price", "trigger_level", "exit_date", "exit_price", "status", "holding_bars", "holding_days", "pnl_per_share", "return_pct", "mark_price"] if c in display_trades.columns]
    st.dataframe(display_trades[shown].sort_values(["entry_date", "ticker"], ascending=False), use_container_width=True, hide_index=True)
    closed = trades[trades["status"] == "CLOSED"]
    if not closed.empty:
        a, b, c = st.columns(3)
        a.metric("Sum of trade returns*", f"{closed['return_pct'].sum():.2f}%")
        b.metric("Average trade return", f"{closed['return_pct'].mean():.2f}%")
        c.metric("Closed-trade win rate", f"{(closed['return_pct'] > 0).mean() * 100:.1f}%")
        st.caption("*Returns exclude fees, taxes, slippage, liquidity, and position sizing.")

st.subheader("📈 Stock chart and news")
available_tickers = sorted({ticker for ticker, _ in tested_by_ticker_timeframe})
if available_tickers:
    selected_ticker = st.selectbox("Choose a stock", available_tickers, format_func=lambda value: value.removesuffix(".NS"))
    chart_timeframe = st.selectbox("Chart timeframe", timeframes, key="chart_timeframe")
    chart_bars = tested_by_ticker_timeframe.get((selected_ticker, chart_timeframe))
    if chart_bars is not None and not chart_bars.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=chart_bars.index, open=chart_bars["Open"], high=chart_bars["High"], low=chart_bars["Low"], close=chart_bars["Close"], name=selected_ticker.removesuffix(".NS")))
        green = chart_bars["st_green"]
        fig.add_trace(go.Scatter(x=chart_bars.index, y=chart_bars["supertrend"].where(green), name="Supertrend green", mode="lines", line={"color": "#16a34a", "width": 2}))
        fig.add_trace(go.Scatter(x=chart_bars.index, y=chart_bars["supertrend"].where(~green), name="Supertrend red", mode="lines", line={"color": "#dc2626", "width": 2}))
        fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📰 Stock news")
    st.caption("Recent news headlines matching this stock symbol. Availability depends on Google News indexing.")
    with st.spinner("Looking for recent news…"):
        stock_news = fetch_stock_news(selected_ticker)
    if not stock_news:
        st.info("No recent matching news was found.")
    else:
        for article in stock_news:
            st.markdown(f"### [{article['headline']}]({article['url']})")
            st.caption(f"{article['source']} · {article['published']}")
            if article["summary"]:
                st.write(article["summary"][:400])
            st.divider()
else:
    st.info("No stocks have enough price history to display a chart or news.")
