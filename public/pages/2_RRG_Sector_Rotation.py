from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty Scanner + RRG Rotation", page_icon="🧭", layout="wide")
st.title("🧭 Nifty Scanner with Sector Rotation Filter")
st.caption("Applies the existing Supertrend + TC/green-arrow breakout scan first, then filters those scan results by RRG sector rotation. Research tool only; not investment advice.")

UNIVERSES = {
    "Nifty 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "Nifty 100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "Nifty 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}


def clean_ticker(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    if not value or value in {"NAN", "-", "SYMBOL", "TICKER"}:
        return None
    return value if value.endswith(".NS") else f"{value}.NS"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_constituents(url: str) -> pd.DataFrame:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def parse_constituents(raw: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(c).strip().lower(): c for c in raw.columns}
    symbol_col = next((col_map[k] for k in ("symbol", "ticker", "tradingsymbol") if k in col_map), None)
    sector_col = next((col_map[k] for k in ("sector", "industry", "sector name") if k in col_map), None)
    if symbol_col is None or sector_col is None:
        return pd.DataFrame(columns=["ticker", "sector"])
    result = pd.DataFrame({"ticker": raw[symbol_col].map(clean_ticker), "sector": raw[sector_col].astype("string").str.strip()})
    return result.dropna().query("sector != ''").drop_duplicates("ticker").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_ohlc(tickers: tuple[str, ...], start: date) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(list(tickers), start=start.isoformat(), auto_adjust=True, group_by="ticker", threads=True, progress=False, actions=False)
    if raw is None or raw.empty:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        first, second = set(raw.columns.get_level_values(0)), set(raw.columns.get_level_values(1))
        for ticker in tickers:
            try:
                frame = raw[ticker].copy() if ticker in first else raw.xs(ticker, axis=1, level=1).copy() if ticker in second else pd.DataFrame()
                if {"Open", "High", "Low", "Close"}.issubset(frame.columns):
                    frames[ticker] = frame[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
            except (KeyError, ValueError):
                continue
    elif len(tickers) == 1 and {"Open", "High", "Low", "Close"}.issubset(raw.columns):
        frames[tickers[0]] = raw[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    return frames


def resample_bars(daily: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "Daily":
        return daily
    return daily.resample("W-FRI", label="right", closed="right").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()


def add_scan_indicators(bars: pd.DataFrame, sma_len: int, atr_len: int, factor: float, pivot_bars: int) -> pd.DataFrame:
    df = bars.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    sma_low = low.rolling(sma_len, min_periods=sma_len).mean()
    sma_high = high.rolling(sma_len, min_periods=sma_len).mean()
    above = (close > sma_low) & (close > sma_high)
    df["green_arrow"] = above & ~above.shift(1, fill_value=False)
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_len, adjust=False, min_periods=atr_len).mean()
    mid = (high+low)/2
    upper, lower = (mid+factor*atr).to_numpy(), (mid-factor*atr).to_numpy()
    closes = close.to_numpy()
    final_u, final_l = np.full(len(df), np.nan), np.full(len(df), np.nan)
    st_line, green = np.full(len(df), np.nan), np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            continue
        if i == 0 or not np.isfinite(final_u[i-1]):
            final_u[i], final_l[i], st_line[i] = upper[i], lower[i], upper[i]
            continue
        final_u[i] = upper[i] if upper[i] < final_u[i-1] or closes[i-1] > final_u[i-1] else final_u[i-1]
        final_l[i] = lower[i] if lower[i] > final_l[i-1] or closes[i-1] < final_l[i-1] else final_l[i-1]
        if st_line[i-1] == final_u[i-1]:
            if closes[i] <= final_u[i]:
                st_line[i] = final_u[i]
            else:
                st_line[i], green[i] = final_l[i], True
        elif closes[i] >= final_l[i]:
            st_line[i], green[i] = final_l[i], True
        else:
            st_line[i] = final_u[i]
    df["supertrend"], df["st_green"] = st_line, green

    window = pivot_bars*2+1
    pivot = high.where(high.eq(high.rolling(window, center=True, min_periods=window).max())).shift(pivot_bars).ffill()
    previous_level = pivot.shift(1)
    cross = (close > previous_level) & (close.shift(1) <= previous_level.shift(1))
    state = False
    bos, mss = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    for i, crossed in enumerate(cross.fillna(False).to_numpy()):
        if crossed:
            if state:
                bos[i] = True
            else:
                mss[i] = True
            state = True
    df["tc_buy"] = bos | mss
    return df


def calculate_rrg(ohlc: dict[str, pd.DataFrame], members: pd.DataFrame, benchmark: pd.Series, lookback: int) -> pd.DataFrame:
    closes = pd.DataFrame({ticker: frame["Close"] for ticker, frame in ohlc.items() if ticker in set(members.ticker)})
    weekly = closes.resample("W-FRI").last()
    bench = benchmark.resample("W-FRI").last()
    # Exclude an unfinished current week so sector and benchmark values align to completed bars.
    if len(weekly) and len(closes) and weekly.index[-1] > closes.index[-1]:
        weekly = weekly.iloc[:-1]
    if len(bench) and len(benchmark) and bench.index[-1] > benchmark.index[-1]:
        bench = bench.iloc[:-1]
    common = weekly.index.intersection(bench.dropna().index)
    weekly, bench = weekly.loc[common], bench.reindex(common)
    if len(common) < lookback*2+2:
        return pd.DataFrame()
    stock_returns, bench_returns = weekly.pct_change(fill_method=None), bench.pct_change(fill_method=None)
    sector_returns = {}
    for sector, rows in members.groupby("sector"):
        available = [t for t in rows.ticker if t in stock_returns]
        if available:
            sector_returns[str(sector)] = stock_returns[available].mean(axis=1)
    if not sector_returns:
        return pd.DataFrame()
    sector_index = (1 + pd.DataFrame(sector_returns).fillna(0)).cumprod()
    benchmark_index = (1 + bench_returns.fillna(0)).cumprod()
    relative = sector_index.div(benchmark_index, axis=0)
    ratio = relative.div(relative.rolling(lookback, min_periods=lookback).mean())*100
    momentum = ratio.div(ratio.shift(lookback))*100
    latest_date = ratio.dropna(how="all").index[-1]
    prior_pos = max(0, ratio.index.get_loc(latest_date)-min(12, ratio.index.get_loc(latest_date)))
    prior_date = ratio.index[prior_pos]
    result = []
    for sector in ratio.columns:
        x, y = ratio.at[latest_date, sector], momentum.at[latest_date, sector]
        if pd.isna(x) or pd.isna(y):
            continue
        quadrant = "Leading" if x >= 100 and y >= 100 else "Weakening" if x >= 100 else "Improving" if y >= 100 else "Lagging"
        result.append({"Sector": sector, "RS-Ratio": float(x), "RS-Momentum": float(y), "Quadrant": quadrant,
                       "12-week RS-Ratio change": float(x-ratio.at[prior_date, sector]) if pd.notna(ratio.at[prior_date, sector]) else np.nan,
                       "As of": latest_date})
    return pd.DataFrame(result)


with st.sidebar:
    universe = st.selectbox("Index constituents", list(UNIVERSES))
    timeframe = st.selectbox("Existing scan timeframe", ["Daily", "Weekly"])
    years = st.selectbox("History (years)", [2, 3, 5], index=1)
    sma_len = st.number_input("SMA band length", min_value=1, value=65)
    atr_len = st.number_input("Supertrend ATR length", min_value=1, value=15)
    factor = st.number_input("Supertrend factor", min_value=0.1, value=3.0, step=0.1)
    pivot_bars = st.number_input("Structure pivot bars", min_value=1, value=2)
    rrg_lookback = st.number_input("RRG lookback (weeks)", min_value=4, max_value=26, value=12)
    quadrants = st.multiselect("Keep stocks in these RRG quadrants", ["Leading", "Improving", "Weakening", "Lagging"], default=["Leading", "Improving"])
    uploaded = st.file_uploader("Optional constituents file (Symbol and Sector/Industry required)", type=["csv", "xlsx"])

try:
    constituents_raw = (pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)) if uploaded else load_constituents(UNIVERSES[universe])
    members = parse_constituents(constituents_raw)
except Exception as exc:
    st.error(f"Could not load constituents: {exc}")
    st.stop()
if members.empty:
    st.error("Constituent source must include Symbol and Sector/Industry columns for sector filtering.")
    st.stop()

members = members.head(500)
benchmark_ticker = "^NSEI"
all_tickers = tuple(sorted(set(members.ticker.tolist() + [benchmark_ticker])))
start = date.today()-timedelta(days=365*int(years)+90)
with st.spinner(f"Downloading OHLC data for {len(members)} stocks and Nifty 50 benchmark…"):
    ohlc = download_ohlc(all_tickers, start)
if benchmark_ticker not in ohlc or not ohlc:
    st.error("Price history or Nifty 50 benchmark is unavailable.")
    st.stop()

rrg = calculate_rrg(ohlc, members, ohlc[benchmark_ticker]["Close"], int(rrg_lookback))
if rrg.empty:
    st.warning("Insufficient overlapping weekly history for RRG calculation.")
    st.stop()

colors = {"Leading": "#16a34a", "Weakening": "#eab308", "Lagging": "#dc2626", "Improving": "#3b82f6"}
fig = go.Figure()
for quadrant, group in rrg.groupby("Quadrant"):
    fig.add_trace(go.Scatter(x=group["RS-Ratio"], y=group["RS-Momentum"], mode="markers+text", text=group["Sector"], textposition="top center", name=quadrant,
                             marker={"size": 13, "color": colors[quadrant]}, customdata=group[["12-week RS-Ratio change"]],
                             hovertemplate="%{text}<br>RS-Ratio %{x:.2f}<br>RS-Momentum %{y:.2f}<br>12-week ratio change %{customdata[0]:.2f}<extra>%{fullData.name}</extra>"))
fig.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
fig.add_vline(x=100, line_dash="dash", line_color="#94a3b8")
fig.update_layout(title="Sector RRG vs Nifty 50", xaxis_title="RS-Ratio", yaxis_title="RS-Momentum", template="plotly_dark", height=600, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(rrg.drop(columns="As of").sort_values(["Quadrant", "RS-Ratio"], ascending=[True, False]), use_container_width=True, hide_index=True)

# First run the original scan rules without the RRG filter.
scan_rows = []
minimum = max(int(sma_len), int(atr_len)) + int(pivot_bars)*2 + 5
for ticker, frame in ohlc.items():
    if ticker == benchmark_ticker:
        continue
    bars = resample_bars(frame, timeframe)
    if len(bars) < minimum:
        continue
    tested = add_scan_indicators(bars, int(sma_len), int(atr_len), float(factor), int(pivot_bars))
    tested["confirmation"] = tested["tc_buy"] | tested["green_arrow"]
    tested["setup"] = tested["st_green"] & tested["confirmation"]
    if len(tested) < 2:
        continue
    latest, previous = tested.iloc[-1], tested.iloc[-2]
    triggered = bool(previous["setup"]) and bool(latest["st_green"]) and float(latest["High"]) > float(previous["High"])
    status = "ENTRY TRIGGERED" if triggered else "ARMED" if bool(latest["setup"]) else "—"
    sector = members.loc[members.ticker.eq(ticker), "sector"]
    scan_rows.append({"Stock": ticker.removesuffix(".NS"), "ticker": ticker, "Sector": sector.iloc[0] if len(sector) else "Unknown",
                      "Timeframe": timeframe, "Date": tested.index[-1], "Old-rule status": status,
                      "Supertrend": "Green" if bool(latest["st_green"]) else "Red",
                      "TC/green-arrow confirmation": bool(latest["confirmation"]), "Close": float(latest["Close"]),
                      "Breakout level": float(previous["High"]), "Supertrend level": float(latest["supertrend"]) if pd.notna(latest["supertrend"]) else np.nan})

raw_scan = pd.DataFrame(scan_rows)
if raw_scan.empty:
    st.warning("No symbols had enough data to evaluate the existing scan rules.")
    st.stop()
sector_rrg = rrg[["Sector", "Quadrant", "RS-Ratio", "RS-Momentum"]]
raw_scan = raw_scan.merge(sector_rrg, on="Sector", how="left")
raw_scan["Quadrant"] = raw_scan["Quadrant"].fillna("Unclassified")
active_scan = raw_scan[raw_scan["Old-rule status"] != "—"].copy()
filtered = active_scan[active_scan["Quadrant"].isin(quadrants)].copy()

st.subheader("Final filtered list — existing scan rules, then RRG sector filter")
st.caption(f"Scanned {len(raw_scan)} stocks with the existing setup/breakout logic; {len(active_scan)} had a setup; {len(filtered)} remain after filtering by {', '.join(quadrants) if quadrants else 'no selected quadrants'}. RRG date: {rrg['As of'].iloc[0]:%Y-%m-%d}.")
if filtered.empty:
    st.info("No stocks passed both the existing scan rules and the selected RRG-sector filter.")
else:
    st.dataframe(filtered.drop(columns="ticker").sort_values(["Quadrant", "RS-Ratio", "Old-rule status"], ascending=[True, False, True]), use_container_width=True, hide_index=True)
with st.expander("Show all stocks passing the existing scan before RRG filtering"):
    st.dataframe(active_scan.drop(columns=[c for c in ("ticker",) if c in active_scan]).sort_values(["Quadrant", "RS-Ratio"], ascending=[True, False]), use_container_width=True, hide_index=True)
st.info("RRG is a relative-performance filter, not an entry signal. This page filters scan results by sector-level rotation and does not change the original scanner or guarantee returns. Current index constituents may introduce survivorship bias.")
