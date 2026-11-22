from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Nifty RRG Sector Rotation", page_icon="🧭", layout="wide")
st.title("🧭 Sector Rotation — Relative Rotation Graph (RRG)")
st.caption(
    "Compares equal-weighted baskets of current index constituents with the Nifty 50 benchmark. "
    "This is a sector-basket proxy, not an official Nifty sector-index RRG or a return guarantee."
)

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


def constituents_with_sector(raw: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(c).strip().lower(): c for c in raw.columns}
    symbol_col = next((col_map[k] for k in ("symbol", "ticker", "tradingsymbol") if k in col_map), None)
    sector_col = next((col_map[k] for k in ("sector", "industry", "sector name") if k in col_map), None)
    if symbol_col is None or sector_col is None:
        return pd.DataFrame(columns=["ticker", "sector"])
    rows = pd.DataFrame({
        "ticker": raw[symbol_col].map(clean_ticker),
        "sector": raw[sector_col].astype("string").str.strip(),
    })
    rows = rows.dropna().query("sector != ''")
    return rows.drop_duplicates("ticker").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_closes(tickers: tuple[str, ...], start: date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        list(tickers), start=start.isoformat(), auto_adjust=True,
        group_by="ticker", threads=True, progress=False, actions=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        out = {}
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        for ticker in tickers:
            if ticker in level0 and "Close" in raw[ticker]:
                out[ticker] = raw[ticker]["Close"]
            elif ticker in level1:
                frame = raw.xs(ticker, axis=1, level=1)
                if "Close" in frame:
                    out[ticker] = frame["Close"]
        return pd.DataFrame(out).sort_index() if out else pd.DataFrame()
    if len(tickers) == 1:
        close_col = "Close" if "Close" in raw else None
        return raw[[close_col]].rename(columns={close_col: tickers[0]}) if close_col else pd.DataFrame()
    return pd.DataFrame()


def build_rrg(closes: pd.DataFrame, members: pd.DataFrame, benchmark: pd.Series, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = closes.resample("W-FRI").last().dropna(how="all")
    benchmark_weekly = benchmark.resample("W-FRI").last().rename("NIFTY50").dropna()
    common = weekly.index.intersection(benchmark_weekly.index)
    weekly = weekly.loc[common]
    benchmark_weekly = benchmark_weekly.loc[common]
    if len(common) < lookback * 2 + 2:
        return pd.DataFrame(), weekly

    weekly_returns = weekly.pct_change(fill_method=None)
    benchmark_returns = benchmark_weekly.pct_change(fill_method=None)
    sector_returns: dict[str, pd.Series] = {}
    for sector, rows in members.groupby("sector"):
        tickers = [t for t in rows["ticker"] if t in weekly_returns.columns]
        if tickers:
            sector_returns[str(sector)] = weekly_returns[tickers].mean(axis=1)
    if not sector_returns:
        return pd.DataFrame(), weekly

    sector_index = pd.DataFrame(sector_returns).fillna(0).add(1).cumprod()
    benchmark_index = (1 + benchmark_returns.fillna(0)).cumprod()
    relative_strength = sector_index.div(benchmark_index, axis=0)
    rs_ratio = relative_strength.div(relative_strength.rolling(lookback, min_periods=lookback).mean()) * 100
    rs_momentum = rs_ratio.div(rs_ratio.shift(lookback)) * 100
    latest_date = rs_ratio.dropna(how="all").index[-1]
    prior = rs_ratio.index[rs_ratio.index.get_loc(latest_date) - min(12, rs_ratio.index.get_loc(latest_date))]
    rows = []
    for sector in rs_ratio.columns:
        ratio, momentum = rs_ratio.at[latest_date, sector], rs_momentum.at[latest_date, sector]
        if pd.isna(ratio) or pd.isna(momentum):
            continue
        if ratio >= 100 and momentum >= 100:
            quadrant = "Leading"
        elif ratio >= 100:
            quadrant = "Weakening"
        elif momentum >= 100:
            quadrant = "Improving"
        else:
            quadrant = "Lagging"
        rows.append({
            "Sector": sector,
            "RS-Ratio": float(ratio),
            "RS-Momentum": float(momentum),
            "Quadrant": quadrant,
            "12-week RS-Ratio change": float(ratio - rs_ratio.at[prior, sector]) if pd.notna(rs_ratio.at[prior, sector]) else np.nan,
            "As of": latest_date,
        })
    return pd.DataFrame(rows).sort_values(["Quadrant", "RS-Ratio"], ascending=[True, False]), pd.DataFrame({
        "ratio": rs_ratio.loc[:latest_date].tail(13),
        "momentum": rs_momentum.loc[:latest_date].tail(13),
    })


with st.sidebar:
    st.header("RRG settings")
    universe_name = st.selectbox("Constituent universe", list(UNIVERSES))
    history_years = st.selectbox("History", [2, 3, 5], index=1)
    lookback_weeks = st.number_input("RS-Ratio lookback (weeks)", min_value=4, max_value=26, value=12)
    quadrants = st.multiselect("Include rotating sectors", ["Leading", "Improving", "Weakening", "Lagging"], default=["Leading", "Improving"])
    uploaded = st.file_uploader("Optional constituents file (needs Symbol and Sector/Industry)", type=["csv", "xlsx"])

try:
    raw_constituents = (
        pd.read_excel(uploaded) if uploaded.name.lower().endswith(".xlsx") else pd.read_csv(uploaded)
    ) if uploaded is not None else load_constituents(UNIVERSES[universe_name])
    members = constituents_with_sector(raw_constituents)
except Exception as exc:
    st.error(f"Could not load constituent data: {exc}")
    st.stop()

if members.empty:
    st.error("Could not find Symbol and Sector/Industry columns. Upload a constituents file with both fields.")
    st.stop()

members = members.head(500)
benchmark_ticker = "^NSEI"
start_date = date.today() - timedelta(days=365 * int(history_years) + 60)
tickers = tuple(sorted(set(members["ticker"].tolist() + [benchmark_ticker])))
with st.spinner(f"Loading weekly price history for {len(tickers) - 1} constituents and Nifty 50 benchmark…"):
    close_data = download_closes(tickers, start_date)
if close_data.empty or benchmark_ticker not in close_data:
    st.error("Price history or Nifty 50 benchmark was unavailable. Try again later.")
    st.stop()

benchmark_prices = close_data[benchmark_ticker].dropna()
stock_closes = close_data.drop(columns=[benchmark_ticker], errors="ignore")
rrg, rrg_path = build_rrg(stock_closes, members, benchmark_prices, int(lookback_weeks))
if rrg.empty:
    st.warning("Not enough overlapping weekly data to calculate RRG values.")
    st.stop()

latest = rrg.iloc[0]["As of"]
st.caption(f"Latest completed weekly reading: {latest:%Y-%m-%d}. Benchmark: Nifty 50 (^NSEI).")

fig = go.Figure()
colors = {"Leading": "#16a34a", "Weakening": "#eab308", "Lagging": "#dc2626", "Improving": "#3b82f6"}
for quadrant, group in rrg.groupby("Quadrant"):
    fig.add_trace(go.Scatter(
        x=group["RS-Ratio"], y=group["RS-Momentum"], mode="markers+text",
        text=group["Sector"], textposition="top center", name=quadrant,
        marker={"size": 13, "color": colors[quadrant]},
        customdata=group[["12-week RS-Ratio change"]],
        hovertemplate="%{text}<br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br>12-week ratio change: %{customdata[0]:.2f}<extra>%{fullData.name}</extra>",
    ))
fig.add_hline(y=100, line_dash="dash", line_color="#94a3b8")
fig.add_vline(x=100, line_dash="dash", line_color="#94a3b8")
fig.update_layout(
    title="Sector RRG — relative strength vs momentum",
    xaxis_title="RS-Ratio (100 = relative strength trend baseline)",
    yaxis_title="RS-Momentum (100 = momentum baseline)",
    template="plotly_dark", height=650, legend_orientation="h",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sector rotation table")
st.dataframe(rrg.drop(columns="As of").sort_values(["Quadrant", "RS-Ratio"], ascending=[True, False]), use_container_width=True, hide_index=True)

eligible = rrg[rrg["Quadrant"].isin(quadrants)]["Sector"].tolist()
sector_members = members[members["sector"].isin(eligible)].copy()
if not sector_members.empty:
    weekly = stock_closes.resample("W-FRI").last()
    common_weeks = weekly.index.intersection(benchmark_prices.resample("W-FRI").last().dropna().index)
    weekly = weekly.loc[common_weeks]
    bench_weekly = benchmark_prices.resample("W-FRI").last().reindex(common_weeks)
    stock_relative = weekly.div(weekly.iloc[0]).div(bench_weekly / bench_weekly.iloc[0], axis=0)
    stock_12w = stock_relative.pct_change(12).iloc[-1] * 100
    sector_members["12-week relative return vs Nifty 50 (%)"] = sector_members["ticker"].map(stock_12w)
    sector_members["Latest close"] = sector_members["ticker"].map(stock_closes.ffill().iloc[-1])
    sector_members["Stock"] = sector_members["ticker"].str.removesuffix(".NS")
    sector_members = sector_members.sort_values("12-week relative return vs Nifty 50 (%)", ascending=False)
    st.subheader("Stocks in selected rotating sectors")
    st.caption("The default filter selects Leading and Improving sectors. Stock ranking is 12-week relative return vs Nifty 50 within those sectors; it is a research filter, not a buy signal.")
    st.dataframe(sector_members[["Sector" if "Sector" in sector_members else "sector", "Stock", "Latest close", "12-week relative return vs Nifty 50 (%)"]], use_container_width=True, hide_index=True)
else:
    st.info("No sectors match the selected RRG quadrants. Select another quadrant or review the constituent sector data.")

st.info("RRG quadrants describe relative performance and momentum, not future returns. Sector baskets use the current uploaded/index constituents, equal-weighted weekly returns, and may have survivorship bias. This page does not alter or validate the trading backtest.")
