# streamlit_app.py
"""
NIFTY Live Option Chain Dashboard (Streamlit)
- Fetches NSE option-chain for NIFTY directly
- Computes OI/LTP changes vs previous snapshot kept in memory (per session)
- Shows near-ATM strikes, CE/PE crossover, mini charts, top movers
- Auto-refresh via streamlit_autorefresh
Notes:
- If NSE returns 403/blocks, the app will show a message. Wait and try again.
- Adjust refresh interval from the sidebar.
"""

import time
import io
import math
from typing import Optional
import requests
import pandas as pd
import numpy as np
import datetime as dt
import pytz

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# ------------------------
# CONFIG / DEFAULTS
# ------------------------
st.set_page_config(layout="wide", page_title="NIFTY Options Live", page_icon="📈")

SYMBOL = "NIFTY"
TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(TIMEZONE)
DEFAULT_REFRESH_SECS = 180  # 3 minutes
DEFAULT_NEAR_STRIKES = 3
STRIKE_STEP = 50

# ------------------------
# UTILITIES
# ------------------------
def make_session():
    s = requests.Session()
    s.headers.update({
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "accept-language": "en-US,en;q=0.9",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "referer": "https://www.nseindia.com/",
        "connection": "keep-alive",
    })
    # warm up cookies
    try:
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:
        pass
    return s

def fetch_option_chain(symbol: str = SYMBOL, tries: int = 6, backoff: float = 1.5) -> pd.DataFrame:
    """
    Fetch option chain JSON from NSE and convert to a normalized DataFrame
    Returns rows for both CE and PE with columns:
    symbol, strike, option_type (CE/PE), ltp, oi, volume, iv, ts, spot
    """
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = make_session()
    headers = {"accept": "application/json, text/plain, */*", "referer": "https://www.nseindia.com/option-chain"}
    for attempt in range(tries):
        try:
            r = s.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                data = r.json()
                break
            else:
                # sometimes returns 403 or 401; wait and retry
                time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    else:
        # all attempts failed
        return pd.DataFrame()

    records = data.get("records", {})
    rows = []
    ts = dt.datetime.now(IST)
    spot = records.get("underlyingValue") or 0.0
    for item in records.get("data", []):
        strike = item.get("strikePrice")
        if strike is None:
            continue
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not isinstance(leg, dict):
                continue
            rows.append({
                "symbol": symbol,
                "strike": int(strike),
                "option_type": side,
                "ltp": float(leg.get("lastPrice") or 0.0),
                "oi": float(leg.get("openInterest") or 0.0),
                "volume": float(leg.get("totalTradedVolume") or 0.0),
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "vwap": np.nan,  # not provided by endpoint
                "ts": ts,
                "spot": float(spot or 0.0),
            })
    df = pd.DataFrame(rows)
    return df

def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)

def classify_buildup(oi_change: float, ltp_change: float) -> str:
    if oi_change > 0 and ltp_change > 0:
        return "Long Buildup"
    if oi_change > 0 and ltp_change < 0:
        return "Short Buildup"
    if oi_change < 0 and ltp_change < 0:
        return "Long Unwinding"
    if oi_change < 0 and ltp_change > 0:
        return "Short Covering"
    return "Neutral"

def enrich_with_prev(curr: pd.DataFrame, prev: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curr.empty:
        return curr
    df = curr.copy()
    if prev is None or prev.empty:
        df["prev_ltp"] = df["ltp"]
        df["prev_oi"] = df["oi"]
    else:
        m = prev[["symbol","strike","option_type","ltp","oi"]].rename(columns={"ltp":"prev_ltp","oi":"prev_oi"})
        df = df.merge(m, on=["symbol","strike","option_type"], how="left")
        df["prev_ltp"] = df["prev_ltp"].fillna(df["ltp"])
        df["prev_oi"] = df["prev_oi"].fillna(df["oi"])
    df["oi_chg"] = df["oi"] - df["prev_oi"]
    df["ltp_chg"] = df["ltp"] - df["prev_ltp"]
    df["oi_chg_pct"] = np.where(df["prev_oi"]>0, 100*df["oi_chg"]/df["prev_oi"], 0.0)
    df["ltp_chg_pct"] = np.where(df["prev_ltp"]>0, 100*df["ltp_chg"]/df["prev_ltp"], 0.0)
    df["buildup"] = [classify_buildup(o, p) for o, p in zip(df["oi_chg"], df["ltp_chg"])]
    df["above_vwap"] = df["ltp"] > df["vwap"]
    return df

def select_near_atm(df: pd.DataFrame, spot: float, n: int = DEFAULT_NEAR_STRIKES) -> pd.DataFrame:
    if df.empty:
        return df
    atm = nearest_strike(spot)
    lo, hi = atm - n*STRIKE_STEP, atm + n*STRIKE_STEP
    return df[(df["strike"]>=lo) & (df["strike"]<=hi)].copy()

def compute_crossover(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame(out)
    for (symbol, strike), g in df.groupby(["symbol","strike"]):
        ce = g[g.option_type=="CE"]["ltp"].values
        pe = g[g.option_type=="PE"]["ltp"].values
        if ce.size and pe.size:
            ce_val, pe_val = float(ce[0]), float(pe[0])
            out.append({
                "symbol": symbol,
                "strike": int(strike),
                "ce_gt_pe": bool(ce_val > pe_val),
                "pe_gt_ce": bool(pe_val > ce_val),
                "diff_pct": float((ce_val - pe_val) / max(1e-6, pe_val) * 100),
            })
    return pd.DataFrame(out)

# ------------------------
# UI / Sidebar settings
# ------------------------
st.sidebar.title("Settings")
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=DEFAULT_REFRESH_SECS, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM (±)", 1, 6, DEFAULT_NEAR_STRIKES)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold (alert)", 10, 500, 50)
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=False)
if enable_telegram:
    tg_token = st.sidebar.text_input("Telegram Bot Token")
    tg_chat = st.sidebar.text_input("Telegram Chat ID")
else:
    tg_token = tg_chat = ""

st.sidebar.markdown("---")
st.sidebar.markdown("Repo: `sbwakte556-a11y/NIFTY`")
st.sidebar.markdown("Live fetch from NSE. If NSE blocks, try again later.")

# client-side auto refresh mechanism
# We trigger a frontend auto-refresh; streaming server will re-run script and fetch again.
st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh_nifty")

# ------------------------
# Fetch data
# ------------------------
st.title("📈 NIFTY Option Chain — Live (NSE)")
status_placeholder = st.empty()

# Use session_state to store previous snapshot across runs (per user session)
if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None

try:
    status_placeholder.info("Fetching option-chain from NSE...")
    curr = fetch_option_chain(SYMBOL)
    if curr.empty:
        status_placeholder.error("Failed to fetch data from NSE (empty response). Try again in a minute.")
        st.stop()
    status_placeholder.success("Data fetched successfully.")
except Exception as e:
    status_placeholder.error(f"Fetch failed: {e}")
    st.stop()

# enrich with previous snapshot kept in session_state
prev = st.session_state.prev_snapshot
df_en = enrich_with_prev(curr, prev)

# update prev for next run
st.session_state.prev_snapshot = curr[["symbol","strike","option_type","ltp","oi"]].copy()

# header metrics
spot = float(df_en["spot"].dropna().iloc[0] if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0)
snapshot_time = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else dt.datetime.now(IST)
col1, col2, col3, col4 = st.columns([1.2,1,1,1])
with col1:
    st.metric("Spot (approx)", f"{spot:.2f}")
with col2:
    st.metric("Snapshot time", f"{snapshot_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
with col3:
    # buyer/seller strength proxy
    ce = df_en[df_en.option_type == "CE"].copy()
    pe = df_en[df_en.option_type == "PE"].copy()
    ce["score"] = ce["volume"].fillna(0) * ce["ltp_chg"].abs().fillna(0)
    pe["score"] = pe["volume"].fillna(0) * pe["ltp_chg"].abs().fillna(0)
    ce_strength = float(ce["score"].sum())
    pe_strength = float(pe["score"].sum())
    tot = (ce_strength + pe_strength) or 1
    st.metric("Buyer % (CE proxy)", f"{100*ce_strength/tot:.1f}%")
with col4:
    up_score = ((df_en["buildup"] == "Long Buildup").sum() + (df_en["buildup"] == "Short Covering").sum())
    dn_score = ((df_en["buildup"] == "Short Buildup").sum() + (df_en["buildup"] == "Long Unwinding").sum())
    st.metric("Directional Bias", f"Up:{up_score} Down:{dn_score}")

st.markdown("---")

# Near ATM table and crossover
near = select_near_atm(df_en, spot, n=near_strikes)
st.subheader(f"Strikes around ATM (±{near_strikes * STRIKE_STEP} points) — total {len(near)//2} strikes")
if near.empty:
    st.warning("No near-ATM data available.")
else:
    show_cols = ["symbol","strike","option_type","ltp","iv","oi","oi_chg_pct","ltp_chg_pct","buildup"]
    st.dataframe(near.sort_values(["strike","option_type"])[show_cols], use_container_width=True)

    cross = compute_crossover(near)
    st.subheader("CE vs PE Crossover (near ATM)")
    st.dataframe(cross.sort_values("strike"), use_container_width=True)

    # Exceptional OI changes
    exc = near[near["oi_chg_pct"].abs() >= oi_alert_pct]
    if not exc.empty:
        st.warning(f"⚡ Exceptional OI changes (|OI%| >= {oi_alert_pct})")
        st.dataframe(exc[["strike","option_type","oi_chg_pct","ltp_chg_pct","buildup"]].sort_values("oi_chg_pct", ascending=False), use_container_width=True)
        if enable_telegram and tg_token and tg_chat:
            # send a simple alert (best-effort; avoid spamming)
            try:
                txt = [f"⚡ Exceptional OI changes at {snapshot_time.strftime('%H:%M:%S')}"]
                for _, r in exc.iterrows():
                    txt.append(f"{r.strike} {r.option_type}: OI% {r.oi_chg_pct:.1f} LTP% {r.ltp_chg_pct:.1f} {r.buildup}")
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={"chat_id": tg_chat, "text": "\n".join(txt)})
            except Exception:
                pass

# mini charts for each strike
if not near.empty:
    st.subheader("Mini Charts (CE vs PE LTP by strike)")
    # group by strike and show bars
    strikes_sorted = sorted(near["strike"].unique())
    for strike in strikes_sorted:
        g = near[near.strike == strike]
        ce = g[g.option_type == "CE"]
        pe = g[g.option_type == "PE"]
        fig = go.Figure()
        if not ce.empty:
            fig.add_trace(go.Bar(name=f"CE {int(strike)}", x=["CE"], y=[float(ce.ltp.iloc[0])]))
        if not pe.empty:
            fig.add_trace(go.Bar(name=f"PE {int(strike)}", x=["PE"], y=[float(pe.ltp.iloc[0])]))
        fig.update_layout(barmode="group", height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

# Top movers
st.subheader("Top 10 Rising Options (last interval % LTP)")
if "ltp_chg_pct" in df_en.columns:
    top = df_en.assign(pct=df_en["ltp_chg_pct"]).sort_values("pct", ascending=False).head(10)
    st.dataframe(top[["symbol","strike","option_type","ltp","ltp_chg_pct","oi_chg_pct","volume"]], use_container_width=True)
else:
    st.info("No LTP change data available yet (first snapshot).")

# Export snapshot to Excel
def to_excel(dframe: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dframe.to_excel(writer, index=False, sheet_name="snapshot")
        writer.save()
    return output.getvalue()

st.markdown("---")
col_a, col_b = st.columns([1,2])
with col_a:
    st.download_button("Download current snapshot (Excel)", data=to_excel(df_en) if not df_en.empty else b"", file_name=f"nifty_snapshot_{snapshot_time.strftime('%Y%m%d_%H%M%S')}.xlsx")
with col_b:
    st.info("App auto-refreshes every {} seconds. You can change the interval from Settings.".format(refresh_secs))

# Footer / helpful message
st.markdown("----")
st.caption("If NSE blocks requests (403), wait a few minutes. NSE sometimes blocks automated requests — reducing frequency or using a proxy/IP rotate helps.")
