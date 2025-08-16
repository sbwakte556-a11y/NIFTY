# app.py — Streamlit dashboard (NSE-only CSV snapshots)
import glob
import os
import io
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import DATA_DIR, NEAR_STRIKES
from utils import enrich_with_prev, select_near_atm, compute_crossover

st.set_page_config(page_title="NIFTY Options — NSE Dashboard", layout="wide", page_icon="📊")

@st.cache_data
def list_csvs():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "snap_*.csv")))
    return files

def load_latest_and_prev():
    files = list_csvs()
    if not files:
        return pd.DataFrame(), None
    latest = files[-1]
    prev = files[-2] if len(files) > 1 else None
    df_latest = pd.read_csv(latest)
    df_prev = pd.read_csv(prev) if prev else None
    return df_latest, df_prev

def download_excel(df_dict, filename="nifty_options_snapshot.xlsx"):
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for name, df in df_dict.items():
                if df is None: 
                    df = pd.DataFrame()
                df.to_excel(writer, sheet_name=name[:31], index=False)
        data = output.getvalue()
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">Download Excel snapshot</a>'
    st.markdown(href, unsafe_allow_html=True)

st.title("📊 NIFTY Options Dashboard — NSE (No Zerodha)")
st.caption("Reads CSV snapshots created by the auto-downloader every 3 minutes.")

# Auto-refresh every 3 minutes (180000 ms) to pick up new CSVs
st_autorefresh(interval=180000, key="auto_refresh")

df, df_prev = load_latest_and_prev()
if df.empty:
    st.warning("No CSV snapshots found in ./data yet. Start the fetcher (fetch_loop.py) and wait for the first file.")
    st.stop()

# Enrich with previous snapshot for 1-interval changes & buildup
df_en = enrich_with_prev(df, df_prev)

# Header metrics
spot = float(df_en["spot"].dropna().iloc[0] if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0)
ts = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else None

c1, c2 = st.columns([1,1])
with c1:
    st.metric("Spot (approx)", f"{spot:.2f}")
with c2:
    st.metric("Snapshot time", str(ts))

# Near-ATM table
near = select_near_atm(df_en, spot)
st.subheader(f"Strikes around ATM (±{NEAR_STRIKES})")
show_cols = ["symbol","strike","option_type","ltp","oi","iv","oi_chg_pct","ltp_chg_pct","buildup"]
st.dataframe(near.sort_values(["strike","option_type"])[show_cols], use_container_width=True, hide_index=True)

# Crossover view
cross = compute_crossover(near)
st.subheader("CE vs PE Crossover (near ATM)")
st.dataframe(cross.sort_values("strike"), use_container_width=True, hide_index=True)

# Mini bar charts by strike
st.subheader("Mini charts (CE vs PE by strike)")
for k, g in near.groupby("strike"):
    fig = go.Figure()
    ce = g[g.option_type=="CE"]["ltp"].values
    pe = g[g.option_type=="PE"]["ltp"].values
    if ce.size:
        fig.add_trace(go.Bar(name=f"CE {int(k)}", x=["CE"], y=[float(ce[0])]))
    if pe.size:
        fig.add_trace(go.Bar(name=f"PE {int(k)}", x=["PE"], y=[float(pe[0])]))
    fig.update_layout(barmode="group", height=250, margin=dict(l=20,r=20,t=35,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Top movers (based on last-interval % LTP change)
if "ltp_chg_pct" in df_en.columns:
    st.subheader("Top 10 rising options (last interval %)")
    rises = df_en.assign(pct=df_en["ltp_chg_pct"]).sort_values("pct", ascending=False).head(10)
    st.dataframe(rises[["symbol","strike","option_type","ltp_chg_pct","oi_chg_pct","volume"]], use_container_width=True, hide_index=True)

# Export
download_excel({"NearATM": near, "Crossover": cross, "FullSnapshot": df_en})
