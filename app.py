import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, os
from datetime import datetime, time as dtime, timezone, timedelta

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("IDX PRO Scanner — FINAL", layout="wide")
st.title("📈 IDX PRO Scanner — Yahoo Finance (ALL IDX STOCKS)")

# =====================================================
# PATH
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "Stock List - Main - 20260122.xlsx")
SIGNAL_FILE = os.path.join(BASE_DIR, "signal_history.csv")
TRADE_FILE = os.path.join(BASE_DIR, "trade_results.csv")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# CONFIG
# =====================================================
ENTRY_INTERVAL = "1h"
DAILY_INTERVAL = "1d"
LOOKBACK_1H = "6mo"
LOOKBACK_1D = "3y"

MIN_VOLUME = 1_000_000

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01

TP1_R = 0.8
TP2_R = 2.0
MIN_RISK_PCT = 0.01

RETEST_TOL = 0.005
TP_EXTEND = 0.9

# =====================================================
# INIT FILES
# =====================================================
def init_files():
    if not os.path.exists(SIGNAL_FILE):
        pd.DataFrame(columns=[
            "Time","Symbol","Phase","Score","Rating",
            "Entry","SL","TP1","TP2",
            "Status","Label"
        ]).to_csv(SIGNAL_FILE,index=False)

    if not os.path.exists(TRADE_FILE):
        pd.DataFrame(columns=["Time","Symbol","R"]).to_csv(TRADE_FILE,index=False)

init_files()

# =====================================================
# MARKET HOURS IDX
# =====================================================
def is_market_open():
    now = datetime.now(WIB)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (
        dtime(9,0) <= t <= dtime(11,30)
        or dtime(13,30) <= t <= dtime(15,50)
    )

# =====================================================
# LOAD IDX SYMBOLS FROM EXCEL
# =====================================================
@st.cache_data(ttl=3600)
def load_idx_symbols_from_excel(path):
    df = pd.read_excel(path)

    possible_cols = ["Kode Saham","Kode","Ticker","Symbol"]
    col = next(c for c in possible_cols if c in df.columns)

    symbols = (
        df[col]
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )

    return [s + ".JK" for s in symbols if s.isalnum()]

# =====================================================
# FILTER BY VOLUME
# =====================================================
@st.cache_data(ttl=1800)
def filter_by_volume(symbols, min_volume):
    liquid = []
    for s in symbols:
        try:
            df = yf.download(s, period="5d", interval="1d", progress=False)
            if df.empty:
                continue
            vol = df["Volume"].iloc[-1]
            if vol >= min_volume:
                liquid.append(s)
            time.sleep(0.15)
        except:
            continue
    return liquid

# =====================================================
# YAHOO DATA (SAFE)
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        group_by="column",
        progress=False
    )
    if df.empty:
        raise RuntimeError("No data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df[["open","high","low","close","volume"]].dropna()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna()

# =====================================================
# INDICATORS (NUMPY SAFE)
# =====================================================
def supertrend(df, period, mult):
    h = df.high.values
    l = df.low.values
    c = df.close.values

    tr = np.maximum.reduce([
        h-l, np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))
    ])
    tr[0] = h[0]-l[0]

    atr = pd.Series(tr).ewm(span=period,adjust=False).mean().values
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = np.zeros(len(df))
    trend = np.ones(len(df))
    stl[0] = lower[0]

    for i in range(1,len(df)):
        if trend[i-1] == 1:
            stl[i] = max(lower[i], stl[i-1])
            trend[i] = 1 if c[i] > stl[i] else -1
        else:
            stl[i] = min(upper[i], stl[i-1])
            trend[i] = -1 if c[i] < stl[i] else 1

    return pd.Series(stl, df.index), pd.Series(trend, df.index)

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    mfm = ((c-l)-(h-c))/(h-l)
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df,lb):
    lv=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i] == min(df.low.iloc[i-lb:i+lb+1]):
            lv.append(df.low.iloc[i])
    clean=[]
    for s in sorted(lv):
        if not clean or abs(s-clean[-1])/clean[-1] > 0.02:
            clean.append(s)
    return clean

# =====================================================
# LOGIC
# =====================================================
def calculate_score(df1h, df1d):
    score = 0
    ema20 = df1h.close.ewm(span=20).mean()
    ema50 = df1h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()

    p = df1h.close.iloc[-1]

    if p > ema20.iloc[-1]: score+=1
    if ema20.iloc[-1] > ema50.iloc[-1]: score+=1
    if ema50.iloc[-1] > ema200.iloc[-1]: score+=1
    if p > ema200.iloc[-1]: score+=1

    vo = volume_osc(df1h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 5: score+=1
    if vo > 10: score+=1
    if vo > 20: score+=1

    adl = accumulation_distribution(df1h)
    if adl.iloc[-1] > adl.iloc[-5]: score+=1
    if adl.iloc[-1] > adl.iloc[-10]: score+=1
    if adl.iloc[-1] > adl.iloc[-20]: score+=1

    return score

def trade_levels(df1d):
    entry = df1d.close.iloc[-1]
    sup = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not sup:
        return None
    sl = max(sup) * (1-ZONE_BUFFER)
    risk = entry - sl
    if risk < entry * MIN_RISK_PCT:
        return None
    return {
        "Entry": round(entry,2),
        "SL": round(sl,2),
        "TP1": round(entry + risk*TP1_R,2),
        "TP2": round(entry + risk*TP2_R,2)
    }

def auto_label(sig, price):
    if not is_market_open(): return "WAIT"
    if price <= sig["SL"]: return "INVALID"
    if price >= sig["TP1"] * TP_EXTEND: return "EXTENDED"
    if abs(price - sig["Entry"]) / sig["Entry"] <= RETEST_TOL:
        return "RETEST"
    if price > sig["Entry"]: return "HOLD"
    return ""

# =====================================================
# HISTORY & TRADE RESULT
# =====================================================
def save_signal(sig):
    df = pd.read_csv(SIGNAL_FILE)
    if ((df.Symbol==sig["Symbol"]) & (df.Status=="OPEN")).any():
        return
    df = pd.concat([df, pd.DataFrame([sig])], ignore_index=True)
    df.to_csv(SIGNAL_FILE, index=False)

def update_trade_outcome():
    df = pd.read_csv(SIGNAL_FILE)
    res = pd.read_csv(TRADE_FILE)

    for i,r in df.iterrows():
        if r.Status != "OPEN":
            continue
        try:
            price = fetch_ohlcv(r.Symbol, ENTRY_INTERVAL, "5d").close.iloc[-1]
        except:
            continue

        R=None; status=None
        if price <= r.SL:
            R=-1; status="SL HIT"
        elif price >= r.TP2:
            R=TP2_R; status="TP2 HIT"
        elif price >= r.TP1:
            R=TP1_R; status="TP1 HIT"

        if R is not None:
            df.at[i,"Status"] = status
            res = pd.concat([res, pd.DataFrame([{
                "Time": now_wib(),
                "Symbol": r.Symbol,
                "R": R
            }])], ignore_index=True)

    df.to_csv(SIGNAL_FILE,index=False)
    res.to_csv(TRADE_FILE,index=False)

# =====================================================
# MONTE CARLO
# =====================================================
def monte_carlo(r, initial, risk, trades, sims):
    curves=[]
    for _ in range(sims):
        bal=initial
        eq=[bal]
        for _ in range(trades):
            bal += bal * risk
            bal += bal * risk * np.random.choice(r)
            eq.append(bal)
        curves.append(eq)
    return np.array(curves)

# =====================================================
# UI
# =====================================================
ALL_SYMBOLS = load_idx_symbols_from_excel(EXCEL_PATH)
IDX_SYMBOLS = filter_by_volume(ALL_SYMBOLS, MIN_VOLUME)

st.caption(f"📊 Saham IDX likuid: {len(IDX_SYMBOLS)} / {len(ALL_SYMBOLS)}")

update_trade_outcome()

tab1, tab2, tab3 = st.tabs(["📡 Scanner","📜 Riwayat","🎲 Monte Carlo"])

with tab1:
    if st.button("🔍 Scan Seluruh Saham IDX"):
        found=[]
        for s in IDX_SYMBOLS:
            try:
                df1h = fetch_ohlcv(s, ENTRY_INTERVAL, LOOKBACK_1H)
                df1d = fetch_ohlcv(s, DAILY_INTERVAL, LOOKBACK_1D)

                stl, trend = supertrend(df1h, ATR_PERIOD, MULTIPLIER)
                if trend.iloc[-1] != 1:
                    continue

                score = calculate_score(df1h, df1d)
                if score < 6:
                    continue

                trade = trade_levels(df1d)
                if not trade:
                    continue

                price = df1h.close.iloc[-1]
                label = auto_label(trade, price)

                sig = {
                    "Time": now_wib(),
                    "Symbol": s,
                    "Phase": "AKUMULASI_KUAT",
                    "Score": score,
                    "Rating": "⭐"*score,
                    **trade,
                    "Status": "OPEN",
                    "Label": label
                }

                save_signal(sig)
                found.append(sig)

                time.sleep(0.2)

            except:
                continue

        st.success(f"🔥 {len(found)} SIGNAL AKUMULASI_KUAT")

with tab2:
    st.dataframe(pd.read_csv(SIGNAL_FILE), use_container_width=True)

with tab3:
    trades = pd.read_csv(TRADE_FILE)
    if len(trades) < 10:
        st.warning("Trade belum cukup untuk Monte Carlo (min 10).")
    else:
        r = trades.R.values
        risk = st.slider("Risk / Trade (%)",0.5,3.0,1.0)/100
        t = st.slider("Jumlah Trade",50,300,150)

        if st.button("🎲 Run Monte Carlo"):
            curves = monte_carlo(r,100_000_000,risk,t,500)
            st.metric("Median Balance",f"Rp {np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin",f"{(curves[:,-1]<50_000_000).mean()*100:.2f}%")
