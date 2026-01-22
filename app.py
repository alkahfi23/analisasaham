import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("IDX PRO Scanner — FINAL STABLE", layout="wide")
st.title("📈 IDX PRO Scanner — Yahoo Finance (FINAL STABLE)")

DEBUG = st.sidebar.toggle("🧪 Debug Mode", value=True)

# =====================================================
# FILES
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_FILE = os.path.join(BASE_DIR, "signal_history.csv")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))
def now_wib():
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# CONFIG (IDX REALISTIC)
# =====================================================
ENTRY_INTERVAL = "1h"
DAILY_INTERVAL = "1d"
LOOKBACK_1H = "6mo"
LOOKBACK_1D = "3y"

MIN_AVG_VOLUME = 300_000
MIN_SCORE = 6

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01
MIN_RISK_PCT = 0.01

# =====================================================
# INIT CSV
# =====================================================
if not os.path.exists(SIGNAL_FILE):
    pd.DataFrame(columns=[
        "Time","Symbol","Phase","Score","Rating",
        "Entry","SL","TP1","TP2","Label"
    ]).to_csv(SIGNAL_FILE, index=False)

# =====================================================
# SIDEBAR — EXCEL
# =====================================================
st.sidebar.header("📂 Master Saham IDX")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (1 kolom kode saham IDX)",
    type=["xlsx"]
)

# =====================================================
# LOAD SYMBOLS
# =====================================================
@st.cache_data(ttl=3600)
def load_idx_symbols_from_excel(file):
    df = pd.read_excel(file)
    col = df.columns[0]
    symbols = (
        df[col]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .unique()
        .tolist()
    )
    return [s + ".JK" for s in symbols if len(s) >= 3]

# =====================================================
# FILTER BY VOLUME
# =====================================================
@st.cache_data(ttl=1800)
def filter_by_volume(symbols, min_volume):
    liquid = []
    debug = []

    for s in symbols:
        try:
            df = yf.download(s, period="10d", interval="1d", progress=False)

            if df.empty or "Volume" not in df.columns:
                debug.append({"Symbol": s, "Reason": "No volume data"})
                continue

            avg_vol = float(df["Volume"].dropna().tail(5).mean())

            if np.isnan(avg_vol):
                debug.append({"Symbol": s, "Reason": "Volume NaN"})
                continue

            if avg_vol >= min_volume:
                liquid.append(s)
            else:
                debug.append({
                    "Symbol": s,
                    "Reason": f"Low volume ({int(avg_vol):,})"
                })

            time.sleep(0.05)

        except Exception as e:
            debug.append({"Symbol": s, "Reason": str(e)})

    return liquid, pd.DataFrame(debug)

# =====================================================
# FETCH OHLCV
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No OHLC data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    return df[["open","high","low","close","volume"]].astype(float).dropna()

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high.values, df.low.values, df.close.values
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - np.roll(c, 1)),
        np.abs(l - np.roll(c, 1))
    ])
    tr[0] = h[0] - l[0]

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    hl2 = (h + l) / 2

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    trend = 1
    st_line = lower[0]

    for i in range(1, len(c)):
        if trend == 1:
            st_line = max(lower[i], st_line)
            if c[i] < st_line:
                trend = -1
        else:
            st_line = min(upper[i], st_line)
            if c[i] > st_line:
                trend = 1

    return trend

def volume_osc(v, f, s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high, df.low, df.close, df.volume
    denom = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / denom
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    return (mfm * v).cumsum()

def find_support(df, lb):
    lows = df["low"].values.astype(float)
    supports = []
    for i in range(lb, len(lows) - lb):
        if lows[i] == min(lows[i-lb:i+lb+1]):
            supports.append(lows[i])
    return supports

# =====================================================
# SCORE & TRADE
# =====================================================
def calculate_score(df1h, df1d):
    score = 0
    ema20 = df1h.close.ewm(span=20).mean()
    ema50 = df1h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()
    p = df1h.close.iloc[-1]

    if p > ema20.iloc[-1]: score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]: score += 1
    if ema50.iloc[-1] > ema200.iloc[-1]: score += 1
    if p > ema200.iloc[-1]: score += 1

    vo = volume_osc(df1h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 5: score += 1
    if vo > 10: score += 1
    if vo > 20: score += 1

    adl = accumulation_distribution(df1h)
    if adl.iloc[-1] > adl.iloc[-5]: score += 1
    if adl.iloc[-1] > adl.iloc[-10]: score += 1
    if adl.iloc[-1] > adl.iloc[-20]: score += 1

    return score

def trade_levels(df1d):
    entry = float(df1d.close.iloc[-1])
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None
    sl = max(supports) * (1 - ZONE_BUFFER)
    if entry - sl < entry * MIN_RISK_PCT:
        return None
    return entry, sl

# =====================================================
# AUTO LABEL (BARU)
# =====================================================
def auto_label(close_price, entry, sl):
    risk = entry - sl
    if risk <= 0:
        return "NO TRADE"

    dist_pct = abs(close_price - entry) / entry

    if dist_pct <= 0.005:
        return "ENTRY NOW"

    if close_price > entry:
        return "WAIT PULLBACK"

    return "NO TRADE"

# =====================================================
# MAIN FLOW
# =====================================================
if not uploaded_file:
    st.warning("⬅️ Upload Excel kode saham IDX")
    st.stop()

ALL_SYMBOLS = load_idx_symbols_from_excel(uploaded_file)
IDX_SYMBOLS, VOL_DEBUG = filter_by_volume(ALL_SYMBOLS, MIN_AVG_VOLUME)

st.caption(f"📊 Saham likuid: {len(IDX_SYMBOLS)} / {len(ALL_SYMBOLS)}")

# =====================================================
# SCANNER
# =====================================================
if st.button("🔍 Scan Saham IDX"):
    found = []

    for s in IDX_SYMBOLS:
        try:
            df1h = fetch_ohlcv(s, ENTRY_INTERVAL, LOOKBACK_1H)
            df1d = fetch_ohlcv(s, DAILY_INTERVAL, LOOKBACK_1D)

            if supertrend(df1h, ATR_PERIOD, MULTIPLIER) != 1:
                continue

            score = calculate_score(df1h, df1d)
            if score < MIN_SCORE:
                continue

            trade = trade_levels(df1d)
            if trade is None:
                continue

            entry, sl = trade
            close_price = df1d.close.iloc[-1]
            label = auto_label(close_price, entry, sl)

            found.append({
                "Time": now_wib(),
                "Symbol": s,
                "Phase": "AKUMULASI_KUAT",
                "Score": score,
                "Rating": "⭐" * score,
                "Entry": round(entry, 2),
                "SL": round(sl, 2),
                "TP1": round(entry + (entry - sl) * 0.8, 2),
                "TP2": round(entry + (entry - sl) * 2.0, 2),
                "Label": label
            })

        except Exception as e:
            if DEBUG:
                st.write(f"{s} ❌ {e}")

    if found:
        df = pd.DataFrame(found).sort_values("Score", ascending=False)
        st.success(f"🔥 {len(df)} SIGNAL AKUMULASI_KUAT")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("🔥 0 SIGNAL AKUMULASI_KUAT")
