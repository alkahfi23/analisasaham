import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("IDX PRO Scanner — REFACTORED", layout="wide")
st.title("📈 IDX PRO Scanner — Yahoo Finance (REFACTORED)")

DEBUG = st.sidebar.toggle("🧪 Debug Mode", value=False)

# =====================================================
# PATH & FILE
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
def load_idx_symbols(file):
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
# LIQUIDITY FILTER
# =====================================================
@st.cache_data(ttl=1800)
def filter_by_volume(symbols, min_volume):
    liquid, debug = [], []

    for s in symbols:
        try:
            df = yf.download(s, period="10d", interval="1d", progress=False)
            if df.empty or "Volume" not in df.columns:
                continue

            avg_vol = df["Volume"].tail(5).mean()
            if avg_vol >= min_volume:
                liquid.append(s)
            else:
                debug.append({"Symbol": s, "AvgVolume": int(avg_vol)})

            time.sleep(0.05)

        except Exception as e:
            debug.append({"Symbol": s, "Error": str(e)})

    return liquid, pd.DataFrame(debug)

# =====================================================
# FETCH OHLCV
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No OHLC data")

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].astype(float).dropna()

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high, df.low, df.close
    tr = np.maximum.reduce([
        h - l,
        abs(h - c.shift(1)),
        abs(l - c.shift(1))
    ])

    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h + l) / 2

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    trend = [1]
    st_line = [lower.iloc[0]]

    for i in range(1, len(c)):
        if trend[-1] == 1:
            st_line.append(max(lower.iloc[i], st_line[-1]))
            trend.append(-1 if c.iloc[i] < st_line[-1] else 1)
        else:
            st_line.append(min(upper.iloc[i], st_line[-1]))
            trend.append(1 if c.iloc[i] > st_line[-1] else -1)

    return trend[-1]

def volume_osc(volume, fast, slow):
    fast_ma = volume.ewm(span=fast).mean()
    slow_ma = volume.ewm(span=slow).mean().replace(0, np.nan)
    return ((fast_ma - slow_ma) / slow_ma * 100).fillna(0)

def accumulation_distribution(df):
    h,l,c,v = df.high, df.low, df.close, df.volume
    denom = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / denom
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    return (mfm * v).cumsum()

def find_support(df, lb):
    lows = df.low.values
    supports = []
    for i in range(lb, len(lows)-lb):
        zone = lows[i-lb:i+lb+1]
        if lows[i] <= min(zone) * 1.001:
            supports.append(lows[i])
    return sorted(set(supports))

# =====================================================
# SCORE SYSTEM
# =====================================================
def calculate_score(df1h, df1d):
    score = 0

    ema20 = df1h.close.ewm(span=20).mean()
    ema50 = df1h.close.ewm(span=50).mean()
    ema200 = df1h.close.ewm(span=200).mean()

    price = df1h.close.iloc[-1]

    if price > ema20.iloc[-1]: score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]: score += 1
    if ema50.iloc[-1] > ema200.iloc[-1]: score += 1
    if price > ema200.iloc[-1]: score += 1

    vo = volume_osc(df1h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 5: score += 1
    if vo > 10: score += 1
    if vo > 20: score += 1

    adl = accumulation_distribution(df1h)
    if adl.iloc[-1] > adl.iloc[-5]: score += 1
    if adl.iloc[-1] > adl.iloc[-10]: score += 1
    if adl.iloc[-1] > adl.iloc[-20]: score += 1

    return score

# =====================================================
# TRADE LEVELS
# =====================================================
def trade_levels(df1d):
    entry = float(df1d.close.iloc[-1])
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]

    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    if entry - sl < entry * MIN_RISK_PCT:
        return None

    return entry, sl

def auto_label(close_price, entry):
    dist = abs(close_price - entry) / entry

    if dist <= 0.005:
        return "ENTRY NOW"
    if close_price > entry and dist < 0.02:
        return "BREAKOUT BUY"
    if close_price > entry:
        return "WAIT PULLBACK"

    return "NO TRADE"

# =====================================================
# MAIN
# =====================================================
if not uploaded_file:
    st.warning("⬅️ Upload Excel kode saham IDX")
    st.stop()

ALL_SYMBOLS = load_idx_symbols(uploaded_file)
IDX_SYMBOLS, _ = filter_by_volume(ALL_SYMBOLS, MIN_AVG_VOLUME)

st.caption(f"📊 Saham likuid: {len(IDX_SYMBOLS)} / {len(ALL_SYMBOLS)}")

# =====================================================
# SCANNER
# =====================================================
if st.button("🔍 Scan Saham IDX"):
    results = []

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
            if not trade:
                continue

            entry, sl = trade
            label = auto_label(df1d.close.iloc[-1], entry)

            results.append({
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

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.success(f"🔥 {len(df)} SIGNAL AKUMULASI_KUAT")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("🔥 0 SIGNAL AKUMULASI_KUAT")
