import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os, math
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("IDX PRO Scanner — FINAL v2", layout="wide")
st.title("📈 IDX PRO Scanner — FINAL v2 (IHSG + POSITION SIZING)")

DEBUG = st.sidebar.toggle("🧪 Debug Mode", value=False)

# =====================================================
# ACCOUNT SETTINGS (NEW)
# =====================================================
st.sidebar.header("💼 Account Settings")
ACCOUNT_EQUITY = st.sidebar.number_input(
    "Account Equity (Rp)", value=100_000_000, step=5_000_000
)
RISK_PCT = st.sidebar.slider(
    "Risk per Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25
)

# =====================================================
# TIME & FILE
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_FILE = os.path.join(BASE_DIR, "signal_history.csv")

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

MIN_AVG_VOLUME = 300_000
BASE_MIN_SCORE = 6

ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01
MIN_RISK_PCT = 0.01

ADX_PERIOD = 14
ADX_TREND_MIN = 20

IHSG_SYMBOL = "^JKSE"

# =====================================================
# INIT CSV
# =====================================================
if not os.path.exists(SIGNAL_FILE):
    pd.DataFrame(columns=[
        "Time","Symbol","IHSG_Regime","Stock_Regime","Phase",
        "Score","Rating","Entry","SL","TP1","TP2",
        "Lot","Risk_Rp","Label"
    ]).to_csv(SIGNAL_FILE, index=False)

# =====================================================
# UTILITIES
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No data")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].astype(float).dropna().copy()

# =====================================================
# INDICATORS (CLEAN)
# =====================================================
def calculate_adx(df, period=14):
    df = df.copy()
    high, low, close = df.high, df.low, df.close

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

def market_regime(df):
    df = df.copy()
    ema50 = df.close.ewm(span=50, adjust=False).mean()
    ema200 = df.close.ewm(span=200, adjust=False).mean()
    adx = calculate_adx(df, ADX_PERIOD)
    price = df.close.iloc[-1]

    if price > ema200.iloc[-1] and ema50.iloc[-1] > ema200.iloc[-1] and adx.iloc[-1] >= ADX_TREND_MIN:
        return "TRENDING_BULL"
    if price < ema200.iloc[-1]:
        return "DISTRIBUTION"
    return "RANGING"

# =====================================================
# IHSG REGIME (GLOBAL)
# =====================================================
@st.cache_data(ttl=600)
def ihsg_regime():
    df = fetch_ohlcv(IHSG_SYMBOL, DAILY_INTERVAL, LOOKBACK_1D)
    return market_regime(df)

# =====================================================
# SCORE
# =====================================================
def volume_osc(v, fast, slow):
    v = pd.Series(v)
    fast_ma = v.ewm(span=fast, adjust=False).mean()
    slow_ma = v.ewm(span=slow, adjust=False).mean().replace(0, np.nan)
    return ((fast_ma - slow_ma) / slow_ma * 100).fillna(0)

def accumulation_distribution(df):
    df = df.copy()
    h,l,c,v = df.high, df.low, df.close, df.volume
    denom = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / denom
    return (mfm.fillna(0) * v).cumsum()

def calculate_score(df1h):
    df1h = df1h.copy()
    score = 0

    ema20 = df1h.close.ewm(span=20, adjust=False).mean()
    ema50 = df1h.close.ewm(span=50, adjust=False).mean()
    ema200 = df1h.close.ewm(span=200, adjust=False).mean()
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
    if adl.iloc[-1] > adl.iloc[-10]: score += 1
    if adl.iloc[-1] > adl.iloc[-20]: score += 1

    return score

# =====================================================
# TRADE & POSITION SIZING
# =====================================================
def find_support(df, lb):
    lows = df.low.values
    supports = []
    for i in range(lb, len(lows)-lb):
        if lows[i] <= min(lows[i-lb:i+lb+1]) * 1.001:
            supports.append(lows[i])
    return sorted(set(supports))

def trade_levels(df1d):
    entry = float(df1d.close.iloc[-1])
    supports = [s for s in find_support(df1d, SR_LOOKBACK) if s < entry]
    if not supports:
        return None
    sl = max(supports) * (1 - ZONE_BUFFER)
    if entry - sl < entry * MIN_RISK_PCT:
        return None
    return entry, sl

def position_size(entry, sl):
    risk_per_share = entry - sl
    if risk_per_share <= 0:
        return 0, 0

    risk_rp = ACCOUNT_EQUITY * (RISK_PCT / 100)
    shares = risk_rp / risk_per_share
    lots = math.floor(shares / 100)

    return lots, int(risk_rp)

# =====================================================
# MAIN SCAN
# =====================================================
ihsg_state = ihsg_regime()
st.subheader(f"📊 IHSG Regime: **{ihsg_state}**")

if ihsg_state == "DISTRIBUTION":
    st.error("🚫 IHSG DISTRIBUTION — NO TRADE ZONE")
    st.stop()

# =====================================================
# SYMBOL INPUT
# =====================================================
uploaded_file = st.sidebar.file_uploader("Upload Excel Kode Saham IDX", type=["xlsx"])
if not uploaded_file:
    st.stop()

symbols = pd.read_excel(uploaded_file).iloc[:,0].astype(str).str.upper().str.strip()
symbols = [s + ".JK" for s in symbols if len(s) >= 3]

# =====================================================
# SCANNER
# =====================================================
if st.button("🔍 Scan Saham IDX"):
    results = []

    for s in symbols:
        try:
            df1h = fetch_ohlcv(s, ENTRY_INTERVAL, LOOKBACK_1H)
            df1d = fetch_ohlcv(s, DAILY_INTERVAL, LOOKBACK_1D)

            stock_regime = market_regime(df1d)
            if stock_regime == "DISTRIBUTION":
                continue

            score = calculate_score(df1h)
            min_score = BASE_MIN_SCORE if stock_regime == "TRENDING_BULL" else BASE_MIN_SCORE + 1
            if score < min_score:
                continue

            trade = trade_levels(df1d)
            if not trade:
                continue

            entry, sl = trade
            lots, risk_rp = position_size(entry, sl)
            if lots <= 0:
                continue

            results.append({
                "Time": now_wib(),
                "Symbol": s,
                "IHSG_Regime": ihsg_state,
                "Stock_Regime": stock_regime,
                "Phase": "AKUMULASI_KUAT",
                "Score": score,
                "Rating": "⭐" * score,
                "Entry": round(entry,2),
                "SL": round(sl,2),
                "TP1": round(entry + (entry-sl)*0.8,2),
                "TP2": round(entry + (entry-sl)*2.0,2),
                "Lot": lots,
                "Risk_Rp": risk_rp,
                "Label": "ENTRY NOW"
            })

        except Exception as e:
            if DEBUG:
                st.write(s, e)

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.success(f"🔥 {len(df)} SIGNAL (FINAL v2)")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("🔥 0 SIGNAL")
