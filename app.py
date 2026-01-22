import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os
from datetime import datetime, timezone, timedelta, time as dtime

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("IDX PRO Scanner — DEBUG FINAL", layout="wide")
st.title("📈 IDX PRO Scanner — Yahoo Finance (DEBUG FINAL)")

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
# CONFIG (REALISTIS IDX)
# =====================================================
ENTRY_INTERVAL = "1h"
DAILY_INTERVAL = "1d"
LOOKBACK_1H = "6mo"
LOOKBACK_1D = "3y"

MIN_AVG_VOLUME = 300_000   # lebih realistis IDX
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
    ]).to_csv(SIGNAL_FILE,index=False)

# =====================================================
# SIDEBAR — EXCEL
# =====================================================
st.sidebar.header("📂 Master Saham IDX")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (1 kolom kode saham IDX)",
    type=["xlsx"]
)

# =====================================================
# LOAD SYMBOLS (KOLOM PERTAMA SAJA)
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
# FILTER VOLUME + DEBUG
# =====================================================
@st.cache_data(ttl=1800)
def filter_by_volume(symbols, min_volume):
    liquid=[]
    debug=[]

    for s in symbols:
        try:
            df = yf.download(
                s, period="10d", interval="1d", progress=False
            )

            if df.empty:
                debug.append({"Symbol": s, "Reason": "No data"})
                continue

            if "Volume" not in df.columns:
                debug.append({"Symbol": s, "Reason": "No Volume column"})
                continue

            avg_vol = df["Volume"].dropna().tail(5).mean()

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

            time.sleep(0.1)

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

    return df[["open","high","low","close","volume"]].dropna()

# =====================================================
# INDICATORS (FIXED)
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

    return trend   # ⬅️ INT ONLY (AMAN)

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    mfm=((c-l)-(h-c))/(h-l)
    mfm=mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df,lb):
    lv=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i]==min(df.low.iloc[i-lb:i+lb+1]):
            lv.append(df.low.iloc[i])
    return sorted(set(lv))

# =====================================================
# SCORE & TRADE
# =====================================================
def calculate_score(df1h,df1d):
    score=0
    ema20=df1h.close.ewm(span=20).mean()
    ema50=df1h.close.ewm(span=50).mean()
    ema200=df1d.close.ewm(span=200).mean()
    p=df1h.close.iloc[-1]

    if p>ema20.iloc[-1]: score+=1
    if ema20.iloc[-1]>ema50.iloc[-1]: score+=1
    if ema50.iloc[-1]>ema200.iloc[-1]: score+=1
    if p>ema200.iloc[-1]: score+=1

    vo=volume_osc(df1h.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo>5: score+=1
    if vo>10: score+=1
    if vo>20: score+=1

    adl=accumulation_distribution(df1h)
    if adl.iloc[-1]>adl.iloc[-5]: score+=1
    if adl.iloc[-1]>adl.iloc[-10]: score+=1
    if adl.iloc[-1]>adl.iloc[-20]: score+=1

    return score

def trade_levels(df1d):
    entry = float(df1d.close.iloc[-1])

    supports = find_support(df1d, SR_LOOKBACK)

    # ⬇️ FIX PALING PENTING
    if supports is None or len(supports) == 0:
        return None

    # pastikan semua support float
    supports = [float(s) for s in supports if s < entry]

    if len(supports) == 0:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    risk = entry - sl

    if risk < entry * MIN_RISK_PCT:
        return None

    return entry, sl


# =====================================================
# MAIN FLOW
# =====================================================
if not uploaded_file:
    st.warning("⬅️ Upload Excel kode saham IDX")
    st.stop()

ALL_SYMBOLS = load_idx_symbols_from_excel(uploaded_file)

if DEBUG:
    st.write("📄 Total symbol Excel:", len(ALL_SYMBOLS))
    st.write("📄 Sample:", ALL_SYMBOLS[:10])

IDX_SYMBOLS, VOL_DEBUG = filter_by_volume(ALL_SYMBOLS, MIN_AVG_VOLUME)

st.caption(f"📊 Saham likuid: {len(IDX_SYMBOLS)} / {len(ALL_SYMBOLS)}")

if DEBUG:
    st.subheader("🧪 Debug Volume Filter")
    st.dataframe(VOL_DEBUG.head(50), use_container_width=True)

# =====================================================
# SCANNER
# =====================================================
if st.button("🔍 Scan Saham IDX"):
    found=[]
    for s in IDX_SYMBOLS:
        try:
            df1h=fetch_ohlcv(s,ENTRY_INTERVAL,LOOKBACK_1H)
            df1d=fetch_ohlcv(s,DAILY_INTERVAL,LOOKBACK_1D)

            trend=supertrend(df1h,ATR_PERIOD,MULTIPLIER)
            if trend != 1:
                if DEBUG: st.write(f"{s} ❌ Supertrend bearish")
                continue

            score=calculate_score(df1h,df1d)
            if score<MIN_SCORE:
                if DEBUG: st.write(f"{s} ❌ Score {score}")
                continue

            trade = trade_levels(df1d)
            if trade is None:
                if DEBUG: st.write(f"{s} ❌ Risk / support invalid")
                continue

            entry, sl = trade
            sig={
                "Time":now_wib(),
                "Symbol":s,
                "Phase":"AKUMULASI_KUAT",
                "Score":score,
                "Rating":"⭐"*score,
                "Entry":round(entry,2),
                "SL":round(sl,2),
                "TP1":round(entry+(entry-sl)*0.8,2),
                "TP2":round(entry+(entry-sl)*2.0,2),
                "Label":"NEW"
            }
            found.append(sig)

        except Exception as e:
            if DEBUG:
                st.write(f"{s} ❌ Error: {e}")

    if found:
        df=pd.DataFrame(found).sort_values("Score",ascending=False)
        st.success(f"🔥 {len(df)} SIGNAL AKUMULASI_KUAT")
        st.dataframe(df,use_container_width=True)
    else:
        st.warning("🔥 0 SIGNAL AKUMULASI_KUAT")

    if DEBUG:
        st.info(f"""
        DEBUG SUMMARY
        - Excel symbols : {len(ALL_SYMBOLS)}
        - Lolos volume  : {len(IDX_SYMBOLS)}
        - Signal final  : {len(found)}
        """)
