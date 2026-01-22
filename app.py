import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os, time
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("IDX PRO Scanner — FULL FIXED", layout="wide")
st.title("🚀 IDX PRO Scanner — Yahoo Finance (FULL SUITE)")

DEBUG = st.sidebar.toggle("🧪 Debug Mode", False)

# =====================================================
# FILES & TIME
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
        "Entry","SL","TP1","TP2",
        "Label","AutoLabel","Status","R"
    ]).to_csv(SIGNAL_FILE, index=False)

# =====================================================
# SIDEBAR — EXCEL UPLOAD (GUARD WAJIB)
# =====================================================
st.sidebar.header("📂 Master Saham IDX")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (1 kolom kode saham IDX)",
    type=["xlsx"]
)

# 🔥 GUARD PALING PENTING (FIX ERROR 0/256)
if uploaded_file is None:
    st.warning("⬅️ Upload Excel kode saham IDX dulu")
    st.stop()

# =====================================================
# LOAD SYMBOLS
# =====================================================
@st.cache_data(ttl=3600)
def load_symbols(file):
    df = pd.read_excel(file)
    col = df.columns[0]
    syms = (
        df[col]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .unique()
        .tolist()
    )
    return [s + ".JK" for s in syms if len(s) >= 3]

@st.cache_data(ttl=1800)
def filter_by_volume(symbols):
    liquid = []
    for s in symbols:
        try:
            df = yf.download(s, period="10d", interval="1d", progress=False)
            if df.empty or "Volume" not in df.columns:
                continue
            if df["Volume"].tail(5).mean() >= MIN_AVG_VOLUME:
                liquid.append(s)
            time.sleep(0.03)
        except:
            pass
    return liquid

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].astype(float).dropna()

# =====================================================
# INDICATORS (SAFE)
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high.values, df.low.values, df.close.values
    tr = np.maximum.reduce([
        h-l, np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))
    ])
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    trend = 1
    stl = lower[0]
    for i in range(1,len(c)):
        if trend==1:
            stl = max(lower[i], stl)
            if c[i]<stl: trend=-1
        else:
            stl = min(upper[i], stl)
            if c[i]>stl: trend=1
    return trend

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    denom = (h-l).replace(0,np.nan)
    mfm = ((c-l)-(h-c))/denom
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df, lb):
    lows = df.low.values
    sup=[]
    for i in range(lb,len(lows)-lb):
        if lows[i]==np.min(lows[i-lb:i+lb+1]):
            sup.append(lows[i])
    return sup

# =====================================================
# SCORE & TRADE
# =====================================================
def calculate_score(df1h, df1d):
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
    entry=float(df1d.close.iloc[-1])
    sup=[s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
    if len(sup)==0: return None
    sl=max(sup)*(1-ZONE_BUFFER)
    if entry-sl<entry*MIN_RISK_PCT: return None
    return entry, sl

def auto_label(price, entry, sl, tp2):
    if price < sl:
        return "NO REENTRY"
    if abs(price-entry)/entry <= 0.003:
        return "RETEST"
    if price > tp2:
        return "EXTENDED"
    if price > entry:
        return "HOLD"
    return ""

# =====================================================
# LOAD SYMBOLS
# =====================================================
ALL = load_symbols(uploaded_file)
LIQ = filter_by_volume(ALL)

st.caption(f"📊 Saham likuid: {len(LIQ)} / {len(ALL)}")

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔍 Scanner", "📜 Riwayat", "🎲 Monte Carlo"])

# =====================================================
# TAB 1 — SCANNER
# =====================================================
with tab1:
    if st.button("🔍 Scan Saham IDX (Rating 10)"):
        found=[]
        for s in LIQ:
            try:
                df1h=fetch_ohlcv(s,ENTRY_INTERVAL,LOOKBACK_1H)
                df1d=fetch_ohlcv(s,DAILY_INTERVAL,LOOKBACK_1D)

                if supertrend(df1h,ATR_PERIOD,MULTIPLIER)!=1: continue
                score=calculate_score(df1h,df1d)
                if score!=10: continue

                trade=trade_levels(df1d)
                if trade is None: continue
                entry, sl = trade

                price=df1h.close.iloc[-1]
                tp2 = entry+(entry-sl)*2.0

                found.append({
                    "Time":now_wib(),
                    "Symbol":s,
                    "Phase":"AKUMULASI_KUAT",
                    "Score":10,
                    "Rating":"⭐"*10,
                    "Entry":round(entry,2),
                    "SL":round(sl,2),
                    "TP1":round(entry+(entry-sl)*0.8,2),
                    "TP2":round(tp2,2),
                    "Label":"NEW",
                    "AutoLabel":auto_label(price,entry,sl,tp2),
                    "Status":"OPEN",
                    "R":np.nan
                })
            except:
                pass

        if found:
            df=pd.DataFrame(found)
            hist=pd.read_csv(SIGNAL_FILE)
            hist=pd.concat([hist,df]).drop_duplicates(["Symbol","Entry"])
            hist.to_csv(SIGNAL_FILE,index=False)
            st.success(f"🔥 {len(df)} SIGNAL RATING 10")
            st.dataframe(df,use_container_width=True)
        else:
            st.warning("0 SIGNAL")

# =====================================================
# TAB 2 — RIWAYAT
# =====================================================
with tab2:
    hist=pd.read_csv(SIGNAL_FILE)
    if hist.empty:
        st.info("Belum ada riwayat")
    else:
        st.dataframe(hist,use_container_width=True)
        st.download_button("⬇️ Download CSV", hist.to_csv(index=False), "signal_history.csv")

# =====================================================
# TAB 3 — MONTE CARLO
# =====================================================
with tab3:
    hist=pd.read_csv(SIGNAL_FILE)
    r = hist["R"].dropna().values

    if len(r) < 10:
        st.warning("Belum cukup data untuk Monte Carlo (min 10 trade)")
    else:
        risk = st.slider("Risk per Trade (%)",0.25,3.0,1.0)/100
        trades = st.slider("Trades / Simulation",50,500,200)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000
                eq=[bal]
                for _ in range(trades):
                    bal+=bal*risk*np.random.choice(r)
                    eq.append(bal)
                curves.append(eq)

            curves=np.array(curves)
            st.metric("Median Balance",f"${np.median(curves[:,-1]):,.0f}")
            st.metric("Risk of Ruin (<$5k)",f"{(curves[:,-1]<5000).mean()*100:.2f}%")
