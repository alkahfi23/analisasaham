import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config("IDX PRO Scanner — FULL SUITE (FIXED)", layout="wide")
st.title("🚀 IDX PRO Scanner — Yahoo Finance (FULL SUITE)")

DEBUG = st.sidebar.toggle("🧪 Debug Mode", value=False)

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
# CONFIG
# =====================================================
ENTRY_INTERVAL = "1h"
DAILY_INTERVAL = "1d"
LOOKBACK_1H = "6mo"
LOOKBACK_1D = "3y"

MIN_AVG_VOLUME = 1_000_000
MIN_SCORE = 9

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
        "Time","Symbol","Score",
        "Entry","SL","TP1","TP2","R"
    ]).to_csv(SIGNAL_FILE, index=False)

# =====================================================
# SIDEBAR — EXCEL
# =====================================================
st.sidebar.header("📂 Master Saham IDX")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (1 kolom kode saham IDX)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.warning("⬅️ Upload Excel kode saham IDX dulu")
    st.stop()

# =====================================================
# LOAD SYMBOLS (CACHE AMAN)
# =====================================================
@st.cache_data(ttl=3600)
def load_idx_symbols_from_excel(file):
    df = pd.read_excel(file)
    col = df.columns[0]
    syms = (
        df[col].astype(str)
        .str.upper().str.strip()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .unique().tolist()
    )
    return [s + ".JK" for s in syms if len(s) >= 3]

# =====================================================
# FILTER VOLUME (❌ NO CACHE — FIX UTAMA)
# =====================================================
def filter_by_volume(symbols):
    liquid = []
    for s in symbols:
        try:
            df = yf.download(
                s, period="10d", interval="1d", progress=False
            )
            if df.empty or "Volume" not in df.columns:
                continue
            avg_vol = df["Volume"].tail(5).mean()
            if avg_vol >= MIN_AVG_VOLUME:
                liquid.append(s)
            time.sleep(0.1)  # anti Yahoo ban
        except:
            continue
    return liquid

# =====================================================
# FETCH OHLCV
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise RuntimeError("No data")
    df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].astype(float)

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high.values, df.low.values, df.close.values
    tr = np.maximum.reduce([
        h-l, np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))
    ])
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).ewm(span=period).mean().values
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr
    trend = 1
    st_line = lower[0]
    for i in range(1,len(c)):
        if trend==1:
            st_line=max(lower[i],st_line)
            if c[i]<st_line: trend=-1
        else:
            st_line=min(upper[i],st_line)
            if c[i]>st_line: trend=1
    return trend

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
    return lv

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
    entry=float(df1d.close.iloc[-1])
    sups=[s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
    if not sups: return None
    sl=max(sups)*(1-ZONE_BUFFER)
    if entry-sl<entry*MIN_RISK_PCT: return None
    return entry,sl

# =====================================================
# LOAD SYMBOLS
# =====================================================
ALL = load_idx_symbols_from_excel(uploaded_file)
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
    if st.button("🔍 Scan Saham IDX (Rating ≥ 9)"):
        found=[]
        for s in LIQ:
            try:
                df1h=fetch_ohlcv(s,ENTRY_INTERVAL,LOOKBACK_1H)
                df1d=fetch_ohlcv(s,DAILY_INTERVAL,LOOKBACK_1D)

                if supertrend(df1h,ATR_PERIOD,MULTIPLIER)!=1: continue

                score=calculate_score(df1h,df1d)
                if score<MIN_SCORE: continue

                trade=trade_levels(df1d)
                if not trade: continue

                entry,sl=trade
                r=(entry+(entry-sl)*2-entry)/(entry-sl)

                found.append({
                    "Time":now_wib(),
                    "Symbol":s,
                    "Score":score,
                    "Entry":round(entry,2),
                    "SL":round(sl,2),
                    "TP1":round(entry+(entry-sl)*0.8,2),
                    "TP2":round(entry+(entry-sl)*2,2),
                    "R":round(r,2)
                })

            except Exception as e:
                if DEBUG:
                    st.write(s,e)

        if found:
            df=pd.DataFrame(found).sort_values("Score",ascending=False)
            st.success(f"🔥 {len(df)} SIGNAL AKUMULASI_KUAT")
            st.dataframe(df,use_container_width=True)

            hist=pd.read_csv(SIGNAL_FILE)
            merged=pd.concat([hist,df]).drop_duplicates(
                subset=["Symbol","Entry"],keep="first"
            )
            merged.to_csv(SIGNAL_FILE,index=False)
        else:
            st.warning("0 SIGNAL")

# =====================================================
# TAB 2 — RIWAYAT
# =====================================================
with tab2:
    hist=pd.read_csv(SIGNAL_FILE)
    st.dataframe(
        hist.sort_values("Time",ascending=False),
        use_container_width=True
    )

# =====================================================
# TAB 3 — MONTE CARLO
# =====================================================
with tab3:
    hist=pd.read_csv(SIGNAL_FILE)
    if len(hist)<10:
        st.warning("Belum cukup data Monte Carlo (min 10)")
    else:
        r=hist["R"].values
        risk=st.slider("Risk / Trade (%)",0.5,3.0,1.0)/100
        trades=st.slider("Trades / Simulasi",50,300,150)

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
            st.metric(
                "Risk of Ruin (<50%)",
                f"{(curves[:,-1]<5000).mean()*100:.2f}%"
            )
