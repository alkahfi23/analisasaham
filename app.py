import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time, os
from datetime import datetime, timezone, timedelta

# =====================================================
# STREAMLIT
# =====================================================
st.set_page_config(
    page_title="IDX PRO Scanner — Yahoo Finance (FINAL)",
    layout="wide"
)
st.title("🚀 IDX PRO Scanner — Yahoo Finance (FINAL STABLE)")

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
# CONFIG IDX (AMAN & REALISTIS)
# =====================================================
INTERVAL = "1d"
LOOKBACK = "2y"

MIN_AVG_VOLUME = 300_000
MIN_SCORE = 7

ATR_PERIOD = 10
SUPPORT_LOOKBACK = 5
ZONE_BUFFER = 0.01
MIN_RISK_PCT = 0.01

VO_FAST = 14
VO_SLOW = 28

# =====================================================
# INIT CSV
# =====================================================
if not os.path.exists(SIGNAL_FILE):
    pd.DataFrame(columns=[
        "Time","Symbol","Score","Entry","SL","TP1","TP2","R"
    ]).to_csv(SIGNAL_FILE, index=False)

# =====================================================
# SIDEBAR — UPLOAD EXCEL
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

ALL_SYMBOLS = load_symbols(uploaded_file)

# =====================================================
# FILTER BY VOLUME (NO CACHE — AMAN)
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

            time.sleep(0.05)
        except:
            continue
    return liquid

LIQUID = filter_by_volume(ALL_SYMBOLS)
st.caption(f"📊 Saham likuid: {len(LIQUID)} / {len(ALL_SYMBOLS)}")

# =====================================================
# FETCH OHLCV
# =====================================================
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol):
    df = yf.download(
        symbol, interval=INTERVAL, period=LOOKBACK, progress=False
    )
    if df.empty:
        raise RuntimeError("No data")

    df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].astype(float)

# =====================================================
# INDICATORS
# =====================================================
def volume_osc(v,f,s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    denom = (h-l).replace(0,np.nan)
    mfm = ((c-l)-(h-c))/denom
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df, lb):
    lows = df.low.values
    supports=[]
    for i in range(lb,len(lows)-lb):
        if lows[i]==min(lows[i-lb:i+lb+1]):
            supports.append(lows[i])
    return supports

# =====================================================
# SCORE & TRADE
# =====================================================
def calculate_score(df):
    score=0
    ema20=df.close.ewm(span=20).mean()
    ema50=df.close.ewm(span=50).mean()
    ema200=df.close.ewm(span=200).mean()
    p=df.close.iloc[-1]

    if p>ema20.iloc[-1]: score+=1
    if ema20.iloc[-1]>ema50.iloc[-1]: score+=1
    if ema50.iloc[-1]>ema200.iloc[-1]: score+=1
    if p>ema200.iloc[-1]: score+=1

    vo=volume_osc(df.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo>5: score+=1
    if vo>10: score+=1
    if vo>20: score+=1

    adl=accumulation_distribution(df)
    if adl.iloc[-1]>adl.iloc[-5]: score+=1
    if adl.iloc[-1]>adl.iloc[-10]: score+=1
    if adl.iloc[-1]>adl.iloc[-20]: score+=1

    return score

def trade_levels(df):
    entry=float(df.close.iloc[-1])
    supports=[s for s in find_support(df,SUPPORT_LOOKBACK) if s<entry]
    if not supports: return None

    sl=max(supports)*(1-ZONE_BUFFER)
    if entry-sl<entry*MIN_RISK_PCT:
        return None

    r=(entry+(entry-sl)*2-entry)/(entry-sl)
    return entry,sl,r

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔍 Scanner", "📜 Riwayat", "🎲 Monte Carlo"])

# =====================================================
# TAB 1 — SCANNER
# =====================================================
with tab1:
    if st.button("🔍 Scan Saham IDX (Rating ≥ 7)"):
        results=[]
        for s in LIQUID:
            try:
                df=fetch_ohlcv(s)
                score=calculate_score(df)
                if score<MIN_SCORE: continue

                trade=trade_levels(df)
                if not trade: continue

                entry,sl,r=trade
                results.append({
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

        if results:
            df_res=pd.DataFrame(results).sort_values("Score",ascending=False)
            st.success(f"🔥 {len(df_res)} SIGNAL DITEMUKAN")
            st.dataframe(df_res,use_container_width=True)

            hist=pd.read_csv(SIGNAL_FILE)
            merged=pd.concat([hist,df_res]).drop_duplicates(
                subset=["Symbol","Entry"],keep="first"
            )
            merged.to_csv(SIGNAL_FILE,index=False)
        else:
            st.warning("❌ Tidak ada saham lolos filter")

# =====================================================
# TAB 2 — RIWAYAT
# =====================================================
with tab2:
    hist=pd.read_csv(SIGNAL_FILE)
    if hist.empty:
        st.info("Belum ada riwayat")
    else:
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
        st.warning("Minimal 10 trade untuk Monte Carlo")
    else:
        r=hist["R"].values
        risk=st.slider("Risk / Trade (%)",0.5,3.0,1.0)/100
        trades=st.slider("Trades / Simulasi",50,300,150)

        if st.button("🎲 Run Monte Carlo"):
            curves=[]
            for _ in range(500):
                bal=10000
                for _ in range(trades):
                    bal+=bal*risk*np.random.choice(r)
                curves.append(bal)

            curves=np.array(curves)
            st.metric("Median Balance",f"${np.median(curves):,.0f}")
            st.metric(
                "Risk of Ruin (<50%)",
                f"{(curves<5000).mean()*100:.2f}%"
            )
