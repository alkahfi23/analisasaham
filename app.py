import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, time as dtime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("IDX PRO Scanner (Alpha Vantage)", layout="wide")
st.title("📈 IDX PRO Scanner — Alpha Vantage")

# =====================================================
# API KEY
# =====================================================
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    st.error("❌ ALPHAVANTAGE_API_KEY belum diset")
    st.stop()

BASE_URL = "https://www.alphavantage.co/query"

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# CONFIG
# =====================================================
ATR_PERIOD = 10
MULTIPLIER = 3.0

VO_FAST = 14
VO_SLOW = 28
VO_MIN = 5

SR_LOOKBACK = 5
ZONE_BUFFER = 0.01

TP1_R = 0.8
TP2_R = 2.0

MIN_RISK_PCT = 0.01

RETEST_TOL = 0.005
TP_EXTEND = 0.9

IDX_SYMBOLS = [
    "BBRI.JK","BMRI.JK","BBCA.JK","TLKM.JK","ASII.JK"
]

# =====================================================
# MARKET HOURS
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
# ALPHA VANTAGE DATA
# =====================================================
def fetch_daily(symbol):
    r = requests.get(
        BASE_URL,
        params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": API_KEY,
            "outputsize": "compact"
        },
        timeout=20
    )
    data = r.json().get("Time Series (Daily)")
    if not data:
        raise RuntimeError("No daily data")

    df = pd.DataFrame(data).T.astype(float)
    df.columns = ["open","high","low","close","adj_close","volume","div","split"]
    return df[["open","high","low","close","volume"]].iloc[::-1]

def fetch_intraday_60(symbol):
    r = requests.get(
        BASE_URL,
        params={
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "60min",
            "apikey": API_KEY,
            "outputsize": "compact"
        },
        timeout=20
    )
    data = r.json().get("Time Series (60min)")
    if not data:
        raise RuntimeError("No intraday data")

    df = pd.DataFrame(data).T.astype(float)
    df.columns = ["open","high","low","close","volume"]
    return df.iloc[::-1]

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, p, m):
    h,l,c = df.high,df.low,df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=p,adjust=False).mean()
    hl2 = (h+l)/2
    upper = hl2 + m*atr
    lower = hl2 - m*atr
    stl = pd.Series(index=df.index,dtype=float)
    trend = pd.Series(index=df.index,dtype=int)
    trend.iloc[0] = 1
    stl.iloc[0] = lower.iloc[0]
    for i in range(1,len(df)):
        if trend.iloc[i-1]==1:
            stl.iloc[i]=max(lower.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=1 if c.iloc[i]>stl.iloc[i] else -1
        else:
            stl.iloc[i]=min(upper.iloc[i],stl.iloc[i-1])
            trend.iloc[i]=-1 if c.iloc[i]<stl.iloc[i] else 1
    return stl,trend

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean()-v.ewm(span=s).mean())/v.ewm(span=s).mean()*100

def accumulation_distribution(df):
    h,l,c,v = df.high,df.low,df.close,df.volume
    mfm = ((c-l)-(h-c))/(h-l)
    mfm = mfm.replace([np.inf,-np.inf],0).fillna(0)
    return (mfm*v).cumsum()

def find_support(df,lb):
    levels=[]
    for i in range(lb,len(df)-lb):
        if df.low.iloc[i]==min(df.low.iloc[i-lb:i+lb+1]):
            levels.append(df.low.iloc[i])
    levels=sorted(set(levels))
    clean=[]
    for s in levels:
        if not clean or abs(s-clean[-1])/clean[-1]>0.02:
            clean.append(s)
    return clean

# =====================================================
# SCORING & TRADE
# =====================================================
def calculate_score(df):
    score=0
    ema20=df.close.ewm(span=20).mean()
    ema50=df.close.ewm(span=50).mean()
    if df.close.iloc[-1]>ema20.iloc[-1]: score+=1
    if ema20.iloc[-1]>ema50.iloc[-1]: score+=1
    vo=volume_osc(df.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo>5: score+=1
    if vo>10: score+=1
    adl=accumulation_distribution(df)
    if adl.iloc[-1]>adl.iloc[-10]: score+=2
    return score

def trade_levels(df):
    entry=df.close.iloc[-1]
    sup=[s for s in find_support(df,SR_LOOKBACK) if s<entry]
    if not sup: return None
    sl=max(sup)*(1-ZONE_BUFFER)
    risk=entry-sl
    if risk<entry*MIN_RISK_PCT: return None
    return {
        "Entry":round(entry,2),
        "SL":round(sl,2),
        "TP1":round(entry+risk*TP1_R,2),
        "TP2":round(entry+risk*TP2_R,2)
    }

def auto_label(sig,price):
    if not is_market_open(): return "WAIT"
    if price<=sig["SL"]: return "INVALID"
    if price>=sig["TP1"]*TP_EXTEND: return "EXTENDED"
    if abs(price-sig["Entry"])/sig["Entry"]<=RETEST_TOL:
        return "RETEST"
    if price>sig["Entry"]: return "HOLD"
    return ""

# =====================================================
# CHART
# =====================================================
def render_chart(df,stl,adl,sig):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3])
    fig.add_candlestick(
        x=df.index,open=df.open,high=df.high,
        low=df.low,close=df.close,row=1,col=1
    )
    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime")),row=1,col=1)
    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=float(sig[k]),line_color=c,row=1)
    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan")),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=520,xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# UI
# =====================================================
if st.button("🔍 Scan Saham IDX (Alpha Vantage)"):
    rows=[]
    for s in IDX_SYMBOLS:
        try:
            df_d = fetch_daily(s)
            df_i = fetch_intraday_60(s)

            stl,trend=supertrend(df_i,ATR_PERIOD,MULTIPLIER)
            if trend.iloc[-1]!=1: continue

            score=calculate_score(df_i)
            if score<5: continue

            trade=trade_levels(df_d)
            if not trade: continue

            price=df_i.close.iloc[-1]
            label=auto_label(trade,price)

            rows.append({
                "Time":now_wib(),
                "Symbol":s,
                "Score":score,
                "Rating":"⭐"*score,
                "Last":round(price,2),
                "Label":label,
                **trade
            })

            time.sleep(15)  # RATE LIMIT FREE

        except Exception as e:
            st.warning(f"{s} error: {e}")

    if not rows:
        st.warning("⚠️ Tidak ada saham yang lolos filter.")
    else:
        df=pd.DataFrame(rows).sort_values("Score",ascending=False)
        st.dataframe(df,use_container_width=True)

        for _,r in df.iterrows():
            with st.expander(f"{r.Symbol} | {r.Label} | {r.Rating}"):
                dfc=fetch_intraday_60(r.Symbol)
                stl,_=supertrend(dfc,ATR_PERIOD,MULTIPLIER)
                adl=accumulation_distribution(dfc)
                st.plotly_chart(render_chart(dfc,stl,adl,r),use_container_width=True)
