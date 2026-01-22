import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, time as dtime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("IDX PRO Scanner (TradingView)", layout="wide")
st.title("📈 IDX PRO Scanner — TradingView (Render VPS)")

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# CONFIG
# =====================================================
ENTRY_TF = "4h"
DAILY_TF = "1d"

LIMIT_4H = 200
LIMIT_1D = 200

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
    "IDX:BBRI","IDX:BMRI","IDX:BBCA",
    "IDX:TLKM","IDX:ASII"
]

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
# TRADINGVIEW DATA
# =====================================================
TV_TF_MAP = {"4h": "240", "1d": "D"}

class TradingViewData:
    BASE_URL = "https://scanner.tradingview.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json"
        })

    def get_ohlcv(self, symbol, tf, limit=200):
        r = self.session.post(
            f"{self.BASE_URL}/history",
            json={
                "symbol": symbol,
                "resolution": TV_TF_MAP[tf],
                "count": limit
            },
            timeout=15
        )
        r.raise_for_status()
        d = r.json()
        if d.get("s") != "ok":
            raise RuntimeError(d)

        return pd.DataFrame({
            "open": d["o"],
            "high": d["h"],
            "low": d["l"],
            "close": d["c"],
            "volume": d["v"]
        })

    def get_price(self, symbol):
        r = self.session.post(
            f"{self.BASE_URL}/quotes",
            json={"symbols": [{"symbol": symbol}]},
            timeout=10
        )
        r.raise_for_status()
        d = r.json()["data"][0]["d"]
        price = d.get("lp")
        if price is None:
            raise ValueError("Price unavailable")
        return float(price)

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high,df.low,df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index,dtype=float)
    trend = pd.Series(index=df.index,dtype=int)

    trend.iloc[0] = 1 if c.iloc[0] > hl2.iloc[0] else -1
    stl.iloc[0] = lower.iloc[0] if trend.iloc[0]==1 else upper.iloc[0]

    for i in range(1,len(df)):
        if trend.iloc[i-1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

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
        if not clean or abs(s-clean[-1])/clean[-1]>0.015:
            clean.append(s)
    return clean

# =====================================================
# LOGIC
# =====================================================
def reject_ara_arb(df):
    pc,lc = df.close.iloc[-2],df.close.iloc[-1]
    return lc >= pc*1.25 or lc <= pc*0.75

def calculate_score(df4h,df1d):
    score=0
    ema20=df4h.close.ewm(span=20).mean()
    ema50=df4h.close.ewm(span=50).mean()
    ema200=df1d.close.ewm(span=200).mean()
    price=df4h.close.iloc[-1]

    if price>ema20.iloc[-1]: score+=1
    if ema20.iloc[-1]>ema50.iloc[-1]: score+=1
    if ema50.iloc[-1]>ema200.iloc[-1]: score+=1
    if price>ema200.iloc[-1]: score+=1

    vo=volume_osc(df4h.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo>5: score+=1
    if vo>10: score+=1
    if vo>20: score+=1

    adl=accumulation_distribution(df4h)
    if adl.iloc[-1]>adl.iloc[-5]: score+=1
    if adl.iloc[-1]>adl.iloc[-10]: score+=1
    if adl.iloc[-1]>adl.iloc[-20]: score+=1

    return score

def trade_levels(df4h,df1d):
    entry=df4h.close.iloc[-1]
    supports=[s for s in find_support(df1d,SR_LOOKBACK) if s<entry]
    if not supports:
        return None
    sl=max(supports)*(1-ZONE_BUFFER)
    risk=entry-sl
    if risk<entry*MIN_RISK_PCT:
        return None
    return {
        "Entry":round(entry,2),
        "SL":round(sl,2),
        "TP1":round(entry+risk*TP1_R,2),
        "TP2":round(entry+risk*TP2_R,2)
    }

def auto_label(sig,price):
    if not is_market_open(): return "WAIT"
    if price<=sig["SL"]: return "INVALID"
    if price>=sig["Entry"]+(sig["TP1"]-sig["Entry"])*TP_EXTEND:
        return "EXTENDED"
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
# MONTE CARLO
# =====================================================
def monte_carlo(rvals,initial,risk,trades,sims):
    curves=[]
    for _ in range(sims):
        bal=initial; eq=[bal]
        for _ in range(trades):
            r=np.random.choice(rvals)
            bal+=bal*risk*r
            eq.append(bal)
        curves.append(eq)
    return np.array(curves)

def mc_metrics(curves,initial):
    final=curves[:,-1]
    return {
        "Median":np.median(final),
        "Worst5":np.percentile(final,5),
        "Ruin%":(final<initial*0.5).mean()*100
    }

# =====================================================
# UI
# =====================================================
tab1,tab2=st.tabs(["📡 Scanner IDX","🎲 Monte Carlo"])

tv=TradingViewData()

with tab1:
    if st.button("🔍 Scan Saham IDX (TradingView)"):
        rows=[]
        for s in IDX_SYMBOLS:
            try:
                df4h=tv.get_ohlcv(s,ENTRY_TF)
                df1d=tv.get_ohlcv(s,DAILY_TF)
                if reject_ara_arb(df1d): continue

                stl,trend=supertrend(df4h,ATR_PERIOD,MULTIPLIER)
                if trend.iloc[-1]!=1: continue

                vo=volume_osc(df4h.volume,VO_FAST,VO_SLOW)
                adl=accumulation_distribution(df4h)
                if vo.iloc[-1]<VO_MIN or adl.iloc[-1]<=adl.iloc[-10]:
                    continue

                score=calculate_score(df4h,df1d)
                if score<6: continue

                trade=trade_levels(df4h,df1d)
                if not trade: continue

                price=tv.get_price(s)
                label=auto_label(trade,price)

                rows.append({
                    "Time":now_wib(),
                    "Symbol":s,
                    "Phase":"AKUMULASI_KUAT",
                    "Score":score,
                    "Rating":"⭐"*score,
                    "Last":round(price,2),
                    "Label":label,
                    **trade
                })

                time.sleep(0.4)

            except Exception as e:
                st.warning(f"{s} error: {e}")

        df=pd.DataFrame(rows).sort_values("Score",ascending=False)
        st.dataframe(df,use_container_width=True)

        for _,r in df.iterrows():
            with st.expander(f"{r.Symbol} | {r.Label} | {r.Rating}"):
                dfc=tv.get_ohlcv(r.Symbol,ENTRY_TF,120)
                stl,_=supertrend(dfc,ATR_PERIOD,MULTIPLIER)
                adl=accumulation_distribution(dfc)
                st.plotly_chart(render_chart(dfc,stl,adl,r),use_container_width=True)

with tab2:
    st.subheader("🎲 Monte Carlo — Saham IDX")
    r_values=[-1,-1,0.8,2.0,0.8,-1,2.0]
    initial=st.number_input("Modal Awal (IDR)",100_000_000,step=10_000_000)
    risk=st.slider("Risk / Trade (%)",0.25,1.0,0.5)/100
    trades=st.slider("Trades / Simulation",50,300,150)
    sims=st.slider("Simulations",300,1000,500)

    if st.button("Run Monte Carlo"):
        curves=monte_carlo(r_values,initial,risk,trades,sims)
        m=mc_metrics(curves,initial)
        c1,c2,c3=st.columns(3)
        c1.metric("Median",f"Rp {m['Median']:,.0f}")
        c2.metric("Worst 5%",f"Rp {m['Worst5']:,.0f}")
        c3.metric("Risk of Ruin",f"{m['Ruin%']:.2f}%")

        fig=go.Figure()
        for i in range(min(30,len(curves))):
            fig.add_trace(go.Scatter(y=curves[i],opacity=0.3,showlegend=False))
        fig.update_layout(template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
