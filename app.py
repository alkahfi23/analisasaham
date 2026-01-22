import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config("IDX PRO Scanner (Cloud Safe)", layout="wide")
st.title("📈 IDX PRO Scanner — Streamlit Cloud SAFE")

# =====================================================
# MODE (WAJIB TRUE UNTUK STREAMLIT CLOUD)
# =====================================================
USE_LIVE_DATA = False   # <-- JANGAN DIUBAH DI STREAMLIT CLOUD

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
    "IDX:BBRI", "IDX:BMRI", "IDX:BBCA",
    "IDX:TLKM", "IDX:ASII"
]

# =====================================================
# DUMMY DATA ENGINE (CLOUD SAFE)
# =====================================================
def dummy_ohlcv(n=200):
    base = np.cumsum(np.random.randn(n)) * 10 + 5000
    return pd.DataFrame({
        "open": base + np.random.randn(n)*5,
        "high": base + np.random.rand(n)*20,
        "low": base - np.random.rand(n)*20,
        "close": base + np.random.randn(n)*5,
        "volume": np.random.randint(1_000_000, 10_000_000, n)
    })

def dummy_price(df):
    return float(df.close.iloc[-1])

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h,l,c = df.high, df.low, df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    hl2 = (h+l)/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    stl = pd.Series(index=df.index,dtype=float)
    trend = pd.Series(index=df.index,dtype=int)

    trend.iloc[0] = 1
    stl.iloc[0] = lower.iloc[0]

    for i in range(1,len(df)):
        if trend.iloc[i-1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i-1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

def volume_osc(v,f,s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h,l,c,v = df.high, df.low, df.close, df.volume
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
# SCORE & TRADE
# =====================================================
def calculate_score(df):
    score = 0
    ema20 = df.close.ewm(span=20).mean()
    ema50 = df.close.ewm(span=50).mean()

    if df.close.iloc[-1] > ema20.iloc[-1]: score+=1
    if ema20.iloc[-1] > ema50.iloc[-1]: score+=1

    vo = volume_osc(df.volume,VO_FAST,VO_SLOW).iloc[-1]
    if vo > 5: score+=1
    if vo > 10: score+=1

    adl = accumulation_distribution(df)
    if adl.iloc[-1] > adl.iloc[-10]: score+=2

    return score

def trade_levels(df):
    entry = df.close.iloc[-1]
    supports = [s for s in find_support(df,SR_LOOKBACK) if s < entry]
    if not supports:
        return None
    sl = max(supports)*(1-ZONE_BUFFER)
    risk = entry-sl
    if risk < entry*MIN_RISK_PCT:
        return None
    return {
        "Entry":round(entry,2),
        "SL":round(sl,2),
        "TP1":round(entry+risk*TP1_R,2),
        "TP2":round(entry+risk*TP2_R,2)
    }

def auto_label(sig, price):
    if price <= sig["SL"]: return "INVALID"
    if price >= sig["TP1"]*TP_EXTEND: return "EXTENDED"
    if abs(price-sig["Entry"])/sig["Entry"] <= RETEST_TOL:
        return "RETEST"
    if price > sig["Entry"]: return "HOLD"
    return ""

# =====================================================
# CHART
# =====================================================
def render_chart(df, stl, adl, sig):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3])
    fig.add_candlestick(
        x=df.index,
        open=df.open,high=df.high,
        low=df.low,close=df.close,
        row=1,col=1
    )
    fig.add_trace(go.Scatter(x=df.index,y=stl,line=dict(color="lime")),row=1,col=1)

    for k,c in [("Entry","cyan"),("SL","red"),("TP1","orange"),("TP2","purple")]:
        fig.add_hline(y=float(sig[k]), line_color=c, row=1)

    fig.add_trace(go.Scatter(x=df.index,y=adl,line=dict(color="cyan")),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=520,
                      xaxis_rangeslider_visible=False)
    return fig

# =====================================================
# MONTE CARLO
# =====================================================
def monte_carlo(rvals, initial, risk, trades, sims):
    curves=[]
    for _ in range(sims):
        bal=initial
        eq=[bal]
        for _ in range(trades):
            r=np.random.choice(rvals)
            bal+=bal*risk*r
            eq.append(bal)
        curves.append(eq)
    return np.array(curves)

def mc_metrics(curves, initial):
    final=curves[:,-1]
    return {
        "Median":np.median(final),
        "Worst5":np.percentile(final,5),
        "Ruin%":(final<initial*0.5).mean()*100
    }

# =====================================================
# UI
# =====================================================
tab1, tab2 = st.tabs(["📡 Scanner", "🎲 Monte Carlo"])

with tab1:
    if st.button("🔍 Scan IDX (Cloud Safe)"):
        rows=[]
        for s in IDX_SYMBOLS:
            df = dummy_ohlcv()
            stl, trend = supertrend(df,ATR_PERIOD,MULTIPLIER)
            if trend.iloc[-1] != 1:
                continue
            score = calculate_score(df)
            if score < 5:
                continue
            trade = trade_levels(df)
            if not trade:
                continue
            price = dummy_price(df)
            label = auto_label(trade, price)

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

        df_sig = pd.DataFrame(rows).sort_values("Score",ascending=False)
        st.dataframe(df_sig, use_container_width=True)

        for _,r in df_sig.iterrows():
            with st.expander(f"{r.Symbol} | {r.Label} | {r.Rating}"):
                dfc = dummy_ohlcv(120)
                stl,_ = supertrend(dfc,ATR_PERIOD,MULTIPLIER)
                adl = accumulation_distribution(dfc)
                st.plotly_chart(render_chart(dfc,stl,adl,r),
                                use_container_width=True)

with tab2:
    st.subheader("🎲 Monte Carlo — Saham IDX")
    dummy_R = [-1,-1,0.8,2.0,0.8,-1,2.0]
    initial = st.number_input("Modal Awal (IDR)",100_000_000,step=10_000_000)
    risk = st.slider("Risk / Trade (%)",0.25,1.0,0.5)/100
    trades = st.slider("Trades / Simulation",50,300,150)
    sims = st.slider("Simulations",300,1000,500)

    if st.button("Run Monte Carlo"):
        curves = monte_carlo(dummy_R,initial,risk,trades,sims)
        m = mc_metrics(curves,initial)
        c1,c2,c3 = st.columns(3)
        c1.metric("Median",f"Rp {m['Median']:,.0f}")
        c2.metric("Worst 5%",f"Rp {m['Worst5']:,.0f}")
        c3.metric("Risk of Ruin",f"{m['Ruin%']:.2f}%")

        fig = go.Figure()
        for i in range(min(30,len(curves))):
            fig.add_trace(go.Scatter(y=curves[i],opacity=0.3,showlegend=False))
        fig.update_layout(template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
