import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, time as dtime, timezone, timedelta

# =====================================================
# TIMEZONE
# =====================================================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(WIB).strftime("%Y-%m-%d %H:%M WIB")

# =====================================================
# IDX CONFIG
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

MIN_RISK_PCT = 0.01  # 1% minimal risk

IDX_SYMBOLS = [
    "IDX:BBRI",
    "IDX:BMRI",
    "IDX:BBCA",
    "IDX:TLKM",
    "IDX:ASII"
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
        dtime(9, 0) <= t <= dtime(11, 30)
        or
        dtime(13, 30) <= t <= dtime(15, 50)
    )

# =====================================================
# TRADINGVIEW DATA WRAPPER
# =====================================================
TV_TF_MAP = {
    "4h": "240",
    "1d": "D"
}

class TradingViewData:
    BASE_URL = "https://scanner.tradingview.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json"
        })

    def get_ohlcv(self, symbol, timeframe, limit=200):
        payload = {
            "symbol": symbol,
            "resolution": TV_TF_MAP[timeframe],
            "count": limit
        }

        r = self.session.post(
            f"{self.BASE_URL}/history",
            json=payload,
            timeout=15
        )
        r.raise_for_status()
        data = r.json()

        if data.get("s") != "ok":
            raise RuntimeError(f"TradingView error: {data}")

        return pd.DataFrame({
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })

# =====================================================
# INDICATORS
# =====================================================
def supertrend(df, period, mult):
    h, l, c = df.high, df.low, df.close
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (h + l) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    stl = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)

    trend.iloc[0] = 1 if c.iloc[0] > hl2.iloc[0] else -1
    stl.iloc[0] = lower.iloc[0] if trend.iloc[0] == 1 else upper.iloc[0]

    for i in range(1, len(df)):
        if trend.iloc[i - 1] == 1:
            stl.iloc[i] = max(lower.iloc[i], stl.iloc[i - 1])
            trend.iloc[i] = 1 if c.iloc[i] > stl.iloc[i] else -1
        else:
            stl.iloc[i] = min(upper.iloc[i], stl.iloc[i - 1])
            trend.iloc[i] = -1 if c.iloc[i] < stl.iloc[i] else 1

    return stl, trend

def volume_osc(v, f, s):
    return (v.ewm(span=f).mean() - v.ewm(span=s).mean()) / v.ewm(span=s).mean() * 100

def accumulation_distribution(df):
    h, l, c, v = df.high, df.low, df.close, df.volume
    mfm = ((c - l) - (h - c)) / (h - l)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    return (mfm * v).cumsum()

def find_support(df, lb):
    levels = []
    for i in range(lb, len(df) - lb):
        if df.low.iloc[i] == min(df.low.iloc[i - lb:i + lb + 1]):
            levels.append(df.low.iloc[i])

    levels = sorted(set(levels))
    clean = []
    for s in levels:
        if not clean or abs(s - clean[-1]) / clean[-1] > 0.015:
            clean.append(s)
    return clean

# =====================================================
# ARA / ARB FILTER
# =====================================================
def reject_ara_arb(df):
    prev_close = df.close.iloc[-2]
    last_close = df.close.iloc[-1]

    ara = prev_close * 1.25
    arb = prev_close * 0.75

    return last_close >= ara or last_close <= arb

# =====================================================
# SCORE ENGINE
# =====================================================
def calculate_score(df4h, df1d):
    score = 0

    ema20 = df4h.close.ewm(span=20).mean()
    ema50 = df4h.close.ewm(span=50).mean()
    ema200 = df1d.close.ewm(span=200).mean()

    price = df4h.close.iloc[-1]

    if price > ema20.iloc[-1]:
        score += 1
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1
    if ema50.iloc[-1] > ema200.iloc[-1]:
        score += 1
    if price > ema200.iloc[-1]:
        score += 1

    vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW).iloc[-1]
    if vo > 5:
        score += 1
    if vo > 10:
        score += 1
    if vo > 20:
        score += 1

    adl = accumulation_distribution(df4h)
    if adl.iloc[-1] > adl.iloc[-5]:
        score += 1
    if adl.iloc[-1] > adl.iloc[-10]:
        score += 1
    if adl.iloc[-1] > adl.iloc[-20]:
        score += 1

    return score

# =====================================================
# TRADE LEVELS (IDX)
# =====================================================
def calculate_trade_levels(df4h, df1d):
    entry = df4h.close.iloc[-1]

    supports = find_support(df1d, SR_LOOKBACK)
    supports = [s for s in supports if s < entry]

    if not supports:
        return None

    sl = max(supports) * (1 - ZONE_BUFFER)
    risk = entry - sl

    if risk < entry * MIN_RISK_PCT:
        return None

    return {
        "Entry": round(entry, 2),
        "SL": round(sl, 2),
        "TP1": round(entry + risk * TP1_R, 2),
        "TP2": round(entry + risk * TP2_R, 2)
    }

# =====================================================
# IDX SCANNER
# =====================================================
def scan_idx_market(debug=False):
    if not is_market_open():
        print("❌ Market IDX sedang tutup")
        return []

    tv = TradingViewData()
    signals = []

    for symbol in IDX_SYMBOLS:
        try:
            df4h = tv.get_ohlcv(symbol, ENTRY_TF, LIMIT_4H)
            df1d = tv.get_ohlcv(symbol, DAILY_TF, LIMIT_1D)

            if len(df4h) < 100 or len(df1d) < 100:
                continue

            if reject_ara_arb(df1d):
                continue

            stl, trend = supertrend(df4h, ATR_PERIOD, MULTIPLIER)
            vo = volume_osc(df4h.volume, VO_FAST, VO_SLOW)
            adl = accumulation_distribution(df4h)

            if trend.iloc[-1] != 1:
                continue
            if vo.iloc[-1] < VO_MIN:
                continue
            if adl.iloc[-1] <= adl.iloc[-10]:
                continue

            score = calculate_score(df4h, df1d)
            if score < 6:
                continue

            trade = calculate_trade_levels(df4h, df1d)
            if not trade:
                continue

            signals.append({
                "Time": now_wib(),
                "Symbol": symbol,
                "Phase": "AKUMULASI_KUAT",
                "Score": score,
                **trade
            })

            time.sleep(0.3)  # rate limit safety

        except Exception as e:
            if debug:
                print(symbol, e)

    return signals

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("🔍 Scan IDX Market...")
    signals = scan_idx_market(debug=True)

    if not signals:
        print("Tidak ada setup valid.")
    else:
        df = pd.DataFrame(signals).sort_values("Score", ascending=False)
        print(df.to_string(index=False))
