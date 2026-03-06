# strategies_hourly.py - UPDATED: Replaced slow Heikin Ashi implementation

import pandas as pd
import pandas_ta as ta
import numpy as np


def find_col(df, search_str):
    try:
        return next(col for col in df.columns if search_str in col)
    except StopIteration:
        return None


# --- Original Hourly Strategies ---
def hourly_ema_cross(df, fast=12, slow=24):
    df.ta.ema(length=fast, append=True)
    df.ta.ema(length=slow, append=True)
    fast_ema, slow_ema = f"EMA_{fast}", f"EMA_{slow}"
    df["signal"] = 0
    if fast_ema in df.columns and slow_ema in df.columns:
        df.loc[df[fast_ema] > df[slow_ema], "signal"] = 1
        df.loc[df[fast_ema] < df[slow_ema], "signal"] = -1
    return df["signal"]


def hourly_rsi(df, length=7, upper=75, lower=25):
    rsi_col = f"RSI_{length}"
    df.ta.rsi(length=length, append=True)
    df["signal"] = 0
    if rsi_col in df.columns:
        df.loc[df[rsi_col] > upper, "signal"] = -1
        df.loc[df[rsi_col] < lower, "signal"] = 1
    return df["signal"]


def volume_anomaly(df, length=20, threshold=2.5):
    volume_sma = df["Volume"].rolling(window=length).mean()
    df["signal"] = 0
    df.loc[df["Volume"] > (volume_sma * threshold), "signal"] = 1
    return df["signal"]


def hourly_bollinger(df, length=20, std=2.0):
    df.ta.bbands(length=length, std=std, append=True)
    bbl_col, bbu_col = find_col(df, f"BBL_{length}_{std}"), find_col(
        df, f"BBU_{length}_{std}"
    )
    df["signal"] = 0
    if bbl_col in df.columns and bbu_col in df.columns:
        df.loc[df["Close"] < df[bbl_col], "signal"] = 1
        df.loc[df["Close"] > df[bbu_col], "signal"] = -1
    return df["signal"]


def hourly_stochastic(df, k=14, d=3, upper=80, lower=20):
    df.ta.stoch(k=k, d=d, append=True)
    stoch_k_col = find_col(df, f"STOCHk_{k}_{d}")
    df["signal"] = 0
    if stoch_k_col in df.columns:
        df.loc[df[stoch_k_col] > upper, "signal"] = -1
        df.loc[df[stoch_k_col] < lower, "signal"] = 1
    return df["signal"]


def hourly_macd(df, fast=12, slow=26, signal=9):
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    macd_col, signal_col = find_col(df, f"MACD_{fast}_{slow}_{signal}"), find_col(
        df, f"MACDs_{fast}_{slow}_{signal}"
    )
    df["signal"] = 0
    if macd_col in df.columns and signal_col in df.columns:
        df.loc[df[macd_col] > df[signal_col], "signal"] = 1
        df.loc[df[macd_col] < df[signal_col], "signal"] = -1
    return df["signal"]


def hourly_cmf(df, length=20):
    df.ta.cmf(length=length, append=True)
    cmf_col = find_col(df, f"CMF_{length}")
    df["signal"] = 0
    if cmf_col in df.columns:
        df.loc[df[cmf_col] > 0, "signal"] = 1
        df.loc[df[cmf_col] < 0, "signal"] = -1
    return df["signal"]


def hourly_roc(df, length=12):
    df.ta.roc(length=length, append=True)
    roc_col = find_col(df, f"ROC_{length}")
    df["signal"] = 0
    if roc_col in df.columns:
        df.loc[df[roc_col] > 0, "signal"] = 1
        df.loc[df[roc_col] < 0, "signal"] = -1
    return df["signal"]


def hourly_adx(df, length=14, threshold=25):
    df.ta.adx(length=length, append=True)
    adx_col = find_col(df, f"ADX_{length}")
    df["signal"] = 0
    if adx_col in df.columns:
        df.loc[df[adx_col] > threshold, "signal"] = 1
    return df["signal"]


def hourly_psar(df, af=0.02, max_af=0.2):
    df.ta.psar(af=af, max=max_af, append=True)
    long_col, short_col = find_col(df, "PSARl_"), find_col(df, "PSARs_")
    df["signal"] = 0
    if long_col in df.columns and short_col in df.columns:
        df.loc[df[long_col].notna(), "signal"] = 1
        df.loc[df[short_col].notna(), "signal"] = -1
    return df["signal"]


def hourly_tema(df, length=10):
    df.ta.tema(length=length, append=True)
    tema_col = find_col(df, f"TEMA_{length}")
    df["signal"] = 0
    if tema_col in df.columns:
        df.loc[df["Close"] > df[tema_col], "signal"] = 1
        df.loc[df["Close"] < df[tema_col], "signal"] = -1
    return df["signal"]


def hourly_long_ema_cross(df, fast=50, slow=200):
    df.ta.ema(length=fast, append=True)
    df.ta.ema(length=slow, append=True)
    fast_ema, slow_ema = f"EMA_{fast}", f"EMA_{slow}"
    df["signal"] = 0
    if fast_ema in df.columns and slow_ema in df.columns:
        df.loc[df[fast_ema] > df[slow_ema], "signal"] = 1
        df.loc[df[fast_ema] < df[slow_ema], "signal"] = -1
    return df["signal"]


# --- NEW HOURLY STRATEGIES ---
def heikin_ashi_signal(df):
    """Generates a signal based on Heikin Ashi candle patterns. (High-performance version)"""
    # FIX: This is a custom, high-performance implementation that avoids pandas_ta's slow version.
    ha_df = df.copy()

    # Calculate HA Close
    ha_df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # Calculate HA Open
    # Initialize first HA Open
    ha_df["HA_Open"] = 0.0
    ha_df.iloc[0, ha_df.columns.get_loc("HA_Open")] = (
        df["Open"].iloc[0] + df["Close"].iloc[0]
    ) / 2
    # Use a faster loop for the iterative calculation
    ha_open_values = ha_df["HA_Open"].values.copy()
    ha_close_values = ha_df["HA_Close"].values
    for i in range(1, len(df)):
        ha_open_values[i] = (ha_open_values[i - 1] + ha_close_values[i - 1]) / 2
    ha_df["HA_Open"] = ha_open_values

    # Calculate HA High and Low
    ha_df["HA_High"] = ha_df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha_df["HA_Low"] = ha_df[["Low", "HA_Open", "HA_Close"]].min(axis=1)

    # Conditions for signals
    strong_bullish = (ha_df["HA_Close"] > ha_df["HA_Open"]) & (
        ha_df["HA_Open"] == ha_df["HA_Low"]
    )
    strong_bearish = (ha_df["HA_Close"] < ha_df["HA_Open"]) & (
        ha_df["HA_Open"] == ha_df["HA_High"]
    )

    # Use numpy.select for efficient conditional assignment
    df["signal"] = np.select([strong_bullish, strong_bearish], [1, -1], default=0)
    return df["signal"]


def rvi_signal(df, length=14, signal=4):
    """Relative Vigor Index (RVI) Signal Cross"""
    df.ta.rvi(length=length, signal=signal, append=True)
    rvi_col = find_col(df, f"RVI_{length}_{signal}")
    rvis_col = find_col(df, f"RVIs_{length}_{signal}")
    if not rvi_col or not rvis_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[rvi_col] > df[rvis_col], "signal"] = 1
    df.loc[df[rvi_col] < df[rvis_col], "signal"] = -1
    return df["signal"]


def fisher_transform_signal(df, length=9):
    """Fisher Transform Signal Cross"""
    df.ta.fisher(length=length, append=True)
    fisher_col = find_col(df, f"FISHERT_{length}")
    fishers_col = find_col(df, f"FISHERTs_{length}")
    if not fisher_col or not fishers_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[fisher_col] > df[fishers_col], "signal"] = 1
    df.loc[df[fisher_col] < df[fishers_col], "signal"] = -1
    return df["signal"]


HOURLY_STRATEGIES = {
    "HourlyEMACross": hourly_ema_cross,
    "HourlyRSI": hourly_rsi,
    "VolumeAnomaly": volume_anomaly,
    "HourlyBollinger": hourly_bollinger,
    "HourlyStochastic": hourly_stochastic,
    "HourlyMACD": hourly_macd,
    "HourlyCMF": hourly_cmf,
    "HourlyROC": hourly_roc,
    "HourlyADX": hourly_adx,
    "HourlyPSAR": hourly_psar,
    "HourlyTEMA": hourly_tema,
    "HourlyLongEMACross": hourly_long_ema_cross,
    "HeikinAshi": heikin_ashi_signal,
    "RVI": rvi_signal,
    "FisherTransform": fisher_transform_signal,
}
