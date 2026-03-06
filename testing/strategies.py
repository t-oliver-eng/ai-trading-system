# strategies.py

import pandas as pd
import pandas_ta as ta
import warnings


def find_col(df, search_str):
    """Helper function to find the first column in a dataframe that contains a string."""
    try:
        return next(col for col in df.columns if search_str in col)
    except StopIteration:
        return None


# --- Original Set of Strategies ---


def macd_strategy(df, fast=12, slow=26, signal=9):
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    macd_col, signal_col = find_col(df, f"MACD_{fast}_{slow}_{signal}"), find_col(
        df, f"MACDs_{fast}_{slow}_{signal}"
    )
    if not macd_col or not signal_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[macd_col] > df[signal_col], "signal"] = 1
    df.loc[df[macd_col] < df[signal_col], "signal"] = -1
    return df["signal"]


def rsi_strategy(df, length=14, upper=70, lower=30):
    df.ta.rsi(length=length, append=True)
    rsi_col = find_col(df, f"RSI_{length}")
    if not rsi_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[rsi_col] > upper, "signal"] = -1
    df.loc[df[rsi_col] < lower, "signal"] = 1
    return df["signal"]


def bollinger_bands_strategy(df, length=20, std=2):
    df.ta.bbands(length=length, std=std, append=True)
    bbl_col, bbu_col = find_col(df, f"BBL_{length}_{std}"), find_col(
        df, f"BBU_{length}_{std}"
    )
    if not bbl_col or not bbu_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df["Close"] < df[bbl_col], "signal"] = 1
    df.loc[df["Close"] > df[bbu_col], "signal"] = -1
    return df["signal"]


def ema_cross_strategy(df, fast=50, slow=200):
    df.ta.ema(length=fast, append=True)
    df.ta.ema(length=slow, append=True)
    fast_ema_col, slow_ema_col = f"EMA_{fast}", f"EMA_{slow}"
    if fast_ema_col not in df.columns or slow_ema_col not in df.columns:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[fast_ema_col] > df[slow_ema_col], "signal"] = 1
    df.loc[df[fast_ema_col] < df[slow_ema_col], "signal"] = -1
    return df["signal"]


def stochastic_oscillator_strategy(df, k=14, d=3, upper=80, lower=20):
    df.ta.stoch(k=k, d=d, append=True)
    stoch_k_col = find_col(df, f"STOCHk_{k}_{d}")
    if not stoch_k_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[stoch_k_col] > upper, "signal"] = -1
    df.loc[df[stoch_k_col] < lower, "signal"] = 1
    return df["signal"]


def vwap_strategy(df):
    """Volume-Weighted Average Price"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df.ta.vwap(append=True)

    vwap_col = find_col(df, "VWAP")
    if not vwap_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df["Close"] > df[vwap_col], "signal"] = 1
    df.loc[df["Close"] < df[vwap_col], "signal"] = -1
    return df["signal"]


def obv_strategy(df, length=20):
    df.ta.obv(append=True)
    if "OBV" not in df.columns:
        return pd.Series(0, index=df.index)
    obv_ema = ta.ema(df["OBV"], length=length)
    df["signal"] = 0
    df.loc[df["OBV"] > obv_ema, "signal"] = 1
    df.loc[df["OBV"] < obv_ema, "signal"] = -1
    return df["signal"]


def donchian_channels_strategy(df, length=20):
    df.ta.donchian(lower_length=length, upper_length=length, append=True)
    upper_col, lower_col = find_col(df, f"DCU_{length}"), find_col(df, f"DCL_{length}")
    if not upper_col or not lower_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df["Close"] > df[upper_col].shift(1), "signal"] = 1
    df.loc[df["Close"] < df[lower_col].shift(1), "signal"] = -1
    return df["signal"]


def keltner_channels_strategy(df, length=20, multiplier=2):
    df.ta.kc(length=length, scalar=multiplier, append=True)
    kcu_col, kcl_col = find_col(df, f"KCUe_{length}"), find_col(df, f"KCLe_{length}")
    if not kcu_col or not kcl_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df["Close"] > df[kcu_col], "signal"] = 1
    df.loc[df["Close"] < df[kcl_col], "signal"] = -1
    return df["signal"]


def rate_of_change_strategy(df, length=12):
    df.ta.roc(length=length, append=True)
    roc_col = find_col(df, f"ROC_{length}")
    if not roc_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[roc_col] > 0, "signal"] = 1
    df.loc[df[roc_col] < 0, "signal"] = -1
    return df["signal"]


def chaikin_money_flow_strategy(df, length=20):
    df.ta.cmf(length=length, append=True)
    cmf_col = find_col(df, f"CMF_{length}")
    if not cmf_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[cmf_col] > 0, "signal"] = 1
    df.loc[df[cmf_col] < 0, "signal"] = -1
    return df["signal"]


def ichimoku_cloud_strategy(df, tenkan=9, kijun=26):
    try:
        ichimoku_df, _ = df.ta.ichimoku(tenkan=tenkan, kijun=kijun, append=True)
        kijun_col = find_col(ichimoku_df, f"IKS_{kijun}")
        if kijun_col not in ichimoku_df.columns:
            return pd.Series(0, index=df.index)
        df["signal"] = 0
        df.loc[df["Close"] > ichimoku_df[kijun_col], "signal"] = 1
        df.loc[df["Close"] < ichimoku_df[kijun_col], "signal"] = -1
        return df["signal"]
    except Exception as e:
        # Return neutral signals if pandas-ta fails (common with Ichimoku date math)
        return pd.Series(0, index=df.index)


def awesome_oscillator_strategy(df, fast=5, slow=34):
    df.ta.ao(fast=fast, slow=slow, append=True)
    ao_col = find_col(df, f"AO_{fast}_{slow}")
    if not ao_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[ao_col] > 0, "signal"] = 1
    df.loc[df[ao_col] < 0, "signal"] = -1
    return df["signal"]


def adx_strategy(df, length=14, threshold=25):
    df.ta.adx(length=length, append=True)
    adx_col = find_col(df, f"ADX_{length}")
    if not adx_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[adx_col] > threshold, "signal"] = 1
    return df["signal"]


# --- NEW STRATEGIES ---
def coppock_curve_strategy(df, length=10, fast=11, slow=14):
    """Coppock Curve - Zero Line Cross"""
    df.ta.coppock(length=length, fast=fast, slow=slow, append=True)
    cc_col = find_col(df, f"COPC_{length}")
    if not cc_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[cc_col] > 0, "signal"] = 1
    df.loc[df[cc_col] < 0, "signal"] = -1
    return df["signal"]


def chande_momentum_oscillator_strategy(df, length=9, upper=50, lower=-50):
    """Chande Momentum Oscillator (CMO)"""
    df.ta.cmo(length=length, append=True)
    cmo_col = find_col(df, f"CMO_{length}")
    if not cmo_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[cmo_col] > upper, "signal"] = -1
    df.loc[df[cmo_col] < lower, "signal"] = 1
    return df["signal"]


def trix_strategy(df, length=30, signal=9):
    """TRIX - Triple Exponentially Smoothed Moving Average"""
    df.ta.trix(length=length, signal=signal, append=True)
    trix_col = find_col(df, f"TRIX_{length}_{signal}")
    trix_signal_col = find_col(df, f"TRIXs_{length}_{signal}")
    if not trix_col or not trix_signal_col:
        return pd.Series(0, index=df.index)
    df["signal"] = 0
    df.loc[df[trix_col] > df[trix_signal_col], "signal"] = 1
    df.loc[df[trix_col] < df[trix_signal_col], "signal"] = -1
    return df["signal"]


ALL_STRATEGIES = {
    "MACD": macd_strategy,
    "RSI": rsi_strategy,
    "BollingerBands": bollinger_bands_strategy,
    "EMACross": ema_cross_strategy,
    "Stochastic": stochastic_oscillator_strategy,
    "VWAP": vwap_strategy,
    "OBV": obv_strategy,
    "Donchian": donchian_channels_strategy,
    "KeltnerChannels": keltner_channels_strategy,
    "ROC": rate_of_change_strategy,
    "CMF": chaikin_money_flow_strategy,
    "Ichimoku": ichimoku_cloud_strategy,
    "AwesomeOscillator": awesome_oscillator_strategy,
    "ADX": adx_strategy,
    "CoppockCurve": coppock_curve_strategy,
    "CMO": chande_momentum_oscillator_strategy,
    "TRIX": trix_strategy,
}

TOP_STRATEGIES = ALL_STRATEGIES
