# portfolio_simulator.py - Instrumented + robust feature-alignment for XGBoost prediction

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime, timedelta
from utils import config
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from strategies import TOP_STRATEGIES, find_col
from strategies_hourly import HOURLY_STRATEGIES
import pandas_ta as ta
import warnings
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)


class PortfolioSimulator:
    def __init__(self, start_capital, start_date, end_date, selected_models):
        self.start_capital = float(start_capital)
        self.start_date = pd.to_datetime(start_date, utc=True).tz_convert(
            "America/New_York"
        )
        self.end_date = pd.to_datetime(end_date, utc=True).tz_convert(
            "America/New_York"
        )
        self.selected_models = selected_models

        self.cash = float(start_capital)
        self.positions = {}
        self.portfolio_history = []
        self.trades = []
        self.num_trades = 0

        # Diagnostics container to return back to caller for debugging
        self.diagnostics = {
            "models_checked": {},
            "signals": [],
            "skipped_reasons": {},
            "errors": [],
            "summary": {"signals_above_threshold": 0, "trades_executed": 0},
        }

    def _diag_inc(self, key):
        self.diagnostics["skipped_reasons"].setdefault(key, 0)
        self.diagnostics["skipped_reasons"][key] += 1

    def load_model_and_params(self, ticker, timeframe):
        STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), timeframe)
        model_path = os.path.join(STOCK_DIR, f"{timeframe}_model.joblib")
        params_path = os.path.join(STOCK_DIR, f"{timeframe}_params.json")
        settings_path = os.path.join(STOCK_DIR, f"{timeframe}_settings.json")

        info = {"model_path": model_path, "params_path": params_path, "loaded": False}
        if not all(os.path.exists(p) for p in [model_path, params_path]):
            info["reason"] = "missing_model_or_params"
            self.diagnostics["models_checked"][f"{ticker}_{timeframe}"] = info
            self._diag_inc("missing_model_or_params")
            return None, None, None

        try:
            model = joblib.load(model_path)
        except Exception as e:
            info["reason"] = f"model_load_failed: {repr(e)}"
            self.diagnostics["models_checked"][f"{ticker}_{timeframe}"] = info
            self._diag_inc("model_load_failed")
            self.diagnostics["errors"].append(traceback.format_exc())
            return None, None, None

        try:
            with open(params_path, "r") as f:
                params = json.load(f)
        except Exception as e:
            info["reason"] = f"params_load_failed: {repr(e)}"
            self.diagnostics["models_checked"][f"{ticker}_{timeframe}"] = info
            self._diag_inc("params_load_failed")
            return None, None, None

        threshold = 0.55 if timeframe == "daily" else 0.80
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    threshold = json.load(f).get("optimal_threshold", threshold)
            except Exception:
                pass

        # detect feature names
        feat_names = None
        try:
            if hasattr(model, "get_booster"):
                try:
                    feat_names = model.get_booster().feature_names
                except Exception:
                    feat_names = getattr(model, "feature_names", None)
            if feat_names is None and hasattr(model, "feature_names_in_"):
                feat_names = list(model.feature_names_in_)
            if feat_names is None:
                feat_names = getattr(model, "feature_names", None)
            info["feature_names_detected"] = bool(feat_names)
            info["feature_count"] = len(feat_names) if feat_names else 0
        except Exception as e:
            info["feature_detection_error"] = repr(e)

        info["loaded"] = True
        self.diagnostics["models_checked"][f"{ticker}_{timeframe}"] = info
        return model, params, threshold

    def get_signal_for_ticker(self, ticker, timeframe, date, all_data):
        model, params, threshold = self.load_model_and_params(ticker, timeframe)
        if model is None:
            return 0.0, threshold

        ticker_data = all_data.get(ticker, {}).get(timeframe)
        market_data = all_data.get(config.MARKET_TICKER, {}).get(timeframe)
        if ticker_data is None or market_data is None:
            self._diag_inc("missing_market_or_ticker_data")
            return 0.0, threshold

        historical_ticker = ticker_data[ticker_data.index < date]
        if len(historical_ticker) < 200:
            self._diag_inc("insufficient_history_len")
            return 0.0, threshold

        historical_ticker_copy = historical_ticker.copy()
        historical_market_copy = (
            all_data.get(config.MARKET_TICKER, {})
            .get(timeframe)[lambda df: df.index < date]
            .copy()
        )

        feature_dict = {}
        try:
            if timeframe == "daily":
                historical_market_copy.ta.rsi(length=14, append=True)
                historical_market_copy.ta.ema(length=50, append=True)
                historical_market_copy.ta.ema(length=200, append=True)
                feature_dict["SPY_RSI"] = historical_market_copy["RSI_14"].iloc[-1]
                feature_dict["SPY_50MA_Ratio"] = (
                    historical_market_copy["Close"].iloc[-1]
                    / historical_market_copy["EMA_50"].iloc[-1]
                )
                feature_dict["SPY_200MA_Ratio"] = (
                    historical_market_copy["Close"].iloc[-1]
                    / historical_market_copy["EMA_200"].iloc[-1]
                )
                historical_ticker_copy.ta.rsi(length=14, append=True)
                historical_ticker_copy.ta.atr(length=14, append=True)
                feature_dict["RSI_14"] = historical_ticker_copy["RSI_14"].iloc[-1]
                atr_col = find_col(historical_ticker_copy, "ATRr_14")
                feature_dict["ATR_14"] = (
                    historical_ticker_copy[atr_col].iloc[-1] if atr_col else 0
                )
                for name, func in TOP_STRATEGIES.items():
                    try:
                        feature_dict[name] = func(
                            historical_ticker_copy, **params.get(name, {})
                        ).iloc[-1]
                    except Exception:
                        feature_dict[name] = np.nan
            else:  # hourly
                daily_context_data = all_data.get(ticker, {}).get("daily_context")
                if daily_context_data is None:
                    self._diag_inc("missing_daily_context_for_hourly")
                    return 0.0, threshold
                historical_market_copy.ta.rsi(length=14, append=True)
                feature_dict["SPY_RSI_Hourly"] = historical_market_copy["RSI_14"].iloc[
                    -1
                ]
                historical_ticker_copy.ta.atr(length=14, append=True)
                atr_col = find_col(historical_ticker_copy, "ATRr_14")
                feature_dict["atr"] = (
                    historical_ticker_copy[atr_col].iloc[-1] if atr_col else 0
                )
                for name, func in HOURLY_STRATEGIES.items():
                    try:
                        feature_dict[name] = func(
                            historical_ticker_copy, **params.get(name, {})
                        ).iloc[-1]
                    except Exception:
                        feature_dict[name] = np.nan
                daily_features_for_hour = daily_context_data[
                    daily_context_data.index <= date
                ].iloc[-1]
                feature_dict.update(daily_features_for_hour.to_dict())

            # Build a defensive input DataFrame
            input_df = pd.DataFrame([feature_dict]).copy()

            # Detect expected feature names from model
            feat_names = None
            try:
                if hasattr(model, "get_booster"):
                    try:
                        feat_names = model.get_booster().feature_names
                    except Exception:
                        feat_names = None
                if not feat_names and hasattr(model, "feature_names_in_"):
                    feat_names = list(model.feature_names_in_)
                if not feat_names:
                    feat_names = getattr(model, "feature_names", None)
            except Exception:
                feat_names = None

            # If the model provides expected feature names, construct a row exactly matching them.
            if feat_names:
                missing = []
                row = {}
                available_cols = list(input_df.columns)

                # Helper: fuzzy match attempt (simple heuristic)
                def find_candidate(expected):
                    expected_key = expected.lower()
                    # exact or startswith or contains
                    for c in available_cols:
                        cl = c.lower()
                        if cl == expected_key:
                            return c
                    for c in available_cols:
                        cl = c.lower()
                        if (
                            cl.startswith(expected_key.split("_")[0])
                            or expected_key in cl
                            or cl in expected_key
                        ):
                            return c
                    return None

                for f in feat_names:
                    if f in input_df.columns:
                        row[f] = input_df.iloc[0][f]
                    else:
                        candidate = find_candidate(f)
                        if candidate:
                            row[f] = input_df.iloc[0][candidate]
                        else:
                            # fill missing with zero (safe fallback)
                            row[f] = 0
                            missing.append(f)

                # Put into DataFrame in the exact order model expects
                input_df = pd.DataFrame([row], columns=list(feat_names))

                if missing:
                    # Log which features were absent and were zero-filled
                    msg = f"Missing features for {ticker}_{timeframe}: {missing}"
                    self.diagnostics["errors"].append(msg)

            else:
                # No feature names detected — use all available features (sorted) as best-effort
                input_df = input_df.reindex(sorted(input_df.columns), axis=1)

            # Coerce to numeric, fill NaN with 0 for safety
            input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Perform prediction (try predict_proba, fallback to predict)
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(input_df)[0][1])
                elif hasattr(model, "predict"):
                    prob = float(model.predict(input_df)[0])
                else:
                    self.diagnostics["errors"].append(
                        f"Model for {ticker}_{timeframe} has no predict/predict_proba"
                    )
                    return 0.0, threshold
            except Exception as e:
                # Log full traceback and fail safely (0.0)
                self.diagnostics["errors"].append(
                    f"Prediction error for {ticker}_{timeframe} on {date}: {repr(e)}\n{traceback.format_exc()}"
                )
                return 0.0, threshold

            # Record a brief signal sample in diagnostics
            self.diagnostics["signals"].append(
                {
                    "date": str(date),
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "prob": prob,
                    "threshold": threshold,
                }
            )

            return prob, threshold

        except Exception:
            self.diagnostics["errors"].append(traceback.format_exc())
            return 0.0, threshold

    def fetch_all_data(self):
        api = tradeapi.REST(
            config.API_KEY,
            config.SECRET_KEY,
            base_url="https://paper-api.alpaca.markets",
        )
        all_data, tickers_needed = {}, set(
            [m["ticker"] for m in self.selected_models] + [config.MARKET_TICKER]
        )
        timeframes_needed = set([m["timeframe"] for m in self.selected_models])
        if "hourly" in timeframes_needed:
            timeframes_needed.add("daily_context")
        fetch_start_date = self.start_date - timedelta(days=300)
        safe_end_date = self.end_date
        if self.end_date.date() >= pd.Timestamp.now(tz="America/New_York").date():
            safe_end_date = self.end_date - timedelta(days=1)
        for ticker in tickers_needed:
            all_data[ticker] = {}
            for tf_str in timeframes_needed:
                is_context = tf_str == "daily_context"
                timeframe = (
                    TimeFrame.Day
                    if (tf_str == "daily" or is_context)
                    else TimeFrame.Hour
                )
                try:
                    bars = api.get_bars(
                        ticker,
                        timeframe,
                        start=fetch_start_date.strftime("%Y-%m-%d"),
                        end=safe_end_date.strftime("%Y-%m-%d"),
                        adjustment="raw",
                    ).df
                    if not bars.empty:
                        bars = bars.tz_convert("America/New_York")
                        bars.rename(
                            columns={
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume",
                            },
                            inplace=True,
                        )
                        if is_context:
                            bars.ta.rsi(length=14, append=True)
                            bars.ta.ema(length=50, append=True)
                            bars.ta.ema(length=200, append=True)
                            bars["Daily_50MA_Ratio"] = bars["Close"] / bars["EMA_50"]
                            bars["Daily_200MA_Ratio"] = bars["Close"] / bars["EMA_200"]
                            bars.rename(
                                columns={find_col(bars, "RSI_14"): "Daily_RSI"},
                                inplace=True,
                            )
                            all_data[ticker][tf_str] = bars[
                                ["Daily_RSI", "Daily_50MA_Ratio", "Daily_200MA_Ratio"]
                            ].dropna()
                        else:
                            all_data[ticker][tf_str] = bars
                except Exception as e:
                    self.diagnostics["errors"].append(
                        f"Error fetching {tf_str} data for {ticker}: {repr(e)}"
                    )
                    all_data[ticker][tf_str] = pd.DataFrame()
        return all_data

    def run_simulation(self):
        all_data = self.fetch_all_data()
        primary_tf = (
            "daily"
            if any(m["timeframe"] == "daily" for m in self.selected_models)
            else "hourly"
        )
        first_ticker = self.selected_models[0]["ticker"]
        if all_data.get(first_ticker, {}).get(primary_tf, pd.DataFrame()).empty:
            raise ValueError("Simulation failed: Could not fetch initial market data.")
        trading_dates = all_data[first_ticker][primary_tf].index
        trading_dates = trading_dates[
            (trading_dates >= self.start_date) & (trading_dates <= self.end_date)
        ]
        if trading_dates.empty:
            raise ValueError(
                "Simulation failed: No trading dates found in the selected range."
            )

        for date in trading_dates:
            portfolio_value = self.cash
            for ticker, position in self.positions.items():
                price_data = all_data.get(ticker, {}).get(position["timeframe"])
                if price_data is not None and not price_data.empty:
                    current_prices = price_data[price_data.index <= date]
                    if not current_prices.empty:
                        portfolio_value += (
                            position["shares"] * current_prices.iloc[-1]["Close"]
                        )
            self.portfolio_history.append({"date": date, "value": portfolio_value})

            for model_config in self.selected_models:
                ticker, timeframe = model_config["ticker"], model_config["timeframe"]
                if timeframe == "hourly" and primary_tf == "daily":
                    continue

                buy_prob, threshold = self.get_signal_for_ticker(
                    ticker, timeframe, date, all_data
                )
                try:
                    price_series = all_data.get(ticker, {}).get(
                        timeframe, pd.DataFrame()
                    )
                    current_tick = (
                        price_series[price_series.index <= date].iloc[-1]
                        if not price_series.empty
                        else None
                    )
                except Exception:
                    current_tick = None

                if current_tick is None:
                    self._diag_inc("missing_current_tick")
                    continue

                if buy_prob > threshold and ticker not in self.positions:
                    self.diagnostics["summary"]["signals_above_threshold"] += 1
                    investment_amount = self.start_capital * 0.20
                    if self.cash >= investment_amount and current_tick["Open"] > 0:
                        shares = int(investment_amount / current_tick["Open"])
                        if shares > 0:
                            cost = shares * current_tick["Open"]
                            self.cash -= cost
                            self.positions[ticker] = {
                                "shares": shares,
                                "avg_price": current_tick["Open"],
                                "timeframe": timeframe,
                            }
                            self.num_trades += 1
                            self.diagnostics["summary"]["trades_executed"] += 1
                            self.trades.append(
                                {
                                    "date": date.strftime("%Y-%m-%d %H:%M"),
                                    "ticker": ticker,
                                    "action": "BUY",
                                    "shares": shares,
                                    "price": current_tick["Open"],
                                    "value": cost,
                                }
                            )

                elif ticker in self.positions and buy_prob < (threshold * 0.9):
                    if current_tick["Open"] > 0:
                        proceeds = (
                            self.positions[ticker]["shares"] * current_tick["Open"]
                        )
                        self.cash += proceeds
                        self.trades.append(
                            {
                                "date": date.strftime("%Y-%m-%d %H:%M"),
                                "ticker": ticker,
                                "action": "SELL",
                                "shares": self.positions[ticker]["shares"],
                                "price": current_tick["Open"],
                                "value": proceeds,
                            }
                        )
                        del self.positions[ticker]
                        self.num_trades += 1

        if len(self.portfolio_history) > 0:
            final_date = pd.to_datetime(self.portfolio_history[-1]["date"])
            final_value = self.cash
            for ticker, position in self.positions.items():
                final_prices = all_data.get(ticker, {}).get(
                    position["timeframe"], pd.DataFrame()
                )
                if not final_prices.empty:
                    final_value += (
                        position["shares"]
                        * final_prices[final_prices.index <= final_date].iloc[-1][
                            "Close"
                        ]
                    )
            if self.portfolio_history:
                self.portfolio_history[-1]["value"] = final_value

        results = self.calculate_metrics()
        # attach diagnostics
        results["diagnostics"] = self.diagnostics
        return results

    def calculate_metrics(self):
        if not self.portfolio_history:
            return None
        df = pd.DataFrame(self.portfolio_history).set_index("date")
        if df.empty:
            return None

        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        df = df.sort_index()
        df["value"] = (
            pd.to_numeric(df["value"], errors="coerce")
            .fillna(method="ffill")
            .fillna(self.start_capital)
        )

        final_value = float(df["value"].iloc[-1])
        total_return = (final_value / float(self.start_capital) - 1) * 100.0

        df["returns"] = df["value"].pct_change().fillna(0)
        sharpe = (
            (df["returns"].mean() / df["returns"].std()) * np.sqrt(252)
            if df["returns"].std() > 0
            else 0
        )
        cumulative = (1 + df["returns"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100 if not cumulative.empty else 0.0

        return {
            "history": df,  # keep DataFrame for app.py to call reset_index()
            "final_value": float(final_value),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else 0.0,
            "max_drawdown": float(max_drawdown) if np.isfinite(max_drawdown) else 0.0,
            "num_trades": int(self.num_trades),
            "trades": self.trades,
        }
