# config.py

import os

# --- Alpaca API Credentials ---
API_KEY = "PKUZTMVCVLU75D88AFP7"
SECRET_KEY = "oo6rYGxzRoZ8m3TC9aA3Jn6vzx2EjYnquUcGyCzZ"

# --- Authentication Settings ---
# Set to False to disable login requirement
REQUIRE_LOGIN = False

# Admin credentials - CHANGE THESE!
ADMIN_USERNAME = "tommi123admin"
ADMIN_PASSWORD = "supersecurepassword739!"

# Flask secret key for sessions (change this to a random string)
FLASK_SECRET_KEY = "ff638bf0ef3e07806b982d5d4078bcc0b4587b8f4e9555a15cf094a999a8d365"

# --- Data & Backtesting Configuration ---
TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "MTC",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "JPM",
    "V",
    "PG",
    "JNJ",
    "XOM",
    "AMD",
    "INTC",
    "NFLX",
    "DIS",
    "QCOM",
    "CSCO",
    "AVGO",
    "ORCL",
    "SAP",
    "IBM",
    "ADBE",
    "BAC",
]
MARKET_TICKER = "SPY"
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

# --- File Paths ---
DATA_DIR = "data"
STOCKS_DIR = "data/historical/stocks"
GLOBAL_OPTIMIZED_PARAMS_PATH = os.path.join("models", "optimized_strategy_params.json")
