

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Live Buy Signals", layout="wide")

st.markdown("### 📈 Final Results:")

# Load Booster Model
st.write("📜 Loading model...")
model = lgb.Booster(model_file="lightgbm_model_converted.txt")
st.write("✅ Model loaded.")

# Load screened tickers
screened_df = pd.read_csv("screened_tickers.csv")
tickers = screened_df['Ticker'].unique().tolist()

st.write("Screened Tickers:")
st.write(tickers)

# Feature Extraction
def get_live_features(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="1m", progress=False)
        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")

        df_today = df[df.index.date == datetime.now().date()]
        df_yesterday = df[df.index.date < datetime.now().date()]

        premarket_df = df_today[df_today.index.time < datetime.strptime("09:30", "%H:%M").time()]
        intraday_df = df_today[df_today.index.time >= datetime.strptime("09:30", "%H:%M").time()]

        if intraday_df.empty or df_yesterday.empty:
            raise ValueError("Not enough data for intraday or previous day")

        premarket_open = premarket_df['Open'].iloc[0] if not premarket_df.empty else np.nan
        premarket_close = premarket_df['Close'].iloc[-1] if not premarket_df.empty else np.nan
        premarket_change = (premarket_close - premarket_open) / premarket_open if premarket_open > 0 else np.nan
        volume_spike_ratio = premarket_df['Volume'].sum() / (premarket_df['Volume'].mean() * len(premarket_df)) if not premarket_df.empty else np.nan

        open_price = intraday_df['Open'].iloc[0]
        prev_close_price = df_yesterday['Close'].iloc[-1]

        features = {
            'premarket_change': premarket_change,
            'open_vs_premarket': (open_price - premarket_close) / premarket_close if premarket_close > 0 else np.nan,
            'volume_spike_ratio': volume_spike_ratio,
            'pct_change': (open_price - prev_close_price) / prev_close_price,
            'closed_up': int(open_price > prev_close_price)
        }

        return pd.DataFrame([features])

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {ticker}: {e}")

results = []

for ticker in tickers:
    try:
        X = get_live_features(ticker)
        pred_proba = model.predict(X.values)[0]
        buy_signal = "✅ Buy" if pred_proba > 0.9761 else "❌ No Buy"
        results.append({"Ticker": ticker, "Buy Signal": buy_signal, "Probability": round(pred_proba, 3)})
    except Exception as e:
        results.append({"Ticker": ticker, "Buy Signal": "⚠️ Error", "Probability": str(e)})

st.dataframe(pd.DataFrame(results))

# Auto-refresh every 2 minutes
st.experimental_rerun() if datetime.now().second == 0 and datetime.now().minute % 2 == 0 else None

