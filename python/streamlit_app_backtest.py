"""Streamlit prototype for interactive strategy diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    RiskConfig,
    RiskEngine,
    load_lobster_csv,
    replay_from_lobster,
)
from backtester.order_book import load_order_book
from strategies import MarketMakingConfig, MarketMakingStrategy

st.set_page_config(page_title="HFT Backtester", layout="wide")
st.title("Hawkes-driven HFT Backtester")

uploaded = st.file_uploader("LOBSTER message CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a LOBSTER message file to begin.")
    st.stop()

symbol = st.text_input("Symbol", "BTCUSDT")
spread_ticks = st.slider("Spread (ticks)", 1, 10, 2)
quote_size = st.number_input("Quote size", min_value=1.0, value=10.0)
inventory_skew = st.slider("Inventory skew", -1.0, 1.0, 0.0, 0.1)
update_interval_ms = st.slider("Update interval (ms)", 1, 500, 25)
risk_limit = st.number_input("Risk limit (units)", min_value=10.0, value=500.0)
run_button = st.button("Run backtest")

if not run_button:
    st.stop()

messages_path = Path("docs/images/backtests/uploaded_messages.csv")
messages_path.parent.mkdir(parents=True, exist_ok=True)
messages_path.write_bytes(uploaded.getbuffer())
messages = list(load_lobster_csv(messages_path, symbol))
replay = replay_from_lobster(messages)

book = load_order_book(depth=5)
log_path = Path("docs/images/backtests/streamlit_run.jsonl")
metrics = MetricsLogger(json_path=log_path)
risk_engine = RiskEngine(
    RiskConfig(symbol=symbol, max_long=risk_limit, max_short=-risk_limit)
)
strategy = MarketMakingStrategy(
    MarketMakingConfig(
        spread_ticks=spread_ticks,
        quote_size=quote_size,
        inventory_skew=inventory_skew,
        update_interval_ns=int(update_interval_ms * 1_000_000),
    ),
    risk_engine=risk_engine,
)

backtester = Backtester(
    config=BacktesterConfig(symbol=symbol),
    limit_book=book,
    metrics_logger=metrics,
    risk_engine=risk_engine,
    strategy=strategy,
)
backtester.run(replay)
metrics.close()

st.success(f"Run complete. Logs saved to {log_path}")

records = []
with log_path.open("r", encoding="utf-8") as fh:
    for line in fh:
        records.append(json.loads(line))

st.write("Event head", records[:5])

snapshots = [row for row in records if row["event_type"] == "snapshot"]
if snapshots:
    mids = []
    for row in snapshots:
        bid = row["payload"].get("best_bid")
        ask = row["payload"].get("best_ask")
        if bid is None or ask is None:
            mids.append(0.0)
        else:
            mids.append((bid + ask) / 2.0)
    mid_df = pd.DataFrame(
        {
            "timestamp_ns": [row["timestamp_ns"] for row in snapshots],
            "mid": mids,
            "imbalance": [
                row["payload"].get("imbalance", 0.0) or 0.0 for row in snapshots
            ],
        }
    ).set_index("timestamp_ns")
    st.line_chart(mid_df)
