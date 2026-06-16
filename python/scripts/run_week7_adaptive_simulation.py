#!/usr/bin/env python3
"""Week 7 – Adaptive execution and regime-aware simulation pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "week7"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


DATASET_CONFIGS = {
    "binance_futures": {
        "volatility": 0.65,
        "trend": 0.0008,
        "latency_ms": 3.2,
        "latency_jitter": 1.0,
        "seed": 404,
    },
    "nasdaq_micro": {
        "volatility": 0.35,
        "trend": -0.0002,
        "latency_ms": 1.4,
        "latency_jitter": 0.4,
        "seed": 777,
    },
}


REGIME_PARAMS = {
    "low_vol": {
        "style": "mean_reversion",
        "lambda": 0.08,
        "order_qty": 4.0,
        "aggressiveness": 1.1,
        "risk_cap": 20.0,
    },
    "transition": {
        "style": "mean_reversion",
        "lambda": 0.15,
        "order_qty": 5.0,
        "aggressiveness": 0.9,
        "risk_cap": 25.0,
    },
    "high_vol": {
        "style": "trend",
        "lambda": 0.08,
        "order_qty": 3.0,
        "aggressiveness": 0.65,
        "risk_cap": 15.0,
    },
}


@dataclass
class SlidingWindow:
    window: int
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    def variance(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.var(self.values))

    def drift(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return (self.values[-1] - self.values[0]) / float(len(self.values) - 1)


def generate_market_series(
    config: Dict[str, float], steps: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(config["seed"]))
    price = np.zeros(steps)
    price[0] = 100.0
    vol = float(config["volatility"])
    trend = float(config["trend"])
    latency = np.zeros(steps)
    realized_vol = np.zeros(steps)
    for i in range(1, steps):
        shock = rng.normal(0.0, vol)
        price[i] = max(
            1.0,
            price[i - 1] + price[i - 1] * (trend + shock) + rng.normal(0.0, 0.15),
        )
        latency[i] = max(
            0.2,
            rng.normal(config["latency_ms"], config["latency_jitter"]),
        )
        realized_vol[i] = np.abs(price[i] - price[i - 1]) / max(price[i - 1], 1.0)
    return {
        "price": price,
        "latency": latency,
        "realized_vol": realized_vol,
    }


def _kmeans_1d(
    values: np.ndarray, clusters: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = values.reshape(-1)
    count = values.shape[0]
    if count < clusters:
        clusters = count
    unique = np.unique(values)
    if unique.shape[0] < clusters:
        clusters = unique.shape[0]
    if clusters == 0:
        return np.zeros(values.shape[0], dtype=int), np.zeros(1)
    idx = rng.choice(count, size=clusters, replace=False)
    centres = values[idx]
    for _ in range(60):
        distances = np.abs(values.reshape(-1, 1) - centres.reshape(1, -1))
        labels = np.argmin(distances, axis=1)
        new_centres = centres.copy()
        for i in range(clusters):
            mask = labels == i
            if np.any(mask):
                new_centres[i] = values[mask].mean()
        if np.allclose(new_centres, centres):
            break
        centres = new_centres
    return labels, centres


def classify_regimes(realized_vol: np.ndarray, window: int, seed: int) -> np.ndarray:
    rolling = (
        pd.Series(realized_vol).rolling(window=window, min_periods=1).mean().to_numpy()
    )
    labels, centres = _kmeans_1d(rolling, 3, seed)
    order = np.argsort(centres)
    mapping = {order[0]: "low_vol", order[1]: "transition", order[2]: "high_vol"}
    mapped = np.vectorize(mapping.get)(labels)
    return mapped


@dataclass
class AdaptiveStrategy:
    confirmation_steps: int = 5
    latency_window: int = 40
    pnl_window: int = 60
    vol_window: int = 30

    def __post_init__(self) -> None:
        self.position = 0.0
        self.cash = 0.0
        self.prev_price = 100.0
        self.current_regime = "low_vol"
        self.pending_regime = "low_vol"
        self.pending_since = 0
        self.adaptation_lags: List[int] = []
        self.latency_stats = SlidingWindow(self.latency_window)
        self.pnl_stats = SlidingWindow(self.pnl_window)
        self.vol_stats = SlidingWindow(self.vol_window)
        self.ema = 100.0
        self.momentum = 0.0
        self.turnover = 0.0
        self.trade_attempts = 0
        self.trades = 0
        self.records: List[Dict[str, float | str]] = []

    def step(
        self,
        idx: int,
        price: float,
        detected_regime: str,
        latency_ms: float,
        short_volatility: float,
    ) -> None:
        if detected_regime != self.pending_regime:
            self.pending_regime = detected_regime
            self.pending_since = idx
        if (
            self.pending_regime != self.current_regime
            and idx - self.pending_since >= self.confirmation_steps
        ):
            self.current_regime = self.pending_regime
            self.adaptation_lags.append(idx - self.pending_since)

        params = REGIME_PARAMS[self.current_regime]
        self.latency_stats.add(latency_ms)
        self.vol_stats.add(short_volatility)

        # PnL update before trading
        equity_before = self.cash + self.position * price
        self.pnl_stats.add(equity_before)

        target = self._target_position(price, params)
        max_step = params["order_qty"]
        delta = np.clip(target - self.position, -max_step, max_step)
        risk_cap = params["risk_cap"]
        delta = np.clip(delta, -risk_cap - self.position, risk_cap - self.position)

        latency_penalty = self.latency_stats.variance() / (
            50.0 + self.latency_stats.variance()
        )
        pnl_drift = self.pnl_stats.drift()
        pnl_boost = pnl_drift / (1_000.0 + abs(pnl_drift))
        vol_penalty = self.vol_stats.mean() / (0.8 + self.vol_stats.mean())
        aggressiveness = params["aggressiveness"]
        aggressiveness *= 1.0 - latency_penalty
        aggressiveness *= 1.0 - vol_penalty
        aggressiveness += pnl_boost
        aggressiveness = float(np.clip(aggressiveness, 0.2, 2.0))

        self.trade_attempts += 1
        fill_prob = np.exp(-latency_ms / 6.0) * (1.0 - 0.5 * vol_penalty)
        fill_prob = np.clip(fill_prob, 0.05, 1.0)
        rng = np.random.default_rng(idx + 13)
        executed = 0.0
        if rng.random() < fill_prob and abs(delta) > 1e-6:
            slip = np.sign(delta) * aggressiveness * 0.0006 * price
            exec_price = price + slip
            self.cash -= exec_price * delta
            self.position += delta
            executed = delta
            self.trades += 1
            self.turnover += abs(delta)

        equity_after = self.cash + self.position * price
        pnl_delta = equity_after - equity_before
        self.records.append(
            {
                "step": idx,
                "price": price,
                "regime_signal": detected_regime,
                "regime_active": self.current_regime,
                "latency_ms": latency_ms,
                "short_vol": short_volatility,
                "aggressiveness": aggressiveness,
                "pnl": equity_after,
                "pnl_delta": pnl_delta,
                "executed": executed,
                "fill_prob": fill_prob,
            }
        )
        self.prev_price = price

    def _target_position(self, price: float, params: Dict[str, float | str]) -> float:
        lam = float(params["lambda"])
        if params["style"] == "mean_reversion":
            self.ema = (1.0 - lam) * self.ema + lam * price
            signal = price - self.ema
            return -signal * params["risk_cap"] / 2.0
        self.momentum = (1.0 - lam) * self.momentum + lam * (price - self.prev_price)
        signal = np.tanh(self.momentum / 0.2)
        return signal * params["risk_cap"]

    def summary(self, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(self.records)
        regime_groups = []
        for regime, group in df.groupby("regime_active"):
            pnl_series = group["pnl"].to_numpy()
            returns = np.diff(pnl_series, prepend=pnl_series[0])
            vol = returns.std() if returns.size else 0.0
            sharpe = (returns.mean() / vol * np.sqrt(len(returns))) if vol > 0 else 0.0
            drawdowns = (
                np.maximum.accumulate(pnl_series) - pnl_series
                if pnl_series.size
                else np.array([0.0])
            )
            max_dd = float(drawdowns.max()) if drawdowns.size else 0.0
            regime_groups.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "realized_pnl": float(group["pnl"].iloc[-1] - group["pnl"].iloc[0]),
                    "return_volatility": float(vol),
                    "sharpe": float(sharpe),
                    "max_drawdown": float(max_dd),
                    "avg_latency_ms": float(group["latency_ms"].mean()),
                    "fill_rate": float(self.trades / max(self.trade_attempts, 1)),
                    "adaptation_lag": (
                        float(np.mean(self.adaptation_lags))
                        if self.adaptation_lags
                        else 0.0
                    ),
                }
            )
        lag_series = pd.Series(self.adaptation_lags or [0.0])
        meta = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "avg_lag_steps": float(lag_series.mean()),
                    "p95_lag_steps": float(lag_series.quantile(0.95)),
                    "total_trades": float(self.trades),
                    "fill_rate": float(self.trades / max(self.trade_attempts, 1)),
                }
            ]
        )
        return pd.DataFrame(regime_groups), meta


def run_dataset(
    name: str, steps: int, window: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DATASET_CONFIGS[name]
    series = generate_market_series(cfg, steps)
    regimes = classify_regimes(series["realized_vol"], window, cfg["seed"])
    strategy = AdaptiveStrategy()
    for idx in range(steps):
        strategy.step(
            idx,
            float(series["price"][idx]),
            str(regimes[idx]),
            float(series["latency"][idx]),
            float(series["realized_vol"][idx]),
        )
    detail_path = RESULTS_DIR / f"{name}_regime_trace.csv"
    pd.DataFrame(strategy.records).assign(dataset=name).to_csv(detail_path, index=False)
    return strategy.summary(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=2000, help="Simulation steps per dataset"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=40,
        help="Rolling window for regime classification",
    )
    args = parser.parse_args()

    summaries: List[pd.DataFrame] = []
    diagnostics: List[pd.DataFrame] = []
    for name in DATASET_CONFIGS:
        regime_df, meta_df = run_dataset(name, steps=args.steps, window=args.window)
        summaries.append(regime_df)
        diagnostics.append(meta_df)

    result_df = pd.concat(summaries, ignore_index=True)
    result_path = RESULTS_DIR / "adaptive_sim_results.csv"
    result_df.to_csv(result_path, index=False)

    diag_path = RESULTS_DIR / "adaptive_meta.json"
    diag_payload = pd.concat(diagnostics, ignore_index=True).to_dict(orient="records")
    diag_path.write_text(json.dumps(diag_payload, indent=2))

    print(f"[info] Adaptive results written to {result_path}")
    print(f"[info] Diagnostics written to {diag_path}")


if __name__ == "__main__":
    main()
