#!/usr/bin/env python3
"""Week 7 – PnL decomposition, risk diagnostics, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "week7"
PLOTS_DIR = RESULTS_DIR / "plots" / "pnl_risk"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import os

MPL_CACHE = RESULTS_DIR / "mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _compute_drawdown(returns: np.ndarray) -> Tuple[float, np.ndarray]:
    if returns.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return float(np.max(drawdown, initial=0.0)), cumulative


def _var_cvar(sample: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    if sample.size == 0:
        return 0.0, 0.0
    var_quantile = float(np.quantile(sample, 1.0 - alpha))
    tail = sample[sample <= var_quantile]
    if tail.size == 0:
        return var_quantile, var_quantile
    return var_quantile, float(tail.mean())


def _sortino(returns: np.ndarray, target: float = 0.0) -> float:
    downside = returns[returns < target] - target
    if downside.size == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 1e-9:
        return 0.0
    mean_excess = float(np.mean(returns - target))
    return mean_excess / downside_std


def compute_risk_metrics(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    metrics: List[Dict[str, float]] = []
    cumulative_paths: Dict[str, np.ndarray] = {}
    grouped = df.groupby("strategy")

    for strategy, group in grouped:
        returns = group["realized_pnl"].to_numpy(dtype=float)
        mean_pnl = float(np.mean(returns)) if returns.size else 0.0
        vol = float(np.std(returns, ddof=0)) if returns.size else 0.0
        sharpe = mean_pnl / vol if vol > 1e-9 else 0.0
        sortino = _sortino(returns)
        max_dd, cumulative = _compute_drawdown(returns)
        var95, cvar95 = _var_cvar(returns)
        metrics.append(
            {
                "strategy": strategy,
                "observations": int(returns.size),
                "mean_realized_pnl": mean_pnl,
                "var_95": var95,
                "cvar_95": cvar95,
                "realized_volatility": vol,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "sortino": sortino,
            }
        )
        cumulative_paths[strategy] = cumulative
    metrics_df = pd.DataFrame(metrics).sort_values("strategy").reset_index(drop=True)
    return metrics_df, cumulative_paths


def _plot_distributions(df: pd.DataFrame, output_path: Path) -> None:
    strategies = sorted(df["strategy"].unique())
    fig, axes = plt.subplots(
        1, len(strategies), figsize=(6 * len(strategies), 4), sharey=True
    )
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        sample = df.loc[df["strategy"] == strategy, "realized_pnl"].to_numpy()
        ax.hist(sample, bins=30, alpha=0.85, color="#1b9e77", edgecolor="black")
        ax.set_title(f"{strategy} PnL")
        ax.set_xlabel("Realized PnL")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_cumulative(
    cumulative_paths: Dict[str, np.ndarray], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, path in cumulative_paths.items():
        if path.size == 0:
            continue
        ax.plot(path, label=strategy)
    ax.set_title("Cumulative PnL (synthetic day ordering)")
    ax.set_xlabel("Simulation bucket")
    ax.set_ylabel("Cumulative PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_risk_return(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"mean_reversion": "#d95f02", "trend_following": "#1b9e77"}
    for strategy, group in df.groupby("strategy"):
        ax.scatter(
            group["return_volatility"],
            group["realized_pnl"],
            s=(group["inventory_volatility"] + 1.0) * 4.0,
            alpha=0.6,
            label=strategy,
            color=colors.get(strategy, "#7570b3"),
            edgecolor="white",
        )
    ax.set_title("Risk vs Return")
    ax.set_xlabel("Return volatility")
    ax.set_ylabel("Realized PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _validate_best_configs(df: pd.DataFrame, configs_path: Path) -> None:
    config_data = json.loads(configs_path.read_text(encoding="utf-8"))
    for cfg in config_data:
        mask = (
            (df["strategy"] == cfg["strategy"])
            & np.isclose(df["lambda"], cfg["lambda"])
            & np.isclose(df["order_size"], cfg["order_size"])
            & np.isclose(df["aggressiveness"], cfg["aggressiveness"])
            & np.isclose(df["latency_sensitivity"], cfg["latency_sensitivity"])
            & np.isclose(df["risk_cap"], cfg["risk_cap"])
        )
        subset = df.loc[mask, "realized_pnl"]
        if subset.empty:
            raise ValueError(f"No calibration rows found for config {cfg}")
        mean_pnl = float(subset.mean())
        if abs(mean_pnl - cfg["realized_pnl"]) > 1e-6:
            raise ValueError(
                f"Pnl mismatch for {cfg['strategy']}: expected {cfg['realized_pnl']}, found {mean_pnl}"
            )


def run_analytics(input_path: Path) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No data found in {input_path}")
    metrics_df, cumulative_paths = compute_risk_metrics(df)
    metrics_path = RESULTS_DIR / "pnl_risk_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    _plot_distributions(df, PLOTS_DIR / "pnl_histograms.png")
    _plot_cumulative(cumulative_paths, PLOTS_DIR / "cumulative_pnl.png")
    _plot_risk_return(df, PLOTS_DIR / "risk_return_scatter.png")

    best_config_path = RESULTS_DIR / "best_configs.json"
    if best_config_path.exists():
        _validate_best_configs(df, best_config_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "strategy_calibration.csv",
        help="Path to the calibration CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analytics(args.input)


if __name__ == "__main__":
    main()
