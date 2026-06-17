#!/usr/bin/env python3
"""Monte Carlo execution-cost analysis and visualisation for Week 6."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import os

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "week6"
FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE = FIGS_DIR / "mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = RESULTS_DIR / "execution_cost_analysis.csv"
SUMMARY_PATH = RESULTS_DIR / "execution_cost_summary.json"
CURVE_PATH = RESULTS_DIR / "execution_cost_aggressiveness_curve.csv"


def _default_runner(build_dir: Path) -> Optional[Path]:
    candidates = [
        build_dir / "execution_cost_runner",
        build_dir / "Release" / "execution_cost_runner",
        build_dir / "Debug" / "execution_cost_runner",
        build_dir / "src" / "execution_cost_runner",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_runner(runner: Path) -> None:
    print(f"[info] Running {runner} to generate Monte Carlo samples …")
    result = subprocess.run([str(runner)], cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"execution_cost_runner exited with code {result.returncode}"
        )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Analysis CSV not found at {path}")
    return pd.read_csv(path)


def plot_cost_distribution(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["total_cost"], bins=20, color="#0c7bdc", alpha=0.8)
    plt.title("Execution Cost Distribution")
    plt.xlabel("Total cost (USD)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_slippage_vs_aggressiveness(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df["aggressiveness"],
        df["mean_slippage"],
        c=df["order_size"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.4,
    )
    plt.title("Slippage vs. Aggressiveness")
    plt.xlabel("Aggressiveness")
    plt.ylabel("Mean slippage (USD)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Order size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_impact_breakdown(df: pd.DataFrame, path: Path) -> None:
    grouped = (
        df.groupby("order_size")[["temporary_cost", "permanent_cost"]]
        .mean()
        .sort_index()
    )
    indices = np.arange(grouped.shape[0])
    width = 0.6

    plt.figure(figsize=(8, 5))
    plt.bar(
        indices,
        grouped["temporary_cost"],
        width=width,
        label="Temporary",
        color="#ef6c00",
    )
    plt.bar(
        indices,
        grouped["permanent_cost"],
        width=width,
        bottom=grouped["temporary_cost"],
        label="Permanent",
        color="#00897b",
    )
    plt.title("Impact Cost Breakdown by Order Size")
    plt.xlabel("Order size (shares)")
    plt.ylabel("Average cost (USD)")
    plt.xticks(indices, grouped.index.astype(str))
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_tradeoff_curve(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(df["aggressiveness"].min(), df["aggressiveness"].max(), 12)
    df = df.copy()
    df["aggr_bin"] = pd.cut(df["aggressiveness"], bins=bins, include_lowest=True)
    curve = (
        df.groupby("aggr_bin", observed=False)[
            ["aggressiveness", "total_cost", "shortfall"]
        ]
        .mean()
        .dropna()
    )
    curve = curve.rename(
        columns={"total_cost": "mean_total_cost", "shortfall": "mean_shortfall"}
    )
    curve.reset_index(drop=True, inplace=True)
    return curve


def plot_tradeoff_curve(curve: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve["aggressiveness"], curve["mean_total_cost"], marker="o", color="#3949ab"
    )
    plt.title("Cost vs. Aggressiveness Trade-off")
    plt.xlabel("Aggressiveness bin centre")
    plt.ylabel("Mean total cost (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build",
        help="Directory containing compiled executables (default: ./build)",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=None,
        help="Path to execution_cost_runner executable (overrides --build-dir)",
    )
    parser.add_argument(
        "--skip-runner",
        action="store_true",
        help="Skip running the Monte Carlo binary and reuse existing CSV",
    )
    args = parser.parse_args()

    runner = args.runner if args.runner else _default_runner(args.build_dir)
    if runner is None and not args.skip_runner:
        raise FileNotFoundError(
            "execution_cost_runner binary not found. Specify --runner or build the project."
        )

    if not args.skip_runner:
        run_runner(Path(runner))

    df = load_csv(CSV_PATH)
    avg_shortfall = float(df["shortfall"].mean())
    cost_variance = float(df["total_cost"].var(ddof=0))
    summary = {
        "runs": int(df.shape[0]),
        "average_shortfall": avg_shortfall,
        "cost_variance": cost_variance,
        "mean_aggressiveness": float(df["aggressiveness"].mean()),
        "mean_order_size": float(df["order_size"].mean()),
    }

    with SUMMARY_PATH.open("w") as handle:
        json.dump(summary, handle, indent=2)

    plot_cost_distribution(df, FIGS_DIR / "execution_cost_histogram.png")
    plot_slippage_vs_aggressiveness(df, FIGS_DIR / "slippage_vs_aggressiveness.png")
    plot_impact_breakdown(df, FIGS_DIR / "impact_breakdown.png")

    tradeoff_curve = compute_tradeoff_curve(df)
    tradeoff_curve.to_csv(CURVE_PATH, index=False)
    plot_tradeoff_curve(tradeoff_curve, FIGS_DIR / "cost_vs_aggressiveness.png")

    print("[info] Summary metrics written to", SUMMARY_PATH)
    print("[info] Figures written to", FIGS_DIR)
    print("[info] Trade-off curve saved to", CURVE_PATH)


if __name__ == "__main__":
    main()
