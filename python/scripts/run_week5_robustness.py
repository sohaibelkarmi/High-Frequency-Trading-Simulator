#!/usr/bin/env python3
"""Week 5 robustness benchmarking for Hawkes model calibrations.

This pipeline extends the empirical BTC study by (1) comparing parameter
stability across assets and temporal resolutions, (2) evaluating multiple
kernel families, and (3) stress-testing the calibrations under synthetic data
corruptions.  Results are written to ``results/week5/robustness``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CACHE = ROOT / "results" / "week5" / "robustness" / "mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import expon, kstest  # noqa: E402

from python.order_flow.calibration import (  # noqa: E402
    branching_ratio_sum_exp,
    fit_hawkes_sum_exp_mle,
)
from python.scripts import run_week5_empirical as empirical  # noqa: E402

DEFAULT_ASSETS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_DELTAS = (0.1, 0.5, 1.0)
POWER_VOLUME_THRESHOLD = empirical.POWER_VOLUME_THRESHOLD
POWER_TRUNCATION = empirical.POWER_TRUNCATION


@dataclass(slots=True)
class SumExpFit:
    mu: float
    alphas: Tuple[float, ...]
    betas: Tuple[float, ...]
    loglik: float
    aic: float
    bic: float
    branching_ratio: float
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    residuals: Optional[np.ndarray]
    success: bool
    message: str


def iter_trade_files(symbol: str, processed_root: Path) -> List[Path]:
    pattern = f"{symbol}-trades-*-clean.csv"
    paths = sorted(processed_root.glob(pattern))
    return [path for path in paths if path.is_file()]


def quantise_times(times: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be positive")
    scaled = np.floor(times / delta) * delta
    return np.asarray(scaled, dtype=np.float64)


def select_high_volume_times(
    trades: pd.DataFrame, elapsed_col: str = "elapsed_sec"
) -> np.ndarray:
    qty = trades["signed_qty"].to_numpy(dtype=np.float64)
    mask = np.abs(qty) >= POWER_VOLUME_THRESHOLD
    times = trades[elapsed_col].to_numpy(dtype=np.float64)
    filtered = times[mask]
    if filtered.size == 0:
        return filtered
    rng = np.random.default_rng(empirical.RANDOM_SEED)
    jitter = rng.uniform(-1e-9, 1e-9, size=filtered.size)
    filtered = np.sort(filtered + jitter)
    return np.maximum(filtered, 0.0)


def sum_exp_residuals(
    times: np.ndarray,
    mu: float,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    residuals = np.zeros(times.size, dtype=np.float64)
    state = np.zeros_like(alphas, dtype=np.float64)
    last_time = 0.0
    for idx, t in enumerate(times):
        dt = t - last_time
        if dt < 0.0:
            raise ValueError("event times must be non-decreasing")
        prev_state = state.copy()
        decay = np.exp(-betas * dt) if dt > 0.0 else np.ones_like(betas)
        decayed = prev_state * decay
        integral = mu * dt + np.sum((alphas / betas) * prev_state * (1.0 - decay))
        residuals[idx] = integral
        state = decayed + 1.0
        last_time = t
    return residuals


def fit_sum_of_exponentials(
    times: np.ndarray,
    horizon: float,
    kernels: int = 2,
) -> SumExpFit:
    if times.size < 5:
        return SumExpFit(
            mu=math.nan,
            alphas=tuple(),
            betas=tuple(),
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=math.nan,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
            success=False,
            message="Insufficient events for sum-of-exponentials fit",
        )

    marks = np.ones_like(times, dtype=np.float64)
    result = fit_hawkes_sum_exp_mle(
        times,
        marks=marks,
        horizon=horizon,
        kernels=kernels,
    )
    if not result.success or not np.isfinite(result.log_likelihood_):
        return SumExpFit(
            mu=float(result.x[0]),
            alphas=tuple(result.x[1 : 1 + kernels]),
            betas=tuple(result.x[1 + kernels :]),
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=math.nan,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
            success=False,
            message=result.message,
        )

    mu = float(result.x[0])
    alphas = np.asarray(result.x[1 : 1 + kernels], dtype=np.float64)
    betas = np.asarray(result.x[1 + kernels :], dtype=np.float64)
    loglik = float(result.log_likelihood_)
    k = 1 + 2 * kernels
    aic = 2 * k - 2 * loglik
    bic = k * math.log(times.size) - 2 * loglik
    branching = branching_ratio_sum_exp(alphas, betas)
    residuals = sum_exp_residuals(times, mu, alphas, betas)
    ks_stat, ks_pvalue = empirical.ks_diagnostics(residuals)
    return SumExpFit(
        mu=mu,
        alphas=tuple(float(a) for a in alphas),
        betas=tuple(float(b) for b in betas),
        loglik=loglik,
        aic=float(aic),
        bic=float(bic),
        branching_ratio=float(branching),
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        residuals=residuals,
        success=True,
        message=result.message,
    )


def load_asset(symbol: str, processed_root: Path) -> Optional[pd.DataFrame]:
    paths = iter_trade_files(symbol, processed_root)
    if not paths:
        return None
    try:
        trades = empirical.load_clean_trades(paths)
    except FileNotFoundError:
        return None
    trades, _ = empirical.normalise_timestamps(trades)
    return trades


def compute_horizon(times: np.ndarray) -> float:
    return float(times[-1]) if times.size else 0.0


def summarise_fit(
    asset: str,
    kernel: str,
    delta: float,
    fit,
) -> Dict[str, object]:
    base = {
        "asset": asset,
        "kernel": kernel,
        "delta": float(delta),
    }
    if isinstance(fit, empirical.ExponentialFit):
        base.update(
            mu=float(fit.mu),
            alpha=float(fit.alpha),
            beta=float(fit.beta),
            rho=float(fit.branching_ratio),
            loglik=float(fit.loglik),
            aic=float(fit.aic),
            bic=float(fit.bic),
            ks_stat=fit.ks_stat,
            ks_pvalue=fit.ks_pvalue,
            success=True,
            message="ok",
        )
    elif isinstance(fit, empirical.PowerLawFit):
        base.update(
            mu=float(fit.mu),
            alpha=float(fit.alpha),
            beta=float("nan"),
            rho=float(fit.branching_ratio or math.nan),
            loglik=float(fit.loglik),
            aic=float(fit.aic),
            bic=float(fit.bic),
            ks_stat=fit.ks_stat,
            ks_pvalue=fit.ks_pvalue,
            success=fit.success,
            message=fit.message,
            c=float(fit.c),
            gamma=float(fit.gamma),
        )
    elif isinstance(fit, SumExpFit):
        base.update(
            mu=float(fit.mu),
            alpha=float(np.sum(fit.alphas)),
            beta=float(np.mean(fit.betas)) if fit.betas else float("nan"),
            rho=float(fit.branching_ratio),
            loglik=float(fit.loglik),
            aic=float(fit.aic),
            bic=float(fit.bic),
            ks_stat=fit.ks_stat,
            ks_pvalue=fit.ks_pvalue,
            success=fit.success,
            message=fit.message,
            components=len(fit.alphas),
        )
    else:
        raise TypeError(f"Unsupported fit type {type(fit)!r}")
    return base


def plot_residual_histograms(
    results: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    grouped: Dict[Tuple[str, float], List[np.ndarray]] = {}
    for row in results:
        key = (row["kernel"], row["delta"])
        residuals = row.get("residuals")
        if residuals is None:
            continue
        grouped.setdefault(key, []).append(residuals)

    if not grouped:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for (kernel, delta), arrays in grouped.items():
        if not arrays:
            continue
        combined = np.concatenate(arrays)
        if combined.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(combined, bins=40, density=True, alpha=0.7, color="#4C72B0")
        grid = np.linspace(0.0, combined.max(), 200)
        ax.plot(grid, expon.pdf(grid), color="#DD8452", lw=1.8, label="Exp(1)")
        ax.set_title(f"{kernel} residuals Δt={delta:.2f}s")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"residual_hist_{kernel}_dt{delta:.2f}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_cross_asset_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(
        index="asset",
        columns="delta",
        values=value_col,
        aggfunc="mean",
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), list(pivot.index))
    ax.set_xlabel("Δt (s)")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_kernel_table(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    table_cols = [
        "asset",
        "kernel",
        "delta",
        "mu",
        "alpha",
        "beta",
        "rho",
        "loglik",
        "aic",
        "bic",
        "ks_stat",
        "ks_pvalue",
        "success",
        "message",
    ]
    present = [col for col in table_cols if col in df.columns]
    df.loc[:, present].to_csv(output_path, index=False)


def inject_perturbations(
    times: np.ndarray,
    remove_frac: float,
    seed: int,
) -> np.ndarray:
    if not (0.0 <= remove_frac < 1.0):
        raise ValueError("remove_frac must lie in [0, 1)")
    if times.size == 0:
        return times
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=times.size) >= remove_frac
    perturbed = times[mask]
    return perturbed


def run_sensitivity_suite(
    times: np.ndarray,
    horizon: float,
    deltas: Sequence[float],
    removal_rates: Sequence[float],
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for delta in deltas:
        baseline = quantise_times(times, delta)
        for rate in removal_rates:
            perturbed = inject_perturbations(baseline, rate, seed)
            pert_horizon = compute_horizon(perturbed)
            exp_fit = empirical.fit_exponential(perturbed, pert_horizon)
            rows.append(
                {
                    "delta": delta,
                    "remove_frac": rate,
                    "mu": exp_fit.mu,
                    "alpha": exp_fit.alpha,
                    "beta": exp_fit.beta,
                    "rho": exp_fit.branching_ratio,
                }
            )
    return rows


def run_pipeline(args: argparse.Namespace) -> int:
    processed_root = ROOT / "data" / "runs" / "processed"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, object]] = []
    residual_payloads: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []

    for asset in args.assets:
        trades = load_asset(asset, processed_root)
        if trades is None:
            print(f"[warn] Skipping {asset}: cleaned trades not found.")
            continue

        times_all = trades["elapsed_sec"].to_numpy(dtype=np.float64)
        hv_times = select_high_volume_times(trades)

        for delta in args.deltas:
            times_dt = quantise_times(times_all, delta)
            horizon = compute_horizon(times_dt)

            exp_fit = empirical.fit_exponential(times_dt, horizon)
            power_fit = empirical.fit_powerlaw(hv_times, horizon, POWER_TRUNCATION)
            sum_fit = fit_sum_of_exponentials(
                times_dt, horizon, kernels=args.sumexp_components
            )

            for label, fit in (
                ("exponential", exp_fit),
                ("power_law", power_fit),
                ("sum_of_exponentials", sum_fit),
            ):
                row = summarise_fit(asset, label, delta, fit)
                all_results.append(row)
                residuals = getattr(fit, "residuals", None)
                if residuals is not None and residuals.size:
                    residual_payloads.append(
                        {
                            "asset": asset,
                            "kernel": label,
                            "delta": float(delta),
                            "residuals": residuals,
                        }
                    )

            sensitivity_rows.extend(
                {
                    "asset": asset,
                    **payload,
                }
                for payload in run_sensitivity_suite(
                    times_dt,
                    horizon,
                    deltas=[delta],
                    removal_rates=args.removal_rates,
                    seed=args.seed,
                )
            )

    if not all_results:
        print("No assets processed; nothing to report.")
        return 1

    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "kernel_comparison.csv"
    render_kernel_table(results_df, results_path)

    plot_residual_histograms(residual_payloads, fig_dir)
    plot_cross_asset_heatmap(
        results_df.loc[results_df["kernel"] == "exponential"],
        value_col="rho",
        title="Exponential branching ratio across assets",
        output_path=fig_dir / "heatmap_exponential_rho.png",
    )
    plot_cross_asset_heatmap(
        results_df.loc[results_df["kernel"] == "power_law"],
        value_col="rho",
        title="Power-law branching ratio across assets",
        output_path=fig_dir / "heatmap_powerlaw_rho.png",
    )

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if not sensitivity_df.empty:
        sens_path = output_dir / "sensitivity.csv"
        sensitivity_df.to_csv(sens_path, index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        for asset, group in sensitivity_df.groupby("asset"):
            ax.plot(
                group["remove_frac"],
                group["rho"],
                marker="o",
                label=asset,
            )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:.1f}%"))
        ax.set_xlabel("Trades removed")
        ax.set_ylabel("Branching ratio ρ")
        ax.set_title("Sensitivity of ρ to missing trades")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "sensitivity_branching.png", dpi=200)
        plt.close(fig)

    metadata = {
        "assets": list(args.assets),
        "deltas": list(args.deltas),
        "removal_rates": list(args.removal_rates),
        "sumexp_components": args.sumexp_components,
        "results": {
            "kernel_comparison": str(results_path),
            "figures": str(fig_dir),
        },
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Results written to {output_dir}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        default=list(DEFAULT_ASSETS),
        help="Asset symbols to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DELTAS),
        help="Temporal resolutions Δt in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sumexp-components",
        dest="sumexp_components",
        type=int,
        default=2,
        help="Number of exponentials in the sum-of-exponentials kernel (default: %(default)s)",
    )
    parser.add_argument(
        "--removal-rates",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05],
        help="Fractions of trades to randomly drop for sensitivity analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=empirical.RANDOM_SEED,
        help="Base RNG seed (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "week5" / "robustness"),
        help="Directory to store outputs (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run_pipeline(args)
    except Exception as exc:  # pragma: no cover - surface failure details
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
