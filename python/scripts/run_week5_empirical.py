#!/usr/bin/env python3
"""Empirical Hawkes calibration on Binance BTCUSDT trades (Sept 19–21 2025).

This script:

1. Loads the cleaned trade dumps produced by ``scripts/preprocess_binance.py``.
2. Normalises timestamps to elapsed seconds and segments the stream into 1 hour
   windows with 50 % overlap.
3. Fits exponential Hawkes models on the full tick flow.
4. Fits power-law Hawkes models on high-volume trades (|qty| >= 1e-2) to reduce
   computational burden while retaining >90 % of traded volume.
5. Runs time-rescaling residual diagnostics (KS tests, Q–Q and CDF overlays).
6. Emits window-level metrics, cached window artefacts, and summary figures
   under ``results/week5/empirical/``.
7. Saves helper metadata for downstream reporting.

The power-law likelihood uses a truncated history (default: 600 seconds) and a
Numba-accelerated evaluator with analytic gradients to make calibration across
142 windows tractable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import sys

# Ensure repository root is on sys.path so ``python`` package resolves and point
# Matplotlib to a writable cache directory (needed in restricted environments).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_DIR = ROOT / "results" / "week5" / "empirical" / "mpl_cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import Bounds, OptimizeResult, minimize
from scipy.stats import expon, kstest

from python.order_flow.calibration import fit_hawkes_exponential_mle

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

CLEAN_DATA_PATHS = [
    Path("data/runs/processed/BTCUSDT-trades-2025-09-19-clean.csv.gz"),
    Path("data/runs/processed/BTCUSDT-trades-2025-09-20-clean.csv.gz"),
    Path("data/runs/processed/BTCUSDT-trades-2025-09-21-clean.csv.gz"),
]

OUTPUT_DIR = Path("results/week5/empirical")
WINDOWS_DIR = OUTPUT_DIR / "windows"
FIGS_DIR = OUTPUT_DIR / "figs"
METRICS_PATH = OUTPUT_DIR / "window_metrics.csv"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

WINDOW_SECONDS = 3600.0
WINDOW_OVERLAP = 0.5
POWER_VOLUME_THRESHOLD = 1e-2  # retain ~8% events / 91% volume
POWER_TRUNCATION = 180.0  # history span (seconds) for power-law kernel
EXP_RHO_MAX = 0.98

BRANCHING_WARN_LEVEL = 0.95

RANDOM_SEED = 314159


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class WindowSlice:
    window_id: int
    start_sec: float
    end_sec: float
    times_all: np.ndarray  # seconds from window start
    qty_all: np.ndarray
    side_all: np.ndarray  # +1 for buy, -1 for sell
    times_filtered: np.ndarray  # high-volume trades for power-law fit


@dataclass(slots=True)
class ExponentialFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    aic: float
    bic: float
    branching_ratio: float
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    residuals: Optional[np.ndarray]


@dataclass(slots=True)
class PowerLawFit:
    mu: float
    alpha: float
    c: float
    gamma: float
    loglik: float
    aic: float
    bic: float
    branching_ratio: Optional[float]
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    residuals: Optional[np.ndarray]
    success: bool
    message: str


# -----------------------------------------------------------------------------
# Power-law log-likelihood (Numba accelerated)
# -----------------------------------------------------------------------------


@njit
def _powerlaw_loglik_grad(
    times: np.ndarray,
    mu: float,
    alpha: float,
    c: float,
    gamma: float,
    horizon: float,
    truncation: float,
) -> Tuple[float, float, float, float, float]:
    """Return log-likelihood and gradient for the truncated power-law Hawkes."""
    if mu <= 0.0 or alpha < 0.0 or c <= 0.0 or gamma <= 1.0:
        return -1e308, 0.0, 0.0, 0.0, 0.0

    n = times.size
    loglik = 0.0
    grad_mu = 0.0
    grad_alpha = 0.0
    grad_c = 0.0
    grad_gamma = 0.0

    last_time = 0.0
    active_lags = np.zeros(n)
    active_size = 0
    inv = 1.0 / (1.0 - gamma)
    inv2 = inv * inv

    for i in range(n):
        t = times[i]
        dt = t - last_time
        if dt < -1e-9:
            return -1e308, 0.0, 0.0, 0.0, 0.0

        sum_kernel = 0.0
        sum_kernel_dc = 0.0
        sum_kernel_dgamma = 0.0
        A_curr = 0.0
        B_prev = 0.0
        A_curr_dc = 0.0
        B_prev_dc = 0.0
        A_curr_dgamma = 0.0
        B_prev_dgamma = 0.0
        new_size = 0

        for j in range(active_size):
            lag_prev = active_lags[j]
            lag_curr = lag_prev + dt
            if lag_curr > truncation:
                continue

            denom_curr = c + lag_curr
            denom_prev = c + lag_prev
            g = denom_curr ** (-gamma)
            sum_kernel += g
            sum_kernel_dc -= gamma * denom_curr ** (-(gamma + 1.0))
            sum_kernel_dgamma -= math.log(denom_curr) * g

            power_curr = denom_curr ** (1.0 - gamma)
            power_prev = denom_prev ** (1.0 - gamma)
            A_curr += power_curr
            B_prev += power_prev
            A_curr_dc += (1.0 - gamma) * denom_curr ** (-gamma)
            B_prev_dc += (1.0 - gamma) * denom_prev ** (-gamma)
            A_curr_dgamma -= math.log(denom_curr) * power_curr
            B_prev_dgamma -= math.log(denom_prev) * power_prev

            active_lags[new_size] = lag_curr
            new_size += 1

        active_size = new_size
        intensity = mu + alpha * sum_kernel
        if intensity <= 0.0:
            return -1e308, 0.0, 0.0, 0.0, 0.0

        diff_AB = A_curr - B_prev
        loglik += math.log(intensity)
        loglik -= mu * dt
        loglik -= alpha * diff_AB * inv

        grad_mu += 1.0 / intensity - dt
        grad_alpha += sum_kernel / intensity - diff_AB * inv
        grad_c += (
            alpha * sum_kernel_dc / intensity - alpha * (A_curr_dc - B_prev_dc) * inv
        )
        grad_gamma += (
            alpha * sum_kernel_dgamma / intensity
            - alpha * (A_curr_dgamma - B_prev_dgamma) * inv
            - alpha * diff_AB * inv2
        )

        active_lags[active_size] = 0.0
        active_size += 1
        last_time = t

    tail_dt = horizon - last_time
    if tail_dt < -1e-9:
        return -1e308, 0.0, 0.0, 0.0, 0.0

    if tail_dt > 0.0:
        tail_curr = 0.0
        tail_prev = 0.0
        tail_curr_dc = 0.0
        tail_prev_dc = 0.0
        tail_curr_dgamma = 0.0
        tail_prev_dgamma = 0.0

        for j in range(active_size):
            lag_prev = active_lags[j]
            lag_curr = lag_prev + tail_dt
            if lag_curr > truncation:
                continue
            denom_curr = c + lag_curr
            denom_prev = c + lag_prev
            power_curr = denom_curr ** (1.0 - gamma)
            power_prev = denom_prev ** (1.0 - gamma)

            tail_curr += power_curr
            tail_prev += power_prev
            tail_curr_dc += (1.0 - gamma) * denom_curr ** (-gamma)
            tail_prev_dc += (1.0 - gamma) * denom_prev ** (-gamma)
            tail_curr_dgamma -= math.log(denom_curr) * power_curr
            tail_prev_dgamma -= math.log(denom_prev) * power_prev

        diff_tail = tail_curr - tail_prev
        loglik -= mu * tail_dt
        loglik -= alpha * diff_tail * inv

        grad_mu -= tail_dt
        grad_alpha -= diff_tail * inv
        grad_c -= alpha * (tail_curr_dc - tail_prev_dc) * inv
        grad_gamma -= (
            alpha * (tail_curr_dgamma - tail_prev_dgamma) * inv
            + alpha * diff_tail * inv2
        )

    return loglik, grad_mu, grad_alpha, grad_c, grad_gamma


@njit
def _powerlaw_residuals(
    times: np.ndarray,
    mu: float,
    alpha: float,
    c: float,
    gamma: float,
    horizon: float,
    truncation: float,
) -> np.ndarray:
    """Return time-rescaled residual increments for the power-law Hawkes."""
    n = times.size
    residuals = np.zeros(n)
    last_time = 0.0
    active_lags = np.zeros(n)
    active_size = 0
    inv = 1.0 / (1.0 - gamma)

    for i in range(n):
        t = times[i]
        dt = t - last_time
        sum_kernel = 0.0
        tail_curr = 0.0
        tail_prev = 0.0
        new_size = 0

        for j in range(active_size):
            lag_prev = active_lags[j]
            lag_curr = lag_prev + dt
            if lag_curr > truncation:
                continue
            denom_curr = c + lag_curr
            sum_kernel += denom_curr ** (-gamma)
            tail_curr += denom_curr ** (1.0 - gamma)
            tail_prev += (c + lag_prev) ** (1.0 - gamma)
            active_lags[new_size] = lag_curr
            new_size += 1

        active_size = new_size
        integral = mu * dt + alpha * (tail_curr - tail_prev) * inv
        residuals[i] = integral

        active_lags[active_size] = 0.0
        active_size += 1
        last_time = t

    return residuals


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def load_clean_trades(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned trade file: {path}")
        df = pd.read_csv(
            path,
            usecols=["trade_id", "ts_ms", "side", "signed_qty"],
        )
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades = trades.drop_duplicates(subset="trade_id", keep="first")
    trades = trades.sort_values("ts_ms").reset_index(drop=True)
    return trades


def normalise_timestamps(trades: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    base_ts = int(trades["ts_ms"].iloc[0])
    trades = trades.assign(elapsed_sec=(trades["ts_ms"] - base_ts) / 1000.0)
    trades["side_flag"] = np.where(trades["side"] == "buy", 1, -1)
    return trades, base_ts


def iter_windows(
    trades: pd.DataFrame,
    window_seconds: float,
    overlap: float,
) -> Iterable[WindowSlice]:
    assert 0 <= overlap < 1, "window overlap must be in [0, 1)"
    step = window_seconds * (1.0 - overlap)
    max_elapsed = float(trades["elapsed_sec"].iloc[-1])
    window_id = 0
    start = 0.0
    rng = np.random.default_rng(RANDOM_SEED)

    while start + window_seconds <= max_elapsed + 1e-9:
        end = start + window_seconds
        mask = (trades["elapsed_sec"] >= start) & (trades["elapsed_sec"] < end)
        window_df = trades.loc[mask].copy()
        if window_df.empty:
            start += step
            window_id += 1
            continue

        times_all = window_df["elapsed_sec"].to_numpy(dtype=np.float64) - start
        qty_all = window_df["signed_qty"].to_numpy(dtype=np.float64)
        side_all = window_df["side_flag"].to_numpy(dtype=np.int8)

        filtered_mask = np.abs(qty_all) >= POWER_VOLUME_THRESHOLD
        times_filtered = times_all[filtered_mask]

        # jitter zero gaps minimally to avoid exact duplicates for diagnostics
        jitter = rng.uniform(-1e-9, 1e-9, size=times_filtered.size)
        times_filtered = np.sort(times_filtered + jitter)
        times_filtered = np.maximum(times_filtered, 0.0)

        yield WindowSlice(
            window_id=window_id,
            start_sec=start,
            end_sec=end,
            times_all=times_all,
            qty_all=qty_all,
            side_all=side_all,
            times_filtered=times_filtered,
        )
        start += step
        window_id += 1


def save_window_slice(slice_: WindowSlice) -> None:
    path = WINDOWS_DIR / f"window_{slice_.window_id:04d}.npz"
    np.savez(
        path,
        times_all=slice_.times_all,
        qty_all=slice_.qty_all,
        side_all=slice_.side_all,
        times_filtered=slice_.times_filtered,
        start_sec=slice_.start_sec,
        end_sec=slice_.end_sec,
    )


def exponential_branching_ratio(alpha: float, beta: float) -> float:
    if beta <= 0:
        return math.nan
    return alpha / beta


def powerlaw_branching_ratio(alpha: float, c: float, gamma: float) -> Optional[float]:
    if gamma <= 1.0:
        return None
    return alpha * (c ** (1.0 - gamma)) / (gamma - 1.0)


def exponential_residuals(
    times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    horizon: float,
) -> np.ndarray:
    residuals = np.zeros(times.size, dtype=np.float64)
    state = 0.0
    last_time = 0.0
    for idx, t in enumerate(times):
        dt = t - last_time
        decay = math.exp(-beta * dt)
        state *= decay
        integral = mu * dt + (alpha / beta) * state * (1.0 - decay)
        residuals[idx] = integral
        state += 1.0
        last_time = t
    return residuals


def ks_diagnostics(residuals: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if residuals.size < 8 or not np.isfinite(residuals).all():
        return None, None
    stat, pvalue = kstest(residuals, expon.cdf)
    return float(stat), float(pvalue)


class PowerLawObjective:
    """Wrapper to cache power-law likelihood/gradient evaluations."""

    RHO_MAX = 0.98
    PENALTY_SCALE = 1e4

    def __init__(
        self,
        times: np.ndarray,
        horizon: float,
        truncation: float,
    ) -> None:
        self.times = times
        self.horizon = float(horizon)
        self.truncation = float(truncation)
        self._last_x: Optional[np.ndarray] = None
        self._val = math.inf
        self._grad = np.zeros(4, dtype=np.float64)

    def _evaluate(self, theta: np.ndarray) -> float:
        if self._last_x is not None and np.allclose(theta, self._last_x):
            return self._val
        mu, alpha, c, gamma = theta
        ll, g_mu, g_alpha, g_c, g_gamma = _powerlaw_loglik_grad(
            self.times, mu, alpha, c, gamma, self.horizon, self.truncation
        )
        if ll <= -1e307:
            self._val = math.inf
            self._grad = np.zeros(4, dtype=np.float64)
        else:
            obj = -ll
            grad = np.array([-g_mu, -g_alpha, -g_c, -g_gamma], dtype=np.float64)

            rho = powerlaw_branching_ratio(alpha, c, gamma)
            if rho is not None and rho > self.RHO_MAX:
                diff = rho - self.RHO_MAX
                penalty = self.PENALTY_SCALE * diff * diff
                obj += penalty
                drho_dalpha = (c ** (1.0 - gamma)) / (gamma - 1.0)
                drho_dc = rho * (1.0 - gamma) / c
                drho_dgamma = rho * (-math.log(c) - 1.0 / (gamma - 1.0))
                coeff = 2.0 * self.PENALTY_SCALE * diff
                grad[1] += coeff * drho_dalpha
                grad[2] += coeff * drho_dc
                grad[3] += coeff * drho_dgamma

            self._val = obj
            self._grad = grad
        self._last_x = theta.copy()
        return self._val

    def value(self, theta: np.ndarray) -> float:
        return float(self._evaluate(theta))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        self._evaluate(theta)
        return self._grad.copy()


def fit_powerlaw(
    times: np.ndarray,
    horizon: float,
    truncation: float,
    initial: Optional[Tuple[float, float, float, float]] = None,
) -> PowerLawFit:
    if times.size < 10:
        return PowerLawFit(
            mu=math.nan,
            alpha=math.nan,
            c=math.nan,
            gamma=math.nan,
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=None,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
            success=False,
            message="Insufficient events for power-law fit",
        )

    if initial is None:
        mu0 = max(1e-4, times.size / horizon)
        alpha0 = 0.5 * mu0
        initial = (mu0, alpha0, 0.01, 1.3)

    initial_arr = np.array(initial, dtype=np.float64)
    bounds = Bounds(
        [1e-6, 0.0, 1e-4, 1.01],
        [50.0, 50.0, 10.0, 3.0],
    )
    objective = PowerLawObjective(times, horizon, truncation)

    result = minimize(
        fun=objective.value,
        x0=initial_arr,
        method="L-BFGS-B",
        jac=objective.grad,
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-6, "gtol": 1e-5},
    )

    mu, alpha, c, gamma = result.x
    ll, _, _, _, _ = _powerlaw_loglik_grad(
        times, mu, alpha, c, gamma, horizon, truncation
    )
    loglik = float(ll)
    if not result.success or ll <= -1e307 or not np.isfinite(loglik):
        return PowerLawFit(
            mu=float(mu),
            alpha=float(alpha),
            c=float(c),
            gamma=float(gamma),
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=None,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
            success=False,
            message=result.message,
        )

    params = (mu, alpha, c, gamma)
    k = len(params)
    aic = 2 * k - 2 * loglik
    bic = k * math.log(times.size) - 2 * loglik
    rho = powerlaw_branching_ratio(alpha, c, gamma)
    residuals = _powerlaw_residuals(times, mu, alpha, c, gamma, horizon, truncation)
    ks_stat, ks_pvalue = ks_diagnostics(residuals)
    return PowerLawFit(
        mu=float(mu),
        alpha=float(alpha),
        c=float(c),
        gamma=float(gamma),
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
        branching_ratio=rho,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        residuals=residuals,
        success=True,
        message=result.message,
    )


def fit_exponential(
    times: np.ndarray,
    horizon: float,
) -> ExponentialFit:
    if times.size < 5:
        return ExponentialFit(
            mu=math.nan,
            alpha=math.nan,
            beta=math.nan,
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=math.nan,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
        )
    mu_guess = times.size / max(horizon, 1.0)
    mu_guess = max(mu_guess, 1e-3)
    seeds: List[Optional[Tuple[float, float, float]]] = [
        None,
        (mu_guess, 0.5 * mu_guess, 1.0),
        (mu_guess, 0.3 * mu_guess, 5.0),
        (1.0, 0.5, 1.0),
    ]

    best_overall: Optional[Dict[str, object]] = None
    best_stable: Optional[Dict[str, object]] = None

    for idx, seed in enumerate(seeds):
        try:
            if seed is None:
                opt: OptimizeResult = fit_hawkes_exponential_mle(times, horizon=horizon)
            else:
                opt = fit_hawkes_exponential_mle(times, horizon=horizon, initial=seed)
        except Exception:  # pragma: no cover - defensive
            continue
        mu, alpha, beta = (float(opt.x[0]), float(opt.x[1]), float(opt.x[2]))
        if not (math.isfinite(mu) and math.isfinite(alpha) and math.isfinite(beta)):
            continue
        ratio = exponential_branching_ratio(alpha, beta)
        loglik = float(opt.log_likelihood_)
        candidate = {
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "ratio": ratio,
            "loglik": loglik,
        }
        if best_overall is None or loglik > best_overall["loglik"]:
            best_overall = candidate
        if ratio < EXP_RHO_MAX:
            if best_stable is None or loglik > best_stable["loglik"]:
                best_stable = candidate
            break

    chosen = best_stable or best_overall
    if chosen is None:
        return ExponentialFit(
            mu=math.nan,
            alpha=math.nan,
            beta=math.nan,
            loglik=float("-inf"),
            aic=math.nan,
            bic=math.nan,
            branching_ratio=math.nan,
            ks_stat=None,
            ks_pvalue=None,
            residuals=None,
        )

    mu = chosen["mu"]
    alpha = chosen["alpha"]
    beta = chosen["beta"]
    loglik = chosen["loglik"]
    ratio = chosen["ratio"]

    k = 3
    aic = 2 * k - 2 * loglik
    bic = k * math.log(times.size) - 2 * loglik
    residuals = exponential_residuals(times, mu, alpha, beta, horizon)
    ks_stat, ks_pvalue = ks_diagnostics(residuals)
    return ExponentialFit(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        loglik=float(loglik),
        aic=float(aic),
        bic=float(bic),
        branching_ratio=float(ratio),
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        residuals=residuals,
    )


def summarise_metrics(
    windows: List[WindowSlice],
    exp_fits: List[ExponentialFit],
    pow_fits: List[PowerLawFit],
    base_ts_ms: int,
) -> pd.DataFrame:
    records = []
    base_ts = pd.to_datetime(base_ts_ms, unit="ms", utc=True)
    for win, exp_fit, pow_fit in zip(windows, exp_fits, pow_fits):
        records.append(
            {
                "window_id": win.window_id,
                "start_sec": win.start_sec,
                "end_sec": win.end_sec,
                "midpoint_sec": 0.5 * (win.start_sec + win.end_sec),
                "start_time_utc": base_ts + pd.to_timedelta(win.start_sec, unit="s"),
                "end_time_utc": base_ts + pd.to_timedelta(win.end_sec, unit="s"),
                "events_all": win.times_all.size,
                "events_filtered": win.times_filtered.size,
                "exp_mu": exp_fit.mu,
                "exp_alpha": exp_fit.alpha,
                "exp_beta": exp_fit.beta,
                "exp_loglik": exp_fit.loglik,
                "exp_aic": exp_fit.aic,
                "exp_bic": exp_fit.bic,
                "exp_branching": exp_fit.branching_ratio,
                "exp_ks_stat": exp_fit.ks_stat,
                "exp_ks_pvalue": exp_fit.ks_pvalue,
                "pow_mu": pow_fit.mu,
                "pow_alpha": pow_fit.alpha,
                "pow_c": pow_fit.c,
                "pow_gamma": pow_fit.gamma,
                "pow_loglik": pow_fit.loglik,
                "pow_aic": pow_fit.aic,
                "pow_bic": pow_fit.bic,
                "pow_branching": pow_fit.branching_ratio,
                "pow_ks_stat": pow_fit.ks_stat,
                "pow_ks_pvalue": pow_fit.ks_pvalue,
                "pow_success": pow_fit.success,
                "pow_message": pow_fit.message,
            }
        )
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------


def plot_branching_ratios(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        metrics["midpoint_sec"] / 3600.0,
        metrics["exp_branching"],
        label="Exponential",
        color="C0",
    )
    ax.plot(
        metrics["midpoint_sec"] / 3600.0,
        metrics["pow_branching"],
        label="Power-law",
        color="C1",
    )
    ax.axhline(
        BRANCHING_WARN_LEVEL, color="red", linestyle="--", linewidth=1.0, alpha=0.6
    )
    ax.set_xlabel("Elapsed hours since 2025-09-19 UTC")
    ax.set_ylabel("Branching ratio ρ")
    ax.set_title("Branching ratio evolution (1h windows, 50% overlap)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "branching_ratios.png", dpi=200)
    plt.close(fig)


def plot_parameter_trajectories(metrics: pd.DataFrame) -> None:
    hours = metrics["midpoint_sec"] / 3600.0
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    axes = axes.flatten()
    axes[0].plot(hours, metrics["exp_mu"], color="C0")
    axes[0].set_ylabel("μ")
    axes[0].set_title("Exponential μ")

    axes[1].plot(hours, metrics["exp_alpha"], color="C0")
    axes[1].set_ylabel("α")
    axes[1].set_title("Exponential α")

    axes[2].plot(hours, metrics["exp_beta"], color="C0")
    axes[2].set_ylabel("β")
    axes[2].set_title("Exponential β")

    axes[3].plot(hours, metrics["pow_mu"], color="C1")
    axes[3].set_ylabel("μ")
    axes[3].set_title("Power-law μ")

    axes[4].plot(hours, metrics["pow_alpha"], color="C1")
    axes[4].set_ylabel("α")
    axes[4].set_title("Power-law α")

    axes[5].plot(hours, metrics["pow_gamma"], color="C1")
    axes[5].set_ylabel("γ")
    axes[5].set_title("Power-law γ")

    for ax in axes:
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Elapsed hours since 2025-09-19 UTC")
    axes[2].set_xlabel("Elapsed hours since 2025-09-19 UTC")
    axes[5].set_xlabel("Elapsed hours since 2025-09-19 UTC")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "parameter_trajectories.png", dpi=200)
    plt.close(fig)


def plot_information_criteria(metrics: pd.DataFrame) -> None:
    hours = metrics["midpoint_sec"] / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(hours, metrics["exp_aic"], label="Exponential", color="C0")
    axes[0].plot(hours, metrics["pow_aic"], label="Power-law", color="C1")
    axes[0].set_ylabel("AIC")
    axes[0].set_title("Akaike Information Criterion")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(hours, metrics["exp_bic"], label="Exponential", color="C0")
    axes[1].plot(hours, metrics["pow_bic"], label="Power-law", color="C1")
    axes[1].set_ylabel("BIC")
    axes[1].set_title("Bayesian Information Criterion")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    axes[1].set_xlabel("Elapsed hours since 2025-09-19 UTC")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "information_criteria.png", dpi=200)
    plt.close(fig)


def select_sample_windows(metrics: pd.DataFrame, count: int = 3) -> List[int]:
    valid = metrics[
        metrics["pow_success"]
        & np.isfinite(metrics["exp_branching"])
        & np.isfinite(metrics["pow_branching"])
    ]
    if valid.empty:
        return []
    ids = valid["window_id"].to_list()
    if len(ids) <= count:
        return ids
    step = len(ids) / count
    return [ids[int(round(i * step))] for i in range(count)]


def qq_plot(residuals: np.ndarray, ax: plt.Axes, label: str) -> None:
    residuals = residuals[np.isfinite(residuals)]
    residuals = residuals[residuals > 0]
    residuals.sort()
    if residuals.size == 0:
        ax.text(0.5, 0.5, "Insufficient residuals", ha="center", va="center")
        return
    probs = (np.arange(1, residuals.size + 1) - 0.5) / residuals.size
    theo = expon.ppf(probs)
    ax.scatter(theo, residuals, s=8, alpha=0.6)
    max_val = max(residuals.max(), theo.max())
    ax.plot([0, max_val], [0, max_val], color="k", linewidth=1.0, linestyle="--")
    ax.set_title(label)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.grid(True, alpha=0.2)


def cdf_plot(residuals: np.ndarray, ax: plt.Axes, label: str) -> None:
    residuals = residuals[np.isfinite(residuals)]
    residuals = residuals[residuals >= 0]
    if residuals.size == 0:
        ax.text(0.5, 0.5, "Insufficient residuals", ha="center", va="center")
        return
    sorted_res = np.sort(residuals)
    probs = np.linspace(0, 1, sorted_res.size, endpoint=False)
    ax.step(sorted_res, probs, where="post", color="C0", label="Empirical")
    grid = np.linspace(0, sorted_res.max(), 200)
    ax.plot(grid, expon.cdf(grid), color="C1", label="Exp(1)")
    ax.set_xlabel("Residual value")
    ax.set_ylabel("CDF")
    ax.set_title(label)
    ax.grid(True, alpha=0.2)
    ax.legend()


def plot_residual_diagnostics(
    sample_windows: List[WindowSlice],
    exp_fits: Dict[int, ExponentialFit],
    pow_fits: Dict[int, PowerLawFit],
) -> None:
    if not sample_windows:
        return
    fig, axes = plt.subplots(
        len(sample_windows), 4, figsize=(16, 4 * len(sample_windows))
    )

    if len(sample_windows) == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, window in enumerate(sample_windows):
        exp_fit = exp_fits[window.window_id]
        pow_fit = pow_fits[window.window_id]
        title = f"Window {window.window_id:04d} (events={window.times_all.size})"
        axes[row_idx, 0].set_title(title)
        axes[row_idx, 0].axis("off")

        qq_plot(
            exp_fit.residuals if exp_fit.residuals is not None else np.array([]),
            axes[row_idx, 1],
            "Exponential Q-Q",
        )
        cdf_plot(
            exp_fit.residuals if exp_fit.residuals is not None else np.array([]),
            axes[row_idx, 2],
            "Exponential CDF",
        )
        if pow_fit.residuals is not None:
            qq_plot(pow_fit.residuals, axes[row_idx, 3], "Power-law Q-Q")
        else:
            axes[row_idx, 3].text(
                0.5, 0.5, "Power-law fit failed", ha="center", va="center"
            )
            axes[row_idx, 3].axis("off")

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "residual_diagnostics.png", dpi=200)
    plt.close(fig)


def compute_exponential_intensity_grid(
    times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    horizon: float,
    grid_step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, horizon + grid_step, grid_step)
    state = 0.0
    event_idx = 0
    last_event_time = 0.0
    intensities = np.zeros_like(grid)
    for idx, t in enumerate(grid):
        if event_idx < times.size:
            while event_idx < times.size and times[event_idx] <= t:
                dt_event = times[event_idx] - last_event_time
                state *= math.exp(-beta * dt_event)
                state += 1.0
                last_event_time = times[event_idx]
                event_idx += 1
        dt = t - last_event_time
        intensities[idx] = mu + alpha * state * math.exp(-beta * dt)
    return grid, intensities


def compute_powerlaw_intensity_grid(
    times: np.ndarray,
    mu: float,
    alpha: float,
    c: float,
    gamma: float,
    horizon: float,
    truncation: float,
    grid_step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, horizon + grid_step, grid_step)
    intensities = np.zeros_like(grid)
    active_times: List[float] = []
    event_idx = 0
    for idx, t in enumerate(grid):
        while event_idx < times.size and times[event_idx] <= t:
            active_times.append(times[event_idx])
            event_idx += 1
        active_times = [ts for ts in active_times if t - ts <= truncation]
        if active_times:
            lags = np.array([t - ts for ts in active_times], dtype=np.float64)
            intensities[idx] = mu + alpha * np.sum((c + lags) ** (-gamma))
        else:
            intensities[idx] = mu
    return grid, intensities


def plot_intensity_vs_counts(
    window: WindowSlice,
    exp_fit: ExponentialFit,
    pow_fit: PowerLawFit,
    grid_step: float = 10.0,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    bins = np.arange(0.0, WINDOW_SECONDS + grid_step, grid_step)
    counts, _ = np.histogram(window.times_all, bins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])

    grid_exp, intensity_exp = compute_exponential_intensity_grid(
        window.times_all,
        exp_fit.mu,
        exp_fit.alpha,
        exp_fit.beta,
        WINDOW_SECONDS,
        grid_step,
    )
    axes[0].bar(
        centres, counts, width=grid_step, color="lightgray", label="Observed trades"
    )
    axes[0].plot(
        grid_exp, intensity_exp * grid_step, color="C0", label="Exp intensity × Δt"
    )
    axes[0].set_ylabel("Count per bin")
    axes[0].set_title(f"Window {window.window_id:04d} — Exponential fit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    if pow_fit.success and window.times_filtered.size >= 10:
        grid_pow, intensity_pow = compute_powerlaw_intensity_grid(
            window.times_filtered,
            pow_fit.mu,
            pow_fit.alpha,
            pow_fit.c,
            pow_fit.gamma,
            WINDOW_SECONDS,
            POWER_TRUNCATION,
            grid_step,
        )
        filtered_counts, _ = np.histogram(window.times_filtered, bins=bins)
        axes[1].bar(
            centres,
            filtered_counts,
            width=grid_step,
            color="lightgray",
            label="Filtered trades",
        )
        axes[1].plot(
            grid_pow,
            intensity_pow * grid_step,
            color="C1",
            label="Power-law intensity × Δt",
        )
    else:
        axes[1].text(0.5, 0.5, "Power-law fit unavailable", ha="center", va="center")
    axes[1].set_xlabel("Seconds within window")
    axes[1].set_ylabel("Count per bin")
    axes[1].set_title(f"Window {window.window_id:04d} — Power-law fit (filtered)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"intensity_window_{window.window_id:04d}.png", dpi=200)
    plt.close(fig)


def plot_kernel_comparison(metrics: pd.DataFrame) -> None:
    exp_med = metrics[["exp_alpha", "exp_beta"]].median()
    pow_med = metrics[["pow_alpha", "pow_c", "pow_gamma"]].median()
    alpha_exp, beta_exp = float(exp_med["exp_alpha"]), float(exp_med["exp_beta"])
    alpha_pow, c_pow, gamma_pow = (
        float(pow_med["pow_alpha"]),
        float(pow_med["pow_c"]),
        float(pow_med["pow_gamma"]),
    )
    grid = np.linspace(0, 600, 500)
    kernel_exp = alpha_exp * np.exp(-beta_exp * grid)
    kernel_pow = alpha_pow / ((c_pow + grid) ** gamma_pow)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, kernel_exp, label="Exponential kernel", color="C0")
    ax.plot(grid, kernel_pow, label="Power-law kernel", color="C1")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Kernel value φ(t)")
    ax.set_title("Median kernel shapes across windows")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "kernel_comparison.png", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    trades = load_clean_trades(CLEAN_DATA_PATHS)
    trades, base_ts_ms = normalise_timestamps(trades)
    trades.to_parquet(OUTPUT_DIR / "trades_normalised.parquet", index=False)

    windows: List[WindowSlice] = []
    exp_fits: List[ExponentialFit] = []
    pow_fits: List[PowerLawFit] = []

    max_windows = (
        args.max_windows if getattr(args, "max_windows", None) is not None else None
    )

    prev_pow: Optional[Tuple[float, float, float, float]] = None

    for slice_ in iter_windows(trades, WINDOW_SECONDS, WINDOW_OVERLAP):
        save_window_slice(slice_)
        windows.append(slice_)

        exp_fit = fit_exponential(slice_.times_all, WINDOW_SECONDS)
        exp_fits.append(exp_fit)

        pow_fit = fit_powerlaw(
            slice_.times_filtered, WINDOW_SECONDS, POWER_TRUNCATION, initial=prev_pow
        )
        pow_fits.append(pow_fit)
        if pow_fit.success and np.isfinite(pow_fit.mu):
            prev_pow = (pow_fit.mu, pow_fit.alpha, pow_fit.c, pow_fit.gamma)
        else:
            prev_pow = None

        print(
            f"[window {slice_.window_id:04d}] events={slice_.times_all.size} "
            f"filtered={slice_.times_filtered.size} exp_rho={exp_fit.branching_ratio:.3f} "
            f"pow_success={pow_fit.success} "
            f"pow_rho={pow_fit.branching_ratio if pow_fit.branching_ratio is not None else float('nan'):.3f}",
            flush=True,
        )

        if max_windows is not None and len(windows) >= max_windows:
            break

    metrics = summarise_metrics(windows, exp_fits, pow_fits, base_ts_ms)
    metrics.to_csv(METRICS_PATH, index=False)

    metadata = {
        "base_timestamp_utc": pd.to_datetime(
            base_ts_ms, unit="ms", utc=True
        ).isoformat(),
        "window_seconds": WINDOW_SECONDS,
        "window_overlap": WINDOW_OVERLAP,
        "power_volume_threshold": POWER_VOLUME_THRESHOLD,
        "power_truncation_seconds": POWER_TRUNCATION,
        "num_windows": len(windows),
        "total_events": int(trades.shape[0]),
        "total_filtered_events": int(metrics["events_filtered"].sum()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    plot_branching_ratios(metrics)
    plot_parameter_trajectories(metrics)
    plot_information_criteria(metrics)
    plot_kernel_comparison(metrics)

    exp_map = {win.window_id: fit for win, fit in zip(windows, exp_fits)}
    pow_map = {win.window_id: fit for win, fit in zip(windows, pow_fits)}
    sample_ids = select_sample_windows(metrics)
    sample_windows = [win for win in windows if win.window_id in sample_ids]
    plot_residual_diagnostics(sample_windows, exp_map, pow_map)

    if sample_windows:
        # Plot intensity comparison for the middle sample window.
        target = sample_windows[min(len(sample_windows) // 2, len(sample_windows) - 1)]
        plot_intensity_vs_counts(
            target, exp_map[target.window_id], pow_map[target.window_id]
        )

    print(f"Processed {len(windows)} windows. Metrics saved to {METRICS_PATH}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--power-threshold",
        type=float,
        default=POWER_VOLUME_THRESHOLD,
        help="Absolute signed quantity threshold for power-law fit (default: 1e-2)",
    )
    parser.add_argument(
        "--power-truncation",
        type=float,
        default=POWER_TRUNCATION,
        help="History truncation (seconds) for power-law kernel (default: 180)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of windows to process (for debugging/testing)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    global POWER_VOLUME_THRESHOLD, POWER_TRUNCATION
    POWER_VOLUME_THRESHOLD = args.power_threshold
    POWER_TRUNCATION = args.power_truncation
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
