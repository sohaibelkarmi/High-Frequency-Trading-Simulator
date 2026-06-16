#!/usr/bin/env python3
"""Monte Carlo diagnostics for exponential Hawkes processes.

This script benchmarks the new native Hawkes simulator against a Poisson
baseline, recording clustering statistics, empirical intensities, and
diagnostic plots.  Results are written to ``results/w5/hawkes_validation``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE = PROJECT_ROOT / "results" / ".mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from kernels import ExpKernel  # noqa: E402
from simulate import (  # noqa: E402
    simulate_poisson_process,
    simulate_thinning_exp_fast,
)


@dataclass(frozen=True)
class SimulationConfig:
    label: str
    mu: float
    alpha: float
    beta: float

    @property
    def rho(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else math.inf


CONFIGS: Tuple[SimulationConfig, ...] = (
    SimulationConfig(label="subcritical", mu=0.25, alpha=0.15, beta=1.4),
    SimulationConfig(label="moderate", mu=0.25, alpha=0.35, beta=1.2),
    SimulationConfig(label="near_critical", mu=0.30, alpha=0.60, beta=1.05),
)


def conditional_intensity(
    times: np.ndarray, mu: float, alpha: float, beta: float
) -> np.ndarray:
    """Conditional intensity immediately before each event."""
    if times.size == 0:
        return np.empty(0, dtype=float)
    S = 0.0
    last_time = 0.0
    intensities = np.empty(times.size, dtype=float)
    for i, t in enumerate(times):
        dt = t - last_time
        if dt < 0.0:
            raise ValueError("event times must be non-decreasing")
        if dt > 0.0:
            S *= math.exp(-beta * dt)
        intensities[i] = mu + S
        S += alpha
        last_time = t
    return intensities


def interarrival_acf(deltas: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of inter-arrival times up to ``max_lag``."""
    if deltas.size <= max_lag or deltas.size < 2:
        return np.full(max_lag, np.nan, dtype=float)
    centered = deltas - np.mean(deltas)
    denom = np.dot(centered, centered)
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=float)
    acf = np.empty(max_lag, dtype=float)
    for lag in range(1, max_lag + 1):
        cov = np.dot(centered[:-lag], centered[lag:])
        acf[lag - 1] = cov / denom
    return acf


def compute_intensity_trace(
    times: np.ndarray, mu: float, alpha: float, beta: float, horizon: float, step: float
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, horizon + step, step)
    if times.size == 0:
        return grid, np.full_like(grid, mu, dtype=float)
    intensities = np.empty_like(grid, dtype=float)
    for i, t in enumerate(grid):
        mask = times <= t
        if not np.any(mask):
            intensities[i] = mu
            continue
        lags = t - times[mask]
        intensities[i] = mu + alpha * np.exp(-beta * lags).sum()
    return grid, intensities


def summarise_paths(
    times_list: Sequence[np.ndarray],
    mu: float,
    alpha: float,
    beta: float,
    horizon: float,
    max_lag: int,
) -> Dict[str, np.ndarray]:
    counts = np.array([float(len(times)) for times in times_list], dtype=float)
    mean_intensity = counts / horizon if horizon > 0 else np.full_like(counts, np.nan)
    cond_vars = []
    acfs: List[np.ndarray] = []
    for times in times_list:
        lam = conditional_intensity(times, mu, alpha, beta)
        cond_vars.append(np.var(lam) if lam.size > 1 else 0.0)
        deltas = np.diff(times)
        acfs.append(interarrival_acf(deltas, max_lag))
    return {
        "counts": counts,
        "mean_intensity": mean_intensity,
        "cond_var": np.array(cond_vars, dtype=float),
        "acfs": np.stack(acfs),
    }


def summarise_poisson(
    times_list: Sequence[np.ndarray], mu: float, horizon: float, max_lag: int
) -> Dict[str, np.ndarray]:
    counts = np.array([float(len(times)) for times in times_list], dtype=float)
    mean_intensity = counts / horizon if horizon > 0 else np.full_like(counts, np.nan)
    acfs: List[np.ndarray] = []
    for times in times_list:
        deltas = np.diff(times)
        acfs.append(interarrival_acf(deltas, max_lag))
    return {
        "counts": counts,
        "mean_intensity": mean_intensity,
        "cond_var": np.zeros_like(counts),
        "acfs": np.stack(acfs),
    }


def aggregate_metrics(
    cfg: SimulationConfig,
    hawkes_summary: Dict[str, np.ndarray],
    poisson_summary: Dict[str, np.ndarray],
    horizon: float,
    max_lag: int,
) -> Dict[str, object]:
    rho = cfg.rho
    theoretical_intensity = cfg.mu / (1.0 - rho) if rho < 1.0 else float("inf")
    empirical_mean = float(np.mean(hawkes_summary["mean_intensity"]))
    cond_var = float(np.mean(hawkes_summary["cond_var"]))
    poisson_cond_var = float(np.mean(poisson_summary["cond_var"]))

    hawkes_acf_mean = np.nanmean(hawkes_summary["acfs"], axis=0)
    poisson_acf_mean = np.nanmean(poisson_summary["acfs"], axis=0)

    histogram_bins = np.linspace(
        0,
        max(hawkes_summary["counts"].max(), poisson_summary["counts"].max()) + 1.0,
        31,
    )

    total_count = float(np.sum(hawkes_summary["counts"]))
    poisson_total = float(np.sum(poisson_summary["counts"]))

    payload = {
        "label": cfg.label,
        "mu": cfg.mu,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "rho": rho,
        "stable": rho < 1.0,
        "paths": int(len(hawkes_summary["counts"])),
        "horizon": horizon,
        "empirical_mean_intensity": empirical_mean,
        "theoretical_mean_intensity": theoretical_intensity,
        "relative_bias": (
            (empirical_mean - theoretical_intensity) / theoretical_intensity
            if math.isfinite(theoretical_intensity) and theoretical_intensity != 0.0
            else float("nan")
        ),
        "total_event_count": total_count,
        "poisson_total_event_count": poisson_total,
        "mean_event_count": float(np.mean(hawkes_summary["counts"])),
        "std_event_count": float(np.std(hawkes_summary["counts"])),
        "poisson_mean_event_count": float(np.mean(poisson_summary["counts"])),
        "conditional_intensity_variance": cond_var,
        "poisson_conditional_intensity_variance": poisson_cond_var,
        "mean_acf": hawkes_acf_mean.tolist(),
        "poisson_mean_acf": poisson_acf_mean.tolist(),
        "histogram_bins": histogram_bins.tolist(),
    }
    return payload


def make_intensity_figure(
    cfgs: Sequence[SimulationConfig],
    sample_paths: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    horizon: float,
) -> None:
    cols = len(cfgs)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), sharey=True)
    if cols == 1:
        axes = [axes]
    for ax, cfg in zip(axes, cfgs):
        grid, intensity = sample_paths[cfg.label]
        ax.plot(grid, intensity, label="Hawkes intensity", lw=1.6)
        ax.axhline(
            cfg.mu, color="tab:orange", linestyle="--", label="Poisson intensity"
        )
        ax.set_title(f"{cfg.label} (rho={cfg.rho:.2f})")
        ax.set_xlabel("time")
        ax.set_xlim(0, horizon)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("intensity λ(t)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_acf_figure(
    cfgs: Sequence[SimulationConfig],
    metrics: Sequence[Dict[str, object]],
    output_path: Path,
    max_lag: int,
) -> None:
    cols = len(cfgs)
    fig, axes = plt.subplots(1, cols, figsize=(4.5 * cols, 4), sharey=True)
    if cols == 1:
        axes = [axes]
    lags = np.arange(1, max_lag + 1)
    for ax, cfg, metric in zip(axes, cfgs, metrics):
        hawkes_acf = np.array(metric["mean_acf"], dtype=float)
        poisson_acf = np.array(metric["poisson_mean_acf"], dtype=float)
        ax.stem(
            lags,
            hawkes_acf,
            linefmt="C0-",
            markerfmt="C0o",
            basefmt="k-",
            label="Hawkes",
        )
        ax.stem(
            lags + 0.15,
            poisson_acf,
            linefmt="C1-",
            markerfmt="C1s",
            basefmt="k-",
            label="Poisson",
        )
        ax.set_title(f"{cfg.label}")
        ax.set_xlabel("lag")
        ax.set_ylim(-0.4, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ACF of Δt")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_histogram_figure(
    cfgs: Sequence[SimulationConfig],
    summary_data: Sequence[Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    cols = len(cfgs)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), sharey=True)
    if cols == 1:
        axes = [axes]
    for ax, cfg, data in zip(axes, cfgs, summary_data):
        hawkes_counts = data["hawkes_counts"]
        poisson_counts = data["poisson_counts"]
        bins = np.arange(
            0,
            max(hawkes_counts.max(), poisson_counts.max()) + 1.5,
            1.0,
        )
        ax.hist(
            hawkes_counts,
            bins=bins,
            alpha=0.6,
            label="Hawkes",
            color="C0",
        )
        ax.hist(
            poisson_counts,
            bins=bins,
            alpha=0.6,
            label="Poisson",
            color="C1",
        )
        if cfg.rho < 1.0:
            theo_mean = cfg.mu / (1.0 - cfg.rho) * data["horizon"]
            ax.axvline(
                theo_mean, color="k", linestyle="--", linewidth=1.2, label="Hawkes mean"
            )
        ax.axvline(
            cfg.mu * data["horizon"],
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Poisson mean",
        )
        ax.set_title(cfg.label)
        ax.set_xlabel("event count")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frequency")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metrics_json(
    output: Path,
    metrics: Sequence[Dict[str, object]],
    meta: Dict[str, object],
) -> None:
    payload = {"meta": meta, "configs": metrics}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_metrics_csv(output: Path, metrics: Sequence[Dict[str, object]]) -> None:
    header = [
        "label",
        "mu",
        "alpha",
        "beta",
        "rho",
        "stable",
        "horizon",
        "paths",
        "total_event_count",
        "poisson_total_event_count",
        "empirical_mean_intensity",
        "theoretical_mean_intensity",
        "relative_bias",
        "conditional_intensity_variance",
        "poisson_conditional_intensity_variance",
        "mean_event_count",
        "std_event_count",
        "poisson_mean_event_count",
    ]
    lines = [",".join(header)]
    for metric in metrics:
        row = [metric["label"]]
        row.extend(f"{metric[key]}" for key in header[1:])  # type: ignore[index]
        lines.append(",".join(map(str, row)))
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths", type=int, default=128, help="Monte Carlo paths per configuration"
    )
    parser.add_argument(
        "--horizon", type=float, default=600.0, help="Simulation horizon"
    )
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum ACF lag")
    parser.add_argument("--seed", type=int, default=1729, help="Base random seed")
    parser.add_argument(
        "--step", type=float, default=1.0, help="Step for intensity traces"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "w5" / "hawkes_validation",
        help="Destination directory for artefacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, object]] = []
    histogram_payload: List[Dict[str, np.ndarray]] = []
    sample_paths: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hawkes_total_events = 0.0
    poisson_total_events = 0.0

    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()

    for cfg_idx, cfg in enumerate(CONFIGS):
        kernel = ExpKernel(alpha=cfg.alpha, beta=cfg.beta)
        hawkes_runs: List[np.ndarray] = []
        poisson_runs: List[np.ndarray] = []
        for path_idx in range(args.paths):
            seed = args.seed + cfg_idx * args.paths + path_idx
            times, _ = simulate_thinning_exp_fast(
                cfg.mu,
                kernel,
                mark_sampler=lambda rng: 1.0,
                T=args.horizon,
                seed=seed,
            )
            hawkes_runs.append(times)
            poi_times, _ = simulate_poisson_process(
                cfg.mu,
                mark_sampler=lambda rng: 1.0,
                T=args.horizon,
                seed=seed + 10_000,
            )
            poisson_runs.append(poi_times)

        hawkes_summary = summarise_paths(
            hawkes_runs,
            cfg.mu,
            cfg.alpha,
            cfg.beta,
            args.horizon,
            args.max_lag,
        )
        poisson_summary = summarise_poisson(
            poisson_runs,
            cfg.mu,
            args.horizon,
            args.max_lag,
        )
        metric = aggregate_metrics(
            cfg,
            hawkes_summary,
            poisson_summary,
            args.horizon,
            args.max_lag,
        )
        metrics.append(metric)
        hawkes_total_events += metric["total_event_count"]  # type: ignore[operator]
        poisson_total_events += metric["poisson_total_event_count"]  # type: ignore[operator]
        histogram_payload.append(
            {
                "hawkes_counts": hawkes_summary["counts"],
                "poisson_counts": poisson_summary["counts"],
                "horizon": args.horizon,
            }
        )

        sample_times = hawkes_runs[0]
        grid, intensity = compute_intensity_trace(
            sample_times,
            cfg.mu,
            cfg.alpha,
            cfg.beta,
            args.horizon,
            args.step,
        )
        sample_paths[cfg.label] = (grid, intensity)

    elapsed = time.perf_counter() - t_start
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_delta = max(0, final_rss - baseline_rss)
    latency_per_event_ms = (
        (elapsed / hawkes_total_events) * 1_000.0
        if hawkes_total_events > 0
        else float("nan")
    )

    meta = {
        "paths": args.paths,
        "horizon": args.horizon,
        "max_lag": args.max_lag,
        "step": args.step,
        "elapsed_seconds": elapsed,
        "hawkes_total_events": hawkes_total_events,
        "poisson_total_events": poisson_total_events,
        "mean_latency_per_event_ms": latency_per_event_ms,
        "memory_overhead_kb": rss_delta,
        "configs": [cfg.__dict__ for cfg in CONFIGS],
    }

    save_metrics_json(output_dir / "metrics.json", metrics, meta)
    save_metrics_csv(output_dir / "metrics.csv", metrics)
    make_intensity_figure(
        CONFIGS, sample_paths, figures_dir / "intensity_traces.png", args.horizon
    )
    make_acf_figure(
        CONFIGS, metrics, figures_dir / "interarrival_acf.png", args.max_lag
    )
    make_histogram_figure(
        CONFIGS, histogram_payload, figures_dir / "event_count_hist.png"
    )


if __name__ == "__main__":
    main()
