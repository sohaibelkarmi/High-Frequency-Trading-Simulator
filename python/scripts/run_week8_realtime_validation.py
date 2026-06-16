#!/usr/bin/env python3
"""Week 8 – Real-time loop, OMS validation, and PnL streaming diagnostics."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "week8"
LOGS_DIR = ROOT / "logs" / "week8"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STREAM_LOG = LOGS_DIR / "pnl_stream.log"


@dataclass(frozen=True)
class LoopScenario:
    name: str
    event_rate_hz: float
    jitter_us: float
    packet_loss_bp: float
    duration_s: float
    seed: int


@dataclass(frozen=True)
class OMSScenario:
    name: str
    entry_target_us: float
    cancel_target_us: float
    confirmation_error_bp: float
    position_drift_bp: float
    operations: int
    seed: int


def _lognormal_latencies(
    mean_us: float, sigma: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    mu = math.log(mean_us) - 0.5 * sigma * sigma
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def simulate_realtime_loop() -> Path:
    scenarios = [
        LoopScenario("baseline_rt", 2000.0, 8.0, 5.0, 1.5, 2024101),
        LoopScenario("jittered_rt", 2600.0, 18.0, 15.0, 1.5, 2024102),
        LoopScenario("lossy_rt", 3200.0, 25.0, 40.0, 1.5, 2024103),
    ]
    rows: List[dict[str, float | str]] = []
    for scenario in scenarios:
        rng = np.random.default_rng(scenario.seed)
        events = int(scenario.event_rate_hz * scenario.duration_s)
        base_lat = _lognormal_latencies(55.0, 0.25, events, rng)
        jitter = rng.normal(0.0, scenario.jitter_us, size=events)
        latencies = np.clip(base_lat + jitter, 15.0, None)
        packet_losses = rng.binomial(events, scenario.packet_loss_bp / 10_000.0)
        latency_breaches = np.sum(latencies > 100.0)
        rows.append(
            {
                "scenario": scenario.name,
                "event_rate_hz": scenario.event_rate_hz,
                "events": events,
                "avg_latency_us": float(np.mean(latencies)),
                "p99_latency_us": float(np.percentile(latencies, 99)),
                "max_latency_us": float(np.max(latencies)),
                "latency_breaches": int(latency_breaches),
                "packet_loss_events": int(packet_losses),
            }
        )
    path = RESULTS_DIR / "realtime_loop_metrics.csv"
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "event_rate_hz",
                "events",
                "avg_latency_us",
                "p99_latency_us",
                "max_latency_us",
                "latency_breaches",
                "packet_loss_events",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def simulate_oms_validation() -> Path:
    scenarios = [
        OMSScenario("baseline_oms", 220.0, 180.0, 2.0, 1.5, 1500, 7),
        OMSScenario("stress_oms", 340.0, 260.0, 5.0, 3.5, 2500, 13),
    ]
    rows: List[dict[str, float | str]] = []
    for scenario in scenarios:
        rng = np.random.default_rng(scenario.seed)
        entry = np.clip(
            _lognormal_latencies(
                scenario.entry_target_us, 0.3, scenario.operations, rng
            ),
            50.0,
            None,
        )
        cancel = np.clip(
            _lognormal_latencies(
                scenario.cancel_target_us, 0.28, scenario.operations, rng
            ),
            40.0,
            None,
        )
        confirmations = rng.binomial(
            1, 1.0 - scenario.confirmation_error_bp / 10_000.0, size=scenario.operations
        )
        sync_jitter = rng.normal(
            0.0, scenario.position_drift_bp / 10.0, size=scenario.operations
        )
        rows.append(
            {
                "scenario": scenario.name,
                "orders_tested": scenario.operations,
                "entry_avg_us": float(np.mean(entry)),
                "entry_p99_us": float(np.percentile(entry, 99)),
                "cancel_avg_us": float(np.mean(cancel)),
                "cancel_p99_us": float(np.percentile(cancel, 99)),
                "trade_conf_accuracy": float(np.mean(confirmations)),
                "position_drift_bp": float(np.max(np.abs(sync_jitter))),
            }
        )
    path = RESULTS_DIR / "oms_validation.csv"
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "orders_tested",
                "entry_avg_us",
                "entry_p99_us",
                "cancel_avg_us",
                "cancel_p99_us",
                "trade_conf_accuracy",
                "position_drift_bp",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def simulate_pnl_streaming() -> Path:
    STREAM_LOG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    batches = 6
    with STREAM_LOG.open("w", encoding="utf-8") as handle:
        backlog = 0
        for batch in range(1, batches + 1):
            produced = rng.integers(800, 1200)
            consumed = int(produced * rng.uniform(0.92, 0.99))
            backlog = max(0, backlog + produced - consumed)
            latency_ms = rng.lognormal(mean=math.log(2.0), sigma=0.35)
            payload = {
                "ts": time.time(),
                "batch": batch,
                "produced": int(produced),
                "consumed": int(consumed),
                "backlog": int(backlog),
                "avg_latency_ms": float(latency_ms),
                "max_latency_ms": float(latency_ms * 1.8),
            }
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            time.sleep(0.05)
    return STREAM_LOG


def main() -> None:
    loop_path = simulate_realtime_loop()
    oms_path = simulate_oms_validation()
    stream_path = simulate_pnl_streaming()
    print("Realtime metrics:", loop_path)
    print("OMS metrics:", oms_path)
    print("PnL stream log:", stream_path)


if __name__ == "__main__":
    main()
