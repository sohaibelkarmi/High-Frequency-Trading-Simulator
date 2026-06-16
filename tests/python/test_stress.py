from __future__ import annotations

import json

from pathlib import Path

from python.backtester import (
    PoissonOrderFlowConfig,
    StressConfig,
    run_order_book_stress,
)
from python.scripts.run_stress_suite import run_suite as run_stress_suite


def test_order_book_stress_metrics_smoke(tmp_path: Path) -> None:
    config = StressConfig(
        message_count=500,
        seed=42,
        max_price_jitter=0.5,
        max_size=5.0,
    )
    profile_path = tmp_path / "profile.txt"
    metrics = run_order_book_stress(config, profiler_output=profile_path)

    assert metrics.message_count == config.message_count
    assert metrics.wall_time_s >= 0.0
    assert metrics.peak_memory_kb >= 0.0
    assert metrics.final_depth >= 0
    assert metrics.hotspots, "Expected at least one hotspot entry"
    assert profile_path.exists()
    assert "ncalls" in profile_path.read_text(encoding="utf-8")


def test_order_book_stress_hotspots_ranked() -> None:
    config = StressConfig(message_count=300, seed=7)
    metrics = run_order_book_stress(config)

    assert (
        metrics.hotspots[0].cumulative_time_s >= metrics.hotspots[-1].cumulative_time_s
    )
    assert all(h.total_calls >= h.primitive_calls for h in metrics.hotspots)


def test_poisson_stress_records_latency_and_sequence() -> None:
    poisson_cfg = PoissonOrderFlowConfig(
        message_count=300,
        seed=99,
        base_rate_hz=1_000.0,
    )
    config = StressConfig(
        poisson=poisson_cfg,
        burst=None,
        validate_sequence=True,
        record_latency=True,
        seed=99,
        depth=5,
    )
    metrics = run_order_book_stress(config)

    assert metrics.message_count == poisson_cfg.message_count
    assert metrics.avg_latency_ns is not None
    assert metrics.max_latency_ns is not None
    assert metrics.p95_latency_ns is not None
    assert metrics.latency_histogram is not None
    assert metrics.latency_histogram, "Expected latency histogram bins"
    assert metrics.add_order_events is not None
    assert metrics.execute_order_events is not None
    assert metrics.add_order_events > 0
    assert metrics.execute_order_events > 0
    assert metrics.sequence_report is not None
    assert metrics.sequence_report.ok


def test_stress_suite_runner(tmp_path: Path) -> None:
    results = run_stress_suite(
        tmp_path,
        base_messages=500,
        base_rate_hz=2_000.0,
        seed=17,
    )

    output_file = tmp_path / "stress_suite.json"
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios")
    if scenarios is None:
        scenarios = (data.get("data") or {}).get("scenarios")
    assert scenarios is not None
    assert len(scenarios) == 3
    assert results["scenarios"][0]["multiplier"] == 1
    assert scenarios[0]["latency_histogram"] is not None
    assert scenarios[0]["add_order_events"] is not None
    execute_events = scenarios[0]["execute_order_events"]
    ratio = scenarios[0]["order_to_trade_ratio"]
    if execute_events:
        assert ratio is not None
    else:
        assert ratio is None
