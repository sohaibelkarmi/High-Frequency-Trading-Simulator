"""Helpers for architecture comparison benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

ARCHITECTURE_CSV_FIELDS: tuple[str, ...] = (
    "variant",
    "iteration",
    "message_count",
    "wall_time_s",
    "throughput_msg_s",
    "digest",
    "fill_count",
    "fill_volume",
    "avg_latency_ns",
    "p95_latency_ns",
    "p99_latency_ns",
    "max_latency_ns",
    "matching_avg_ns",
    "message_avg_ns",
)


@dataclass(slots=True)
class ArchitectureRun:
    """Per-run metrics for a backtester architecture variant."""

    variant: str
    iteration: int
    message_count: int
    wall_time_s: float
    throughput_msg_s: float
    digest: str
    fill_count: int
    fill_volume: float
    avg_latency_ns: float | None
    p95_latency_ns: float | None
    p99_latency_ns: float | None
    max_latency_ns: float | None
    matching_avg_ns: float | None
    message_avg_ns: float | None

    def to_row(self) -> Dict[str, object]:
        return {field: getattr(self, field) for field in ARCHITECTURE_CSV_FIELDS}


def rows_for_csv(runs: Iterable[ArchitectureRun]) -> List[Dict[str, object]]:
    return [run.to_row() for run in runs]


def summarise_by_variant(
    runs: Sequence[ArchitectureRun],
) -> Dict[str, Dict[str, float | int]]:
    summary: Dict[str, Dict[str, float | int]] = {}
    grouped: Dict[str, List[ArchitectureRun]] = {}
    for run in runs:
        grouped.setdefault(run.variant, []).append(run)

    for variant, variant_runs in grouped.items():
        digests = {run.digest for run in variant_runs}
        summary[variant] = {
            "runs": len(variant_runs),
            "unique_digests": len(digests),
            "avg_wall_time_s": mean(run.wall_time_s for run in variant_runs),
            "avg_throughput_msg_s": mean(run.throughput_msg_s for run in variant_runs),
        }
        matching = [
            run.matching_avg_ns
            for run in variant_runs
            if run.matching_avg_ns is not None
        ]
        message = [
            run.message_avg_ns for run in variant_runs if run.message_avg_ns is not None
        ]
        if matching:
            summary[variant]["avg_matching_ns"] = mean(matching)
        if message:
            summary[variant]["avg_message_ns"] = mean(message)
    return summary


__all__ = [
    "ArchitectureRun",
    "ARCHITECTURE_CSV_FIELDS",
    "rows_for_csv",
    "summarise_by_variant",
]
