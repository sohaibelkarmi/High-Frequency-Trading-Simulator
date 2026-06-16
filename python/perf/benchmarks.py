"""Shared dataclasses and helpers for benchmark throughput exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

BENCHMARK_CSV_FIELDS: tuple[str, ...] = (
    "label",
    "events",
    "wall_time_s",
    "throughput_msg_s",
    "latency_avg_us",
    "latency_p95_us",
    "latency_p99_us",
    "latency_max_us",
    "matching_avg_ns",
    "matching_p95_ns",
    "matching_p99_ns",
    "matching_max_ns",
    "message_avg_ns",
    "message_p95_ns",
    "message_p99_ns",
    "message_max_ns",
    "digest",
)


@dataclass(slots=True)
class BenchmarkResult:
    """Structured capture of a single throughput experiment."""

    label: str
    events: int
    wall_time_s: float
    throughput_msg_s: float
    latency_avg_us: float | None
    latency_p95_us: float | None
    latency_p99_us: float | None
    latency_max_us: float | None
    matching_avg_ns: float | None
    matching_p95_ns: int | None
    matching_p99_ns: int | None
    matching_max_ns: int | None
    message_avg_ns: float | None
    message_p95_ns: int | None
    message_p99_ns: int | None
    message_max_ns: int | None
    digest: str

    def to_row(self) -> Dict[str, object]:
        """Render the result as a CSV-compatible mapping with stable ordering."""
        return {field: getattr(self, field) for field in BENCHMARK_CSV_FIELDS}


def rows_for_csv(results: Iterable[BenchmarkResult]) -> List[Dict[str, object]]:
    """Materialise results into a list of ordered CSV rows."""
    return [result.to_row() for result in results]


__all__ = ["BenchmarkResult", "BENCHMARK_CSV_FIELDS", "rows_for_csv"]
