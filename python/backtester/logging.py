"""Structured logging for backtests with JSONL and SQLite sinks."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import FillEvent, MarketSnapshot, OrderRequest


@dataclass(slots=True)
class LogRecord:
    timestamp_ns: int
    event_type: str
    payload: Dict[str, object]


@dataclass(slots=True)
class MetricsSnapshot:
    order_count: int
    fill_count: int
    order_volume: float
    fill_volume: float
    avg_latency_ns: Optional[float]
    p95_latency_ns: Optional[int]
    p99_latency_ns: Optional[int]
    max_latency_ns: Optional[int]
    latency_breakdown: "LatencyBreakdown"
    timings: Dict[str, "TimingSummary"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timings:
            return
        converted: Dict[str, TimingSummary] = {}
        for label, entry in self.timings.items():
            if isinstance(entry, TimingSummary):
                converted[label] = entry
            else:
                converted[label] = TimingSummary.from_mapping(label, entry)
        self.timings = converted


@dataclass(slots=True)
class LatencyBreakdown:
    last_market_to_decision_us: Optional[float]
    last_decision_to_submit_us: Optional[float]
    last_market_to_submit_us: Optional[float]
    avg_market_to_decision_us: Optional[float]
    avg_decision_to_submit_us: Optional[float]
    avg_market_to_submit_us: Optional[float]


@dataclass(slots=True)
class TimingSummary:
    label: str
    count: int
    total_ns: int
    avg_ns: float
    p95_ns: Optional[int]
    p99_ns: Optional[int]
    max_ns: Optional[int]

    @classmethod
    def from_mapping(cls, label: str, payload: Mapping[str, object]) -> "TimingSummary":
        count = int(payload.get("count", 0))  # type: ignore[arg-type]
        total_ns = int(payload.get("total_ns", 0))  # type: ignore[arg-type]
        avg_val = payload.get("avg_ns", 0.0)
        avg_ns = float(avg_val) if avg_val is not None else 0.0  # type: ignore[arg-type]
        p95_val = payload.get("p95_ns")
        p99_val = payload.get("p99_ns")
        max_val = payload.get("max_ns")
        p95_ns = int(p95_val) if p95_val is not None else None  # type: ignore[arg-type]
        p99_ns = int(p99_val) if p99_val is not None else None  # type: ignore[arg-type]
        max_ns = int(max_val) if max_val is not None else None  # type: ignore[arg-type]
        return cls(
            label=label,
            count=count,
            total_ns=total_ns,
            avg_ns=avg_ns,
            p95_ns=p95_ns,
            p99_ns=p99_ns,
            max_ns=max_ns,
        )


@dataclass(slots=True)
class RunSummary:
    symbol: str
    realized_pnl: float
    unrealized_pnl: float
    inventory: float
    order_count: int
    fill_count: int
    order_volume: float
    fill_volume: float
    order_to_trade_ratio: Optional[float]
    fill_efficiency: Optional[float]
    avg_latency_ns: Optional[float]
    p95_latency_ns: Optional[int]
    p99_latency_ns: Optional[int]
    max_latency_ns: Optional[int]
    start_timestamp_ns: Optional[int]
    end_timestamp_ns: Optional[int]
    duration_ns: Optional[int]
    digest: Optional[str] = None
    timings: Dict[str, TimingSummary] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timings:
            return
        converted: Dict[str, TimingSummary] = {}
        for label, entry in self.timings.items():
            if isinstance(entry, TimingSummary):
                converted[label] = entry
            else:
                converted[label] = TimingSummary.from_mapping(label, entry)
        self.timings = converted


class MetricsLogger:
    def __init__(
        self,
        json_path: Optional[str | Path] = None,
        sqlite_path: Optional[str | Path] = None,
    ) -> None:
        self.json_path = Path(json_path) if json_path else None
        self.sqlite_path = Path(sqlite_path) if sqlite_path else None
        self._json_handle = None
        if self.json_path:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_handle = self.json_path.open("w", encoding="utf-8")
        self._conn = None
        if self.sqlite_path:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.sqlite_path)
            self._initialise_sqlite()
        self._order_count = 0
        self._fill_count = 0
        self._order_volume = 0.0
        self._fill_volume = 0.0
        self._latencies: List[int] = []
        self._timings: Dict[str, List[int]] = defaultdict(list)
        self._run_start_ns: Optional[int] = None
        self._run_end_ns: Optional[int] = None
        self._summary_logged = False
        self._latency_market_to_decision_ns: List[int] = []
        self._latency_decision_to_submit_ns: List[int] = []
        self._latency_market_to_submit_ns: List[int] = []

    def _initialise_sqlite(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp_ns INTEGER,
                event_type TEXT,
                payload TEXT
            )
            """)
        self._conn.commit()

    def close(self) -> None:
        if self._json_handle:
            self._json_handle.close()
            self._json_handle = None
        if self._conn:
            self._conn.close()
            self._conn = None

    def log_order(
        self, order: "OrderRequest", latency_ns: Optional[int] = None
    ) -> None:
        payload = asdict(order)
        if latency_ns is not None:
            payload["latency_ns"] = latency_ns
            self._latencies.append(int(latency_ns))
        record = LogRecord(order.timestamp_ns, "order", payload)
        self._write(record)
        self._order_count += 1
        self._order_volume += order.size
        self._mark_run_boundary(order.timestamp_ns)

    def log_cancel(self, order_id: int, timestamp_ns: int) -> None:
        record = LogRecord(timestamp_ns, "cancel", {"order_id": order_id})
        self._write(record)
        self._mark_run_boundary(timestamp_ns)

    def log_control_violation(
        self,
        kind: str,
        timestamp_ns: int,
        payload: Optional[Mapping[str, object]] = None,
    ) -> None:
        record_payload: Dict[str, object] = {"kind": kind}
        if payload:
            record_payload.update(payload)
        record = LogRecord(timestamp_ns, "control_violation", record_payload)
        self._write(record)
        self._mark_run_boundary(timestamp_ns)

    def log_fill(self, fill: "FillEvent") -> None:
        record = LogRecord(fill.timestamp_ns, "fill", asdict(fill))
        self._write(record)
        self._fill_count += 1
        self._fill_volume += fill.size
        self._mark_run_boundary(fill.timestamp_ns)

    def log_snapshot(self, snapshot: "MarketSnapshot") -> None:
        record = LogRecord(snapshot.timestamp_ns, "snapshot", asdict(snapshot))
        self._write(record)
        self._mark_run_boundary(snapshot.timestamp_ns)

    def log_run_summary(
        self,
        *,
        symbol: str,
        realized_pnl: float,
        unrealized_pnl: float,
        inventory: float,
        digest: Optional[str] = None,
    ) -> RunSummary:
        summary = self._build_summary(
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            inventory=inventory,
            digest=digest,
        )
        record = LogRecord(
            summary.end_timestamp_ns or 0,
            "run_summary",
            asdict(summary),
        )
        self._write(record)
        self._summary_logged = True
        return summary

    def _write(self, record: LogRecord) -> None:
        payload = {
            "timestamp_ns": record.timestamp_ns,
            "event_type": record.event_type,
            "payload": record.payload,
        }
        if self._json_handle:
            self._json_handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self._json_handle.flush()
        if self._conn:
            self._conn.execute(
                "INSERT INTO metrics(timestamp_ns, event_type, payload) VALUES(?, ?, ?)",
                (
                    record.timestamp_ns,
                    record.event_type,
                    json.dumps(record.payload, sort_keys=True),
                ),
            )
            self._conn.commit()

    def record_timing(self, label: str, duration_ns: int) -> None:
        if duration_ns < 0:
            return
        self._timings[label].append(int(duration_ns))

    @contextmanager
    def time_block(self, label: str):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            duration = time.perf_counter_ns() - start
            self.record_timing(label, duration)

    def _timing_summary(self) -> Dict[str, TimingSummary]:
        summary: Dict[str, TimingSummary] = {}
        for label, samples in self._timings.items():
            if not samples:
                continue
            ordered = sorted(samples)
            total = sum(ordered)
            count = len(ordered)
            p95_index = int(round(0.95 * (count - 1))) if count > 1 else 0
            p99_index = int(round(0.99 * (count - 1))) if count > 1 else 0
            p95_ns = ordered[p95_index]
            p99_ns = ordered[p99_index]
            summary[label] = TimingSummary(
                label=label,
                count=count,
                total_ns=total,
                avg_ns=total / count,
                p95_ns=p95_ns,
                p99_ns=p99_ns,
                max_ns=ordered[-1],
            )
        return summary

    def record_latency(
        self, market_to_decision_ns: int, decision_to_submit_ns: int
    ) -> None:
        total = market_to_decision_ns + decision_to_submit_ns
        self._latency_market_to_decision_ns.append(market_to_decision_ns)
        self._latency_decision_to_submit_ns.append(decision_to_submit_ns)
        self._latency_market_to_submit_ns.append(total)

    def _latency_snapshot(self) -> LatencyBreakdown:
        def _stats(samples: List[int]) -> tuple[Optional[float], Optional[float]]:
            if not samples:
                return None, None
            last = samples[-1] / 1_000.0
            avg = sum(samples) / len(samples) / 1_000.0
            return last, avg

        last_md, avg_md = _stats(self._latency_market_to_decision_ns)
        last_ds, avg_ds = _stats(self._latency_decision_to_submit_ns)
        last_total, avg_total = _stats(self._latency_market_to_submit_ns)
        return LatencyBreakdown(
            last_market_to_decision_us=last_md,
            last_decision_to_submit_us=last_ds,
            last_market_to_submit_us=last_total,
            avg_market_to_decision_us=avg_md,
            avg_decision_to_submit_us=avg_ds,
            avg_market_to_submit_us=avg_total,
        )

    def snapshot(self) -> MetricsSnapshot:
        avg_latency = (
            sum(self._latencies) / len(self._latencies) if self._latencies else None
        )
        p95_latency = None
        p99_latency = None
        max_latency = None
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            index = int(round(0.95 * (len(sorted_lat) - 1)))
            p95_latency = sorted_lat[index]
            index99 = int(round(0.99 * (len(sorted_lat) - 1)))
            p99_latency = sorted_lat[index99]
            max_latency = sorted_lat[-1]
        latency_breakdown = self._latency_snapshot()
        return MetricsSnapshot(
            order_count=self._order_count,
            fill_count=self._fill_count,
            order_volume=self._order_volume,
            fill_volume=self._fill_volume,
            avg_latency_ns=avg_latency,
            p95_latency_ns=p95_latency,
            p99_latency_ns=p99_latency,
            max_latency_ns=max_latency,
            latency_breakdown=latency_breakdown,
            timings=self._timing_summary(),
        )

    def _build_summary(
        self,
        *,
        symbol: str,
        realized_pnl: float,
        unrealized_pnl: float,
        inventory: float,
        digest: Optional[str],
    ) -> RunSummary:
        ratio = self._order_count / self._fill_count if self._fill_count > 0 else None
        fill_eff = (
            self._fill_volume / self._order_volume if self._order_volume > 0 else None
        )
        avg_latency = (
            sum(self._latencies) / len(self._latencies) if self._latencies else None
        )
        p95_latency = None
        p99_latency = None
        max_latency = None
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            index = int(round(0.95 * (len(sorted_lat) - 1)))
            p95_latency = sorted_lat[index]
            index99 = int(round(0.99 * (len(sorted_lat) - 1)))
            p99_latency = sorted_lat[index99]
            max_latency = sorted_lat[-1]
        duration = None
        if self._run_start_ns is not None and self._run_end_ns is not None:
            duration = self._run_end_ns - self._run_start_ns
        return RunSummary(
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            inventory=inventory,
            order_count=self._order_count,
            fill_count=self._fill_count,
            order_volume=self._order_volume,
            fill_volume=self._fill_volume,
            order_to_trade_ratio=ratio,
            fill_efficiency=fill_eff,
            avg_latency_ns=avg_latency,
            p95_latency_ns=p95_latency,
            p99_latency_ns=p99_latency,
            max_latency_ns=max_latency,
            start_timestamp_ns=self._run_start_ns,
            end_timestamp_ns=self._run_end_ns,
            duration_ns=duration,
            digest=digest,
            timings=self._timing_summary(),
        )

    def _mark_run_boundary(self, timestamp_ns: int) -> None:
        if self._run_start_ns is None or timestamp_ns < self._run_start_ns:
            self._run_start_ns = timestamp_ns
        if self._run_end_ns is None or timestamp_ns > self._run_end_ns:
            self._run_end_ns = timestamp_ns

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MetricsAggregator:
    def __init__(self, records: Iterable[LogRecord] | None = None) -> None:
        self.records: List[LogRecord] = list(records or [])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MetricsAggregator":
        records = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                blob = json.loads(line)
                records.append(
                    LogRecord(
                        timestamp_ns=blob["timestamp_ns"],
                        event_type=blob["event_type"],
                        payload=blob["payload"],
                    )
                )
        return cls(records)

    def fill_ratio(self) -> float:
        orders = sum(1 for r in self.records if r.event_type == "order")
        fills = sum(1 for r in self.records if r.event_type == "fill")
        if orders == 0:
            return 0.0
        return fills / orders

    def pnl_curve(self) -> List[Dict[str, float]]:
        realised = 0.0
        unrealised = 0.0
        curve = []
        for record in self.records:
            if record.event_type == "fill":
                realised = float(record.payload.get("liquidity_flag", 0))  # placeholder
            elif record.event_type == "snapshot":
                unrealised = float(
                    record.payload.get("imbalance", 0)
                )  # placeholder until pnl stored
            curve.append(
                {
                    "timestamp_ns": record.timestamp_ns,
                    "realized": realised,
                    "unrealized": unrealised,
                }
            )
        return curve


__all__ = [
    "MetricsLogger",
    "MetricsAggregator",
    "LogRecord",
    "RunSummary",
    "MetricsSnapshot",
    "LatencyBreakdown",
    "TimingSummary",
]
