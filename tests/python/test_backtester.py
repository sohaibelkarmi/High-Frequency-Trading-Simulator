import json
from pathlib import Path

import pytest

from backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    ReplayConfig,
    ReplayEngine,
    RiskConfig,
    RiskEngine,
    RateLimitConfig,
    StrategyCallbacks,
    StrategyError,
    load_lobster_csv,
    replay_from_lobster,
)
from backtester.order_book import PythonOrderBook
from strategies import MarketMakingConfig, MarketMakingStrategy


class CollectingStrategy(StrategyCallbacks):
    def __init__(self) -> None:
        self.snapshots = []

    def on_market_data(self, snapshot, ctx) -> None:
        self.snapshots.append(snapshot)


def test_backtester_digest_deterministic(tmp_path: Path) -> None:
    messages = list(load_lobster_csv("tests/data/itch_sample.csv", symbol="TEST"))
    replay1 = list(replay_from_lobster(messages))
    replay2 = list(replay_from_lobster(messages))

    def run_once(replay):
        logger_path = tmp_path / "run.jsonl"
        with MetricsLogger(json_path=logger_path) as metrics:
            risk = RiskEngine(RiskConfig(symbol="TEST"))
            backtester = Backtester(
                config=BacktesterConfig(symbol="TEST"),
                limit_book=PythonOrderBook(depth=5),
                metrics_logger=metrics,
                risk_engine=risk,
                strategy=CollectingStrategy(),
                seed=42,
            )
            backtester.run(replay)
            digest = backtester.digest
        data = logger_path.read_text().strip().splitlines()
        return digest, data

    digest1, log1 = run_once(replay1)
    digest2, log2 = run_once(replay2)

    assert digest1 == digest2

    def _strip_dynamic(entry: str) -> dict:
        record = json.loads(entry)
        if record.get("event_type") == "run_summary":
            payload = dict(record["payload"])
            payload["timings"] = {}
            record["payload"] = payload
        return record

    assert [_strip_dynamic(line) for line in log1] == [
        _strip_dynamic(line) for line in log2
    ]

    last_record = json.loads(log1[-1])
    assert last_record["event_type"] == "run_summary"
    summary = last_record["payload"]
    assert summary["digest"] == digest1
    assert summary["symbol"] == "TEST"
    assert "order_to_trade_ratio" in summary


def test_market_maker_replay_deterministic(tmp_path: Path) -> None:
    messages = list(load_lobster_csv("tests/data/itch_sample.csv", symbol="TEST"))

    def run_once(seed: int):
        replay = replay_from_lobster(messages)
        log_path = tmp_path / f"run_{seed}.jsonl"
        with MetricsLogger(json_path=log_path) as metrics:
            risk = RiskEngine(RiskConfig(symbol="TEST"))
            strategy = MarketMakingStrategy(
                MarketMakingConfig(
                    spread_ticks=1,
                    quote_size=1.0,
                    tick_size=0.1,
                    update_interval_ns=0,
                ),
                risk_engine=risk,
                seed=seed,
            )
            backtester = Backtester(
                config=BacktesterConfig(symbol="TEST"),
                limit_book=PythonOrderBook(depth=5),
                metrics_logger=metrics,
                risk_engine=risk,
                strategy=strategy,
                seed=seed,
            )
            backtester.run(replay)
            digest = backtester.digest
        summary_record = json.loads(log_path.read_text().strip().splitlines()[-1])
        return digest, summary_record["payload"]

    digest1, summary1 = run_once(seed=99)
    digest2, summary2 = run_once(seed=99)

    assert digest1 == digest2

    def _normalise_summary(summary: dict) -> dict:
        clone = dict(summary)
        clone["timings"] = {}
        return clone

    assert _normalise_summary(summary1) == _normalise_summary(summary2)
    assert summary1["order_count"] > 0
    assert summary1["order_volume"] > 0.0


def test_replay_engine_fast_mode(tmp_path: Path) -> None:
    messages = list(load_lobster_csv("tests/data/itch_sample.csv", symbol="TEST"))
    base_events = list(replay_from_lobster(messages))

    replay_fast = ReplayEngine(ReplayConfig(speed=0.0, real_time=False)).stream(
        base_events
    )

    def run_stream(stream, path: Path) -> str:
        with MetricsLogger(json_path=path) as metrics:
            risk = RiskEngine(RiskConfig(symbol="TEST"))
            backtester = Backtester(
                config=BacktesterConfig(symbol="TEST"),
                limit_book=PythonOrderBook(depth=5),
                metrics_logger=metrics,
                risk_engine=risk,
                strategy=CollectingStrategy(),
                seed=1,
            )
            backtester.run(stream)
            return backtester.digest

    fast_digest = run_stream(replay_fast, tmp_path / "fast.jsonl")
    baseline_digest = run_stream(
        replay_from_lobster(messages), tmp_path / "baseline.jsonl"
    )

    assert fast_digest == baseline_digest


def test_metrics_logger_jsonl(tmp_path: Path) -> None:
    logger_path = tmp_path / "metrics.jsonl"
    with MetricsLogger(json_path=logger_path):
        pass
    assert logger_path.exists()


def test_order_rate_limit_triggers_violation(tmp_path: Path) -> None:
    log_path = tmp_path / "order_rate_limit.jsonl"
    metrics = MetricsLogger(json_path=log_path)
    risk = RiskEngine(RiskConfig(symbol="LIMIT"))
    rate_limit = RateLimitConfig(max_actions=1, interval_ns=1_000_000)
    backtester = Backtester(
        config=BacktesterConfig(symbol="LIMIT"),
        limit_book=PythonOrderBook(depth=1),
        metrics_logger=metrics,
        risk_engine=risk,
        order_rate_limit=rate_limit,
    )
    backtester.clock_ns = 0
    backtester.submit_order("BUY", 100.0, 1.0)
    backtester.clock_ns = 0
    with pytest.raises(StrategyError):
        backtester.submit_order("SELL", 101.0, 1.0)
    metrics.close()
    assert backtester.strategy_halted
    assert backtester.control_stats["order_rate_limit"] == 1
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(
        entry["event_type"] == "control_violation"
        and entry["payload"].get("kind") == "order_rate_limit"
        for entry in events
    )


def test_cancel_rate_limit_throttles(tmp_path: Path) -> None:
    log_path = tmp_path / "cancel_rate_limit.jsonl"
    metrics = MetricsLogger(json_path=log_path)
    risk = RiskEngine(RiskConfig(symbol="LIMIT"))
    cancel_limit = RateLimitConfig(
        max_actions=1,
        interval_ns=1_000_000,
        halt_on_violation=False,
    )
    backtester = Backtester(
        config=BacktesterConfig(symbol="LIMIT"),
        limit_book=PythonOrderBook(depth=1),
        metrics_logger=metrics,
        risk_engine=risk,
        cancel_rate_limit=cancel_limit,
    )
    backtester.clock_ns = 0
    order1 = backtester.submit_order("BUY", 100.0, 1.0)
    order2 = backtester.submit_order("SELL", 101.0, 1.0)
    backtester.clock_ns = 0
    backtester.cancel_order(order1)
    backtester.clock_ns = 0
    backtester.cancel_order(order2)
    metrics.close()
    assert not backtester.strategy_halted
    assert backtester.control_stats["cancel_rate_limit"] == 1
    assert order2 in backtester.active_orders
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert any(
        entry["event_type"] == "control_violation"
        and entry["payload"].get("kind") == "cancel_rate_limit"
        for entry in events
    )
