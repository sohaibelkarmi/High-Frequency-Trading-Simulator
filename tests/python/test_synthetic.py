from __future__ import annotations

from python.backtester import MarketEvent
from python.backtester.synthetic import (
    BurstConfig,
    PoissonOrderFlowConfig,
    PoissonOrderFlowGenerator,
    SequenceValidator,
    validate_sequence,
)


def test_poisson_order_flow_generator_monotonic() -> None:
    config = PoissonOrderFlowConfig(
        message_count=500,
        seed=123,
        base_rate_hz=5_000.0,
        include_metadata=True,
    )
    generator = PoissonOrderFlowGenerator(config)
    events = list(generator.stream())

    assert len(events) == config.message_count
    timestamps = [event.timestamp_ns for event in events]
    assert timestamps == sorted(timestamps)

    report = validate_sequence(events)
    assert report.ok
    assert report.total_events == config.message_count
    assert report.max_timestamp_gap_ns > 0


def test_poisson_burst_flags_present() -> None:
    config = PoissonOrderFlowConfig(
        message_count=200,
        seed=321,
        base_rate_hz=2_000.0,
        include_metadata=True,
    )
    burst = BurstConfig(
        probability=1.0,
        min_duration_us=10_000,
        max_duration_us=20_000,
        rate_multiplier=5.0,
    )
    generator = PoissonOrderFlowGenerator(config, burst_config=burst)
    events = list(generator.stream())

    assert any(event.payload.get("burst") for event in events)


def test_sequence_validator_flags_invalid_order() -> None:
    validator = SequenceValidator()
    events = [
        MarketEvent(timestamp_ns=10, event_type="add_order", payload={"order_id": 1}),
        MarketEvent(timestamp_ns=9, event_type="delete_order", payload={"order_id": 5}),
        MarketEvent(
            timestamp_ns=11,
            event_type="execute_order",
            payload={"order_id": 2},
        ),
    ]

    for event in events:
        validator.observe(event)
    report = validator.report()

    assert not report.ok
    assert report.orphan_executes == 1
    assert report.orphan_cancels == 1
    assert not report.timestamp_monotonic
