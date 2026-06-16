"""Parsers that transform LOBSTER/ITCH feeds into backtester events."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .backtester import MarketEvent

LOBSTER_MESSAGE_TYPES = {
    "1": "add_order",
    "2": "add_order",
    "3": "delete_order",
    "4": "execute_order",
    "5": "execute_order",
    "6": "trade",
}

LOBSTER_SIDES = {"1": "BUY", "2": "SELL", "B": "BUY", "S": "SELL"}


@dataclass(slots=True)
class LOBSTERMessage:
    timestamp_ns: int
    event_type: str
    order_id: int
    size: float
    price: float
    side: str
    symbol: str


class ITCHEvent(MarketEvent):
    """Structured ITCH event passed to the backtester."""

    pass


def load_lobster_csv(
    path: str | Path, symbol: str, time_scale: float = 1e9
) -> Iterator[LOBSTERMessage]:
    """Yield `LOBSTERMessage` instances from the canonical message CSV.

    The default `time_scale` converts seconds to nanoseconds. Adjust if your
    dataset uses microseconds instead (e.g. pass `1e6`).
    """

    with Path(path).open("r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            timestamp = int(float(row[0]) * time_scale)
            raw_type = row[1]
            event_type = LOBSTER_MESSAGE_TYPES.get(raw_type, "unknown")
            order_id = int(row[2])
            size = float(row[3])
            price = float(row[4])
            side = LOBSTER_SIDES.get(row[5], "BUY")
            yield LOBSTERMessage(
                timestamp_ns=timestamp,
                event_type=event_type,
                order_id=order_id,
                size=size,
                price=price,
                side=side,
                symbol=symbol,
            )


def to_market_event(message: LOBSTERMessage) -> ITCHEvent:
    payload = {
        "order_id": message.order_id,
        "size": message.size,
        "price": message.price,
        "side": message.side,
        "symbol": message.symbol,
    }
    return ITCHEvent(
        timestamp_ns=message.timestamp_ns,
        event_type=message.event_type,
        payload=payload,
    )


def replay_from_lobster(messages: Iterable[LOBSTERMessage]) -> Iterator[ITCHEvent]:
    for msg in messages:
        yield to_market_event(msg)


__all__ = [
    "LOBSTERMessage",
    "ITCHEvent",
    "load_lobster_csv",
    "replay_from_lobster",
    "to_market_event",
]
