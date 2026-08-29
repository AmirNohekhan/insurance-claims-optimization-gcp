from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class OperationalEvent:
    event_type: str
    market_id: str
    payload: dict
    event_id: str = ""
    timestamp: str = ""
    schema_version: str = "1.0"
    correlation_id: str = ""

    def normalized(self) -> OperationalEvent:
        return OperationalEvent(
            self.event_type,
            self.market_id,
            self.payload,
            self.event_id or str(uuid4()),
            self.timestamp or datetime.now(UTC).isoformat(),
            self.schema_version,
            self.correlation_id or str(uuid4()),
        )


class EventPublisher(Protocol):
    def publish(self, event: OperationalEvent) -> str: ...


class LocalEventPublisher:
    def __init__(self, path: Path):
        self.path = path
        self.seen: set[str] = set()

    def publish(self, event: OperationalEvent) -> str:
        event = event.normalized()
        if event.event_id in self.seen:
            return event.event_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event)) + "\n")
        self.seen.add(event.event_id)
        return event.event_id


class PubSubEventPublisher:
    def __init__(self, project_id: str, topic: str):
        from google.cloud import pubsub_v1

        self.client = pubsub_v1.PublisherClient()
        self.topic_path = self.client.topic_path(project_id, topic)

    def publish(self, event: OperationalEvent) -> str:
        payload = json.dumps(asdict(event.normalized())).encode()
        return str(self.client.publish(self.topic_path, payload).result())
