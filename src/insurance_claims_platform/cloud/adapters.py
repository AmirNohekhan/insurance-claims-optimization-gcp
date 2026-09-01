from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
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
        self.seen = self._load_seen()
        self._lock = Lock()

    def _load_seen(self) -> set[str]:
        if not self.path.exists():
            return set()
        seen: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    event_id = json.loads(line).get("event_id")
                except (json.JSONDecodeError, AttributeError) as exc:
                    raise ValueError(
                        f"Invalid event record at {self.path}:{line_number}"
                    ) from exc
                if event_id:
                    seen.add(str(event_id))
        return seen

    def publish(self, event: OperationalEvent) -> str:
        event = event.normalized()
        with self._lock:
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
