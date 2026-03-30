from __future__ import annotations

import json
from pathlib import Path

from pkp.schema._types.telemetry import TelemetryEvent


class LocalEventRepo:
    def __init__(self, sink_path: Path | None = None) -> None:
        self._sink_path = sink_path
        self._events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> None:
        self._events.append(event)
        if self._sink_path is None:
            return

        self._sink_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(event.model_dump(mode="json"), ensure_ascii=True)
        with self._sink_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")

    def list_events(self) -> list[TelemetryEvent]:
        return list(self._events)
