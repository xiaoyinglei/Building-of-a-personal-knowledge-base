from __future__ import annotations

from dataclasses import dataclass

from rag.schema.core import ProcessingStateRecord
from rag.storage.storage_lifecycle_service import StorageLifecycleService


@dataclass(slots=True)
class StorageLifecycleWorker:
    service: StorageLifecycleService
    worker_id: str = "storage-lifecycle-worker"

    def run_once(self, *, lease_seconds: int = 60) -> ProcessingStateRecord | None:
        queue = self.service.lifecycle_queue
        if queue is None:
            return None
        return queue.process_next(
            worker_id=self.worker_id,
            lease_seconds=lease_seconds,
            sync_handler=lambda state: self.service.process_state(state),
        )

    def run_until_idle(self, *, max_tasks: int = 8, lease_seconds: int = 60) -> list[ProcessingStateRecord]:
        processed: list[ProcessingStateRecord] = []
        for _ in range(max(max_tasks, 0)):
            state = self.run_once(lease_seconds=lease_seconds)
            if state is None:
                break
            processed.append(state)
            if state.status == "pending":
                break
        return processed


__all__ = ["StorageLifecycleWorker"]
