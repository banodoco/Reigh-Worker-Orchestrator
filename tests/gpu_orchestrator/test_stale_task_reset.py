"""Regression coverage for stale assigned-task timeout classes."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_orchestrator.database import DatabaseClient


class _TaskQuery:
    def __init__(self, rows):
        self.rows = rows
        self.selected_columns = ""
        self.update_payload = None
        self.updated_ids = []

    @property
    def not_(self):
        return self

    def select(self, columns):
        self.selected_columns = columns
        return self

    def eq(self, *_args):
        return self

    def is_(self, *_args):
        return self

    def like(self, *_args):
        return self

    def lt(self, *_args):
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def in_(self, _column, values):
        self.updated_ids = list(values)
        return self

    def execute(self):
        if self.update_payload is not None:
            return SimpleNamespace(data=[{"id": task_id} for task_id in self.updated_ids])
        return SimpleNamespace(data=self.rows)


class _FakeSupabase:
    def __init__(self, rows):
        self.query = _TaskQuery(rows)

    def table(self, table_name):
        assert table_name == "tasks"
        return self.query


@pytest.mark.asyncio
async def test_stale_reset_uses_two_hour_window_for_long_running_tasks():
    now = datetime.now(timezone.utc)
    rows = [
        {
            "id": "ordinary-60m",
            "task_type": "image_generation",
            "worker_id": "worker-1",
            "updated_at": (now - timedelta(minutes=60)).isoformat(),
        },
        {
            "id": "travel-60m",
            "task_type": "travel_segment",
            "worker_id": "worker-2",
            "updated_at": (now - timedelta(minutes=60)).isoformat(),
        },
        {
            "id": "travel-130m",
            "task_type": "travel_segment",
            "worker_id": "worker-3",
            "updated_at": (now - timedelta(minutes=130)).isoformat(),
        },
        {
            "id": "orchestrator-60m",
            "task_type": "travel_orchestrator",
            "worker_id": "worker-4",
            "updated_at": (now - timedelta(minutes=60)).isoformat(),
        },
        {
            "id": "orchestrator-130m",
            "task_type": "travel_orchestrator",
            "worker_id": "worker-5",
            "updated_at": (now - timedelta(minutes=130)).isoformat(),
        },
    ]
    fake_supabase = _FakeSupabase(rows)
    client = DatabaseClient.__new__(DatabaseClient)
    client.supabase = fake_supabase

    reset_count = await client.reset_stale_assigned_tasks(timeout_minutes=30)

    assert "updated_at" in fake_supabase.query.selected_columns
    assert fake_supabase.query.updated_ids == [
        "ordinary-60m",
        "travel-130m",
        "orchestrator-130m",
    ]
    assert reset_count == 3
