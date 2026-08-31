"""Tests for capacity-control database helper methods."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_orchestrator.database import (
    CAPACITY_POOL_FLOOR_ROUTE_KEY,
    DatabaseClient,
    normalize_capacity_route_key,
)


class _Result:
    def __init__(self, data: Any = None) -> None:
        self.data = data


class _Query:
    def __init__(self, db: "_FakeSupabase", table_name: str) -> None:
        self.db = db
        self.table_name = table_name
        self.operation = "select"
        self.payload: Any = None
        self.filters: list[tuple[str, Any]] = []
        self.limit_count: int | None = None
        self.order_by: tuple[str, bool] | None = None
        self.upsert_conflict: str | None = None

    def select(self, *_args: Any, **_kwargs: Any) -> "_Query":
        self.operation = "select"
        return self

    def insert(self, payload: Any) -> "_Query":
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload: dict[str, Any]) -> "_Query":
        self.operation = "update"
        self.payload = payload
        return self

    def upsert(self, payload: dict[str, Any], on_conflict: str | None = None) -> "_Query":
        self.operation = "upsert"
        self.payload = payload
        self.upsert_conflict = on_conflict
        return self

    def delete(self) -> "_Query":
        self.operation = "delete"
        return self

    def eq(self, column: str, value: Any) -> "_Query":
        self.filters.append((column, value))
        return self

    def limit(self, count: int) -> "_Query":
        self.limit_count = count
        return self

    def order(self, column: str, desc: bool = False) -> "_Query":
        self.order_by = (column, desc)
        return self

    def single(self) -> "_Query":
        self.limit_count = 1
        return self

    def execute(self) -> _Result:
        rows = self.db.tables.setdefault(self.table_name, [])
        self.db.calls.append(
            {
                "table": self.table_name,
                "operation": self.operation,
                "payload": deepcopy(self.payload),
                "filters": list(self.filters),
                "on_conflict": self.upsert_conflict,
            }
        )

        if self.operation == "insert":
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            inserted = []
            for payload in payloads:
                row = deepcopy(payload)
                row.setdefault("id", f"{self.table_name}-{len(rows) + 1}")
                rows.append(row)
                inserted.append(deepcopy(row))
            return _Result(inserted)

        if self.operation == "update":
            matched = self._matched_rows(rows)
            for row in matched:
                row.update(deepcopy(self.payload))
            return _Result(deepcopy(matched))

        if self.operation == "upsert":
            conflict_cols = [col.strip() for col in (self.upsert_conflict or "id").split(",")]
            existing = None
            for row in rows:
                if all(row.get(col) == self.payload.get(col) for col in conflict_cols):
                    existing = row
                    break
            if existing is None:
                existing = deepcopy(self.payload)
                existing.setdefault("id", f"{self.table_name}-{len(rows) + 1}")
                rows.append(existing)
            else:
                existing.update(deepcopy(self.payload))
            return _Result([deepcopy(existing)])

        if self.operation == "delete":
            matched = self._matched_rows(rows)
            self.db.tables[self.table_name] = [row for row in rows if row not in matched]
            return _Result(deepcopy(matched))

        matched = self._matched_rows(rows)
        if self.order_by:
            column, desc = self.order_by
            matched = sorted(matched, key=lambda row: row.get(column) or "", reverse=desc)
        if self.limit_count is not None:
            matched = matched[: self.limit_count]
        return _Result(deepcopy(matched))

    def _matched_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matched = rows
        for column, expected in self.filters:
            matched = [row for row in matched if self._value(row, column) == expected]
        return matched

    @staticmethod
    def _value(row: dict[str, Any], column: str) -> Any:
        if column.startswith("metadata->>"):
            key = column.split("metadata->>", 1)[1]
            value = (row.get("metadata") or {}).get(key)
            return str(value) if value is not None else None
        return row.get(column)


class _FakeSupabase:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []

    def table(self, name: str) -> _Query:
        return _Query(self, name)


def _client(fake: _FakeSupabase) -> DatabaseClient:
    db = object.__new__(DatabaseClient)
    db.supabase = fake
    return db


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_capacity_intent_insert_update_and_shadow_reads() -> None:
    fake = _FakeSupabase()
    db = _client(fake)

    inserted = _run(
        db.insert_worker_capacity_intent(
            pool="gpu-main",
            desired_capacity=2,
            reason="queued_coverage",
            observed_queued=3,
            observed_active=1,
            observed_spawning=1,
            cycle_id=42,
            shadow=False,
            actions=[{"type": "spawn", "route_key": "route-a"}],
        )
    )
    assert inserted is not None
    assert inserted["shadow"] is False

    assert _run(db.update_worker_capacity_intent_outcome(inserted["id"], outcome={"spawned": 1}))
    latest = _run(db.get_latest_authoritative_capacity_intent("gpu-main"))
    assert latest is not None
    assert latest["outcome"] == {"spawned": 1}

    shadow = _run(
        db.insert_worker_capacity_intent(
            pool="gpu-main",
            desired_capacity=1,
            reason="shadow",
            observer_id="observer-1",
            observation_id="00000000-0000-0000-0000-000000000001",
            shadow=True,
        )
    )
    assert shadow is not None
    observations = _run(db.get_shadow_capacity_observations(observer_id="observer-1"))
    assert [row["id"] for row in observations] == [shadow["id"]]


def test_route_backoff_normalizes_pool_floor_and_resets_on_success() -> None:
    fake = _FakeSupabase()
    db = _client(fake)
    now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)

    assert normalize_capacity_route_key(None) == CAPACITY_POOL_FLOOR_ROUTE_KEY
    expected_delays = [30, 60, 120, 240]
    for index, delay in enumerate(expected_delays, start=1):
        row = _run(db.record_route_spawn_failure("gpu-main", None, now=now, error="spawn_failed"))
        assert row["route_key"] == CAPACITY_POOL_FLOOR_ROUTE_KEY
        assert row["consecutive_spawn_failures"] == index
        assert row["next_spawn_allowed_at"] == (now + timedelta(seconds=delay)).isoformat()

    assert _run(db.reset_route_backoff("gpu-main", None, now=now + timedelta(minutes=10)))
    backoff = _run(db.get_route_backoff("gpu-main", None))
    assert backoff["consecutive_spawn_failures"] == 0
    assert backoff["next_spawn_allowed_at"] is None
    assert backoff["last_spawn_succeeded_at"] == (now + timedelta(minutes=10)).isoformat()


def test_pool_lease_acquire_renew_reject_and_release() -> None:
    fake = _FakeSupabase()
    db = _client(fake)
    now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)

    assert _run(db.acquire_pool_lease(pool="gpu-main", holder_id="a", ttl_seconds=30, now=now))
    assert not _run(db.acquire_pool_lease(pool="gpu-main", holder_id="b", ttl_seconds=30, now=now + timedelta(seconds=5)))
    assert _run(db.acquire_pool_lease(pool="gpu-main", holder_id="b", ttl_seconds=30, now=now + timedelta(seconds=31)))

    lease = fake.tables["orchestrator_leases"][0]
    assert lease["holder_id"] == "b"

    assert _run(db.release_pool_lease(pool="gpu-main", holder_id="b"))
    assert fake.tables["orchestrator_leases"] == []


def test_system_log_worker_metadata_and_capacity_action_lookup(monkeypatch) -> None:
    fake = _FakeSupabase()
    db = _client(fake)
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_POOL", "gpu-vibecomfy-prod")

    assert _run(
        db.insert_system_log_metadata(
            source_id="orchestrator-1",
            message="shadow action",
            metadata={"shadow_intended_action": "true", "pool": "gpu-vibecomfy-prod"},
            cycle_number=7,
        )
    )
    log = fake.tables["system_logs"][0]
    assert log["source_type"] == "orchestrator_gpu"
    assert log["metadata"]["shadow_intended_action"] == "true"

    assert _run(
        db.create_worker_record(
            "worker-1",
            "RTX4090",
            capacity_intent_id="intent-1",
            capacity_action_ordinal=2,
            spawn_reason_route_key=None,
            metadata_update={"extra": "kept"},
        )
    )
    worker = fake.tables["workers"][0]
    assert worker["metadata"]["worker_pool"] == "gpu-vibecomfy-prod"
    assert worker["metadata"]["capacity_intent_id"] == "intent-1"
    assert worker["metadata"]["capacity_action_ordinal"] == 2
    assert worker["metadata"]["spawn_reason_route_key"] == CAPACITY_POOL_FLOOR_ROUTE_KEY
    assert worker["metadata"]["extra"] == "kept"

    found = _run(db.get_worker_by_capacity_action("intent-1", 2))
    assert found is not None
    assert found["id"] == "worker-1"
