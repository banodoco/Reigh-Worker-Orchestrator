"""Regression tests for SQL migration registration and execution helpers."""

from __future__ import annotations

from pathlib import Path

from scripts import apply_sql_migrations
from scripts.sql_migrations import (
    CAPACITY_MIGRATION_FILE,
    CAPACITY_POOL_FLOOR_ROUTE_KEY,
    MIGRATION_FILES,
    capacity_schema_verification_sql,
    migration_paths,
)


class _FakeRpcResult:
    def execute(self) -> dict[str, bool]:
        return {"ok": True}


class _FakeClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rpc(self, name: str, params: dict[str, str]) -> _FakeRpcResult:
        assert name == "sql"
        self.queries.append(params["query"])
        return _FakeRpcResult()


def test_capacity_migration_is_registered_and_exists() -> None:
    assert CAPACITY_MIGRATION_FILE in MIGRATION_FILES
    assert migration_paths() == sorted(migration_paths())
    for path in migration_paths():
        assert path.exists(), f"registered migration is missing: {path}"


def test_apply_migration_file_preserves_do_block_boundaries(tmp_path: Path) -> None:
    migration = tmp_path / "20260514000000_do_block_fixture.sql"
    sql_text = """
CREATE TABLE IF NOT EXISTS demo_semicolon_split_guard (id int);

DO $$
BEGIN
    IF 1 = 1 THEN
        RAISE NOTICE 'semicolon inside block; must stay in one execute call';
    END IF;
END $$;
""".strip()
    migration.write_text(sql_text, encoding="utf-8")

    client = _FakeClient()
    apply_sql_migrations.apply_migration_file(client, migration)

    assert client.queries == [sql_text]


def test_capacity_schema_verification_covers_required_surfaces() -> None:
    verification_sql = capacity_schema_verification_sql()

    assert "worker_capacity_intents" in verification_sql
    assert "worker_capacity_route_backoffs" in verification_sql
    assert "orchestrator_leases" in verification_sql
    assert CAPACITY_POOL_FLOOR_ROUTE_KEY in verification_sql
    assert "UNIQUE (pool, route_key)" in verification_sql
    assert "valid_source_type" in verification_sql
    assert "capacity_reconciler" in verification_sql
