#!/usr/bin/env python3
"""
Apply registered SQL migrations to Supabase.

The runner executes each migration file as one SQL unit. This preserves
PL/pgSQL blocks such as ``DO $$ ... END $$;``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sql_migrations import (
    capacity_schema_verification_sql,
    migration_paths,
    read_sql,
)


def execute_sql(client: Client, sql_text: str) -> Any:
    """Execute a SQL string through the existing Supabase SQL RPC."""

    result = client.rpc("sql", {"query": sql_text})
    if hasattr(result, "execute"):
        return result.execute()
    return result


def connect_supabase() -> Client:
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        sys.exit(1)

    try:
        client: Client = create_client(supabase_url, supabase_key)
    except Exception as exc:
        print(f"Failed to connect to Supabase: {exc}")
        sys.exit(1)

    print("Connected to Supabase")
    return client


def apply_migration_file(client: Client, path: Path) -> None:
    sql_text = read_sql(path).strip()
    if not sql_text:
        print(f"Skipping empty migration {path.name}")
        return
    execute_sql(client, sql_text)


def apply_sql_migrations(*, verify_capacity_schema: bool = True) -> None:
    client = connect_supabase()

    print("Applying registered SQL migrations...")
    for path in migration_paths():
        if not path.exists():
            print(f"Missing migration: {path}")
            sys.exit(1)

        print(f"Applying {path.name}...")
        try:
            apply_migration_file(client, path)
        except Exception as exc:
            print(f"Failed to apply {path.name}: {exc}")
            print("No further migrations were applied.")
            sys.exit(1)

    if verify_capacity_schema:
        print("Verifying capacity-control schema...")
        try:
            execute_sql(client, capacity_schema_verification_sql())
        except Exception as exc:
            print(f"Capacity schema verification failed: {exc}")
            sys.exit(1)

    print("SQL migrations completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-verify-capacity-schema",
        action="store_true",
        help="Skip the post-migration capacity schema verification block.",
    )
    args = parser.parse_args()
    apply_sql_migrations(verify_capacity_schema=not args.no_verify_capacity_schema)


if __name__ == "__main__":
    main()
