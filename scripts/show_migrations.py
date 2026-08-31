#!/usr/bin/env python3
"""Display registered SQL migrations for manual Supabase execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sql_migrations import (
    capacity_schema_verification_sql,
    migration_paths,
    project_root,
    read_sql,
)


def show_migrations(*, include_capacity_verification: bool = True) -> None:
    root = project_root()

    print("SQL Migrations")
    print("=" * 60)
    print("Copy these scripts into the Supabase SQL Editor and run them in order.")

    for index, path in enumerate(migration_paths(), 1):
        if not path.exists():
            print(f"\nMissing migration file: {path.relative_to(root)}")
            continue

        print(f"\n{'=' * 60}")
        print(f"MIGRATION {index}: {path.name}")
        print(f"File: {path.relative_to(root)}")
        print(f"{'=' * 60}")
        print(read_sql(path))
        print(f"\n{'=' * 60}")
        print(f"END OF MIGRATION {index}")
        print(f"{'=' * 60}")

    if include_capacity_verification:
        print(f"\n{'=' * 60}")
        print("CAPACITY SCHEMA VERIFICATION")
        print(f"{'=' * 60}")
        print(capacity_schema_verification_sql())
        print(f"\n{'=' * 60}")
        print("END CAPACITY SCHEMA VERIFICATION")
        print(f"{'=' * 60}")

    print("\nAfter applying migrations, run:")
    print("   python3 scripts/apply_sql_migrations.py --no-verify-capacity-schema")
    print("or paste only the CAPACITY SCHEMA VERIFICATION block above to verify manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-capacity-verification",
        action="store_true",
        help="Do not print the capacity schema verification SQL block.",
    )
    args = parser.parse_args()
    show_migrations(include_capacity_verification=not args.no_capacity_verification)


if __name__ == "__main__":
    main()
