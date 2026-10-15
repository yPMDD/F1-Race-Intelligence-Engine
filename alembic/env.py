"""Alembic environment.

Deliberately side-effect free: importing this module must NOT create or modify
any table. Schema changes only happen when `alembic upgrade head` is run
explicitly, which is what the CI deploy job does.

The database URL is read from DATABASE_URL at runtime rather than from
alembic.ini, so the same migration scripts run against local Postgres and
against Neon without edits and without a connection string in version control.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project importable when alembic is invoked from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from storage.postgres.database import DATABASE_URL
from storage.postgres.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", DATABASE_URL)

target_metadata = Base.metadata


def _is_sqlite() -> bool:
    return DATABASE_URL.startswith("sqlite")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting ('alembic upgrade head --sql')."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table for us so migrations stay portable between dev and Neon.
        render_as_batch=_is_sqlite(),
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=_is_sqlite(),
            compare_type=True,
            # Neon compute can be suspended/restarted; drop connections the
            # server has already closed instead of erroring out.
            connect_args={"pool_pre_ping": True} if not _is_sqlite() else {},
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
