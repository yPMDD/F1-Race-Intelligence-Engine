"""Guards for the Alembic migration setup.

These run against a throwaway SQLite database so they stay hermetic and fast.
They exist to catch the two ways a migration setup silently rots:

  1. `alembic upgrade head` stops producing the full schema (e.g. someone
     regenerates it against an already-populated database and gets an empty
     migration, which is a no-op on a fresh Neon instance).
  2. `apps/api/main.py` reintroduces `Base.metadata.create_all()`, which would
     quietly take schema ownership back from Alembic.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_alembic(args, db_url, cwd=ROOT):
    env = dict(os.environ, DATABASE_URL=db_url)
    return subprocess.run(
        [sys.executable, "-m", "alembic"] + args,
        capture_output=True, text=True, env=env, cwd=cwd,
    )


@pytest.fixture()
def scratch_db(tmp_path):
    return f"sqlite:///{tmp_path / 'mig.db'}"


def test_upgrade_creates_every_table(scratch_db):
    r = run_alembic(["upgrade", "head"], scratch_db)
    assert r.returncode == 0, r.stderr

    import sqlalchemy
    eng = sqlalchemy.create_engine(scratch_db)
    tables = set(sqlalchemy.inspect(eng).get_table_names())
    assert tables == {
        "races", "drivers", "laps", "results", "predictions", "alembic_version",
    }, f"migration did not create the full schema: {tables}"


def test_downgrade_drops_everything(scratch_db):
    assert run_alembic(["upgrade", "head"], scratch_db).returncode == 0
    r = run_alembic(["downgrade", "base"], scratch_db)
    assert r.returncode == 0, r.stderr

    import sqlalchemy
    eng = sqlalchemy.create_engine(scratch_db)
    tables = set(sqlalchemy.inspect(eng).get_table_names())
    assert tables == {"alembic_version"}, f"downgrade left tables behind: {tables}"


def test_upgrade_is_idempotent(scratch_db):
    """Running it twice must not error -- CI re-runs are common."""
    assert run_alembic(["upgrade", "head"], scratch_db).returncode == 0
    r = run_alembic(["upgrade", "head"], scratch_db)
    assert r.returncode == 0, r.stderr


def test_current_is_head_after_upgrade(scratch_db):
    assert run_alembic(["upgrade", "head"], scratch_db).returncode == 0
    r = run_alembic(["current"], scratch_db)
    assert "head" in r.stdout, r.stdout


def test_app_does_not_create_tables_at_import():
    """create_all() must not run at import time -- Alembic owns the schema."""
    src = open(os.path.join(ROOT, "apps", "api", "main.py"), encoding="utf-8").read()
    unguarded = "Base.metadata.create_all(bind=engine)" in src
    guarded = 'if os.getenv("AUTO_CREATE_TABLES") == "1"' in src
    assert not unguarded or guarded, (
        "apps/api/main.py calls create_all() without the AUTO_CREATE_TABLES "
        "guard. create_all() never alters existing tables, so it would mask "
        "real migrations."
    )


def test_alembic_ini_has_no_committed_connection_string():
    """A connection string in alembic.ini would leak prod credentials."""
    ini = open(os.path.join(ROOT, "alembic.ini"), encoding="utf-8").read()
    for line in ini.splitlines():
        if line.strip().startswith("sqlalchemy.url"):
            assert line.split("=", 1)[1].strip() == "", (
                "alembic.ini must leave sqlalchemy.url blank; env.py resolves "
                "it from DATABASE_URL."
            )


def test_migrations_directory_is_not_gitignored():
    """A .gitignore rule that swallows alembic/ breaks deploys silently."""
    r = subprocess.run(
        ["git", "check-ignore", "alembic/versions"], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert r.returncode != 0, "alembic/ is gitignored - migrations would never deploy"
