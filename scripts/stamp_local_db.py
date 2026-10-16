"""Stamp the existing local database as being at the baseline revision.

The local Postgres was populated by `Base.metadata.create_all()` and by the
ingestion worker before Alembic existed. We mark it as already at `base` so the
initial migration can be generated as a pure schema declaration rather than an
attempt to recreate tables that already exist.
"""
import sys

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, ".")

from storage.postgres.database import DATABASE_URL

cfg = Config("alembic.ini")
cfg.set_main_option("sqlalchemy.url", DATABASE_URL)

engine = create_engine(DATABASE_URL)
existing = set(inspect(engine).get_table_names()) - {"alembic_version"}

print("target :", DATABASE_URL.split("@")[-1])
print("tables :", sorted(existing))

if not existing:
    print("\nDatabase is empty -> nothing to stamp; 'alembic upgrade head'")
    print("will create every table from the initial migration.")
    raise SystemExit(0)

print("\nStamping revision 'base' (pre-Alembic schema).")
command.stamp(cfg, "base")
print("Done. Next: alembic upgrade head")
