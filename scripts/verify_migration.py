"""Verify the Alembic migration reproduces the live schema exactly.

Runs upgrade -> downgrade -> upgrade against a scratch database and diffs the
resulting DDL against the real development database. Catches the classic
mistake of shipping a migration that differs from the schema the app actually
runs on.
"""
import subprocess
import sys
import os

sys.path.insert(0, ".")

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

SCRATCH = "postgresql://f1user:f1password@localhost:5432/f1_migration_scratch"
REAL = "postgresql://f1user:f1password@localhost:5432/f1db"


def schema_snapshot(url):
    insp = inspect(create_engine(url))
    snap = {}
    for table in sorted(insp.get_table_names()):
        if table == "alembic_version":
            continue
        cols = []
        for c in insp.get_columns(table):
            cols.append((c["name"], str(c["type"]), c["nullable"]))
        fks = sorted(
            (tuple(fk["constrained_columns"]), fk["referred_table"],
             tuple(fk["referred_columns"]))
            for fk in insp.get_foreign_keys(table)
        )
        idx = sorted((i["name"], tuple(i["column_names"]), i["unique"])
                     for i in insp.get_indexes(table))
        uqs = sorted(tuple(u["column_names"]) for u in insp.get_unique_constraints(table))
        snap[table] = (cols, fks, idx, uqs)
    return snap


def alembic_cmd(args, url):
    env = dict(os.environ, DATABASE_URL=url)
    r = subprocess.run(
        [".venv\\Scripts\\alembic.exe"] + args,
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        print(f"FAILED: alembic {' '.join(args)}\n{r.stdout}\n{r.stderr}")
        sys.exit(1)
    return r


cfg = lambda url: (lambda c: (c.set_main_option("sqlalchemy.url", url), c)[1])(
    Config("alembic.ini"))

print("1. upgrade head on scratch ...")
alembic_cmd(["upgrade", "head"], SCRATCH)
fresh = schema_snapshot(SCRATCH)
print("   ->", list(fresh))

print("2. downgrade base ...")
alembic_cmd(["downgrade", "base"], SCRATCH)
empty = [t for t in schema_snapshot(SCRATCH)]
print("   ->", empty, "(expect no tables)")

print("3. upgrade head again ...")
alembic_cmd(["upgrade", "head"], SCRATCH)
again = schema_snapshot(SCRATCH)
print("   ->", list(again))

print("4. upgrade/downgrade is idempotent:", fresh == again)

print()
print("5. diff migration-produced schema vs live development schema ...")
live = schema_snapshot(REAL)
all_tables = sorted(set(fresh) | set(live))
drift = False
for t in all_tables:
    a, b = fresh.get(t), live.get(t)
    if a == b:
        print(f"   OK   {t}")
    else:
        drift = True
        print(f"   DRIFT {t}")
        if a is None:
            print("        missing from migration")
        elif b is None:
            print("        missing from live db")
        else:
            for label, x, y in zip(("columns", "fks", "indexes", "uniques"), a, b):
                if x != y:
                    print(f"        {label}:\n          migration={x}\n          live     ={y}")

print()
print("RESULT:", "SCHEMA DRIFT DETECTED" if drift else "MIGRATION MATCHES LIVE SCHEMA")
sys.exit(1 if drift else 0)
