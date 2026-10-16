import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Docker-compose publishes Postgres on 5433; a native install is usually 5432.
DEFAULT_URL = "postgresql://f1user:f1password@localhost:5433/f1db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_URL)


def _engine_kwargs(url: str) -> dict:
    """Connection options tuned for both local Postgres and Neon.

    Neon specifics (per their docs):
      * `sslmode=require` is expected on the connection string itself. We only
        inject it for Neon hosts -- forcing it on a local Postgres would break
        any server not built with SSL support.
      * The pooler endpoint (`-pooler` hostname) must be used for the app, the
        direct hostname for migrations. Nothing to change at the SQLAlchemy
        layer.
      * Computes can be suspended and restarted, which leaves stale sockets.
        `pool_pre_ping` discards a dead connection and opens a fresh one rather
        than surfacing "SSL connection has been closed unexpectedly" to a user.
    """
    kwargs: dict = {"pool_pre_ping": True, "future": True}

    is_postgres = url.startswith("postgresql")
    is_neon = "neon.tech" in url

    if is_postgres and is_neon and "sslmode=" not in url:
        kwargs["connect_args"] = {"sslmode": "require"}

    return kwargs


engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
