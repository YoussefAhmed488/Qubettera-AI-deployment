"""Explicit memory or durable PostgreSQL checkpoint configuration.

Usage:
    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        result = graph.invoke(...)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_PG_VARS = ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")


def _make_conninfo() -> str:
    """Build a psycopg connection string from PG* environment variables."""
    from psycopg.conninfo import make_conninfo  # type: ignore[import]

    return make_conninfo(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        connect_timeout="10",
    )


def _supabase_conninfo() -> str:
    """Build a connection string for the Supabase target.

    Reuses the shared connection helper so the URI is resolved exactly as the
    RAG side resolves it, then pins the search_path: LangGraph's own setup
    statements are unqualified and Supabase keeps pgvector in `extensions`.
    """
    from psycopg.conninfo import conninfo_to_dict, make_conninfo  # type: ignore[import]

    from qubettera.rag.database import get_db_config

    settings = get_db_config(purpose="Postgres checkpoints")
    fields = conninfo_to_dict(settings["dsn"])
    fields.setdefault("connect_timeout", "10")
    fields["sslmode"] = settings.get("sslmode", "require")
    fields["options"] = settings.get("options", "")
    return make_conninfo(**fields)


def _target_conninfo() -> str | None:
    """Return an explicit conninfo for a non-default target, else ``None``.

    ``None`` preserves the local behaviour, including its existing
    ``PGHOST``-based error message, so callers fall back to ``_make_conninfo()``.
    """
    from qubettera.rag.database import active_target

    if active_target() == "local":
        return None
    return _supabase_conninfo()


@contextmanager
def open_postgres_checkpointer(conninfo: str | None = None) -> Iterator:
    """Open and initialise a durable LangGraph PostgresSaver.

    Requires langgraph-checkpoint-postgres and PGHOST/PGDATABASE/PGUSER/
    PGPASSWORD in environment. Calls saver.setup() once to ensure the
    checkpoint schema exists.
    """
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import]

    resolved = conninfo or _target_conninfo()
    with PostgresSaver.from_conn_string(resolved or _make_conninfo()) as saver:
        saver.setup()
        yield saver


@contextmanager
def get_checkpointer() -> Iterator:
    """Open the explicitly selected checkpointer.

    Uses PostgresSaver only when ``CHECKPOINT_BACKEND=postgres`` and all
    connection variables are present; missing variables raise an error. RAG settings must
    not silently change agent-memory behavior.
    """
    backend = os.environ.get("CHECKPOINT_BACKEND", "memory").strip().lower()
    if backend not in {"memory", "postgres"}:
        raise ValueError("CHECKPOINT_BACKEND must be memory or postgres.")
    if backend == "postgres":
        # Supabase resolves its connection from a URI, so the PG* variables are
        # only required for the local target (whose error message is asserted).
        if _target_conninfo() is None:
            missing = [name for name in _REQUIRED_PG_VARS if not os.environ.get(name)]
            if missing:
                raise ValueError("PostgreSQL checkpoints require: " + ", ".join(missing))
        with open_postgres_checkpointer() as saver:
            yield saver
    else:
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
