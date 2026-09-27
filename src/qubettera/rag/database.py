"""Shared PostgreSQL configuration and connection helpers.

Two targets are supported, selected by ``QUBETTERA_DB``:

- ``local`` (default): the ``PGHOST``/``PGPORT``/``PGDATABASE``/``PGUSER``/
  ``PGPASSWORD`` variables, pointing at the bundled local Postgres.
- ``supabase``: the same variables, resolved from a single gitignored
  connection URI (``SUPABASE_DB_URL`` or the file written by
  ``qubettera rag supabase-init``).

Supabase direct connections (``db.<ref>.supabase.co``) are IPv6-only unless the
project has the paid IPv4 add-on; on an IPv4-only network use the shared pooler
endpoint shown in the dashboard. Schema changes and bulk loads need one
long-lived session, so use session mode (port 5432), not transaction mode.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from qubettera.paths import RESOURCES_DIR

load_dotenv()

_ENV_TO_CONNECT_KEY = {
    "PGHOST": "host",
    "PGPORT": "port",
    "PGDATABASE": "dbname",
    "PGUSER": "user",
    "PGPASSWORD": "password",
}

TARGET_FILE = RESOURCES_DIR / "supabase.target.json"
VALID_TARGETS = ("local", "supabase")

CORPUS_SCHEMA = "public"
EXTENSION_SCHEMA = "extensions"
SUPABASE_SCHEMAS = (CORPUS_SCHEMA, EXTENSION_SCHEMA)

# Supabase installs pgvector into the `extensions` schema, while the retrieval
# SQL uses bare `::vector` casts and `<=>` operators. The managed `postgres`
# role does not have `extensions` on its search_path, so every Supabase
# connection pins one explicitly. `SUPABASE_SCHEMAS` is the single source of
# truth for that list; the migration derives its own search_path from it.
#
# `pg_catalog` must NOT be listed: it is already searched implicitly ahead of
# any explicit entry, but naming it makes it the *current* schema, which breaks
# unqualified `CREATE TABLE` (permission denied) and `current_schema()`.
#
# No spaces are allowed around the commas: the value is passed through libpq's
# `options` startup string, which is split on whitespace, so `public, extensions`
# would be truncated to `public,` and rejected by the server.
SUPABASE_SEARCH_PATH = ",".join(SUPABASE_SCHEMAS)

_target_override: ContextVar[str | None] = ContextVar("qubettera_db_target", default=None)


def active_target() -> str:
    """Return the configured target name, rejecting unknown values."""
    override = _target_override.get()
    if override is not None:
        return override
    target = os.environ.get("QUBETTERA_DB", "local").strip().lower()
    if target not in VALID_TARGETS:
        raise ValueError(
            f"QUBETTERA_DB must be one of {', '.join(VALID_TARGETS)}, got {target!r}."
        )
    return target


def describe_target() -> str:
    """Return a redacted, human-readable label for the active target."""
    if active_target() != "supabase":
        host = os.environ.get("PGHOST", "?")
        database = os.environ.get("PGDATABASE", "?")
        return f"local ({host}/{database})"
    source = "SUPABASE_DB_URL" if os.environ.get("SUPABASE_DB_URL") else str(TARGET_FILE)
    return f"supabase (credentials from {source})"


def _read_target_file(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Supabase target file not found at {path}. Run "
            f"'qubettera rag supabase-init --url <connection-uri>' first."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not payload.get("url"):
        raise RuntimeError(
            f"{path} must be a JSON object containing a non-empty 'url' key."
        )
    return {str(key): str(value) for key, value in payload.items()}


def _supabase_settings() -> dict[str, str]:
    """Resolve the Supabase connection URI into libpq keyword arguments."""
    from psycopg2.extensions import parse_dsn

    override = os.environ.get("SUPABASE_DB_URL")
    if override:
        settings: dict[str, str] = {"dsn": override}
    else:
        payload = _read_target_file(TARGET_FILE)
        settings = {"dsn": payload.pop("url")}
        settings.update(payload)

    if isinstance(os.environ.get("PGSSLMODE"), str):
        settings["sslmode"] = str(os.environ["PGSSLMODE"])
    settings.setdefault("sslmode", "require")
    settings["options"] = _merge_options(settings.get("options"), None)

    try:
        fields = parse_dsn(settings["dsn"])
    except psycopg2.ProgrammingError as error:
        raise RuntimeError(f"Invalid Supabase connection URI: {error}") from error
    if not fields.get("host"):
        raise RuntimeError(
            "The Supabase connection URI must include a host (e.g. "
            "aws-0-<region>.pooler.supabase.com)."
        )
    return settings


def _merge_options(existing: str | None, statement_timeout_ms: int | None) -> str:
    """Combine search_path and statement_timeout into one libpq options string.

    Both reach the server as startup parameters, so they must share a single
    ``-c`` option list; passing them separately would drop one.
    """
    parts = [f"-c search_path={SUPABASE_SEARCH_PATH}"]
    if statement_timeout_ms is not None:
        parts.append(f"-c statement_timeout={statement_timeout_ms}")
    elif existing:
        parts.append(existing)
    elif os.environ.get("PGOPTIONS"):
        parts.append(str(os.environ["PGOPTIONS"]))
    return " ".join(parts)


def get_db_config(*, purpose: str = "the pipeline") -> dict[str, str]:
    if active_target() == "supabase":
        return _supabase_settings()

    values = {name: os.environ.get(name) for name in _ENV_TO_CONNECT_KEY}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing PostgreSQL configuration env vars: "
            + ", ".join(missing)
            + f". Set them before running {purpose}."
        )
    config = {
        connect_key: str(values[env_name])
        for env_name, connect_key in _ENV_TO_CONNECT_KEY.items()
    }
    if os.environ.get("PGSSLMODE"):
        config["sslmode"] = str(os.environ["PGSSLMODE"])
    if os.environ.get("PGOPTIONS"):
        config["options"] = str(os.environ["PGOPTIONS"])
    return config


@contextmanager
def use_target(target: str) -> Iterator[str]:
    """Temporarily connect to ``target`` regardless of ``QUBETTERA_DB``.

    Lets one process read from local Postgres and write to Supabase, which is
    what the migration needs. The override is scoped to the calling context.
    """
    if target not in VALID_TARGETS:
        raise ValueError(
            f"target must be one of {', '.join(VALID_TARGETS)}, got {target!r}."
        )
    token = _target_override.set(target)
    try:
        yield target
    finally:
        _target_override.reset(token)


def write_target_file(url: str, *, overwrite: bool = False) -> Path:
    """Persist a Supabase connection URI to the gitignored target file.

    The file is created with owner-only permissions because it contains the
    database password. A direct-connection host is rejected: those endpoints
    resolve to IPv6 only unless the project has the paid IPv4 add-on, which is
    a failure that would otherwise appear much later as a connect timeout.
    """
    from psycopg2.extensions import parse_dsn

    candidate = url.strip()
    try:
        fields = parse_dsn(candidate)
    except psycopg2.ProgrammingError as error:
        raise ValueError(f"Not a valid PostgreSQL connection URI: {error}") from error
    if not fields.get("host"):
        raise ValueError("The connection URI must include a host.")
    if not fields.get("password"):
        raise ValueError(
            "The connection URI must include the database password; copy the full "
            "string from the Supabase dashboard (Connect -> Session pooler)."
        )

    host = fields["host"]
    if host.startswith("db.") and host.endswith(".supabase.co"):
        raise ValueError(
            f"{host!r} is a direct Supabase endpoint, which is IPv6-only unless the "
            "project has the IPv4 add-on enabled. Use the session pooler URI instead "
            "(host aws-0-<region>.pooler.supabase.com, port 5432, "
            "user postgres.<project-ref>)."
        )

    if TARGET_FILE.exists() and not overwrite:
        raise FileExistsError(
            f"{TARGET_FILE} already exists. Pass --force to replace it."
        )
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(
        json.dumps({"url": candidate}, indent=2) + "\n", encoding="utf-8"
    )
    TARGET_FILE.chmod(0o600)
    return TARGET_FILE


def connect(*, purpose: str = "the pipeline", statement_timeout_ms: int | None = None):
    kwargs: dict[str, object] = get_db_config(purpose=purpose)
    kwargs["connect_timeout"] = 10
    if statement_timeout_ms is not None:
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be > 0")
        if active_target() == "supabase":
            kwargs["options"] = _merge_options(None, statement_timeout_ms)
        else:
            kwargs["options"] = f"-c statement_timeout={statement_timeout_ms}"
    return psycopg2.connect(**kwargs)
