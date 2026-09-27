"""Tests for the local/Supabase target switch and the migration guards.

Nothing here touches a real database: the target resolver and the connection
keywords it produces are pure functions, so they can be asserted directly.
"""

from __future__ import annotations

import json
import stat

import pytest

from qubettera.rag import database


@pytest.fixture(autouse=True)
def clean_target_env(monkeypatch):
    monkeypatch.delenv("QUBETTERA_DB", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("PGSSLMODE", raising=False)
    monkeypatch.delenv("PGOPTIONS", raising=False)
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "ragdb")
    monkeypatch.setenv("PGUSER", "postgres")
    monkeypatch.setenv("PGPASSWORD", "change_me")


def test_local_is_the_default_target():
    assert database.active_target() == "local"
    assert database.get_db_config()["host"] == "localhost"


def test_unknown_target_is_rejected(monkeypatch):
    monkeypatch.setenv("QUBETTERA_DB", "sqlite")
    with pytest.raises(ValueError, match="QUBETTERA_DB"):
        database.active_target()


def test_use_target_overrides_and_restores():
    with database.use_target("supabase"):
        assert database.active_target() == "supabase"
    assert database.active_target() == "local"


def test_use_target_rejects_unknown_name():
    with pytest.raises(ValueError, match="target must be one of"):
        with database.use_target("mysql"):
            pass


def test_search_path_omits_pg_catalog_and_spaces():
    """`pg_catalog` must stay implicit, and spaces break the libpq option."""
    assert "pg_catalog" not in database.SUPABASE_SEARCH_PATH
    assert " " not in database.SUPABASE_SEARCH_PATH
    assert "extensions" in database.SUPABASE_SEARCH_PATH.split(",")


def test_supabase_config_pins_search_path_and_ssl(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:pw@host:5432/postgres")
    with database.use_target("supabase"):
        config = database.get_db_config()
    assert config["sslmode"] == "require"
    assert config["options"] == f"-c search_path={database.SUPABASE_SEARCH_PATH}"


def test_supabase_config_rejects_bad_uri(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "not-a-uri")
    with database.use_target("supabase"):
        with pytest.raises(RuntimeError, match="Invalid Supabase connection URI"):
            database.get_db_config()


def test_supabase_config_requires_a_host(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql:///dbname")
    with database.use_target("supabase"):
        with pytest.raises(RuntimeError, match="must include a host"):
            database.get_db_config()


def test_missing_target_file_explains_the_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "TARGET_FILE", tmp_path / "absent.json")
    with database.use_target("supabase"):
        with pytest.raises(RuntimeError, match="supabase-init"):
            database.get_db_config()


def test_statement_timeout_replaces_search_path_once(monkeypatch):
    """Supabase must send search_path and timeout as a single options string."""
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:pw@host:5432/postgres")
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(database.psycopg2, "connect", fake_connect)
    with database.use_target("supabase"):
        with pytest.raises(RuntimeError, match="stop"):
            database.connect(statement_timeout_ms=1000)
    options = captured["options"]
    assert options.count("-c") == 2
    assert "search_path=" in options
    assert "statement_timeout=1000" in options


class TestWriteTargetFile:
    URL = "postgresql://postgres:pw@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

    def test_writes_owner_only_file(self, monkeypatch, tmp_path):
        target = tmp_path / "supabase.target.json"
        monkeypatch.setattr(database, "TARGET_FILE", target)
        database.write_target_file(self.URL)
        assert json.loads(target.read_text()) == {"url": self.URL}
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_refuses_to_clobber_without_force(self, monkeypatch, tmp_path):
        target = tmp_path / "supabase.target.json"
        monkeypatch.setattr(database, "TARGET_FILE", target)
        database.write_target_file(self.URL)
        with pytest.raises(FileExistsError, match="--force"):
            database.write_target_file(self.URL)
        database.write_target_file(self.URL, overwrite=True)

    def test_rejects_direct_ipv6_only_host(self, monkeypatch, tmp_path):
        monkeypatch.setattr(database, "TARGET_FILE", tmp_path / "t.json")
        with pytest.raises(ValueError, match="session pooler"):
            database.write_target_file(
                "postgresql://postgres:pw@db.abcdefghijklm.supabase.co:5432/postgres"
            )

    def test_rejects_uri_without_password(self, monkeypatch, tmp_path):
        monkeypatch.setattr(database, "TARGET_FILE", tmp_path / "t.json")
        with pytest.raises(ValueError, match="password"):
            database.write_target_file("postgresql://postgres@host:5432/postgres")

    def test_rejects_garbage_uri(self, monkeypatch, tmp_path):
        monkeypatch.setattr(database, "TARGET_FILE", tmp_path / "t.json")
        with pytest.raises(ValueError, match="Not a valid PostgreSQL connection URI"):
            database.write_target_file("not-a-uri")


def test_describe_target_redacts_credentials(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://postgres:secret@host:5432/db")
    with database.use_target("supabase"):
        assert "secret" not in database.describe_target()


def test_migrate_refuses_when_target_is_local(monkeypatch):
    """QUBETTERA_DB is set explicitly: importing `qubettera.rag` loads a
    developer's .env, so the ambient default is not reliably `local`."""
    from qubettera.rag import migrate

    monkeypatch.setenv("QUBETTERA_DB", "local")
    with pytest.raises(RuntimeError, match="Refusing to migrate"):
        migrate.run()


def test_migrate_rejects_conflicting_flags(monkeypatch):
    from qubettera.rag import migrate

    monkeypatch.setenv("QUBETTERA_DB", "supabase")
    with pytest.raises(ValueError, match="not both"):
        migrate.run(publish_only=True, stage_only=True)


def test_chunk_columns_exclude_generated_column():
    """`text_search` is GENERATED ALWAYS and cannot appear in a COPY list."""
    from qubettera.rag import store

    assert "text_search" not in store.CHUNK_COLUMNS
    assert "embedding" in store.CHUNK_COLUMNS


class TestMigrateCliReportsFailures:
    """`rag migrate` must report failures as messages, not tracebacks.

    A migration is run by hand at a terminal, so the guards have to be
    actionable on their own rather than wrapped in an interpreter stack trace.
    """

    def test_local_target_exits_two_with_a_clean_message(self, monkeypatch, capsys):
        from qubettera import cli

        monkeypatch.setenv("QUBETTERA_DB", "local")
        exit_code = cli.main(["rag", "migrate"])

        captured = capsys.readouterr()
        assert exit_code == 2
        assert "Refusing to migrate" in captured.err
        assert "Traceback" not in captured.err + captured.out

    def test_conflicting_flags_are_rejected_by_the_parser(self, monkeypatch, capsys):
        """The flags are mutually exclusive, so argparse rejects them cleanly."""
        from qubettera import cli

        monkeypatch.setenv("QUBETTERA_DB", "supabase")
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["rag", "migrate", "--stage-only", "--publish-only"])

        captured = capsys.readouterr()
        assert excinfo.value.code == 2
        assert "not allowed with argument" in captured.err
        assert "Traceback" not in captured.err + captured.out
