"""Unified test defaults isolated from a developer's local .env file."""

import pytest


@pytest.fixture(autouse=True)
def stable_agent_defaults(monkeypatch):
    monkeypatch.setenv("RECENT_EXCHANGES_TO_KEEP", "5")
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")
    # Pin the corpus target. `qubettera.rag` calls load_dotenv() at import, so a
    # developer's .env decides the ambient target; without this a test that
    # assumes the local default can be pointed at their real Supabase project.
    monkeypatch.setenv("QUBETTERA_DB", "local")

