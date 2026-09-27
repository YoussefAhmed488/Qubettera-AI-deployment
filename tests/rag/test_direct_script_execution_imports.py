"""Regression tests for importing every unified RAG module."""

from __future__ import annotations

import importlib

import pytest

@pytest.mark.parametrize(
    "module_name",
    [
        "clean",
        "collection",
        "spider",
        "chunk",
        "embed",
        "retrieve",
        "evaluate",
        "store",
        "migrate",
        "run_pipeline",
    ],
)
def test_unified_rag_module_imports(module_name: str):
    if module_name == "spider":
        pytest.importorskip("scrapling")
    assert importlib.import_module(f"qubettera.rag.{module_name}")
