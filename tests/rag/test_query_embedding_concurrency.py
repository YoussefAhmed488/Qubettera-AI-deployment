from concurrent.futures import ThreadPoolExecutor
import importlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest

retrieve = importlib.import_module("qubettera.rag.retrieve")


@pytest.fixture(autouse=True)
def local_embedding_provider(monkeypatch):
    """These tests cover in-process weight loading, so pin the local provider.

    ``EMBEDDING_PROVIDER`` defaults to ``cloud``, which returns a stateless HTTP
    client and would bypass the SentenceTransformer construction under test.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")


def test_parallel_queries_load_weights_once_and_serialize_encoding(monkeypatch):
    barrier = threading.Barrier(5)
    calls = []
    active = 0
    peak = 0

    class Model:
        prompts = {"query": "prompt"}

        def encode(self, query, **kwargs):
            nonlocal active, peak
            assert kwargs["prompt_name"] == "query"
            active += 1
            peak = max(peak, active)
            time.sleep(.01)
            active -= 1
            return [query]

    model = Model()

    def construct(*args, **kwargs):
        calls.append(kwargs)
        time.sleep(.03)
        return model

    monkeypatch.setattr(retrieve, "_embed_model", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=construct))

    def query(index):
        barrier.wait(timeout=5)
        loaded = retrieve._get_embed_model()
        assert loaded is model
        return retrieve._encode_query(loaded, str(index))

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(query, range(5)))
    assert results == [[str(i)] for i in range(5)]
    assert len(calls) == 1
    assert peak == 1


def test_model_initialization_failure_does_not_poison_cache(monkeypatch):
    calls = []
    model = object()

    def construct(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary model load failure")
        return model

    monkeypatch.setattr(retrieve, "_embed_model", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=construct))
    with pytest.raises(RuntimeError, match="temporary"):
        retrieve._get_embed_model()
    assert retrieve._get_embed_model() is model
    assert len(calls) == 2
