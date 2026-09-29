"""Offline tests for the Modal-hosted embedding and sentiment providers.

Nothing here reaches the network: the HTTP layer is replaced with a fake so the
response parsing, batching and provider dispatch can be asserted directly.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from qubettera.analytics.sentiment import SentimentScorer
from qubettera.rag.cloud_embedding import QUERY_INSTRUCTION, CloudEmbeddingModel
from qubettera.modal_client import (
    CLOUD_MAX_BATCH_SIZE,
    ModalServiceError,
    classify_texts,
    embed_texts,
)

sentiment = importlib.import_module("qubettera.analytics.sentiment")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text_body=None):
        self.status_code = status_code
        self._payload = payload
        self._text_body = text_body

    def json(self):
        if self._text_body is not None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def captured(monkeypatch):
    """Capture outbound POSTs and script the responses."""
    calls = []
    queue = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return queue.pop(0)

    monkeypatch.setattr("qubettera.modal_client.requests.post", fake_post)
    return SimpleNamespace(calls=calls, queue=queue)


# --- modal_client.transport -------------------------------------------------

def test_embed_texts_sends_api_key_header_and_returns_vectors(captured, monkeypatch):
    monkeypatch.setenv("Modal_API_KEY", "secret")
    captured.queue.append(FakeResponse(payload={"status": "success", "embeddings": [[0.1, 0.2]]}))

    assert embed_texts(["hello"]) == [[0.1, 0.2]]
    assert captured.calls[0]["headers"]["X-API-Key"] == "secret"
    assert captured.calls[0]["json"] == {"text": ["hello"]}


def test_embed_texts_rejects_vector_count_mismatch(captured):
    captured.queue.append(FakeResponse(payload={"embeddings": [[0.1], [0.2]]}))
    with pytest.raises(ModalServiceError, match="one vector per input"):
        embed_texts(["only-one"])


def test_http_200_error_payload_is_raised(captured):
    # These services report rejected input as HTTP 200 + {"error": ...}, so
    # raise_for_status() alone would silently accept it.
    captured.queue.append(FakeResponse(payload={"error": "Missing 'text' parameter"}))
    with pytest.raises(ModalServiceError, match="Missing 'text' parameter"):
        embed_texts(["", "  "])


def test_unauthorized_response_explains_the_key(captured):
    captured.queue.append(FakeResponse(status_code=401, payload={}))
    with pytest.raises(ModalServiceError, match="API key"):
        embed_texts(["hello"])


def test_non_json_body_is_reported(captured):
    captured.queue.append(FakeResponse(text_body="<html>oops</html>"))
    with pytest.raises(ModalServiceError, match="non-JSON"):
        embed_texts(["hello"])


def test_classify_texts_sends_max_length_and_unwraps_nested_results(captured):
    captured.queue.append(FakeResponse(payload={"status": "success", "results": {
        "predictions": [[{"label": "LABEL_2", "score": 0.9}]],
        "token_count": 7,
        "truncated": False,
    }}))

    scored = classify_texts(["text"], max_length=512)
    assert captured.calls[0]["json"] == {"text": ["text"], "max_length": 512}
    assert scored == [{"predictions": [{"label": "LABEL_2", "score": 0.9}], "token_count": 7}]


def test_classify_texts_accepts_a_list_of_result_objects(captured):
    # For a list input the service returns `results` as a list, not a dict.
    captured.queue.append(FakeResponse(payload={"results": [
        {"predictions": [[{"label": "LABEL_1", "score": 0.5}]], "token_count": 3},
        {"predictions": [[{"label": "LABEL_0", "score": 0.8}]], "token_count": 4},
    ]}))

    scored = classify_texts(["a", "b"], max_length=2048)
    assert [item["token_count"] for item in scored] == [3, 4]
    assert scored[1]["predictions"] == [{"label": "LABEL_0", "score": 0.8}]


def test_classify_texts_requires_a_token_count(captured):
    captured.queue.append(FakeResponse(payload={"results": {
        "predictions": [[{"label": "LABEL_1", "score": 0.5}]],
    }}))
    with pytest.raises(ModalServiceError, match="token count"):
        classify_texts(["text"], max_length=2048)


# --- CloudEmbeddingModel ----------------------------------------------------

def test_cloud_model_reports_the_configured_dimension():
    model = CloudEmbeddingModel(1024)
    assert model.get_embedding_dimension() == 1024
    assert model.get_sentence_embedding_dimension() == 1024


def test_cloud_model_exposes_the_query_prompt(captured):
    # `_encode_query` only passes prompt_name="query" when the model advertises it.
    assert "query" in CloudEmbeddingModel(1024).prompts


def test_cloud_model_applies_the_instruction_only_for_queries(captured):
    captured.queue.extend([
        FakeResponse(payload={"embeddings": [[0.5]]}),
        FakeResponse(payload={"embeddings": [[0.6]]}),
    ])
    model = CloudEmbeddingModel(1024)

    model.encode(["a document"], prompt_name=None)
    assert captured.calls[0]["json"]["text"] == ["a document"]

    model.encode(["a query"], prompt_name="query")
    assert captured.calls[1]["json"]["text"] == [f"{QUERY_INSTRUCTION}a query"]


def test_cloud_model_does_not_double_apply_the_instruction(captured):
    captured.queue.append(FakeResponse(payload={"embeddings": [[0.5]]}))
    CloudEmbeddingModel(1024).encode([f"{QUERY_INSTRUCTION}q"], prompt_name="query")
    assert captured.calls[0]["json"]["text"] == [f"{QUERY_INSTRUCTION}q"]


def test_cloud_model_splits_batches_at_the_documented_limit(captured):
    texts = [f"doc-{i}" for i in range(CLOUD_MAX_BATCH_SIZE + 3)]
    captured.queue.extend([
        FakeResponse(payload={"embeddings": [[0.0]] * CLOUD_MAX_BATCH_SIZE}),
        FakeResponse(payload={"embeddings": [[0.0]] * 3}),
    ])

    vectors = CloudEmbeddingModel(1024).encode(texts)
    assert len(vectors) == len(texts)
    assert len(captured.calls[0]["json"]["text"]) == CLOUD_MAX_BATCH_SIZE
    assert len(captured.calls[1]["json"]["text"]) == 3


def test_cloud_model_returns_a_single_vector_as_a_list(captured):
    captured.queue.append(FakeResponse(payload={"embeddings": [[0.1, 0.2]]}))
    assert CloudEmbeddingModel(2).encode("one text") == [[0.1, 0.2]]


# --- provider dispatch ------------------------------------------------------

def test_cloud_embedding_is_the_default_provider(monkeypatch):
    from qubettera.rag.settings import get_embedding_provider

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert get_embedding_provider() == "cloud"


def test_embedding_provider_rejects_unknown_values(monkeypatch):
    from qubettera.rag.settings import get_embedding_provider

    monkeypatch.setenv("EMBEDDING_PROVIDER", "modal")
    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER"):
        get_embedding_provider()


def test_cloud_is_the_default_sentiment_provider(monkeypatch):
    monkeypatch.delenv("SENTIMENT_PROVIDER", raising=False)
    assert SentimentScorer().provider == "cloud"


def test_sentiment_provider_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "remote")
    with pytest.raises(RuntimeError, match="SENTIMENT_PROVIDER"):
        SentimentScorer().provider


# --- SentimentScorer cloud path --------------------------------------------

def _cloud_predictions(negative, neutral, positive):
    return [
        {"label": "LABEL_0", "score": negative},
        {"label": "LABEL_1", "score": neutral},
        {"label": "LABEL_2", "score": positive},
    ]


def test_cloud_scorer_uses_bipolar_polarity_and_maps_labels(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")
    monkeypatch.setattr(sentiment, "classify_texts", lambda texts, *, max_length: [
        {"predictions": _cloud_predictions(0.1, 0.2, 0.7), "token_count": 9}
    ])

    result = SentimentScorer().score_one("a happy message")

    assert result.label == "positive"
    assert result.sentiment == pytest.approx(0.6)
    assert result.confidence == pytest.approx(0.7)
    assert result.token_count == 9
    assert result.truncated is False
    assert result.method == "aieng-lab/ModernBERT-large_sentiment"


def test_cloud_scorer_flags_truncation_beyond_max_length(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")
    monkeypatch.setattr(sentiment, "classify_texts", lambda texts, *, max_length: [
        {"predictions": _cloud_predictions(0.7, 0.2, 0.1), "token_count": max_length + 1}
    ])

    result = SentimentScorer(max_length=2048).score_one("a very long message")

    assert result.label == "negative"
    assert result.truncated is True


def test_cloud_scorer_passes_max_length_and_keeps_metadata_aligned(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")
    seen = {}

    def fake_classify(texts, *, max_length):
        seen["texts"] = texts
        seen["max_length"] = max_length
        return [
            {"predictions": _cloud_predictions(0.1, 0.1, 0.8), "token_count": 5},
            {"predictions": _cloud_predictions(0.8, 0.1, 0.1), "token_count": 6},
        ]

    monkeypatch.setattr(sentiment, "classify_texts", fake_classify)
    results = SentimentScorer(max_length=512).score_batch(
        ["good", "bad"], message_ids=["m1", "m2"], agent_ids=["a", "b"], rounds=[0, 1]
    )

    assert seen == {"texts": ["good", "bad"], "max_length": 512}
    assert [r.message_id for r in results] == ["m1", "m2"]
    assert [r.label for r in results] == ["positive", "negative"]
    assert [r.text_length for r in results] == [len("good"), len("bad")]


def test_cloud_scorer_skips_the_model_for_empty_text(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")

    def fail(*args, **kwargs):
        raise AssertionError("empty text must not reach the model")

    monkeypatch.setattr(sentiment, "classify_texts", fail)
    result = SentimentScorer().score_one("   ")

    assert result.note == "empty_text"
    assert result.label == "neutral"
    assert result.method == ""


def test_cloud_scorer_does_not_require_transformers(monkeypatch):
    # The hosted path must not import the optional analytics dependencies.
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")
    monkeypatch.setattr(sentiment, "classify_texts", lambda texts, *, max_length: [
        {"predictions": _cloud_predictions(0.1, 0.2, 0.7), "token_count": 2}
    ])
    scorer = SentimentScorer()
    scorer._load()
    assert scorer._pipeline is None

    assert scorer.score_one("hi").label == "positive"


def test_truncation_summary_still_operates_on_cloud_results(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "cloud")
    monkeypatch.setattr(sentiment, "classify_texts", lambda texts, *, max_length: [
        {"predictions": _cloud_predictions(0.1, 0.2, 0.7), "token_count": max_length + 5}
    ])

    rows = [sentiment._result_to_dict(SentimentScorer(max_length=10).score_one("long text"))]
    json.dumps(rows)  # results must stay JSON-serialisable
    summary = sentiment.truncation_summary(rows)
    assert (summary["total_count"], summary["truncated_count"]) == (1, 1)
