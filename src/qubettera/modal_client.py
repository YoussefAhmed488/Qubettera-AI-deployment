"""Transport for the Modal-hosted embedding and sentiment services.

Both hosted endpoints authenticate with an ``X-API-Key`` header. They answer
HTTP 200 even for rejected input, returning ``{"error": ...}`` in the body, so
``raise_for_status()`` alone cannot detect a failed call: every response is
checked for that key explicitly here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# Document-sized batches are slow on the hosted embedder (~25s per document)
# and trigger an HTTP 500 somewhere between 8 and 16 documents.
CLOUD_MAX_BATCH_SIZE = 8

_DEFAULT_EMBEDDING_URL = (
    "https://gamer7dragon817--dual-nlp-services-qwenembedder-embed.modal.run"
)
_DEFAULT_SENTIMENT_URL = (
    "https://gamer7dragon817--dual-nlp-services-sentimentanalyzer-classify.modal.run"
)
_DEFAULT_TIMEOUT_SECONDS = 600.0

_EMBEDDING_URL_ENV = "MODAL_EMBEDDING_URL"
_SENTIMENT_URL_ENV = "MODAL_SENTIMENT_URL"
_TIMEOUT_ENV = "MODAL_TIMEOUT_SECONDS"
# ``Modal_API_KEY`` is the spelling already present in .env; the uppercase form
# is accepted as the documented alias.
_API_KEY_ENVS = ("Modal_API_KEY", "MODAL_API_KEY")

# Sentiment is scored by the Modal-hosted ModernBERT service by default. Set
# SENTIMENT_PROVIDER=local to load the transformer in-process instead; that
# requires the optional ``analytics`` dependencies.
SENTIMENT_PROVIDER_ENV = "SENTIMENT_PROVIDER"
SENTIMENT_PROVIDER_DEFAULT = "cloud"
_SENTIMENT_PROVIDERS = {"cloud", "local"}


class ModalServiceError(RuntimeError):
    """Raised when a Modal-hosted service cannot serve a request."""


def sentiment_provider() -> str:
    """Return the active sentiment provider ('cloud' or 'local')."""
    provider = os.environ.get(
        SENTIMENT_PROVIDER_ENV, SENTIMENT_PROVIDER_DEFAULT
    ).strip().lower()
    if provider not in _SENTIMENT_PROVIDERS:
        raise RuntimeError(
            f"{SENTIMENT_PROVIDER_ENV} must be one of "
            f"{sorted(_SENTIMENT_PROVIDERS)}, got {provider!r}"
        )
    return provider


def _endpoint(env_name: str, default: str) -> str:
    return os.environ.get(env_name, "").strip() or default


def embedding_endpoint() -> str:
    return _endpoint(_EMBEDDING_URL_ENV, _DEFAULT_EMBEDDING_URL)


def sentiment_endpoint() -> str:
    return _endpoint(_SENTIMENT_URL_ENV, _DEFAULT_SENTIMENT_URL)


def api_key() -> str | None:
    """Return the configured Modal API key, or None for an open endpoint."""
    for name in _API_KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def timeout_seconds() -> float:
    raw_value = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw_value:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{_TIMEOUT_ENV} must be a number, got {raw_value!r}") from error
    if value <= 0:
        raise RuntimeError(f"{_TIMEOUT_ENV} must be > 0, got {value}")
    return value


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["X-API-Key"] = key
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=timeout_seconds()
        )
    except requests.RequestException as error:
        raise ModalServiceError(f"Request to {url} failed: {error}") from error
    if response.status_code == 401:
        raise ModalServiceError(
            f"{url} rejected the API key; set Modal_API_KEY in .env to the value "
            "configured on the deployed service."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise ModalServiceError(
            f"Request to {url} failed with HTTP {response.status_code}."
        ) from error
    try:
        body = response.json()
    except ValueError as error:
        raise ModalServiceError(f"{url} returned a non-JSON response body.") from error
    if not isinstance(body, dict):
        raise ModalServiceError(f"{url} returned an unexpected response body.")
    if "error" in body:
        # These services report rejected input as HTTP 200 + {"error": ...}.
        raise ModalServiceError(f"{url} rejected the request: {body['error']}")
    return body


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Return one embedding vector per input text, in input order."""
    batch = list(texts)
    body = _post(embedding_endpoint(), {"text": batch})
    raw_vectors = body.get("embeddings")
    if not isinstance(raw_vectors, list) or len(raw_vectors) != len(batch):
        raise ModalServiceError(
            "Embedding response did not contain one vector per input text."
        )
    vectors = []
    for raw_vector in raw_vectors:
        if not isinstance(raw_vector, list):
            raise ModalServiceError("Embedding response contained a malformed vector.")
        vectors.append([float(value) for value in raw_vector])
    return vectors


def classify_texts(
    texts: Sequence[str], *, max_length: int
) -> list[dict[str, Any]]:
    """Score sentiment for each text.

    Returns one dict per input with ``predictions`` (the model's label/score
    list) and ``token_count`` (the untruncated token count the service
    measured, which matches the local tokenizer).
    """
    batch = list(texts)
    body = _post(
        sentiment_endpoint(),
        {"text": batch, "max_length": int(max_length)},
    )
    results = body.get("results")
    if isinstance(results, dict):
        # A single-text request returns one object rather than a list.
        results = [results]
    if not isinstance(results, list) or len(results) != len(batch):
        raise ModalServiceError(
            "Sentiment response did not contain one result per input text."
        )
    scored = []
    for result in results:
        predictions = result.get("predictions") if isinstance(result, dict) else None
        # ``predictions`` is batch-shaped: one list of label/score dicts per input.
        if (
            not isinstance(predictions, list)
            or not predictions
            or not isinstance(predictions[0], list)
        ):
            raise ModalServiceError(
                "Sentiment response did not contain per-label predictions."
            )
        token_count = result.get("token_count")
        if not isinstance(token_count, int):
            raise ModalServiceError(
                "Sentiment response did not report the untruncated token count."
            )
        scored.append({"predictions": predictions[0], "token_count": token_count})
    return scored
