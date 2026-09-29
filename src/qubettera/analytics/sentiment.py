"""Score the tone of each participant message."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qubettera.modal_client import classify_texts, sentiment_provider

from .loader import DiscussionLog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label normalisation across model families. The transformer pipeline returns
# whatever string labels the model was trained with; we map them to a stable
# internal vocabulary so downstream code never has to know which model ran.
_LABEL_MAP: dict[str, str] = {
    # ModernBERT (aieng-lab/ModernBERT-large_sentiment)
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # SciBERT (puzzz21/sci-sentiment-classify)
    "p": "positive",
    "n": "negative",
    "o": "neutral",
    # Generic HuggingFace LABEL_n convention (SST-2, IMDB, etc.)
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}

# Default model â€” Tier 1 prototype. Changing this single string is the
# only code change needed to move to a fine-tuned Tier 2 model.
_DEFAULT_MODEL = "aieng-lab/ModernBERT-large_sentiment"
_DEFAULT_BATCH_SIZE = 16

# Maximum token length before the model truncates. Raised from 512 to 2048
# because Week 3 messages routinely exceed 2000 characters (~500 tokens),
# and the sentiment-bearing language often appears later in the message
# rather than in the opening paragraph. ModernBERT supports up to 8192
# tokens, so 2048 is well within its range.
_DEFAULT_MAX_LENGTH = 2048

# Sentiment score bounds (kept as module constants so tests can reference
# them instead of hard-coding Â±1.0).
_MIN_SENTIMENT = -1.0
_MAX_SENTIMENT = 1.0

# Canonical labels in fixed order. Used by `_build_result` to pick the argmax
# class deterministically even when a model emits an unusual label set.
_CANONICAL_LABELS: tuple[str, ...] = ("positive", "neutral", "negative")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentResult:
    """One message's sentiment, model-agnostic.

    Fields
    ------
    message_id : str
        Week 3 message identifier (``<discussion_id>:<sequence>``), used to
        trace the score back to the original event.
    agent_id : str
        Sender of the message.
    round : int
        Discussion round the message belongs to.
    sentiment : float
        Bipolar score in [-1.0, +1.0] = P(positive) - P(negative).
    label : str
        Argmax class: ``"positive"`` | ``"neutral"`` | ``"negative"``.
    confidence : float
        Probability of the argmax class, in [0.0, 1.0]. Low values indicate
        the model was uncertain between classes.
    method : str
        Model identifier that produced this score (e.g.
        ``"aieng-lab/ModernBERT-large_sentiment"``). Recorded so the report
        and Week 5 dashboard can distinguish Tier 1 from Tier 2 scores.
    text_length : int
        Character count of the scored text. Useful for spotting very short
        messages whose scores may be unreliable.
    token_count : int
        Number of tokens in the source text *before* truncation. Zero for
        messages that were not sent to the model (empty text). Useful for
        diagnosing truncation and for filtering very short messages whose
        scores may be unreliable.
    truncated : bool
        ``True`` when ``token_count`` exceeds the scorer's ``max_length``,
        meaning the model only saw the first ``max_length`` tokens of the
        message. A truncated message's score should be interpreted with
        caution â€” the sentiment-bearing language may have been cut off.
    note : str | None
        Populated when the message could not be scored normally
        (e.g. ``"empty_text"``). ``None`` for successful scores.
    """

    message_id: str
    agent_id: str
    round: int
    sentiment: float
    label: str
    confidence: float
    method: str
    text_length: int
    token_count: int = 0
    truncated: bool = False
    note: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_label(raw_label: str) -> str:
    """Map a model-specific label string to our canonical vocabulary.

    Unknown labels are returned lowercased rather than raising, so an
    unfamiliar model does not crash the pipeline â€” the caller can still
    inspect the label, and `_normalize_probs` will simply ignore it.
    """
    return _LABEL_MAP.get(raw_label.lower(), raw_label.lower())


def _clamp(value: float, low: float = _MIN_SENTIMENT, high: float = _MAX_SENTIMENT) -> float:
    """Clamp a float to [low, high] to absorb floating-point overshoot."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# SentimentScorer
# ---------------------------------------------------------------------------

class SentimentScorer:
    """Sentiment scorer with a hosted (Modal) and a local backend.

    The default backend is the Modal-hosted ``aieng-lab/ModernBERT-large_sentiment``
    service, which needs no local weights. Set ``SENTIMENT_PROVIDER=local`` to run
    the model in-process instead, via a transformer pipeline that is loaded lazily
    on first use so that simply importing this class (or the parent package) does
    not require ``transformers`` / ``torch`` to be installed. Construction is cheap
    either way; the expensive model load happens inside ``_load()``.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or local path. Defaults to the Tier 1
        prototype model. To move to a fine-tuned Tier 2 model, pass a
        different ``model_name`` â€” no other code changes are needed.
    device : int
        ``-1`` for CPU (default), ``0`` for the first GPU. GPU is faster
        but requires CUDA-enabled ``torch``.
    batch_size : int
        Number of messages scored per forward pass. Larger batches are
        faster on GPU; on CPU the difference is small.
    max_length : int
        Maximum token length before truncation. Messages longer than this
        are truncated from the end. Defaults to 2048, which comfortably
        covers the vast majority of Week 3 messages. ModernBERT supports
        up to 8192 tokens if the default proves insufficient.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: int = 0,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        # Lazy-loaded on first score_batch() call. Typed as Any because the
        # HuggingFace pipeline's callable signature is intentionally dynamic
        # (varies by task and model); tests replace this with a fake callable.
        self._pipeline: Any = None

    # -- Public API ---------------------------------------------------------

    def score_batch(
        self,
        texts: list[str],
        *,
        message_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        rounds: list[int] | None = None,
    ) -> list[SentimentResult]:
        """Score a batch of message texts.

        Returns one ``SentimentResult`` per input, in the same order as the
        input list. Messages with empty or whitespace-only text receive a
        neutral result with ``note="empty_text"`` and do not reach the model.

        Parameters
        ----------
        texts : list[str]
            Message texts to score.
        message_ids, agent_ids, rounds : list, optional
            Parallel metadata lists. If omitted, the corresponding field on
            each result is left as an empty string / zero. If provided,
            each must be the same length as ``texts``.
        """
        self._load()
        if self.provider == "local":
            assert self._pipeline is not None, "pipeline failed to load"

        n = len(texts)
        # Fill in missing metadata with neutral defaults so the parallel
        # indexing below stays simple.
        message_ids = list(message_ids) if message_ids is not None else [""] * n
        agent_ids = list(agent_ids) if agent_ids is not None else [""] * n
        rounds = list(rounds) if rounds is not None else [0] * n

        if not (len(message_ids) == len(agent_ids) == len(rounds) == n):
            raise ValueError(
                f"Metadata lists must match texts length: "
                f"texts={n}, message_ids={len(message_ids)}, "
                f"agent_ids={len(agent_ids)}, rounds={len(rounds)}"
            )

        # Pre-allocate result slots so we can fill them out-of-order when
        # empty texts are interleaved with valid ones.
        results: list[SentimentResult | None] = [None] * n

        # Separate valid texts from empty/invalid ones. Only valid texts
        # are sent to the model â€” this avoids tokenising empty strings and
        # keeps the batch contiguous.
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, text in enumerate(texts):
            if text is None or not text.strip():
                results[i] = self._empty_result(
                    message_id=message_ids[i],
                    agent_id=agent_ids[i],
                    round_num=rounds[i],
                    note="empty_text",
                )
            else:
                valid_indices.append(i)
                valid_texts.append(text)

        # Run batched inference on the valid subset.
        if valid_texts:
            raw_outputs, token_counts = self._infer(valid_texts)
            for local_idx, raw in enumerate(raw_outputs):
                original_idx = valid_indices[local_idx]
                results[original_idx] = self._build_result(
                    raw=raw,
                    message_id=message_ids[original_idx],
                    agent_id=agent_ids[original_idx],
                    round_num=rounds[original_idx],
                    text_length=len(texts[original_idx]),
                    token_count=token_counts[local_idx],
                )

        # At this point every slot is filled; the type checker cannot know
        # that, so we assert it explicitly.
        assert all(r is not None for r in results), "internal error: unfilled result slot"
        return [r for r in results if r is not None]

    def score_one(
        self,
        text: str,
        *,
        message_id: str = "",
        agent_id: str = "",
        round_num: int = 0,
    ) -> SentimentResult:
        """Convenience wrapper around ``score_batch`` for a single message."""
        return self.score_batch(
            [text],
            message_ids=[message_id],
            agent_ids=[agent_id],
            rounds=[round_num],
        )[0]

    # -- Internal -----------------------------------------------------------

    @property
    def provider(self) -> str:
        """The active backend, read per call so env changes take effect live."""
        return sentiment_provider()

    def _infer(self, texts: list[str]) -> tuple[list[list[dict[str, Any]]], list[int]]:
        """Score ``texts``, returning raw label/score lists and token counts.

        Both backends return one raw label/score list per text plus the
        untruncated token count, so ``_build_result`` is provider-agnostic.
        """
        if self.provider == "cloud":
            scored = classify_texts(texts, max_length=self.max_length)
            return (
                [item["predictions"] for item in scored],
                [item["token_count"] for item in scored],
            )

        raw_outputs = self._pipeline(
            texts,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )
        # Count tokens on the full (untruncated) texts so we can flag
        # which messages exceeded max_length. This is a single batched
        # tokenizer call, negligible next to model inference.
        return raw_outputs, self._count_tokens_batch(texts)

    def _load(self) -> None:
        """Load the local transformer pipeline on first use.

        Importing ``transformers`` here rather than at module top means the
        rest of the analytics package can be imported and used without
        ``transformers`` / ``torch`` installed. The import cost is paid
        only when local sentiment scoring is actually requested; the hosted
        provider needs no local weights and returns immediately.
        """
        if self.provider == "cloud" or self._pipeline is not None:
            return
        # Local import â€” deliberate, see docstring.
        from transformers import pipeline  # type: ignore[import-untyped]

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=self.device,
            top_k=None,  # return all class probabilities, not just the top one
            truncation=True,
            max_length=self.max_length,
        )

    def _count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Return the number of tokens in each text, without truncation.

        Uses the pipeline's tokenizer when available so the count reflects
        what the model actually saw *before* truncation. Falls back to a
        character-based approximation (``len(text) // 4``) when the pipeline
        does not expose a tokenizer â€” this is the case for the fake pipeline
        used in unit tests, and it lets the diagnostic run without blocking
        scoring.

        Never raises. Any tokenizer-API surprise is caught and the fallback
        approximation is used, so a broken tokenizer cannot fail a scoring
        call that would otherwise succeed.
        """
        if not texts:
            return []

        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is None:
            # Fake pipeline (tests) or an unusual real pipeline.
            return [max(1, len(t) // 4) for t in texts]

        try:
            encoded = tokenizer(texts, truncation=False, padding=False)
            ids = encoded["input_ids"]
            # A well-behaved batched tokenizer returns a list of lists. Guard
            # against the single-text shape just in case.
            if ids and isinstance(ids[0], int):
                ids = [ids]
            return [len(x) for x in ids]
        except Exception:
            return [max(1, len(t) // 4) for t in texts]

    def _build_result(
        self,
        raw: list[dict[str, Any]],
        message_id: str,
        agent_id: str,
        round_num: int,
        text_length: int,
        token_count: int,
    ) -> SentimentResult:
        """Convert one model output into a ``SentimentResult``."""
        probs = self._normalize_probs(raw)
        sentiment = _clamp(round(probs["positive"] - probs["negative"], 4))
        # Argmax across the three canonical classes. Iterating over the
        # fixed tuple (rather than the dict) guarantees we always return
        # one of our canonical labels, even if the model emitted an
        # unrecognised label that `_normalize_probs` dropped.
        label = max(_CANONICAL_LABELS, key=lambda k: probs[k])
        confidence = round(float(probs[label]), 4)
        return SentimentResult(
            message_id=message_id,
            agent_id=agent_id,
            round=round_num,
            sentiment=sentiment,
            label=label,
            confidence=confidence,
            method=self.model_name,
            text_length=text_length,
            token_count=token_count,
            truncated=token_count > self.max_length,
            note=None,
        )

    @staticmethod
    def _empty_result(
        message_id: str,
        agent_id: str,
        round_num: int,
        note: str,
    ) -> SentimentResult:
        """Neutral result for a message that cannot be scored."""
        return SentimentResult(
            message_id=message_id,
            agent_id=agent_id,
            round=round_num,
            sentiment=0.0,
            label="neutral",
            confidence=1.0,
            method="",  # no model ran
            text_length=0,
            token_count=0,
            truncated=False,
            note=note,
        )

    @staticmethod
    def _normalize_probs(raw: list[dict[str, Any]]) -> dict[str, float]:
        """Convert a model's raw output list to a canonical probability dict.

        The pipeline with ``top_k=None`` returns one entry per class, e.g.
        ``[{"label": "positive", "score": 0.7}, ...]``. Different model
        families use different label strings, so we normalise each label
        via ``_LABEL_MAP`` before bucketing.

        If a canonical class is missing from the model output, it defaults
        to 0.0 â€” the resulting sentiment is still well-defined because the
        bipolar formula only uses positive and negative.
        """
        probs = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for item in raw:
            canonical = _normalize_label(str(item.get("label", "")))
            if canonical in probs:
                probs[canonical] = float(item.get("score", 0.0))
        return probs


# ---------------------------------------------------------------------------
# Discussion-level scoring
# ---------------------------------------------------------------------------

def score_sentiment(
    log: DiscussionLog,
    scorer: SentimentScorer | None = None,
) -> list[dict[str, Any]]:
    """Score every message in a discussion.

    Parameters
    ----------
    log : DiscussionLog
        Parsed Week 3 discussion, as returned by ``loader.load_discussion``.
    scorer : SentimentScorer, optional
        Pre-constructed scorer. If omitted, a default Tier 1 scorer is
        constructed lazily on first call. Providing a scorer is useful for
        tests (mock pipeline) and for batch runs (reuse one loaded model
        across many discussions).

    Returns
    -------
    list[dict]
        One dict per message, in the order the messages appear in the log.
        Each dict contains: ``message_id``, ``agent_id``, ``round``,
        ``sentiment``, ``label``, ``confidence``, ``method``,
        ``text_length``, ``token_count``, ``truncated``, ``note``.

        Returns ``[]`` if the log has no snapshots.
    """
    if scorer is None:
        scorer = SentimentScorer()

    snapshots = log.snapshots
    if not snapshots:
        return []

    texts = [s.content_text or s.opinion_text for s in snapshots]
    message_ids = [s.message_id for s in snapshots]
    agent_ids = [s.agent_id for s in snapshots]
    rounds = [s.round_number for s in snapshots]

    results = scorer.score_batch(
        texts,
        message_ids=message_ids,
        agent_ids=agent_ids,
        rounds=rounds,
    )

    return [_result_to_dict(r) for r in results]


def _result_to_dict(r: SentimentResult) -> dict[str, Any]:
    """Serialise a ``SentimentResult`` for inclusion in the analytics JSON."""
    return {
        "message_id": r.message_id,
        "agent_id": r.agent_id,
        "round": r.round,
        "sentiment": r.sentiment,
        "label": r.label,
        "confidence": r.confidence,
        "method": r.method,
        "text_length": r.text_length,
        "token_count": r.token_count,
        "truncated": r.truncated,
        "note": r.note,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers (used by engine.py and report.py)
# ---------------------------------------------------------------------------

def aggregate_by_agent(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Per-agent average sentiment and message count.

    Messages with a ``note`` (e.g. empty text) are excluded from the
    average so that a single malformed message does not skew an agent's
    aggregate downward.
    """
    buckets: dict[str, list[float]] = {}
    for r in results:
        if r.get("note"):
            continue
        buckets.setdefault(r["agent_id"], []).append(float(r["sentiment"]))
    return {
        agent: {
            "avg_sentiment": round(sum(vals) / len(vals), 4),
            "message_count": len(vals),
        }
        for agent, vals in buckets.items()
    }


def aggregate_by_round(
    results: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """Per-round average sentiment and message count.

    Rounds are returned sorted ascending. Messages with a ``note`` are
    excluded, same as ``aggregate_by_agent``.
    """
    buckets: dict[int, list[float]] = {}
    for r in results:
        if r.get("note"):
            continue
        buckets.setdefault(int(r["round"]), []).append(float(r["sentiment"]))
    return {
        round_num: {
            "avg_sentiment": round(sum(vals) / len(vals), 4),
            "message_count": len(vals),
        }
        for round_num, vals in sorted(buckets.items())
    }


def sentiment_distribution(
    results: list[dict[str, Any]],
) -> dict[str, int]:
    """Count of messages by label.

    Always returns all three keys (``positive``, ``neutral``, ``negative``)
    even when a label has zero count, so downstream code can rely on the
    shape without null checks.
    """
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        label = r.get("label")
        if label in dist:
            dist[label] += 1
    return dist


def truncation_summary(
    results: list[dict[str, Any]],
    *,
    sample_limit: int = 10,
) -> dict[str, Any]:
    """Summarize how many messages were truncated during scoring.

    A message is considered truncated when its full token count exceeds
    the scorer's ``max_length`` â€” the model only saw the first portion
    of the message when producing its sentiment score.

    Parameters
    ----------
    results : list[dict]
        Per-message results from ``score_sentiment``.
    sample_limit : int
        Maximum number of truncated message IDs to include in the
        ``sample_message_ids`` field, so the summary stays compact even
        when truncation is widespread.

    Returns
    -------
    dict
        ::

            {
                "total_count":       int,     # messages scored
                "truncated_count":   int,     # messages exceeding max_length
                "truncated_share":   float,   # fraction in [0.0, 1.0]
                "sample_message_ids": list[str],
            }
    """
    total = len(results)
    truncated = [r for r in results if r.get("truncated")]
    return {
        "total_count": total,
        "truncated_count": len(truncated),
        "truncated_share": (len(truncated) / total) if total else 0.0,
        "sample_message_ids": [
            str(r.get("message_id", "")) for r in truncated[:sample_limit]
        ],
    }

