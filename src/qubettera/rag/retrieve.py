"""Hybrid retrieval against the PostgreSQL + pgvector knowledge base."""

import argparse
import json
import logging
import math
import re
import sys
import threading
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import psycopg2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qubettera.rag.database import connect as connect_database
    from qubettera.rag.database import get_db_config as read_db_config
    from qubettera.rag.settings import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        EMBEDDING_MODEL_REVISION,
        get_embedding_provider,
        IVFFLAT_PROBES as CONFIGURED_IVFFLAT_PROBES,
        PIPELINE_VERSION,
        PREPROCESSING_VERSION,
        ADAPTIVE_EXPANSION_MIN_SIMILARITY,
    )
else:
    from .database import connect as connect_database
    from .database import get_db_config as read_db_config
    from .settings import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        EMBEDDING_MODEL_REVISION,
        get_embedding_provider,
        IVFFLAT_PROBES as CONFIGURED_IVFFLAT_PROBES,
        PIPELINE_VERSION,
        PREPROCESSING_VERSION,
        ADAPTIVE_EXPANSION_MIN_SIMILARITY,
    )

def get_db_config() -> dict:
    return read_db_config(purpose="retrieval")


RRF_K = 30
CANDIDATE_POOL = 150
EXPANSION_COUNT = 3
EXPANSION_WEIGHT = 0.7
# Storage caps IVFFlat at 100 lists. Probing every list makes evaluation and
# local retrieval stable across index rebuilds at the current corpus scale.
IVFFLAT_PROBES = CONFIGURED_IVFFLAT_PROBES
SOURCE_CANDIDATE_LIMIT = 2
FINAL_SOURCE_LIMIT = 1
_embed_model = None
_embed_model_lock = threading.Lock()
_query_encode_lock = threading.Lock()
logger = logging.getLogger(__name__)

QueryExpander = Callable[[str, int], list[str]]


def get_connection():
    return connect_database(purpose="retrieval", statement_timeout_ms=10_000)


def _get_embed_model():
    global _embed_model
    # Initial discussion turns start concurrently. Only one worker may load
    # weights; publish the model only after construction completes successfully.
    with _embed_model_lock:
        if _embed_model is None:
            if get_embedding_provider() == "cloud":
                # Stateless HTTP client: no weights to load or serialize.
                from qubettera.rag.cloud_embedding import CloudEmbeddingModel

                _embed_model = CloudEmbeddingModel(EMBEDDING_DIM)
                return _embed_model
            from sentence_transformers import SentenceTransformer
            try:
                _embed_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    revision=EMBEDDING_MODEL_REVISION,
                    local_files_only=True,
                )
            except (OSError, ValueError):
                _embed_model = SentenceTransformer(
                    EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION
                )
        return _embed_model


def _encode_query(model, query: str):
    """Encode a query with the model's retrieval prompt when it provides one."""
    encode_kwargs = {
        "normalize_embeddings": True,
        "show_progress_bar": False,
        "batch_size": 1,
    }
    if "query" in getattr(model, "prompts", {}):
        encode_kwargs["prompt_name"] = "query"
    # Shared tokenizer/model state and device memory must not be used by
    # simultaneous encode calls. Database queries and LLM calls stay parallel.
    with _query_encode_lock:
        return model.encode(query, **encode_kwargs)


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be a non-empty string")
    if len(normalized) > 512:
        raise ValueError("query is too long; limit is 512 characters")
    return normalized


def _validate_top_k(top_k: int, candidate_pool: int) -> tuple[int, int]:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if not isinstance(candidate_pool, int) or isinstance(candidate_pool, bool):
        raise TypeError("candidate_pool must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be > 0")
    if top_k > 50:
        raise ValueError("top_k must be <= 50")
    if candidate_pool < top_k:
        raise ValueError("candidate_pool must be >= top_k")
    if candidate_pool > 1000:
        raise ValueError("candidate_pool must be <= 1000")
    return top_k, candidate_pool


def _read_index_manifest(cur, include_urls: bool = True) -> dict:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'chunks'
        """
    )
    columns = {row[0] for row in cur.fetchall()}
    required = {
        "embedding_text",
        "embedding_model",
        "embedding_model_revision",
        "preprocessing_version",
        "pipeline_version",
    }
    missing_columns = sorted(required - columns)
    if "chunk_id" not in columns:
        raise RuntimeError("The chunks table is missing; run src/store.py first.")

    optional_selects = []
    for column in (
        "embedding_model",
        "embedding_model_revision",
        "preprocessing_version",
        "pipeline_version",
    ):
        if column in columns:
            optional_selects.append(f"array_remove(array_agg(DISTINCT {column}), NULL)")
        else:
            optional_selects.append("ARRAY[]::text[]")
    cur.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT url), {', '.join(optional_selects)}
        FROM chunks
        """
    )
    row = cur.fetchone()
    manifest = {
        "row_count": row[0],
        "source_count": row[1],
        "embedding_models": row[2] or [],
        "embedding_model_revisions": row[3] or [],
        "preprocessing_versions": row[4] or [],
        "pipeline_versions": row[5] or [],
        "missing_identity_columns": missing_columns,
    }
    if include_urls:
        cur.execute("SELECT DISTINCT url FROM chunks WHERE url IS NOT NULL")
        manifest["source_urls"] = [item[0] for item in cur.fetchall()]
    return manifest


def _validate_index_identity(manifest: dict) -> None:
    expected = {
        "embedding_models": [EMBEDDING_MODEL],
        "embedding_model_revisions": [EMBEDDING_MODEL_REVISION],
        "preprocessing_versions": [PREPROCESSING_VERSION],
        "pipeline_versions": [PIPELINE_VERSION],
    }
    mismatches = []
    if manifest.get("missing_identity_columns"):
        mismatches.append(
            "missing columns: " + ", ".join(manifest["missing_identity_columns"])
        )
    for field, wanted in expected.items():
        actual = sorted(manifest.get(field) or [])
        if actual != wanted:
            mismatches.append(f"{field}={actual!r}, expected {wanted!r}")
    if mismatches:
        raise RuntimeError(
            "The retrieval index is incompatible with the active embedding configuration ("
            + "; ".join(mismatches)
            + "). Re-run src/embed.py and src/store.py."
        )


def get_index_manifest() -> dict:
    """Return auditable corpus/model metadata and validate query-vector compatibility."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                manifest = _read_index_manifest(cur, include_urls=True)
    except psycopg2.Error as exc:
        # The HTTP layer replaces this message with a generic 502, so without a
        # server-side log the cause (wrong target, missing table, timeout) is lost.
        logger.error("Index metadata query failed: %s", exc)
        raise RuntimeError("Database query failed while reading index metadata.") from exc
    _validate_index_identity(manifest)
    return manifest


def _vector_search(cur, query_embedding: list, top_k: int) -> list[dict]:
    cur.execute(
        """
        SELECT chunk_id, text, embedding_text, url, title, headers,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )
    return [
        {
            "chunk_id": row[0],
            "text": row[1],
            "embedding_text": row[2],
            "url": row[3],
            "title": row[4],
            "headers": row[5],
            "similarity": float(row[6]),
        }
        for row in cur.fetchall()
    ]


def _keyword_search(cur, query: str, top_k: int) -> list[dict]:
    cur.execute(
        """
        SELECT chunk_id, text, embedding_text, url, title, headers,
               ts_rank_cd(text_search, plainto_tsquery('english', %s)) AS rank_score
        FROM chunks
        WHERE text_search @@ plainto_tsquery('english', %s)
        ORDER BY rank_score DESC
        LIMIT %s
        """,
        (query, query, top_k),
    )
    return [
        {
            "chunk_id": row[0],
            "text": row[1],
            "embedding_text": row[2],
            "url": row[3],
            "title": row[4],
            "headers": row[5],
            "text_rank_score": float(row[6]),
        }
        for row in cur.fetchall()
    ]


def _rrf_fuse(vector_results, keyword_results, k=RRF_K):
    return _weighted_rrf_fuse(
        [
            (vector_results, 1.0, "vector"),
            (keyword_results, 1.0, "keyword"),
        ],
        k=k,
    )


def _weighted_rrf_fuse(ranked_lists, k=RRF_K):
    """Fuse any number of ranked lists, optionally down-weighting expansions."""
    scores = {}
    chunk_data = {}
    matched_queries: dict[str, list[str]] = {}

    for ranked_results, weight, query_label in ranked_lists:
        seen_in_list = set()
        for rank, result in enumerate(ranked_results):
            cid = result["chunk_id"]
            if cid in seen_in_list:
                continue
            seen_in_list.add(cid)
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            stored = chunk_data.setdefault(cid, {})
            for key, value in result.items():
                if value is not None:
                    if key in {"similarity", "text_rank_score"} and stored.get(key) is not None:
                        stored[key] = max(float(stored[key]), float(value))
                    else:
                        stored[key] = value
            labels = matched_queries.setdefault(cid, [])
            if query_label not in labels:
                labels.append(query_label)

    results = []
    for cid, score in sorted(scores.items(), key=lambda x: -x[1]):
        data = chunk_data[cid]
        results.append({
            "chunk_id": cid,
            "text": data.get("text", ""),
            "embedding_text": data.get("embedding_text", data.get("text", "")),
            "url": data.get("url", ""),
            "title": data.get("title", ""),
            "headers": data.get("headers", {}),
            "rrf_score": score,
            "similarity": data.get("similarity"),   # None if missing
            "text_rank_score": data.get("text_rank_score"),
            "matched_queries": matched_queries.get(cid, []),
        })
    return results


def _query_variants(
    query: str,
    *,
    expand: bool,
    expansion_count: int,
    query_expander: QueryExpander | None,
) -> list[str]:
    if not expand:
        return [query]
    if not isinstance(expansion_count, int) or isinstance(expansion_count, bool):
        raise TypeError("expansion_count must be an integer")
    if not 1 <= expansion_count <= 5:
        raise ValueError("expansion_count must be between 1 and 5")

    if query_expander is None:
        if __package__ in {None, ""}:
            from qubettera.rag.query_expansion import expand_query_with_llm
        else:
            from .query_expansion import expand_query_with_llm
        query_expander = expand_query_with_llm

    try:
        candidates = query_expander(query, expansion_count)
    except Exception as exc:
        logger.warning("Query expansion failed; using the original query: %s", exc)
        return [query]
    if not isinstance(candidates, list):
        logger.warning("Query expansion returned a non-list; using the original query")
        return [query]

    variants = [query]
    seen = {query.casefold()}
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        try:
            candidate = _validate_query(candidate)
        except (TypeError, ValueError):
            continue
        identity = candidate.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        variants.append(candidate)
        if len(variants) > expansion_count:
            break
    return variants


def _source_key(result: dict) -> str:
    url = result.get("url") or ""
    if not url:
        return result.get("chunk_id") or ""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    path = parts.path.rstrip("/").lower()
    if host in {"arxiv.org", "www.arxiv.org"} and path.startswith("/abs/"):
        host = "arxiv.org"
        path = re.sub(r"v\d+$", "", path)
    return f"{host}{path}"


def _limit_per_source(results: list[dict], limit: int) -> list[dict]:
    """Prevent overlapping chunks from a long paper consuming the whole top-k."""
    if limit <= 0:
        raise ValueError("source limit must be > 0")
    selected, counts = [], {}
    for result in results:
        source = _source_key(result)
        if counts.get(source, 0) >= limit:
            continue
        counts[source] = counts.get(source, 0) + 1
        selected.append(result)
    return selected


def _retrieval_is_weak(
    results: list[dict],
    min_similarity: float = ADAPTIVE_EXPANSION_MIN_SIMILARITY,
) -> bool:
    """Use the strongest dense match to gate the expensive LLM expansion pass."""
    if not results:
        return True
    similarities = [
        float(result["similarity"])
        for result in results[:5]
        if result.get("similarity") is not None
    ]
    return bool(similarities) and max(similarities) < min_similarity


def _retrieve_once(
    query: str,
    top_k: int = 20,
    candidate_pool: int = CANDIDATE_POOL,
    expand: bool = False,
    expansion_count: int = EXPANSION_COUNT,
    query_expander: QueryExpander | None = None,
    adaptive_expand: bool = False,
) -> list[dict]:
    query = _validate_query(query)
    top_k, candidate_pool = _validate_top_k(top_k, candidate_pool)
    if adaptive_expand and not expand:
        initial = _retrieve_once(
            query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            expand=False,
            expansion_count=expansion_count,
            query_expander=query_expander,
            adaptive_expand=False,
        )
        if not _retrieval_is_weak(initial):
            for result in initial:
                result["query_expansion_used"] = False
            return initial
        expanded = _retrieve_once(
            query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            expand=True,
            expansion_count=expansion_count,
            query_expander=query_expander,
            adaptive_expand=False,
        )
        chosen = expanded or initial
        for result in chosen:
            result["query_expansion_used"] = bool(expanded)
        return chosen
    variants = _query_variants(
        query,
        expand=expand,
        expansion_count=expansion_count,
        query_expander=query_expander,
    )

    try:
        model = _get_embed_model()
        query_vectors = []
        for variant in variants:
            query_vec = _encode_query(model, variant)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()
            if not isinstance(query_vec, list) or len(query_vec) != EMBEDDING_DIM:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {EMBEDDING_DIM}, "
                    f"got {len(query_vec) if isinstance(query_vec, list) else 'non-list'}"
                )
            if not all(math.isfinite(float(value)) for value in query_vec):
                raise ValueError("Query embedding contains NaN or infinity")
            query_vectors.append(query_vec)
    except Exception as exc:
        raise RuntimeError(f"Embedding model failed for query: {query}") from exc

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                manifest = _read_index_manifest(cur, include_urls=False)
                _validate_index_identity(manifest)
                cur.execute("SELECT set_config('ivfflat.probes', %s, true)", (str(IVFFLAT_PROBES),))
                ranked_lists = []
                for index, (variant, query_vec) in enumerate(zip(variants, query_vectors)):
                    weight = 1.0 if index == 0 else EXPANSION_WEIGHT
                    ranked_lists.extend(
                        (
                            (_vector_search(cur, query_vec, candidate_pool), weight, variant),
                            (_keyword_search(cur, variant, candidate_pool), weight, variant),
                        )
                    )
    except psycopg2.Error as exc:
        logger.error("Retrieval query failed: %s", exc)
        raise RuntimeError("Database query failed during retrieval.") from exc

    fused = _weighted_rrf_fuse(ranked_lists)
    selection_pool = _limit_per_source(
        fused, limit=SOURCE_CANDIDATE_LIMIT
    )[: max(top_k * 6, 40)]
    final = _limit_per_source(selection_pool, limit=FINAL_SOURCE_LIMIT)[:top_k]
    for i, r in enumerate(final):
        r["rank"] = i + 1
    return final


class RetrievalService:
    """Reusable public boundary for Qwen3 + PostgreSQL hybrid retrieval.

    The embedding model is a process-level lazy singleton, so constructing a
    service is cheap and repeated discussion turns do not reload model weights.
    """

    def __init__(
        self,
        query_expander: QueryExpander | None = None,
        *,
        adaptive_expand: bool = False,
    ):
        self._query_expander = query_expander
        self._adaptive_expand = adaptive_expand

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        candidate_pool: int = CANDIDATE_POOL,
        expand: bool = False,
        expansion_count: int = EXPANSION_COUNT,
        adaptive_expand: bool | None = None,
    ) -> list[dict]:
        return _retrieve_once(
            query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            expand=expand,
            expansion_count=expansion_count,
            query_expander=self._query_expander,
            adaptive_expand=(
                self._adaptive_expand if adaptive_expand is None else adaptive_expand
            ),
        )

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 20,
        candidate_pool: int = CANDIDATE_POOL,
        expand: bool = False,
        expansion_count: int = EXPANSION_COUNT,
        adaptive_expand: bool | None = None,
    ) -> list[list[dict]]:
        """Retrieve several independent queries through one reusable service."""
        return [
            self.retrieve(
                query,
                top_k=top_k,
                candidate_pool=candidate_pool,
                expand=expand,
                expansion_count=expansion_count,
                adaptive_expand=adaptive_expand,
            )
            for query in queries
        ]


_DEFAULT_RETRIEVAL_SERVICE = RetrievalService()


def retrieve(
    query: str,
    top_k: int = 20,
    candidate_pool: int = CANDIDATE_POOL,
    expand: bool = False,
    expansion_count: int = EXPANSION_COUNT,
    adaptive_expand: bool = False,
) -> list[dict]:
    """Retrieve evidence using the shared process-level retrieval service."""
    return _DEFAULT_RETRIEVAL_SERVICE.retrieve(
        query,
        top_k=top_k,
        candidate_pool=candidate_pool,
        expand=expand,
        expansion_count=expansion_count,
        adaptive_expand=adaptive_expand,
    )


def format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"{'='*80}")
        lines.append(f"  Rank:       {r['rank']}")
        lines.append(f"  URL:        {r.get('url', 'N/A')}")
        lines.append(f"  Title:      {r.get('title', 'N/A')}")
        if r.get("similarity") is not None:
            lines.append(f"  Cosine sim: {r['similarity']:.4f}")
        if r.get("text_rank_score") is not None:
            lines.append(f"  Text rank:  {r['text_rank_score']:.4f}")
        lines.append(f"  RRF score:  {r.get('rrf_score', 0):.6f}")
        lines.append(f"  Text:       {r['text'][:300]}...")
        lines.append("")
    return "\n".join(lines)


def _console_safe(value: str) -> str:
    """Replace characters unsupported by a legacy Windows console encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding)


def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from the knowledge base.")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=20, help="Number of results (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    parser.add_argument(
        "--adaptive-expand",
        action="store_true",
        help="Expand only when the first-pass dense similarity is weak",
    )
    parser.add_argument("--expansions", type=int, default=EXPANSION_COUNT, help="Number of query variants (default: 3)")
    args = parser.parse_args()

    results = retrieve(
        args.query,
        top_k=args.top_k,
        expansion_count=args.expansions,
        adaptive_expand=args.adaptive_expand,
    )

    if args.json:
        for r in results:
            if "headers" in r and not isinstance(r["headers"], (dict, list, str)):
                r["headers"] = str(r["headers"])
        print(_console_safe(json.dumps(results, indent=2, ensure_ascii=False)))
    else:
        output = "\n".join(
            (
                f"\nQuery: {args.query}",
                f"Strategy: Hybrid (vector + PostgreSQL text rank) "
                f"{'+ adaptive query expansion' if args.adaptive_expand else ''}",
                f"Results: {len(results)}\n",
                format_results(results),
            )
        )
        print(_console_safe(output))


if __name__ == "__main__":
    main()
