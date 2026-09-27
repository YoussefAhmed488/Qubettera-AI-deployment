"""Copy the retrieval index from local Postgres to Supabase.

The migration is repeatable: it rebuilds the target's ``chunks`` table through
the same staging-then-swap flow used locally, so a failed or interrupted run
leaves the previously published table untouched. Two phases are involved:

1. ``stage``: stream the local corpus into ``chunks_staging`` on the target and
   build its indexes. The live table stays readable throughout.
2. ``publish``: atomically rename staging over the live table.

``run`` performs both. ``--publish-only`` exists separately so a long upload can
be verified before the target's live table changes.

Both phases are committed independently, which is why they are separate
functions rather than one transaction: a single transaction over an
inter-region ``COPY`` would hold one snapshot for the whole transfer.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import closing
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psycopg2 import sql

from qubettera.rag import database, store
from qubettera.rag.database import active_target, describe_target, use_target

# The target's schema names and its search_path both come from the shared
# constants in `database`, so the migration and the runtime connection can never
# disagree. Index builds on a managed instance need more memory than the local
# 64 MB default provides.
TARGET_SCHEMA = database.CORPUS_SCHEMA
EXPECTED_EXTENSION_SCHEMA = database.EXTENSION_SCHEMA
MAINTENANCE_WORK_MEM = "256MB"
COPY_PAGE_ROWS = 20_000

# The migration always reads the local corpus and writes to Supabase; the two
# are pinned so a stray QUBETTERA_DB value can never copy a database onto itself.
MIGRATION_SOURCE = "local"
MIGRATION_TARGET = "supabase"


def _target_search_path() -> sql.Composed:
    # Derived from the shared constant so the migration and the runtime
    # connection always agree on the schema list.
    return sql.SQL(", ").join(
        sql.Identifier(part) for part in database.SUPABASE_SCHEMAS
    )


def _prepare_target(cur, target: str) -> None:
    """Install pgvector and pin the session settings needed for the build.

    The target is passed explicitly rather than read from ``active_target()``:
    the caller may be inside a scoped override (``use_target("local")``) that
    does not describe the connection this cursor belongs to.
    """
    if target == "supabase":
        # Supabase ships pgvector but leaves it disabled per project, and it
        # must live in `extensions` for the pinned search_path to resolve it.
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(EXPECTED_EXTENSION_SCHEMA)
            )
        )
        cur.execute(
            sql.SQL("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA {}").format(
                sql.Identifier(EXPECTED_EXTENSION_SCHEMA)
            )
        )
        cur.execute(sql.SQL("SET LOCAL search_path TO {}").format(_target_search_path()))
    else:
        # Locally pgvector lives in `public`, which is already first on the
        # default search_path, so no override is needed.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        sql.SQL("SET LOCAL maintenance_work_mem = {}").format(
            sql.Literal(MAINTENANCE_WORK_MEM)
        )
    )


def _source_manifest(conn) -> dict:
    """Read the local corpus size and its single embedding identity."""
    live = store.DEFAULT_TABLES.live
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT COUNT(*), COUNT(DISTINCT embedding_model),"
                " COUNT(DISTINCT embedding_model_revision),"
                " COUNT(DISTINCT preprocessing_version),"
                " COUNT(DISTINCT pipeline_version),"
                " COUNT(DISTINCT vector_dims(embedding)) FROM {}"
            ).format(sql.Identifier(live))
        )
        count, models, revisions, preprocessing, pipeline, dimensions = cur.fetchone()
        cur.execute(
            sql.SQL("SELECT vector_dims(embedding) FROM {} LIMIT 1").format(
                sql.Identifier(live)
            )
        )
        dimension = cur.fetchone()[0]
    if not count:
        raise RuntimeError(
            f"Local {live!r} table is empty; run 'qubettera rag pipeline' first."
        )
    if dimensions != 1:
        raise RuntimeError(
            f"Local {live!r} table mixes {dimensions} embedding dimensions; expected 1."
        )
    if (models, revisions, preprocessing, pipeline) != (1, 1, 1, 1):
        raise RuntimeError(
            f"Local {live!r} table mixes embedding identities "
            f"(models={models}, revisions={revisions}, preprocessing={preprocessing}, "
            f"pipeline={pipeline}); expected 1 of each."
        )
    return {"count": count, "dimension": int(dimension)}


def _stream_rows(source_conn, target_conn, expected_count: int) -> int:
    """Copy every row page by page using keyset pagination on ``chunk_id``.

    Paging by primary key keeps peak memory bounded by one page instead of the
    whole corpus, and ``chunk_id`` is a text primary key so the order is total.
    """
    columns = store.CHUNK_COLUMNS
    column_list = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    source_table = sql.Identifier(store.DEFAULT_TABLES.live)
    target_spec = sql.SQL("{} ({})").format(
        sql.Identifier(store.DEFAULT_TABLES.staging), column_list
    )

    copied = 0
    last_key = ""
    buffer = io.StringIO()
    with target_conn.cursor() as target_cur:
        target_cur.execute(sql.SQL("SET LOCAL search_path TO {}").format(_target_search_path()))
        load_statement = sql.SQL("COPY {} FROM STDIN WITH (FORMAT csv)").format(
            target_spec
        ).as_string(target_cur)
        with source_conn.cursor() as key_cur:
            while True:
                key_cur.execute(
                    sql.SQL(
                        "SELECT chunk_id FROM {} WHERE chunk_id > %s"
                        " ORDER BY chunk_id LIMIT %s"
                    ).format(source_table),
                    (last_key, COPY_PAGE_ROWS),
                )
                keys = [row[0] for row in key_cur.fetchall()]
                if not keys:
                    break
                high_key = keys[-1]
                with source_conn.cursor() as copy_cur:
                    page_statement = sql.SQL(
                        "COPY (SELECT {} FROM {} WHERE chunk_id > {} AND chunk_id <= {}"
                        " ORDER BY chunk_id) TO STDOUT WITH (FORMAT csv)"
                    ).format(
                        column_list,
                        source_table,
                        sql.Literal(last_key),
                        sql.Literal(high_key),
                    ).as_string(copy_cur)
                    buffer.seek(0)
                    buffer.truncate()
                    copy_cur.copy_expert(page_statement, buffer)
                    buffer.seek(0)
                    target_cur.copy_expert(load_statement, buffer)
                copied += len(keys)
                last_key = high_key
                print(f"  copied {copied}/{expected_count} rows", flush=True)

    buffer.seek(0)
    buffer.truncate()
    return copied


def stage() -> dict:
    """Copy the local corpus into the target's staging table and index it."""
    with use_target(MIGRATION_TARGET):
        target_conn = store.connect_database(purpose="migration target")
    with closing(target_conn) as target_conn:
        target_conn.autocommit = False
        try:
            with use_target(MIGRATION_SOURCE):
                with closing(
                    store.connect_database(purpose="migration source")
                ) as source_conn:
                    source_conn.autocommit = True
                    manifest = _source_manifest(source_conn)
                    with target_conn.cursor() as cur:
                        _prepare_target(cur, MIGRATION_TARGET)
                        store._create_staging_schema(cur, manifest["dimension"])
                    target_conn.commit()

                    copied = _stream_rows(source_conn, target_conn, manifest["count"])
                    if copied != manifest["count"]:
                        raise RuntimeError(
                            f"Copied {copied} rows but the source reported {manifest['count']}."
                        )
                    target_conn.commit()

            # Indexing runs after the load commits: building an IVFFlat index
            # inside the same transaction as a multi-minute COPY would hold it
            # open and bloat WAL on the managed instance.
            with target_conn.cursor() as cur:
                _prepare_target(cur, MIGRATION_TARGET)
                store._build_and_validate_indexes(cur, manifest["count"])
            target_conn.commit()
        except Exception:
            target_conn.rollback()
            raise

    print(f"Staged {copied} rows into {store.STAGING_TABLE} on {describe_target()}.")
    return {"count": copied, "dimension": manifest["dimension"]}


def publish() -> None:
    """Atomically swap the staged table over the target's live table."""
    with use_target(MIGRATION_TARGET):
        conn = store.connect_database(purpose="migration target")
    with closing(conn) as target_conn:
        target_conn.autocommit = False
        try:
            with target_conn.cursor() as cur:
                _prepare_target(cur, MIGRATION_TARGET)
                cur.execute(
                    "SELECT to_regclass(current_schema() || '.' || %s)",
                    (store.DEFAULT_TABLES.staging,),
                )
                if cur.fetchone()[0] is None:
                    raise RuntimeError(
                        f"No {store.DEFAULT_TABLES.staging!r} table on "
                        f"{describe_target()} to publish. Run 'qubettera rag migrate' "
                        "or '--stage-only' first."
                    )
                store._atomic_swap(cur)
            target_conn.commit()
        except Exception:
            target_conn.rollback()
            raise
    print(f"Published staging table on {describe_target()}.")
    print(
        "Local Postgres was not modified. To make the app use Supabase, set "
        "QUBETTERA_DB=supabase (and CHECKPOINT_BACKEND if needed)."
    )


def run(*, publish_only: bool = False, stage_only: bool = False) -> dict | None:
    if active_target() != MIGRATION_TARGET:
        raise RuntimeError(
            f"Refusing to migrate: QUBETTERA_DB must be {MIGRATION_TARGET!r} so the "
            "local database is never treated as the migration target. Run "
            "'qubettera rag supabase-init' first."
        )
    if publish_only and stage_only:
        raise ValueError("Choose either --publish-only or --stage-only, not both.")
    result = None
    if not publish_only:
        result = stage()
    if not stage_only:
        publish()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stage-only",
        action="store_true",
        help="Copy and index into chunks_staging without touching the live table.",
    )
    group.add_argument(
        "--publish-only",
        action="store_true",
        help="Swap an already staged table over the live table.",
    )
    args = parser.parse_args(argv)
    run(publish_only=args.publish_only, stage_only=args.stage_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
