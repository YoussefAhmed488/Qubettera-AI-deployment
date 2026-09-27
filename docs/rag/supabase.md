# Running against Supabase

The retrieval corpus lives in PostgreSQL with pgvector. It runs in the bundled
container by default, but it can also be hosted on Supabase. Both targets use
the same schema and SQL, so switching is a configuration change.

## Which target is active

`QUBETTERA_DB` selects the target:

| Value | Source of credentials |
| --- | --- |
| `local` (default) | `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` |
| `supabase` | one connection URI, from `SUPABASE_DB_URL` or, if unset, the gitignored `resources/supabase.target.json` |

The target is resolved at connection time, so no code path can accidentally
fall back to the local database when Supabase is selected. An unknown value
raises immediately instead of silently using a default.

## Create the Supabase project

1. Create a project at <https://supabase.com/dashboard> and wait for it to
   finish provisioning.
2. Enable pgvector. In the SQL editor run:

   ```sql
   create extension if not exists vector with schema extensions;
   ```

   Supabase installs extensions into the `extensions` schema rather than
   `public`. Every Qubettera connection therefore pins
   `search_path = public,extensions`, because the retrieval SQL uses bare
   `::vector` casts and the `<=>` operator. Listing `pg_catalog` in that path
   would make it the current schema and break unqualified `CREATE TABLE`, so it
   is left implicit; spaces around the comma are also omitted because libpq
   splits the option string on whitespace.

   The `chunks` table, its indexes, and the corpus rows are all created by the
   migration, so no other DDL is needed.

   Only two schemas are involved: `public` holds the corpus (`chunks` plus its
   indexes) on both targets, and `extensions` holds pgvector on Supabase only
   (locally pgvector lives in `public`). These names live in one place,
   `database.CORPUS_SCHEMA`/`EXTENSION_SCHEMA`, which both the runtime
   connection and the migration read.

## Choose the connection string

Open **Connect** in the dashboard and copy the **Session pooler** URI:

```
postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

Use the session pooler (port `5432`), not the transaction pooler (port `6543`).
Schema changes and the bulk `COPY` need one long-lived session, and the
transaction pooler cannot provide that.

The **direct connection** (`db.<project-ref>.supabase.co`) is the documented
choice for migrations, but it resolves to IPv6 only unless the project has the
paid IPv4 add-on. On an IPv4-only network the session pooler is the correct
endpoint; `supabase-init` rejects a direct-connection host for exactly this
reason, rather than letting it surface later as a connection timeout.

## Save the connection string

```powershell
qubettera rag supabase-init --url "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
```

The URI is written to `resources/supabase.target.json` with owner-only
permissions and is gitignored. It can also be supplied per-invocation with
`SUPABASE_DB_URL`, which takes precedence over the file. Pass `--force` to
replace an existing file.

## Migrate the corpus

```powershell
$env:QUBETTERA_DB = "supabase"
qubettera rag migrate
```

The migration reads the local corpus and writes to Supabase, never the reverse,
and refuses to run unless `QUBETTERA_DB=supabase`. It runs in two committed
phases:

1. **Stage** — create `chunks_staging` with the canonical schema, stream every
   row across with `COPY` in `chunk_id`-ordered pages, then build and validate
   the IVFFlat and GIN indexes.
2. **Publish** — atomically rename `chunks_staging` over `chunks`.

Because the two phases commit separately, an interrupted upload leaves the
existing `chunks` table untouched. Run `--stage-only` to prepare Supabase while
local stays authoritative, then `--publish-only` once the staged copy is
verified. Both phases validate the row count and require a single embedding
identity, so a partial or mixed copy is rejected rather than published.

The local database is never modified. To switch back, set `QUBETTERA_DB=local`.

## Durable agent threads

`CHECKPOINT_BACKEND=postgres` stores LangGraph threads. On the Supabase target
the checkpoint connection is resolved from the same URI and the same
`search_path`, so no extra variables are needed. The `PGHOST`/`PGDATABASE`/
`PGUSER`/`PGPASSWORD` variables are still required for the local target.

Supabase's schema is owned by `postgres`; the checkpoint tables are created by
`saver.setup()` on first use.

## Verify a migration

Compare the source and target without dumping either table:

```sql
select md5(string_agg(chunk_id || '|' || content_hash, ',' order by chunk_id))
from chunks;
```

Then confirm retrieval works against the new target and that the identity
manifest matches:

```powershell
$env:QUBETTERA_DB = "supabase"
qubettera rag retrieve "mixture of experts routing" --top-k 5
```

The manifest records the row count and the embedding, preprocessing, and
pipeline versions. Coworkers whose local corpus is built from the same pipeline
version read the same rows.

## Notes and limits

- Expect higher latency than the local container. The corpus is small
  (~20k rows), so the planner may prefer a sequential scan over the IVFFlat
  index on either target; the index still bounds latency as the corpus grows.
- Storage, connection limits, and backups follow the Supabase plan. The local
  container's `qubettera-postgres` volume is unaffected by a migration.
- Rows are copied exactly as stored, so the embedding model, revision,
  preprocessing version, and pipeline version of the local corpus are preserved.
  Re-run `qubettera rag pipeline` locally and migrate again to refresh content.
