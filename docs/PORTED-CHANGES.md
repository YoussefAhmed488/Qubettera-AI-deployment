# Ported Changes — Parallel Debate & Resilience

## Files Changed in Qubettera-AI

| File | Status |
|------|--------|
| `src/qubettera/discussion/orchestrator.py` | **Updated** — thread-safe event writes + DiscussionExecutionError merge |
| `src/qubettera/discussion/demo.py` | **Updated** — ConsoleTurnStream, --parallel, --no-agent-tools, persona names, timestamped output |
| `src/qubettera/discussion/retrieval_provider.py` | **No changes needed** — Qubettera-AI uses local MiniLM, not Kaggle Ollama |
| `src/qubettera/rag/retrieve.py` | **No changes needed** — Qubettera-AI uses local SentenceTransformer embeddings |
| `tests/discussion/test_demo.py` | **Created** — fake mode streaming tests |

---

## Change Details

### 1. `src/qubettera/discussion/orchestrator.py`

| Change | Why |
|--------|-----|
| Added `import threading` | Needed for `threading.RLock()` |
| Added `self._write_lock = threading.RLock()` in `__init__` | Thread-safe console output when parallel |
| `_persist_outcomes` now acquires `_write_lock` around `_write_event` | Prevents garbled interleaved output from concurrent agents |
| `_write_event` itself also holds `_write_lock` | Double protection for direct calls |

> **Bug fix (2026-09-23):** The original `threading.Lock()` caused a self-deadlock
> because `_persist_outcomes` acquires the lock then calls `_write_event`, which also
> acquires it. Since `Lock` is non-reentrant, the same thread deadlocked — even in
> single-threaded mode. Switched to `threading.RLock()` which allows reentrant acquisition.
| `run()` merges `DiscussionExecutionError.partial_result` messages into failure report | Already present — confirmed working |

**Note:** Qubettera-AI already had `_execute_stage` + `max_workers` + `ThreadPoolExecutor` — this is a cleaner architecture than the `parallel=True` flag. The `--parallel` flag in demo.py maps to `max_workers=len(participants)`, sequential to `max_workers=1`.

### 2. `src/qubettera/discussion/demo.py`

| Change | Why |
|--------|-----|
| Added `ConsoleTurnStream` class | Prints each turn's opinion, retrieval query, evidence count live to console while also writing to JSONL |
| Added `_persona_names()` helper | Resolves human-readable names from `personas/<id>.json` |
| Added `--parallel` flag | `max_workers=len(participants)` → all agents run concurrently per round |
| Added `--no-agent-tools` flag | Passes `tools=[]` to `Week2AgentRuntime` for LLM-only turns |
| Added `datetime`/`timezone` imports | Timestamp in output filename |
| Output filename now includes mode, topology, agent count, timestamp | Examples: `demo-live-json-5agents-20260122T143000.jsonl` |
| Added participants/objective header and summary footer | Better console UX |
| Added `EventSink` import | Required by `ConsoleTurnStream` |
| Wired `--parallel` → `max_workers` in orchestrator | `len(participants)` when parallel, `1` otherwise |

### 3. `src/qubettera/discussion/retrieval_provider.py`

**No changes needed.** Qubettera-AI's version already uses:
- Local `MiniLM` / `SentenceTransformer` embeddings (not Kaggle Ollama)
- `RetrievalService` wrapper around PostgreSQL pgvector
- Graceful degradation on errors (returns `()` instead of crashing)
- `_clip()` component truncation for query building
- `retrieval_focus` from persona loader

The Kaggle Ollama semaphore + retry logic from the Transformer-Architecture-Debate-Framework is **not applicable** because Qubettera-AI runs embeddings locally.

### 4. `tests/discussion/test_demo.py` (**New file**)

Two tests ported from the original `tests/week3/test_demo.py`:
- `test_fake_mode_streams_every_turn` — runs `--mode fake`, verifies `[opening]`, `[round 1]`, `[round 2]`, `[round 3]` appear in console output
- `test_fake_mode_logs_events_while_streaming` — verifies the JSONL file has 20 `turn_completed` events

---

## How to Run

### From the Qubettera-AI project root:

```bash
cd /home/Herofis/projects/Qubettera_AI/Qubettera-AI

# Fake mode (deterministic, no API keys):
python -m qubettera.discussion.demo --mode fake
python -m qubettera.discussion.demo --mode fake --topology ring
python -m qubettera.discussion.demo --mode fake --topology full

# Live mode (requires LLM API keys in .env):
python -m qubettera.discussion.demo --mode live
python -m qubettera.discussion.demo --mode live --parallel          # Concurrent agents
python -m qubettera.discussion.demo --mode live --no-retrieval       # No knowledge base
python -m qubettera.discussion.demo --mode live --no-agent-tools     # LLM-only, no tools

# Run tests:
python -m pytest tests/discussion/test_demo.py -v
python -m pytest tests/discussion/ -v

# Output goes to:
outputs/discussions/demo-fake-json-5agents-<timestamp>.jsonl
```

---

## Architecture Note: Kaggle vs Local Embeddings

| Aspect | Transformer-Architecture-Debate-Framework | Qubettera-AI |
|--------|-------------------------------------------|--------------|
| Embedding model | `qwen3-embedding:8b` on Kaggle Ollama (remote GPU) | `MiniLM` via `SentenceTransformer` (local CPU/GPU) |
| Connection | ngrok tunnel → Kaggle notebook | Direct PostgreSQL connection |
| Concurrency guard | `Semaphore(2)` + retry + backoff | Not needed (local embedding is fast) |
| Failure mode | Returns `()` if Kaggle is unreachable | Returns `()` on DB errors |

This means **Qubettera-AI does not have the Kaggle retrieval failures** that the other project experiences. Retrieval is always available as long as PostgreSQL is reachable.