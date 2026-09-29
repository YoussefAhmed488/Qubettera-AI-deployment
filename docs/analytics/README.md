# Discussion analytics

Analytics consumes a completed `outputs/discussions/<id>.jsonl` log. It writes
`outputs/analytics/analytics_<id>.json`, `report_<id>.md`, and two PNG files in
`outputs/analytics/visuals/`. The JSON preserves the existing `opinion_change`,
`agreement`, `influence`, and `sentiment` keys and adds `influence_edges` for
the interaction graph. An invalid or incomplete log fails before scoring.

## Reproduce

```powershell
python -m pip install -e ".[dev,analytics]"
qubettera analytics run tests/analytics/example.jsonl
qubettera analytics run outputs/discussions/72e40778-3f08-4442-9f0e-b5e7a217a0f0.jsonl
qubettera discuss run --mode live --analyze
python -m pytest tests/analytics -q
```

Configure the same `LLM_PROVIDER` and provider credentials or local endpoint
used by discussion generation. Stance scoring uses that model at temperature
zero. Sentiment scoring downloads its classifier on first use. `--no-visuals`
and `--no-report` omit those artifacts. Fake discussion logs contain placeholder
opinions and cannot be analyzed. The second command replays a completed local
discussion when that output file is present; the checked-in fixture always
exists and exercises the same pipeline with shorter history.

## Metrics

| Metric | Input and method | Output and interpretation | Limits |
|---|---|---|---|
| Opinion change | Score each opinion against one fixed proposition derived from the discussion brief; subtract the previous round's score. | One stance in `[-1, 1]` and a signed change per agent and round. Positive stance supports the proposition. | An LLM may misread mixed views; one axis cannot describe every architecture tradeoff. Temperature zero does not guarantee identical output across providers. |
| Agreement | Compute `1 - mean(|stance_i - stance_j| / 2)` across every distinct agent pair in a round. | One score in `[0, 1]` per round; higher means closer positions. | With fewer than two agents, the score is `null` with `insufficient_agents`. Agreement does not measure correctness. |
| Influence | For each routed edge, correlate the sender–recipient stance gap in round `t` with the recipient's stance change in `t+1` (Pearson `r`). Multiply by mean Jaccard overlap of non-stopword terms in the sender message and recipient's next message. Average usable outgoing edge scores for each agent. | Signed edge and agent scores in `[-1, 1]`; positive values indicate movement toward the sender's position, negative values indicate movement away, and both are scaled by shared terms. | At least three routed transitions with varying values are needed. Missing data yields `null` and a reason. Lexical overlap misses paraphrases and can make a correlation appear as zero; these scores are associations, not causal evidence. |
| Sentiment | Classify each participant message's content; use `P(positive) - P(negative)`. | One value in `[-1, 1]` per message plus per-agent and per-round summaries. Higher means more positive tone. | The classifier may truncate long text or misread technical language. Tone does not indicate stance or factual quality. |

The moderator synthesis is excluded from participant metrics. A complete log
contains one substantive participant turn per configured round, unique message
IDs, a valid graph, and a matching completion event. The report describes
missing influence scores explicitly. Week 5 can read the analytics JSON and
PNG files without importing metric internals.
