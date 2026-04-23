# Deferred Work

Pre-existing issues surfaced during review but not caused by the triggering story.

## From spec-add-long-term-memory.md review (2026-04-23)

### `metadata` YAML parameter declared as string but never parsed as JSON

**Tools affected:** `tools/add_memory.py`, `tools/add_long_term_memory.py`.

**Issue:** Both tools' YAML manifests declare `metadata` with `type: string` and
`label: "Metadata (JSON)"`, instructing users to supply JSON. The Python code does
not parse the string — it is passed straight through (in `add_memory`) or silently
dropped unless it is already a dict (in `add_long_term_memory`). As a result,
workflow users supplying metadata as documented see it either ignored or flattened
into a single string field in Mem0.

**Why deferred:** The `add_long_term_memory` spec explicitly scopes this tool to
mirror `add_memory.yaml`'s parameters verbatim, and fixing this requires touching
`add_memory.py` (which the spec marks as out-of-scope). Needs its own story to
(a) add a JSON-string parser helper in `utils/memory_tool_helpers.py` and
(b) apply it consistently across both add tools and any other YAML string-typed
metadata fields.

**Suggested fix sketch:** parse with `json.loads`; on parse failure, log at
warning level and fall back to `None` (do not crash the write).
