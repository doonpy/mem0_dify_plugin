---
title: 'Add classified long-term memory tool'
type: 'feature'
created: '2026-04-23'
status: 'done'
baseline_commit: 'd87a418bfa09238fd166b6b1c8a56d45edd0950e'
context:
  - '{project-root}/tools/extract_long_term_memory.py'
  - '{project-root}/utils/mem0_extraction.py'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `tools/add_memory.py` unconditionally writes every user/assistant turn to Mem0, which is wasteful for low-value chatter and bypasses the semantic/episodic/procedural classification that `extract_long_term_memory` uses to build long-term memory. There is no single-turn counterpart to the classification+extraction flow.

**Approach:** Add a new Dify tool `add_long_term_memory` that mirrors `add_memory`'s input shape (user/assistant text), classifies the turn via `MemoryClassificationManager`, and on `(type, should_extract=True)` writes to the matching subtype client with `memory_origin="explicit"`. On `(None, False)` or `should_extract=False` it yields a `SKIPPED` result without writing. Supports sync and async modes. Leaves `add_memory` untouched.

## Boundaries & Constraints

**Always:**
- Build **only one** subtype client per invocation (the classified type's) — the LLM chooses the subtype first, then we allocate that subtype's client. Applies to both sync and async paths.
- Use `memory_origin="explicit"` in metadata (this is user-driven, not an implicit extraction).
- Always close the subtype client in `finally` (both sync `.close()` and async `.aclose()`).
- Async mode: enqueue classification+write as a single coroutine on the background event loop, return `ACCEPTED` immediately (same shape as `add_memory` async path).
- Sync mode: classify and write in-request, return `SUCCESS`/`SKIPPED` synchronously.
- Reuse `SyncMemoryClassificationManager` / `AsyncMemoryClassificationManager` / `SyncMemoryWriter` / `AsyncMemoryWriter` from `utils/mem0_extraction.py`.
- Reuse `init_request_context`, `validate_user_id`, `yield_error`, `log_thread_info`, `parse_timeout` helpers.

**Ask First:**
- Any change to `tools/add_memory.py` itself (out of scope — this is a new tool).
- Adding a top-level credential toggle (e.g. "always classify"). Out of scope.

**Never:**
- Do not modify `add_memory.py` or its YAML.
- Do not change `MEMORY_CLASSIFICATION_PROMPT` or existing classification managers.
- Do not build all three subtype clients when only one is needed.
- Do not invent a new metadata schema — use `build_memory_metadata`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Classification returns (semantic, True) | valid user+assistant turn | build semantic subtype client, write via `SyncMemoryWriter.add_memory` with `memory_origin="explicit"`, yield `SUCCESS` with `classified_type="semantic"` and mem0 results | N/A |
| Classification returns (None, False) | vague chatter | no write, yield `SKIPPED` with `reason="not_classified_or_low_value"` | N/A |
| Classification returns (episodic, False) | type detected but low-value | no write, yield `SKIPPED` with `reason="not_classified_or_low_value"` and `classified_type="episodic"` | N/A |
| Empty messages (blank user + blank assistant) | empty strings | no classify/write, yield `SKIPPED` with `reason="empty_messages"` | N/A |
| Missing user_id | `user_id=""` | yield ERROR via `yield_error` | `user_id is required` |
| Async mode happy path | valid turn, `async_mode=true` | enqueue background task, yield `ACCEPTED` immediately with `task_info` | N/A |
| Async overload | pending tasks > threshold | yield `OVERLOAD` before enqueue | same pattern as `add_memory._execute_async_add` |
| Classification raises (network/LLM error) | LLM call fails | caught inside manager → returns `(None, False)` → yields `SKIPPED` (no retry) | existing behavior of `MemoryClassificationManager` |
| Writer raises (mem0 error) | classification ok, `.add()` raises | caught by outer `try/except`, yields ERROR via `yield_error` | standard error path |

</frozen-after-approval>

## Code Map

- `tools/add_long_term_memory.py` -- new tool, single-turn classified add
- `tools/add_long_term_memory.yaml` -- new Dify tool manifest
- `utils/mem0_extraction.py` -- add `build_single_subtype_sync_client` and `build_single_subtype_async_client` helpers (extract shared logic from existing `build_subtype_sync_clients` / `build_subtype_async_clients`)
- `tests/unit/tools/test_add_long_term_memory.py` -- new unit tests
- `tools/add_memory.py` -- reference only, not modified
- `tools/extract_long_term_memory.py` -- reference only (classification+write pattern)
- `utils/memory_tool_helpers.py` -- reused helpers
- `utils/constants.py` -- reused `WRITE_OPERATION_TIMEOUT`, `MAX_PENDING_TASKS_MULTIPLIER`

## Tasks & Acceptance

**Execution:**
- [x] `utils/mem0_extraction.py` -- add `build_single_subtype_sync_client(credentials, subtype)` and `async build_single_subtype_async_client(credentials, subtype)` returning a single `SyncMem0Client` / `AsyncMem0Client` configured with that subtype's `custom_instructions`. Refactor existing `build_subtype_sync_clients` / `build_subtype_async_clients` to call the new helpers for each subtype (no behavior change).
- [x] `tools/add_long_term_memory.py` -- implement `AddLongTermMemoryTool` with `_build_messages`, `_should_skip_empty`, `_execute_sync` (classify → build single subtype client → write → close), `_execute_async` (enqueue combined coroutine on background loop), `_invoke`.
- [x] `tools/add_long_term_memory.yaml` -- Dify manifest mirroring `add_memory.yaml` parameters (user, assistant, user_id, agent_id, run_id, metadata, timeout).
- [x] `tests/unit/tools/test_add_long_term_memory.py` -- cover I/O matrix rows: classified+extract, classified+skip, unclassified, empty messages, writer error. Use `unittest.mock` to stub classification manager and writer (mirror pattern in `tests/unit/tools/test_add_memory_overload_guard.py`).

**Acceptance Criteria:**
- Given a valid user+assistant turn in sync mode, when `AddLongTermMemoryTool._invoke` runs and classification returns `("semantic", True)`, then exactly one `SyncMem0Client` for the `semantic` subtype is built, `SyncMemoryWriter.add_memory` is called with `metadata["memory_subtype"]=="semantic"` and `metadata["memory_origin"]=="explicit"`, the subtype client's `.close()` is called in `finally`, and the JSON result contains `status="SUCCESS"` and `classified_type="semantic"`.
- Given a valid turn but classification returns `(None, False)`, when `_invoke` runs, then no subtype client is built, no writer is called, and the JSON result contains `status="SKIPPED"` with `reason="not_classified_or_low_value"`.
- Given empty user and assistant strings, when `_invoke` runs, then classification is not invoked and the JSON result contains `status="SKIPPED"` with `reason="empty_messages"`.
- Given `async_mode=true` and classification+write would succeed, when `_invoke` runs, then the tool yields `ACCEPTED` immediately (before classification runs) and a background task is submitted via `asyncio.run_coroutine_threadsafe`.
- Given async mode with pending tasks > `max_ops * MAX_PENDING_TASKS_MULTIPLIER`, when `_invoke` runs, then the tool yields `OVERLOAD` and does not enqueue.
- Given `pytest tests/unit/tools/test_add_long_term_memory.py`, when run, then all tests pass.

## Design Notes

**Why build_single_subtype_*_client helpers:** `build_subtype_sync_clients` unconditionally builds all three. For this single-turn tool we know the classified subtype before we write, so building 3 pools (each with its own connection pool per `build_local_mem0_config_without_pool`) is wasteful — we only need one. Extracting a single-subtype helper is a small refactor that keeps the 3-subtype builder as a thin loop.

**Classification reuses base LLM:** In `extract_long_term_memory` the classifier uses `subtype_clients["semantic"].memory` (they share LLM config). For this tool, the classifier also needs an `Memory`/`AsyncMemory` instance. Options: (a) build a minimal base client via `get_sync_client`/`get_async_client` for classification, then build the chosen subtype client for writing — 2 clients; (b) build the semantic client first for classification, and reuse it if classified type is semantic, or swap otherwise — saves 1 pool in the semantic case only. **Choose (a)** — simpler, predictable, and `get_sync_client`/`get_async_client` returns the shared singleton (no extra pool). The extra subtype client is unavoidable because the writer needs the subtype-specific `custom_instructions`.

**Async flow sketch:**
```python
async def _bg_task() -> None:
    subtype_client = None
    try:
        classifier = AsyncMemoryClassificationManager(base_client.memory)
        classified, should = await classifier.classify(messages=..., context=...)
        if classified is None or not should:
            return  # status already sent as ACCEPTED; caller polls nothing
        subtype_client = await build_single_subtype_async_client(creds, classified)
        writer = AsyncMemoryWriter(subtype_client)
        await writer.add_memory(messages=..., user_id=..., metadata=..., timeout_s=...)
    finally:
        if subtype_client is not None:
            await subtype_client.aclose()
```
The async response is fire-and-forget like `add_memory` async mode — the caller gets `ACCEPTED` and the classification+write happen in the background. Any classification/write errors are logged but not surfaced (consistent with `add_memory` async). Track the future via `client.track_bg_task`.

## Verification

**Commands:**
- `cd /home/poonnguyen/personal/mem0_dify_plugin && python -m pytest tests/unit/tools/test_add_long_term_memory.py -v` -- expected: all tests pass
- `cd /home/poonnguyen/personal/mem0_dify_plugin && python -m pytest tests/unit/ -v` -- expected: existing suite still green (no regressions from the `mem0_extraction.py` refactor)
- `cd /home/poonnguyen/personal/mem0_dify_plugin && python -c "import tools.add_long_term_memory; import tools.add_long_term_memory as m; print(m.AddLongTermMemoryTool.__name__)"` -- expected: prints `AddLongTermMemoryTool`

**Manual checks:**
- Load `tools/add_long_term_memory.yaml` in a Dify workflow — all 7 parameters visible with correct labels.

## Suggested Review Order

**Entry point — tool orchestration**

- Top-down flow: validate → skip-empty → classify → dispatch sync/async.
  [`add_long_term_memory.py:315`](../../tools/add_long_term_memory.py#L315)

**Classification + single-subtype write (sync path)**

- Sync path: classify with shared base client, build one subtype client, write, close in `finally`.
  [`add_long_term_memory.py:86`](../../tools/add_long_term_memory.py#L86)

- Metadata precedence: caller extras first, then `build_memory_metadata` wins — protects `memory_subtype` / `memory_origin="explicit"`.
  [`add_long_term_memory.py:133`](../../tools/add_long_term_memory.py#L133)

**Async path — fire-and-forget with overload guard**

- Pre-enqueue overload check mirrors `add_memory`; background coroutine owns subtype client lifecycle.
  [`add_long_term_memory.py:176`](../../tools/add_long_term_memory.py#L176)

**Single-subtype helper refactor**

- New sync helper; 3-subtype builder now delegates (no behavior change).
  [`mem0_extraction.py:40`](../../utils/mem0_extraction.py#L40)

- New async helper symmetric with sync; eagerly calls `client.create()`.
  [`mem0_extraction.py:477`](../../utils/mem0_extraction.py#L477)

**Dify manifest**

- 7 parameters mirroring `add_memory.yaml` — no behavioural knobs added.
  [`add_long_term_memory.yaml:1`](../../tools/add_long_term_memory.yaml#L1)

**Tests — covers the I/O matrix + override-prevention regression**

- Happy path: one subtype client, writer gets `memory_origin="explicit"`.
  [`test_add_long_term_memory.py:38`](../../tests/unit/tools/test_add_long_term_memory.py#L38)

- Regression: caller metadata cannot override `memory_subtype` / `memory_origin`.
  [`test_add_long_term_memory.py:94`](../../tests/unit/tools/test_add_long_term_memory.py#L94)

- Async happy path: ACCEPTED yielded before classification; coroutine enqueued.
  [`test_add_long_term_memory.py:267`](../../tests/unit/tools/test_add_long_term_memory.py#L267)

- Async overload: no `ensure_bg_loop`, no `track_bg_task`, no coroutine enqueue.
  [`test_add_long_term_memory.py:322`](../../tests/unit/tools/test_add_long_term_memory.py#L322)
