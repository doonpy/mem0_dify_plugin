"""Unit tests for the add_long_term_memory tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_tool(monkeypatch: pytest.MonkeyPatch, *, async_mode: bool = False):
    """Build an AddLongTermMemoryTool with patched runtime and message factories."""
    import tools.add_long_term_memory as mod
    from tools.add_long_term_memory import AddLongTermMemoryTool

    monkeypatch.setattr(mod, "is_async_mode", lambda _c: async_mode)

    # Make create_json_message / create_text_message return the raw payload so we
    # can assert on them without constructing Dify SDK objects.
    mock_runtime = MagicMock()
    mock_runtime.credentials = {}
    tool = AddLongTermMemoryTool(runtime=mock_runtime, session=MagicMock())
    tool.create_json_message = lambda d: d  # type: ignore[method-assign]
    tool.create_text_message = lambda t: t  # type: ignore[method-assign]
    return tool, mod


def _params(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "user_id": "u1",
        "user": "I live in Hanoi and love rainy afternoons.",
        "assistant": "Got it, I'll remember that.",
    }
    base.update(overrides)
    return base


def test_sync_classified_and_extracted_writes_with_explicit_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: (semantic, True) -> build semantic client, write, SUCCESS."""
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    # Classifier returns (semantic, True)
    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = ("semantic", True)
    monkeypatch.setattr(
        mod, "SyncMemoryClassificationManager", lambda _mem: fake_classifier
    )

    # get_sync_client returns a fake base client (only .memory is used here)
    base_client = MagicMock()
    base_client.memory = object()
    monkeypatch.setattr(mod, "get_sync_client", lambda _c: base_client)

    # build_single_subtype_sync_client returns a closable fake
    subtype_client = MagicMock()
    subtype_client.close = MagicMock()
    built_subtypes: list[str] = []

    def fake_build_sync(_creds: dict[str, Any], subtype: str):
        built_subtypes.append(subtype)
        return subtype_client

    monkeypatch.setattr(mod, "build_single_subtype_sync_client", fake_build_sync)

    # Writer records the add_memory call
    writer = MagicMock()
    writer.add_memory.return_value = {"results": [{"id": "m1", "event": "ADD"}]}
    monkeypatch.setattr(mod, "SyncMemoryWriter", lambda _c: writer)

    msgs = list(tool._invoke(_params()))

    # Must have built exactly one client, for the classified subtype
    assert built_subtypes == ["semantic"]

    # Writer called with explicit origin and correct subtype
    writer.add_memory.assert_called_once()
    kwargs = writer.add_memory.call_args.kwargs
    assert kwargs["user_id"] == "u1"
    assert kwargs["metadata"]["memory_subtype"] == "semantic"
    assert kwargs["metadata"]["memory_origin"] == "explicit"

    # Subtype client closed exactly once
    subtype_client.close.assert_called_once()

    # JSON result = SUCCESS with classified_type
    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "SUCCESS"
    assert json_msg["classified_type"] == "semantic"


def test_sync_caller_metadata_cannot_override_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-supplied metadata must not override memory_subtype / memory_origin.

    Spec constraints: the classifier chooses the subtype, and memory_origin must be
    "explicit" because this is user-driven. Custom extras are preserved.
    """
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = ("semantic", True)
    monkeypatch.setattr(
        mod, "SyncMemoryClassificationManager", lambda _mem: fake_classifier
    )
    monkeypatch.setattr(mod, "get_sync_client", lambda _c: MagicMock(memory=object()))

    subtype_client = MagicMock()
    subtype_client.close = MagicMock()
    monkeypatch.setattr(
        mod, "build_single_subtype_sync_client", lambda _c, _s: subtype_client
    )

    writer = MagicMock()
    writer.add_memory.return_value = {"results": []}
    monkeypatch.setattr(mod, "SyncMemoryWriter", lambda _c: writer)

    params = _params(
        metadata={
            "memory_subtype": "procedural",  # attempt to override
            "memory_origin": "implicit",  # attempt to override
            "custom_tag": "keep_me",  # should survive
        }
    )
    list(tool._invoke(params))

    writer.add_memory.assert_called_once()
    md = writer.add_memory.call_args.kwargs["metadata"]
    assert md["memory_subtype"] == "semantic"
    assert md["memory_origin"] == "explicit"
    assert md["custom_tag"] == "keep_me"


def test_sync_classified_but_low_value_skips_without_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(episodic, False) -> SKIPPED with classified_type, no client built."""
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = ("episodic", False)
    monkeypatch.setattr(
        mod, "SyncMemoryClassificationManager", lambda _mem: fake_classifier
    )
    monkeypatch.setattr(mod, "get_sync_client", lambda _c: MagicMock(memory=object()))

    def _boom_build(*_a, **_kw):  # pragma: no cover
        raise AssertionError("subtype client must not be built when should_extract=False")

    monkeypatch.setattr(mod, "build_single_subtype_sync_client", _boom_build)

    def _boom_writer(*_a, **_kw):  # pragma: no cover
        raise AssertionError("writer must not be constructed when skipping")

    monkeypatch.setattr(mod, "SyncMemoryWriter", _boom_writer)

    msgs = list(tool._invoke(_params()))

    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "SKIPPED"
    assert json_msg["reason"] == "not_classified_or_low_value"
    assert json_msg["classified_type"] == "episodic"


def test_sync_unclassified_skips_without_classified_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(None, False) -> SKIPPED with no classified_type field, no build, no write."""
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = (None, False)
    monkeypatch.setattr(
        mod, "SyncMemoryClassificationManager", lambda _mem: fake_classifier
    )
    monkeypatch.setattr(mod, "get_sync_client", lambda _c: MagicMock(memory=object()))

    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("must not build or write for unclassified turn")

    monkeypatch.setattr(mod, "build_single_subtype_sync_client", _boom)
    monkeypatch.setattr(mod, "SyncMemoryWriter", _boom)

    msgs = list(tool._invoke(_params()))

    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "SKIPPED"
    assert json_msg["reason"] == "not_classified_or_low_value"
    assert "classified_type" not in json_msg


def test_empty_messages_skip_without_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty user + empty assistant -> SKIPPED with reason=empty_messages and no classify."""
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    def _boom_classifier(*_a, **_kw):  # pragma: no cover
        raise AssertionError("classification must not run on empty input")

    monkeypatch.setattr(mod, "SyncMemoryClassificationManager", _boom_classifier)
    monkeypatch.setattr(mod, "get_sync_client", _boom_classifier)

    msgs = list(tool._invoke(_params(user="", assistant="   ")))

    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "SKIPPED"
    assert json_msg["reason"] == "empty_messages"


def test_missing_user_id_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool, _mod = _make_tool(monkeypatch, async_mode=False)

    msgs = list(tool._invoke(_params(user_id="")))

    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "ERROR"
    assert "user_id" in json_msg["messages"]


def test_sync_writer_error_surfaces_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer raising propagates to the outer try/except as ERROR; subtype client still closed."""
    tool, mod = _make_tool(monkeypatch, async_mode=False)

    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = ("procedural", True)
    monkeypatch.setattr(
        mod, "SyncMemoryClassificationManager", lambda _mem: fake_classifier
    )
    monkeypatch.setattr(mod, "get_sync_client", lambda _c: MagicMock(memory=object()))

    subtype_client = MagicMock()
    subtype_client.close = MagicMock()
    monkeypatch.setattr(
        mod,
        "build_single_subtype_sync_client",
        lambda _c, _s: subtype_client,
    )

    writer = MagicMock()
    writer.add_memory.side_effect = RuntimeError("mem0 boom")
    monkeypatch.setattr(mod, "SyncMemoryWriter", lambda _c: writer)

    msgs = list(tool._invoke(_params()))

    # Client must still be closed despite the error
    subtype_client.close.assert_called_once()

    # Error path yields ERROR status
    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "ERROR"
    assert "mem0 boom" in json_msg["messages"]


def test_async_happy_path_accepts_immediately_and_enqueues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """async_mode=True -> yield ACCEPTED before classification, enqueue a coroutine."""
    tool, mod = _make_tool(monkeypatch, async_mode=True)

    # Base client used only for ensure_bg_loop
    base_client = MagicMock()
    base_client.ensure_bg_loop.return_value = "FAKE_LOOP"
    monkeypatch.setattr(mod, "get_async_client", lambda _c: base_client)

    # TaskTracker: low pending, capture track_bg_task calls
    fake_tracker = MagicMock()
    fake_tracker.get_pending_tasks_count.return_value = 0
    fake_tracker.track_bg_task = MagicMock()
    monkeypatch.setattr(mod, "TaskTracker", fake_tracker)

    # parse_positive_int just returns the default
    monkeypatch.setattr(
        mod,
        "parse_positive_int",
        lambda _v, default, **_kw: default,
    )

    # Capture run_coroutine_threadsafe without executing the coroutine
    captured: dict[str, Any] = {}

    def fake_run_coro(coro: Any, loop: Any) -> Any:
        captured["loop"] = loop
        # close the coroutine so Python doesn't warn about unawaited coroutines
        coro.close()
        return "FAKE_FUTURE"

    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", fake_run_coro)

    # Classifier / writer / builder should not be invoked on the foreground path.
    def _boom_sync(*_a, **_kw):  # pragma: no cover
        raise AssertionError("sync-path helpers must not be called in async mode")

    monkeypatch.setattr(mod, "SyncMemoryClassificationManager", _boom_sync)
    monkeypatch.setattr(mod, "SyncMemoryWriter", _boom_sync)
    monkeypatch.setattr(mod, "build_single_subtype_sync_client", _boom_sync)

    msgs = list(tool._invoke(_params()))

    # ACCEPTED response
    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "ACCEPTED"

    # Task submitted to the background loop and tracked
    assert captured["loop"] == "FAKE_LOOP"
    fake_tracker.track_bg_task.assert_called_once()


def test_async_overload_yields_overload_and_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pending tasks exceed max_ops * multiplier, yield OVERLOAD without enqueue."""
    tool, mod = _make_tool(monkeypatch, async_mode=True)

    base_client = MagicMock()

    def _boom_loop():  # pragma: no cover
        raise AssertionError("ensure_bg_loop must not be called on overload")

    base_client.ensure_bg_loop = _boom_loop
    monkeypatch.setattr(mod, "get_async_client", lambda _c: base_client)

    fake_tracker = MagicMock()
    fake_tracker.get_pending_tasks_count.return_value = 10_000

    def _boom_track(*_a, **_kw):  # pragma: no cover
        raise AssertionError("track_bg_task must not be called on overload")

    fake_tracker.track_bg_task = _boom_track
    monkeypatch.setattr(mod, "TaskTracker", fake_tracker)

    monkeypatch.setattr(
        mod,
        "parse_positive_int",
        lambda _v, default, **_kw: default,
    )

    def _boom_coro(*_a, **_kw):  # pragma: no cover
        raise AssertionError("run_coroutine_threadsafe must not be called on overload")

    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", _boom_coro)

    msgs = list(tool._invoke(_params()))

    json_msg = msgs[0]
    assert isinstance(json_msg, dict)
    assert json_msg["status"] == "OVERLOAD"
