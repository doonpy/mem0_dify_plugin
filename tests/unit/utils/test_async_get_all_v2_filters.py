"""Regression: AsyncMem0Client.get_all must pass entity IDs inside filters (Mem0 v2)."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from utils.mem0_client import AsyncMem0Client


class FakeAsyncMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        # Mimic Mem0 v2's strict rejection of top-level entity IDs so tests
        # surface regressions even if the assertion below is forgotten.
        forbidden = {"user_id", "agent_id", "run_id"} & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Top-level entity parameters frozenset({forbidden!r}) are not "
                "supported in get_all(). Use filters={'user_id': '...'} instead."
            )
        return {"results": []}


def _make_client(monkeypatch: pytest.MonkeyPatch, mem: FakeAsyncMemory) -> AsyncMem0Client:
    import utils.mem0_client as mem0_client

    monkeypatch.setattr(mem0_client, "build_local_mem0_config", lambda _c: {})
    client = AsyncMem0Client({})
    client.memory = mem
    return client


def test_get_all_merges_entity_ids_into_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    mem = FakeAsyncMemory()
    client = _make_client(monkeypatch, mem)

    async def _run() -> None:
        await client.get_all({"user_id": "u1", "agent_id": "a1"})

    asyncio.run(_run())

    assert mem.calls, "get_all should have been called"
    kwargs = mem.calls[0]
    assert "user_id" not in kwargs
    assert "agent_id" not in kwargs
    assert "run_id" not in kwargs
    assert kwargs.get("filters", {}).get("user_id") == "u1"
    assert kwargs.get("filters", {}).get("agent_id") == "a1"


def test_get_all_preserves_user_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    mem = FakeAsyncMemory()
    client = _make_client(monkeypatch, mem)

    user_filters = {"AND": [{"category": "work"}]}

    async def _run() -> None:
        await client.get_all({"user_id": "u1", "filters": user_filters})

    asyncio.run(_run())

    kwargs = mem.calls[0]
    filters = kwargs.get("filters", {})
    assert filters.get("user_id") == "u1"
    assert filters.get("AND") == [{"category": "work"}]
