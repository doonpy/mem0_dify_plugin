"""Unit tests for the search score threshold feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.helpers import parse_score_threshold
from utils.mem0_client import AsyncMem0Client, _apply_score_threshold


def test_parse_score_threshold_empty_returns_default():
    assert parse_score_threshold(None, default=None) is None
    assert parse_score_threshold("", default=None) is None
    assert parse_score_threshold(None, default=0.5) == 0.5


def test_parse_score_threshold_accepts_numeric_and_string():
    assert parse_score_threshold(0.42) == 0.42
    assert parse_score_threshold("0.7") == 0.7
    assert parse_score_threshold("0.0") == 0.0
    assert parse_score_threshold("1.0") == 1.0


def test_parse_score_threshold_rejects_out_of_range():
    assert parse_score_threshold(-0.1, default=0.3) == 0.3
    assert parse_score_threshold(1.5, default=None) is None


def test_parse_score_threshold_rejects_garbage():
    assert parse_score_threshold("abc", default=0.2) == 0.2
    assert parse_score_threshold(object(), default=None) is None


def test_apply_score_threshold_drops_below_cutoff():
    results = [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.5},
        {"id": "c", "score": 0.1},
    ]
    assert _apply_score_threshold(results, 0.5) == [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.5},
    ]


def test_apply_score_threshold_none_is_passthrough():
    results = [{"id": "a", "score": 0.1}]
    assert _apply_score_threshold(results, None) == results


def test_apply_score_threshold_invalid_is_passthrough():
    results = [{"id": "a", "score": 0.1}]
    assert _apply_score_threshold(results, "bad") == results
    assert _apply_score_threshold(results, 1.5) == results


def test_apply_score_threshold_missing_score_treated_as_zero():
    results = [{"id": "a"}, {"id": "b", "score": 0.8}]
    assert _apply_score_threshold(results, 0.1) == [{"id": "b", "score": 0.8}]


@pytest.mark.asyncio
async def test_async_search_applies_threshold():
    """End-to-end: payload threshold filters out low-scoring results."""
    config = {
        "mem0_api_key": "",
        "mem0_base_url": "",
        "mem0_org_id": "",
        "mem0_project_id": "",
        "provider": "local",
        "local_llm_json_secret": {
            "provider": "ollama",
            "config": {"model": "qwen2.5:latest"},
        },
        "local_embedder_json_secret": {
            "provider": "ollama",
            "config": {"model": "nomic-embed-text:latest"},
        },
        "local_vector_db_json_secret": {
            "provider": "qdrant",
            "config": {"collection_name": "test_threshold"},
        },
        "version": "v1.1",
    }

    raw_results = {
        "results": [
            {"id": "1", "memory": "high", "score": 0.95},
            {"id": "2", "memory": "mid", "score": 0.55},
            {"id": "3", "memory": "low", "score": 0.15},
        ]
    }
    mock_memory = MagicMock()
    mock_memory.search = AsyncMock(return_value=raw_results)

    with patch("utils.mem0_client.AsyncMemory") as mock_cls:
        mock_cls.from_config = AsyncMock(return_value=mock_memory)

        client = AsyncMem0Client(config)
        try:
            payload = {
                "query": "q",
                "user_id": "u1",
                "limit": 5,
                "threshold": 0.5,
            }
            results = await client.search(payload)
            ids = [r["id"] for r in results]
            assert ids == ["1", "2"]
        finally:
            await client.aclose()
