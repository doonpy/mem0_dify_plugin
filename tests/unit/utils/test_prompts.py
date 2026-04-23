from __future__ import annotations

from utils.prompts import (
    EPISODIC_FACT_EXTRACTION_PROMPT,
    MEMORY_CLASSIFICATION_PROMPT,
    PROCEDURAL_FACT_EXTRACTION_PROMPT,
    SEMANTIC_FACT_EXTRACTION_PROMPT,
    build_update_memory_prompt,
)


def test_fact_prompts_carry_repo_specific_semantics() -> None:
    # Output format (JSON shape, key names) is owned by Mem0's built-in additive
    # extraction system prompt. These assertions guard the repo-specific content
    # that Mem0 will NEVER supply: subtype scope, USER-vs-ASSISTANT guidance, the
    # adopt-assistant-suggestion exception, and the atomic/language invariants
    # from _common_rules().
    scope_markers = {
        SEMANTIC_FACT_EXTRACTION_PROMPT: "SEMANTIC long-term memories",
        EPISODIC_FACT_EXTRACTION_PROMPT: "EPISODIC memories",
        PROCEDURAL_FACT_EXTRACTION_PROMPT: "PROCEDURAL memories",
    }
    for prompt, marker in scope_markers.items():
        assert marker in prompt
        assert "DISCRETE, ATOMIC memory units" in prompt
        assert "SAME language" in prompt
        assert "ABOUT THE USER" in prompt
        assert "ASSISTANT messages" in prompt
        assert "User Adopting Assistant Suggestions" in prompt
        assert "secrets" in prompt


def test_classification_prompt_defines_its_own_schema() -> None:
    # This prompt is NOT sent through Mem0 — it drives a separate LLM call whose
    # output is parsed strictly by json.loads. Its schema and hardening must stay
    # in the prompt itself.
    prompt = MEMORY_CLASSIFICATION_PROMPT
    for key in ("memory_type", "should_extract", "reason"):
        assert f'"{key}"' in prompt
    assert "SEMANTIC|EPISODIC|PROCEDURAL|NONE" in prompt
    assert "Return ONLY the JSON object" in prompt
    assert "no trailing commas" in prompt


def test_update_prompt_requires_minimal_changes_only() -> None:
    prompt = build_update_memory_prompt(subtype="semantic")
    assert "Include ONLY items that require a change" in prompt
    assert "return { \"memory\": [] }" in prompt
    assert "Memory text must be a single line" in prompt

