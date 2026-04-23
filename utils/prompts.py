"""Prompt templates for Dify history -> Mem0 long-term memory extraction.

Mem0 local mode behavior notes:
- For infer-based extraction, Mem0 uses a single config-level ``custom_instructions``
  string (the legacy ``custom_fact_extraction_prompt`` / ``custom_update_memory_prompt``
  keys were removed in mem0 v2).
- The `prompt=` argument on `add()` is NOT applied to infer extraction.

Therefore, the extraction tool builds 3 separate Mem0 configs (semantic/episodic/procedural),
each with its own extraction prompt assigned to ``custom_instructions``.

Design principles (aligned with mem0 best practices):
1. Extract user-related facts from ALL messages (user + assistant)
2. Support multilingual extraction (detect and preserve conversation language)
3. Leverage mem0's built-in deduplication via infer=True

Note: For the FACT extraction prompts (semantic/episodic/procedural) the output format
(JSON shape, key names) and few-shot examples are owned by Mem0's built-in additive
extraction system prompt — our `custom_instructions` is appended as one section, not
substituted. So these templates only describe WHAT to extract.

This does NOT apply to `MEMORY_CLASSIFICATION_PROMPT`, which drives a separate LLM call
owned by this repo and must continue to define its own output schema.
"""

from __future__ import annotations

from datetime import datetime


def _get_current_date() -> str:
    """Get current date in YYYY-MM-DD format for temporal context."""
    return datetime.now().strftime("%Y-%m-%d")


def _common_rules() -> str:
    return f"""
General rules:
- Extract facts as DISCRETE, ATOMIC memory units (not long narrative paragraphs).
- Each fact should be a compact, self-contained unit (1-2 sentences, <= 160 characters).
- Split related information into separate addressable units for flexible retrieval and updates.
- Focus ONLY on user-related information (preferences, events, knowledge).
- If there is nothing worth remembering, return an empty list.
- Do not include system messages, meta instructions, or assistant's self-descriptions.
- Do not include secrets (API keys, passwords, tokens, private URLs).
- Today's date is {_get_current_date()}. Use this for temporal context when relevant.
- Detect the conversation language and record facts in the SAME language.
"""


SEMANTIC_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer specialized in
extracting SEMANTIC long-term memories.

[CRITICAL]: Extract facts about the USER from the conversation. Primary source is USER messages,
but you may also reference ASSISTANT messages for context, confirmation, or when assistant
summarizes/confirms user information. Always ensure extracted facts are ABOUT THE USER, not
about the assistant itself.

SEMANTIC memories are STABLE, ENDURING facts about the user:
- Personal Preferences: Likes, dislikes, consistent habits
- Profile Facts: Name, occupation, relationships, identity
- Long-term Goals: Career aspirations, life objectives
- Enduring Constraints: Dietary restrictions, accessibility needs, schedules
- Core Values: Principles, beliefs

Extract if the information:
- User ACTUALLY stated facts about themselves (not intentions or plans to share)
- Remains true across multiple conversations/contexts
- Describes inherent characteristics, not one-time states

EXCEPTION - User Adopting Assistant Suggestions:
If assistant suggests a preference/approach and user EXPLICITLY commits to adopt it, EXTRACT.
Vague acknowledgments ("okay", "thanks", "got it") do NOT count as commitment.

EXCLUDE: One-off events (->episodic), procedures (->procedural), temporary plans.

{_common_rules()}
Following is a conversation between the user and the assistant. Extract SEMANTIC long-term
memories about the user from the conversation.
"""


EPISODIC_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer specialized in
extracting EPISODIC long-term memories.

[CRITICAL]: Extract facts about the USER from the conversation. Primary source is USER messages,
but you may also reference ASSISTANT messages for context, confirmation, or when assistant
summarizes/confirms user experiences. Always ensure extracted facts are ABOUT THE USER, not
about the assistant itself.

EPISODIC memories are SPECIFIC, UNIQUE events with contextual details:
- Significant Events: Meetings, milestones, achievements (with WHO)
- Past Experiences: Travel, projects, interactions (with WHERE)
- Temporal Context: Must include specific timeframes (dates, "last week", "yesterday")
- Key Outcomes: Results, decisions made, lessons learned (with WHY)

Extract if the information:
- User ACTUALLY described events they experienced (not intentions to share)
- Happened at a particular time and place (5W context)
- Is a one-time occurrence, not a repeated pattern

EXCEPTION - User Adopting Assistant Suggestions:
If assistant suggests a future action/plan and user EXPLICITLY commits (e.g. "I'll join a gym
next month", "Starting tomorrow, I'll do X"), EXTRACT. Vague consideration ("I'll think about
it") does NOT count as commitment.

EXCLUDE: Stable traits (->semantic), procedures (->procedural), routine updates.

{_common_rules()}
Following is a conversation between the user and the assistant. Extract EPISODIC memories
(notable events and experiences) about the user from the conversation.
"""


PROCEDURAL_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer specialized in
extracting PROCEDURAL long-term memories.

[CRITICAL]: Extract facts about the USER from the conversation. Primary source is USER messages,
but you may also reference ASSISTANT messages for context, confirmation, or when assistant
summarizes/confirms user workflows. Always ensure extracted facts are ABOUT THE USER, not
about the assistant itself.

PROCEDURAL memories are REPEATABLE workflows and methods the user has DEMONSTRATED or
EXPLICITLY states they use regularly:
- Workflows: Multi-step processes the user follows REGULARLY
- Rules & Policies: Decision criteria the user CONSISTENTLY applies
- SOPs: How the user TYPICALLY handles recurring tasks
- Checklists: Systematic checks the user ROUTINELY performs

Extract if the user:
- Describes actual steps, methods, or approaches they follow
- States "I always/usually/regularly do X" with some description of what X is
- Shares actual procedural content (can be across multiple messages)

CRITICAL: Only extract based on USER's actual shared content, NOT intentions or plans:
- "I want to share my workflow" = Intent only, NO extraction (user hasn't shared yet)
- "My workflow is: step 1, step 2" = Actual content, EXTRACT
- "I always do X by doing Y" = Actual method, EXTRACT
- User asks questions = NO extraction (questions are not sharing)

EXCEPTION - User Adopting Assistant Suggestions:
If assistant suggests a workflow/method and user EXPLICITLY commits to adopt it (e.g.
"I'll use this method from now on", "I'll adopt this approach: [restates the method]"),
EXTRACT. Vague acknowledgments ("thanks, sounds good") do NOT count as commitment.

Be generous with ACTUAL content but strict with mere intentions. If user only expresses
desire to share but hasn't shared actual procedural content, return empty list.

EXCLUDE: Preferences (->semantic), one-time events (->episodic), vague intentions
without any procedural content.

{_common_rules()}
Following is a conversation between the user and the assistant. Extract PROCEDURAL memories
(reusable workflows, rules, procedures) about the user from the conversation.
"""


MEMORY_CLASSIFICATION_PROMPT = f"""You are a Memory Classification Expert.
Analyze the conversation and determine:
1. Which SINGLE memory type is MOST relevant
2. Whether the content is WORTH extracting

[CRITICAL]: Analyze ONLY USER messages, NOT assistant/system messages.

## Memory Types:

1. **SEMANTIC**: Stable, enduring facts about the user
   - Preferences, profile, goals, constraints, values
   - Remains true across multiple conversations
   - Describes inherent characteristics, not one-time states

2. **EPISODIC**: Specific, unique events with contextual details
   - Meetings, experiences, temporal context, outcomes
   - Happened at particular time/place (WHO, WHEN, WHERE)
   - One-time occurrences, not repeated patterns

3. **PROCEDURAL**: Repeatable workflows and methods user ACTUALLY DESCRIBED
   - User shares actual steps, processes, rules, or methods they use
   - User states "I always/usually/regularly do X" with description of what X is
   - User demonstrates their approach with concrete examples
   - EXCLUDE: Mere intentions ("I want to share") without actual content

## Extract if:
- User ACTUALLY shared concrete information (not just intentions)
- User explicitly commits to adopt assistant's suggestion (with clear language)
- Actionable/reusable content for future interactions
- Sufficient context to be meaningful

## Do NOT extract if:
- User only expresses INTENTION to share ("I want to share X") without actual content
- User gives vague acknowledgment ("thanks") without commitment
- Generic questions (not user's own information)
- Pure greetings/small talk
- Too vague or lacks actionable user-provided information

{_common_rules()}
Analyze the following conversation and return ONLY valid JSON in this exact format:
{{
  "memory_type": "SEMANTIC|EPISODIC|PROCEDURAL|NONE",
  "should_extract": true|false,
  "reason": "brief explanation"
}}

Output format:
- Return ONLY the JSON object. No markdown, no code fences, no extra text.
- Use double quotes for keys and string values; no trailing commas.

Conversation:
"""

