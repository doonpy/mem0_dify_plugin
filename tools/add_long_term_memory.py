"""Dify tool for adding a single user/assistant turn as a classified long-term memory.

This tool mirrors ``add_memory``'s input shape but adds LLM-driven classification:
- The turn is classified into semantic / episodic / procedural (or skipped).
- Only the classified subtype's Mem0 client is built (one pool per invocation).
- Writes use ``memory_origin="explicit"`` — this is user-driven, not implicit extraction.
- Sync mode: classify + write in-request; return SUCCESS or SKIPPED.
- Async mode: enqueue a single coroutine (classify + write) on the background event
  loop and return ACCEPTED immediately (fire-and-forget, same as add_memory async).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from dify_plugin import Tool

from utils.config_builder import is_async_mode
from utils.constants import (
    ADD_SKIP_RESULT,
    MAX_CONCURRENT_MEMORY_OPERATIONS,
    MAX_PENDING_TASKS_MULTIPLIER,
    WRITE_OPERATION_TIMEOUT,
)
from utils.helpers import log_thread_info, parse_positive_int, parse_timeout
from utils.logger import get_logger
from utils.mem0_client import (
    AsyncMem0Client,
    SyncMem0Client,
    get_async_client,
    get_sync_client,
)
from utils.mem0_extraction import (
    AsyncMemoryClassificationManager,
    AsyncMemoryWriter,
    SyncMemoryClassificationManager,
    SyncMemoryWriter,
    build_memory_metadata,
    build_single_subtype_async_client,
    build_single_subtype_sync_client,
)
from utils.memory_tool_helpers import (
    init_request_context,
    validate_user_id,
    yield_error,
)
from utils.task_tracker import TaskTracker

if TYPE_CHECKING:
    from collections.abc import Generator

    from dify_plugin.entities.tool import ToolInvokeMessage

logger = get_logger(__name__)


class AddLongTermMemoryTool(Tool):
    """Tool to classify a user/assistant turn and write it as long-term memory."""

    def _build_messages(
        self,
        tool_parameters: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Build messages list from user and assistant text (mirrors add_memory)."""
        user_text = (tool_parameters.get("user") or "").strip()
        assistant_text = (tool_parameters.get("assistant") or "").strip()

        messages: list[dict[str, str]] = []
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text and assistant_text != user_text:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def _should_skip_empty(
        self,
        messages: list[dict[str, str]],
    ) -> bool:
        """True when there is no non-blank content to classify."""
        return not messages or not any(
            isinstance(m.get("content"), str) and m["content"].strip() for m in messages
        )

    def _execute_sync(  # noqa: PLR0913
        self,
        messages: list[dict[str, str]],
        user_id: str,
        agent_id: str | None,
        metadata_extra: dict[str, Any] | None,
        request_id: str,
        start_time: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """Classify and write synchronously."""
        base_client: SyncMem0Client = get_sync_client(self.runtime.credentials)
        classifier = SyncMemoryClassificationManager(base_client.memory)
        classified_type, should_extract = classifier.classify(
            messages=messages,
            context={"user_id": user_id, "request_id": request_id},
        )

        if classified_type is None or not should_extract:
            logger.info(
                "[req:%s] Long-term memory skipped "
                "(classified_type=%s, should_extract=%s, user_id=%s)",
                request_id,
                classified_type,
                should_extract,
                user_id,
            )
            payload: dict[str, Any] = {
                "status": "SKIPPED",
                "messages": messages,
                "reason": "not_classified_or_low_value",
                **ADD_SKIP_RESULT,
            }
            if classified_type is not None:
                payload["classified_type"] = classified_type
            yield self.create_json_message(payload)
            yield self.create_text_message(
                "Memory skipped: content not valuable for long-term storage."
            )
            log_thread_info(logger, request_id, "COMPLETED (SKIPPED)", start_time)
            return

        subtype_client: SyncMem0Client | None = None
        try:
            subtype_client = build_single_subtype_sync_client(
                self.runtime.credentials, classified_type
            )
            md: dict[str, Any] = dict(metadata_extra) if metadata_extra else {}
            md.update(
                build_memory_metadata(
                    subtype=classified_type,
                    memory_origin="explicit",
                )
            )

            writer = SyncMemoryWriter(subtype_client)
            result = writer.add_memory(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=md,
            )
            elapsed = time.time() - start_time
            logger.info(
                "[req:%s] Long-term memory added (subtype=%s, user_id=%s, duration=%.2fs)",
                request_id,
                classified_type,
                user_id,
                elapsed,
            )
            yield self.create_json_message(
                {
                    "status": "SUCCESS",
                    "messages": messages,
                    "classified_type": classified_type,
                    "results": result,
                }
            )
            yield self.create_text_message(
                f"Long-term memory added (subtype: {classified_type})."
            )
            log_thread_info(logger, request_id, "COMPLETED", start_time)
        finally:
            if subtype_client is not None:
                try:
                    subtype_client.close()
                except Exception:
                    logger.exception(
                        "[req:%s] Error closing subtype client", request_id
                    )

    def _execute_async(  # noqa: PLR0913
        self,
        messages: list[dict[str, str]],
        user_id: str,
        agent_id: str | None,
        metadata_extra: dict[str, Any] | None,
        timeout: float,
        request_id: str,
        start_time: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """Enqueue classify+write on the background loop and return immediately."""
        credentials = self.runtime.credentials
        base_client: AsyncMem0Client = get_async_client(credentials)

        # Pre-enqueue overload guard
        pending = TaskTracker.get_pending_tasks_count()
        max_ops = parse_positive_int(
            credentials.get("max_concurrent_memory_operations"),
            MAX_CONCURRENT_MEMORY_OPERATIONS,
            logger=logger,
            config_name="max_concurrent_memory_operations",
        )
        max_pending = max_ops * MAX_PENDING_TASKS_MULTIPLIER
        if pending > max_pending:
            logger.warning(
                "[req:%s] Long-term memory add skipped before enqueue: "
                "queue overloaded (pending=%d, max=%d, user_id=%s)",
                request_id,
                pending,
                max_pending,
                user_id,
            )
            yield self.create_json_message(
                {
                    "status": "OVERLOAD",
                    "messages": messages,
                    **ADD_SKIP_RESULT,
                }
            )
            yield self.create_text_message(
                "System overloaded, long-term memory add skipped (not enqueued)."
            )
            log_thread_info(
                logger, request_id, "COMPLETED (OVERLOAD SKIPPED)", start_time
            )
            return

        async def _bg_task() -> None:
            subtype_client: AsyncMem0Client | None = None
            try:
                await base_client.create()
                classifier = AsyncMemoryClassificationManager(base_client.memory)
                classified_type, should_extract = await classifier.classify(
                    messages=messages,
                    context={"user_id": user_id, "request_id": request_id},
                )
                if classified_type is None or not should_extract:
                    logger.info(
                        "[req:%s] Long-term memory (async) skipped "
                        "(classified_type=%s, should_extract=%s, user_id=%s)",
                        request_id,
                        classified_type,
                        should_extract,
                        user_id,
                    )
                    return

                subtype_client = await build_single_subtype_async_client(
                    credentials, classified_type
                )
                md: dict[str, Any] = dict(metadata_extra) if metadata_extra else {}
                md.update(
                    build_memory_metadata(
                        subtype=classified_type,
                        memory_origin="explicit",
                    )
                )

                writer = AsyncMemoryWriter(subtype_client)
                await writer.add_memory(
                    messages=messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    metadata=md,
                    timeout_s=int(timeout),
                )
                logger.info(
                    "[req:%s] Long-term memory (async) added "
                    "(subtype=%s, user_id=%s)",
                    request_id,
                    classified_type,
                    user_id,
                )
            except Exception:
                logger.exception(
                    "[req:%s] Background long-term memory task failed (user_id=%s)",
                    request_id,
                    user_id,
                )
            finally:
                if subtype_client is not None:
                    try:
                        await subtype_client.aclose()
                    except Exception:
                        logger.exception(
                            "[req:%s] Error closing subtype async client",
                            request_id,
                        )

        loop = base_client.ensure_bg_loop()
        future = asyncio.run_coroutine_threadsafe(_bg_task(), loop)
        TaskTracker.track_bg_task(
            future,
            f"add_long_term_memory(user_id={user_id}, req_id={request_id})",
        )

        yield self.create_json_message(
            {
                "status": "ACCEPTED",
                "messages": messages,
                "results": [
                    {
                        "id": "",
                        "memory": "",
                        "event": "ACCEPT",
                    }
                ],
                "task_info": {
                    "request_id": request_id,
                    "user_id": user_id,
                },
            }
        )
        yield self.create_text_message(
            "Long-term memory addition accepted and will be classified and "
            "written in the background."
        )
        log_thread_info(logger, request_id, "COMPLETED (ASYNC ACCEPTED)", start_time)

    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        request_id, start_time = init_request_context(tool_parameters)
        log_thread_info(logger, request_id, "STARTED", start_time)

        user_id = validate_user_id(tool_parameters)
        if not user_id:
            yield from yield_error(
                self, request_id, "user_id is required", "add long-term memory", {}
            )
            return

        messages = self._build_messages(tool_parameters)
        if self._should_skip_empty(messages):
            logger.debug(
                "[req:%s] Skipping long-term memory for empty messages (user_id=%s)",
                request_id,
                user_id,
            )
            yield self.create_json_message(
                {
                    "status": "SKIPPED",
                    "messages": messages,
                    "reason": "empty_messages",
                    **ADD_SKIP_RESULT,
                }
            )
            yield self.create_text_message(
                "Skipped long-term memory addition for empty messages."
            )
            log_thread_info(
                logger, request_id, "COMPLETED (EMPTY SKIPPED)", start_time
            )
            return

        agent_id = tool_parameters.get("agent_id") or None
        metadata_extra = tool_parameters.get("metadata") or None
        if metadata_extra is not None and not isinstance(metadata_extra, dict):
            metadata_extra = None

        try:
            async_mode = is_async_mode(self.runtime.credentials)
            mode_str = "async" if async_mode else "sync"
            timeout = parse_timeout(
                tool_parameters.get("timeout"),
                WRITE_OPERATION_TIMEOUT,
                logger,
                "add_long_term_memory",
            )

            logger.info(
                "[req:%s] Add long-term memory started (mode=%s, user_id=%s)",
                request_id,
                mode_str,
                user_id,
            )

            if async_mode:
                yield from self._execute_async(
                    messages,
                    user_id,
                    agent_id,
                    metadata_extra,
                    timeout,
                    request_id,
                    start_time,
                )
            else:
                yield from self._execute_sync(
                    messages,
                    user_id,
                    agent_id,
                    metadata_extra,
                    request_id,
                    start_time,
                )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(
                "[req:%s] Add long-term memory failed (user_id=%s, duration=%.2fs)",
                request_id,
                user_id,
                elapsed,
            )
            error_message = f"Error: {e!s}"
            yield from yield_error(
                self, request_id, error_message, "add long-term memory", {}
            )
            log_thread_info(logger, request_id, "COMPLETED (ERROR)", start_time)
