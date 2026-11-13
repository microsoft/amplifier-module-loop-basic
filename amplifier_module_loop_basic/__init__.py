"""
Basic orchestrator with complete event emissions (desired state).
"""

# Amplifier module metadata
__amplifier_module_type__ = "orchestrator"

import json
import logging
from copy import deepcopy
from typing import Any

from amplifier_core import HookRegistry, HookResult, ModuleCoordinator
from amplifier_core.events import (
    CANCEL_COMPLETED,
    CANCEL_REQUESTED,
    CONTENT_BLOCK_END,
    CONTENT_BLOCK_START,
    ORCHESTRATOR_COMPLETE,
    PROMPT_COMPLETE,
    PROMPT_SUBMIT,
    PROVIDER_ERROR,
    PROVIDER_REQUEST,
    PROVIDER_RESPONSE,
    TOOL_ERROR,
    TOOL_POST,
    TOOL_PRE,
)
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from ._media import (
    capability,
    checkpoint,
    dispatch_context,
    media_call,
    presentation,
    request_messages,
    settle,
    stage,
)

logger = logging.getLogger(__name__)


def _tool_arguments(call):
    # Empty arguments are valid on a Core ToolCall (which has no dict.get).
    return (call.get("arguments") if isinstance(call, dict)
            else getattr(call, "arguments", None)) or {}


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    config = config or {}
    orchestrator = BasicOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    media = capability(coordinator)
    if media is not None:
        await media_call(
            media.report, None, event="consumer-ready", outcome="ready",
            details={"contract": 1, "role": "loop"},
        )
    logger.info("Mounted BasicOrchestrator (desired-state)")
    return


class BasicOrchestrator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        # -1 means unlimited iterations (default)
        max_iter_config = config.get("max_iterations", -1)
        self.max_iterations = int(max_iter_config) if max_iter_config != -1 else -1
        self.default_provider: str | None = config.get("default_provider")
        self.extended_thinking = config.get("extended_thinking", False)
        # Store ephemeral injections from tool:post hooks for next iteration
        self._pending_ephemeral_injections: list[dict[str, Any]] = []
        # Track whether cancel:requested has been emitted for the current execution
        self._cancel_requested_emitted: bool = False

    async def execute(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> str:
        # Reset cancellation event tracking for this execution
        self._cancel_requested_emitted = False
        media = capability(coordinator)

        # Emit and process prompt submit (allows hooks to inject context on session start)
        result = await hooks.emit(PROMPT_SUBMIT, {"prompt": prompt})
        if coordinator:
            result = await coordinator.process_hook_result(
                result, "prompt:submit", "orchestrator"
            )
            if result.action == "deny":
                return f"Operation denied: {result.reason}"

        # Add user message
        if hasattr(context, "add_message"):
            await context.add_message({"role": "user", "content": prompt})

        # Select provider based on priority
        provider = self._select_provider(providers)
        if not provider:
            raise RuntimeError("No provider available")
        provider_name = None
        for name, prov in providers.items():
            if prov is provider:
                provider_name = name
                break

        # Agentic loop: continue until we get a text response (no tool calls)
        iteration = 0
        final_content = ""

        while self.max_iterations == -1 or iteration < self.max_iterations:
            # Check for cancellation at iteration start
            if coordinator and coordinator.cancellation.is_cancelled:
                # Emit cancel:requested on first detection and trigger cleanup callbacks
                if not self._cancel_requested_emitted:
                    self._cancel_requested_emitted = True
                    await hooks.emit(
                        CANCEL_REQUESTED,
                        {
                            "orchestrator": "loop-basic",
                            "state": str(coordinator.cancellation.state),
                            "turn_count": iteration,
                        },
                    )
                    try:
                        await coordinator.cancellation.trigger_callbacks()
                    except Exception as e:
                        logger.warning(f"Error in cancellation callbacks: {e}")
                # Emit cancel:completed — orchestrator is exiting due to cancellation
                await hooks.emit(
                    CANCEL_COMPLETED,
                    {
                        "orchestrator": "loop-basic",
                        "was_immediate": coordinator.cancellation.is_immediate,
                        "turn_count": iteration,
                    },
                )
                # Emit orchestrator complete with cancelled status
                await hooks.emit(
                    ORCHESTRATOR_COMPLETE,
                    {
                        "orchestrator": "loop-basic",
                        "turn_count": iteration,
                        "status": "cancelled",
                    },
                )
                return final_content

            # Emit provider request BEFORE getting messages (allows hook injections)
            result = await hooks.emit(
                PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration}
            )
            if coordinator:
                result = await coordinator.process_hook_result(
                    result, "provider:request", "orchestrator"
                )
                if result.action == "deny":
                    return f"Operation denied: {result.reason}"

            # Get messages for LLM request (context handles compaction internally)
            if self._cancelled(coordinator):
                await self._finish_cancelled(
                    context, hooks, coordinator, iteration, [],
                )
                return final_content
            # Pass provider for dynamic budget calculation based on model's context window
            if hasattr(context, "get_messages_for_request"):
                message_dicts = await context.get_messages_for_request(
                    provider=provider
                )
            else:
                # Fallback for simple contexts without the method
                message_dicts = getattr(
                    context, "messages", [{"role": "user", "content": prompt}]
                )

            # Append ephemeral injection if present (temporary, not stored)
            if (
                result.action == "inject_context"
                and result.ephemeral
                and result.context_injection
            ):
                message_dicts = list(message_dicts)  # Copy to avoid modifying context

                # Check if we should append to last tool result
                if result.append_to_last_tool_result and len(message_dicts) > 0:
                    last_msg = message_dicts[-1]
                    # Append to last message if it's a tool result
                    if last_msg.get("role") == "tool":
                        # Append to existing content
                        original_content = last_msg.get("content", "")
                        message_dicts[-1] = {
                            **last_msg,
                            "content": f"{original_content}\n\n{result.context_injection}",
                        }
                        logger.debug(
                            "Appended ephemeral injection to last tool result message"
                        )
                    else:
                        # Fall back to new message if last message isn't a tool result
                        message_dicts.append(
                            {
                                "role": result.context_injection_role,
                                "content": result.context_injection,
                            }
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    message_dicts.append(
                        {
                            "role": result.context_injection_role,
                            "content": result.context_injection,
                        }
                    )

            # Apply pending ephemeral injections from tool:post hooks
            if self._pending_ephemeral_injections:
                message_dicts = list(message_dicts)  # Ensure we have a mutable list
                for injection in self._pending_ephemeral_injections:
                    if (
                        injection.get("append_to_last_tool_result")
                        and len(message_dicts) > 0
                    ):
                        last_msg = message_dicts[-1]
                        if last_msg.get("role") == "tool":
                            original_content = last_msg.get("content", "")
                            message_dicts[-1] = {
                                **last_msg,
                                "content": f"{original_content}\n\n{injection['content']}",
                            }
                            logger.debug(
                                "Applied pending ephemeral injection to last tool result"
                            )
                        else:
                            message_dicts.append(
                                {
                                    "role": injection["role"],
                                    "content": injection["content"],
                                }
                            )
                            logger.debug(
                                "Last message not a tool result, created new message for injection"
                            )
                    else:
                        message_dicts.append(
                            {"role": injection["role"], "content": injection["content"]}
                        )
                        logger.debug(
                            "Applied pending ephemeral injection as new message"
                        )
                # Clear pending injections after applying
                self._pending_ephemeral_injections = []

            # Convert to ChatRequest with Message objects
            try:
                messages_objects = [
                    Message(**msg) for msg in request_messages(message_dicts, media)
                ]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(
                            name=t.name,
                            description=t.description,
                            parameters=t.input_schema,
                        )
                        for t in tools.values()
                    ]

                chat_request = ChatRequest(
                    messages=messages_objects,
                    tools=tools_list,
                    reasoning_effort=self.config.get("reasoning_effort"),
                )
                logger.debug(
                    f"Created ChatRequest with {len(messages_objects)} messages"
                )
                logger.debug(
                    f"Message roles: {[m.role for m in chat_request.messages]}"
                )
            except Exception as e:
                logger.error(f"Failed to create ChatRequest: {e}")
                logger.error(f"Message dicts: {message_dicts}")
                raise
            try:
                if hasattr(provider, "complete"):
                    if self._cancelled(coordinator):
                        await self._finish_cancelled(
                            context, hooks, coordinator, iteration, [],
                        )
                        return final_content
                    # Pass extended_thinking if enabled in orchestrator config
                    kwargs = {}
                    if self.extended_thinking:
                        kwargs["extended_thinking"] = True
                    response = await provider.complete(chat_request, **kwargs)

                    # Check for immediate cancellation after provider returns
                    # This allows force-cancel to take effect as soon as the blocking
                    # provider call completes, before processing the response
                    if coordinator and coordinator.cancellation.is_immediate:
                        # Emit cancel:requested on first detection and trigger cleanup callbacks
                        if not self._cancel_requested_emitted:
                            self._cancel_requested_emitted = True
                            await hooks.emit(
                                CANCEL_REQUESTED,
                                {
                                    "orchestrator": "loop-basic",
                                    "state": str(coordinator.cancellation.state),
                                    "turn_count": iteration,
                                },
                            )
                            try:
                                await coordinator.cancellation.trigger_callbacks()
                            except Exception as e:
                                logger.warning(f"Error in cancellation callbacks: {e}")
                        # Emit cancel:completed — orchestrator is exiting due to cancellation
                        await hooks.emit(
                            CANCEL_COMPLETED,
                            {
                                "orchestrator": "loop-basic",
                                "was_immediate": coordinator.cancellation.is_immediate,
                                "turn_count": iteration,
                            },
                        )
                        # Emit cancelled status and exit
                        await hooks.emit(
                            ORCHESTRATOR_COMPLETE,
                            {
                                "orchestrator": "loop-basic",
                                "turn_count": iteration,
                                "status": "cancelled",
                            },
                        )
                        return final_content
                else:
                    raise RuntimeError(f"Provider {provider_name} missing 'complete'")

                usage = getattr(response, "usage", None)
                content = getattr(response, "content", None)
                tool_calls = getattr(response, "tool_calls", None)

                await hooks.emit(
                    PROVIDER_RESPONSE,
                    {
                        "provider": provider_name,
                        "usage": usage,
                        "tool_calls": bool(tool_calls),
                    },
                )

                # Emit content block events if present
                content_blocks = getattr(response, "content_blocks", None)
                logger.info(
                    f"Response has content_blocks: {content_blocks is not None} - count: {len(content_blocks) if content_blocks else 0}"
                )
                if content_blocks:
                    total_blocks = len(content_blocks)
                    logger.info(f"Emitting events for {total_blocks} content blocks")
                    for idx, block in enumerate(content_blocks):
                        logger.info(
                            f"Emitting CONTENT_BLOCK_START for block {idx}, type: {block.type.value}"
                        )
                        # Emit block start (without non-serializable raw object)
                        await hooks.emit(
                            CONTENT_BLOCK_START,
                            {
                                "block_type": block.type.value,
                                "block_index": idx,
                                "total_blocks": total_blocks,
                            },
                        )

                        # Emit block end with complete block, usage, and total count
                        event_data = {
                            "block_index": idx,
                            "total_blocks": total_blocks,
                            "block": block.to_dict(),
                        }
                        if usage:
                            event_data["usage"] = (
                                usage.model_dump()
                                if hasattr(usage, "model_dump")
                                else usage
                            )
                        await hooks.emit(CONTENT_BLOCK_END, event_data)

                # Handle tool calls (parallel execution)
                if tool_calls:
                    # Add assistant message with tool calls BEFORE executing them
                    if hasattr(context, "add_message"):
                        # Store structured content from response.content (our Pydantic models)
                        response_content = getattr(response, "content", None)
                        if response_content and isinstance(response_content, list):
                            assistant_msg = {
                                "role": "assistant",
                                "content": [
                                    block.model_dump()
                                    if hasattr(block, "model_dump")
                                    else block
                                    for block in response_content
                                ],
                                "tool_calls": [
                                    {
                                        "id": getattr(tc, "id", None) or tc.get("id"),
                                        "tool": getattr(tc, "name", None)
                                        or tc.get("tool"),
                                        "arguments": _tool_arguments(tc),
                                    }
                                    for tc in tool_calls
                                ],
                            }
                        else:
                            assistant_msg = {
                                "role": "assistant",
                                "content": content if content else "",
                                "tool_calls": [
                                    {
                                        "id": getattr(tc, "id", None) or tc.get("id"),
                                        "tool": getattr(tc, "name", None)
                                        or tc.get("tool"),
                                        "arguments": _tool_arguments(tc),
                                    }
                                    for tc in tool_calls
                                ],
                            }

                        # Preserve provider metadata (provider-agnostic passthrough)
                        # This enables providers to maintain state across steps (e.g., OpenAI reasoning items)
                        if hasattr(response, "metadata") and response.metadata:
                            assistant_msg["metadata"] = response.metadata

                        await context.add_message(assistant_msg)

                    # Execute tools in parallel (user guidance: assume parallel intent when multiple tool calls)
                    import asyncio
                    import uuid

                    # Generate parallel group ID for event correlation
                    parallel_group_id = str(uuid.uuid4())

                    # Create tasks for parallel execution
                    async def execute_single_tool(
                        tc: Any, group_id: str
                    ) -> dict[str, Any]:
                        """Execute one tool, handling all errors gracefully.

                        Returns a serializable, refs-only settled tool message.
                        Never raises - errors become error results.
                        """
                        tool_name = getattr(tc, "name", None) or tc.get("tool")
                        tool_call_id = getattr(tc, "id", None) or tc.get("id")
                        args = _tool_arguments(tc)
                        tool = tools.get(tool_name)

                        # Register tool with cancellation token for visibility
                        if coordinator:
                            # Build semantic display name for delegate calls
                            display_name = tool_name
                            if tool_name == "delegate":
                                try:
                                    _args = (
                                        args
                                        if isinstance(args, dict)
                                        else json.loads(args)
                                    )
                                    _agent = _args.get("agent", "")
                                    if _agent:
                                        display_name = _agent
                                except (
                                    json.JSONDecodeError,
                                    TypeError,
                                    AttributeError,
                                ):
                                    pass
                            coordinator.cancellation.register_tool_start(
                                tool_call_id, display_name
                            )

                        try:
                            try:
                                # Emit and process tool pre (allows hooks to block or request approval)
                                pre_result = await hooks.emit(
                                    TOOL_PRE,
                                    {
                                        "tool_name": tool_name,
                                        "tool_input": args,
                                        "parallel_group_id": group_id,
                                    },
                                )
                                if coordinator:
                                    pre_result = await coordinator.process_hook_result(
                                        pre_result, "tool:pre", tool_name
                                    )
                                    if pre_result.action == "deny":
                                        return {
                                            "role": "tool", "tool_call_id": tool_call_id,
                                            "content": f"Denied by hook: {pre_result.reason}",
                                        }

                                if not tool:
                                    error_msg = f"Error: Tool '{tool_name}' not found"
                                    await hooks.emit(
                                        TOOL_ERROR,
                                        {
                                            "tool_name": tool_name,
                                            "error": {
                                                "type": "RuntimeError",
                                                "msg": error_msg,
                                            },
                                            "parallel_group_id": group_id,
                                        },
                                    )
                                    return {"role": "tool", "tool_call_id": tool_call_id,
                                            "content": error_msg}

                                with dispatch_context(
                                    coordinator, asyncio.current_task(),
                                    tool_call_id, group_id,
                                ):
                                    result = await tool.execute(args)

                                # Stage canonical rich content before any model_dump
                                # or observer sees the result. No loop-side hydration.
                                origin = {
                                    "kind": "tool",
                                    "session_id": getattr(coordinator, "session_id", None)
                                    or getattr(media, "session_id", None),
                                    "id": tool_call_id,
                                }
                                result = await stage(result, media, origin)
                                result_data = presentation(result)
                                original_data = deepcopy(result_data)

                                # Emit and process tool post (allows hooks to inject feedback)
                                post_event = {
                                        "tool_name": tool_name,
                                        "tool_call_id": tool_call_id,
                                        "tool_input": args,
                                        "result": result_data,
                                        "parallel_group_id": group_id,
                                    }
                                post_result = await hooks.emit(
                                    TOOL_POST, post_event,
                                )
                                if coordinator:
                                    post_result = await coordinator.process_hook_result(
                                        post_result, "tool:post", tool_name
                                    )

                                # Store ephemeral injection from tool:post for next iteration
                                if (
                                    post_result.action == "inject_context"
                                    and post_result.ephemeral
                                    and post_result.context_injection
                                ):
                                    self._pending_ephemeral_injections.append(
                                        {
                                            "role": post_result.context_injection_role,
                                            "content": post_result.context_injection,
                                            "append_to_last_tool_result": post_result.append_to_last_tool_result,
                                        }
                                    )
                                    logger.debug(
                                        f"Stored ephemeral injection from tool:post ({tool_name}) for next iteration"
                                    )

                                return await settle(
                                    result, original_data, post_result, media,
                                    origin, tool_call_id, post_event.get("result"),
                                )

                            except Exception as te:
                                # Emit error event
                                await hooks.emit(
                                    TOOL_ERROR,
                                    {
                                        "tool_name": tool_name,
                                        "error": {
                                            "type": type(te).__name__,
                                            "msg": str(te),
                                        },
                                        "parallel_group_id": group_id,
                                    },
                                )

                                # Return failure with error message (don't raise!)
                                error_msg = f"Error executing tool: {str(te)}"
                                logger.error(f"Tool {tool_name} failed: {te}")
                                return {"role": "tool", "tool_call_id": tool_call_id,
                                        "content": error_msg}
                        finally:
                            # Unregister tool from cancellation token
                            if coordinator:
                                coordinator.cancellation.register_tool_complete(
                                    tool_call_id
                                )

                    # Execute all tools in parallel with asyncio.gather
                    # return_exceptions=False because we handle exceptions inside execute_single_tool
                    # Wrap in try/except for CancelledError to handle immediate cancellation
                    tasks = [
                        asyncio.create_task(execute_single_tool(tc, parallel_group_id))
                        for tc in tool_calls
                    ]
                    try:
                        tool_results = await asyncio.gather(*tasks)
                    except asyncio.CancelledError as cancelled:
                        # Completed siblings retain their actual settled evidence.
                        logger.info(
                            "Tool execution cancelled - synthesizing cancelled results"
                        )
                        await asyncio.gather(*tasks, return_exceptions=True)
                        tool_results = []
                        for tc, task in zip(tool_calls, tasks):
                            if task.done() and not task.cancelled() and task.exception() is None:
                                tool_results.append(task.result())
                            else:
                                tool_results.append({
                                    "role": "tool",
                                    "tool_call_id": getattr(tc, "id", None) or tc.get("id"),
                                    "content": json.dumps({
                                        "error": "Tool execution was cancelled by user",
                                        "cancelled": True,
                                    }),
                                })
                        try:
                            await self._finish_cancelled(
                                context, hooks, coordinator, iteration, tool_results,
                            )
                        except Exception as save_error:
                            raise cancelled from save_error
                        # Re-raise to let the cancellation propagate
                        raise

                    # Check for immediate cancellation (graceful path - tools completed)
                    if self._cancelled(coordinator):
                        await self._finish_cancelled(
                            context, hooks, coordinator, iteration, tool_results,
                        )
                        return final_content

                    # Add all tool results to context in original order (deterministic)
                    for message in tool_results:
                        if hasattr(context, "add_message"):
                            await context.add_message(message)
                    try:
                        await checkpoint(coordinator, tool_results)
                    except asyncio.CancelledError:
                        # The host owns reconciliation of an interrupted save.
                        # Preserve memory and cleanup, but never replay that write.
                        await self._finish_cancelled(
                            context, hooks, coordinator, iteration, [],
                            save=False,
                        )
                        raise
                    if self._cancelled(coordinator):
                        await self._finish_cancelled(
                            context, hooks, coordinator, iteration, [],
                        )
                        return final_content

                    # After executing tools, continue loop to get final response
                    iteration += 1
                    continue

                # If we have content (no tool calls), we're done
                if content:
                    # Extract text from content blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            block_type = getattr(block, "type", None)
                            type_value = (
                                getattr(block_type, "value", block_type)
                                if block_type
                                else None
                            )
                            if type_value == "text" and hasattr(block, "text"):
                                text_parts.append(block.text)
                            elif (
                                isinstance(block, dict)
                                and block.get("type", "text") == "text"
                                and "text" in block
                            ):
                                text_parts.append(block["text"])
                        final_content = "\n\n".join(text_parts) if text_parts else ""
                    else:
                        final_content = content
                    if hasattr(context, "add_message"):
                        # Store structured content from response.content (our Pydantic models)
                        response_content = getattr(response, "content", None)
                        if response_content and isinstance(response_content, list):
                            assistant_msg = {
                                "role": "assistant",
                                "content": [
                                    block.model_dump()
                                    if hasattr(block, "model_dump")
                                    else block
                                    for block in response_content
                                ],
                            }
                        else:
                            assistant_msg = {"role": "assistant", "content": content}
                        # Preserve provider metadata (provider-agnostic passthrough)
                        if hasattr(response, "metadata") and response.metadata:
                            assistant_msg["metadata"] = response.metadata
                        await context.add_message(assistant_msg)
                    break

                # No content and no tool calls - this shouldn't happen but handle it
                logger.warning("Provider returned neither content nor tool calls")
                iteration += 1

            except LLMError as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                        "retryable": e.retryable,
                        "status_code": e.status_code,
                    },
                )
                raise
            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                    },
                )
                raise

        # Check if we exceeded max iterations (only if not unlimited)
        if (
            self.max_iterations != -1
            and iteration >= self.max_iterations
            and not final_content
        ):
            if self._cancelled(coordinator):
                await self._finish_cancelled(
                    context, hooks, coordinator, iteration, [],
                )
                return final_content
            logger.warning(
                f"Max iterations ({self.max_iterations}) reached without final response"
            )

            # Inject system reminder to agent before final response
            await hooks.emit(
                PROVIDER_REQUEST,
                {
                    "provider": provider_name,
                    "iteration": iteration,
                    "max_reached": True,
                },
            )
            if coordinator:
                # Inject ephemeral reminder (not stored in context)
                await coordinator.process_hook_result(
                    HookResult(
                        action="inject_context",
                        context_injection="""<system-reminder>
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.
</system-reminder>""",
                        context_injection_role="system",
                        ephemeral=True,
                        suppress_output=True,
                    ),
                    "provider:request",
                    "orchestrator",
                )

            # Get one final response with the reminder (context handles compaction internally)
            if self._cancelled(coordinator):
                await self._finish_cancelled(
                    context, hooks, coordinator, iteration, [],
                )
                return final_content
            if hasattr(context, "get_messages_for_request"):
                message_dicts = await context.get_messages_for_request(
                    provider=provider
                )
            else:
                message_dicts = getattr(
                    context, "messages", [{"role": "user", "content": prompt}]
                )
            message_dicts = list(message_dicts)
            message_dicts.append(
                {
                    "role": "user",
                    "content": """<system-reminder source="orchestrator-loop-limit">
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.

DO NOT mention this iteration limit or reminder to the user explicitly. Simply wrap up naturally.
</system-reminder>""",
                }
            )

            try:
                messages_objects = [
                    Message(**msg) for msg in request_messages(message_dicts, media)
                ]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(
                            name=t.name,
                            description=t.description,
                            parameters=t.input_schema,
                        )
                        for t in tools.values()
                    ]

                chat_request = ChatRequest(
                    messages=messages_objects,
                    tools=tools_list,
                    reasoning_effort=self.config.get("reasoning_effort"),
                )

                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True

                if self._cancelled(coordinator):
                    await self._finish_cancelled(
                        context, hooks, coordinator, iteration, [],
                    )
                    return final_content
                response = await provider.complete(chat_request, **kwargs)
                content = getattr(response, "content", None)
                content_blocks = getattr(response, "content_blocks", None)

                if content:
                    final_content = content
                    if hasattr(context, "add_message"):
                        # Store structured content from response.content (our Pydantic models)
                        response_content = getattr(response, "content", None)
                        if response_content and isinstance(response_content, list):
                            assistant_msg = {
                                "role": "assistant",
                                "content": [
                                    block.model_dump()
                                    if hasattr(block, "model_dump")
                                    else block
                                    for block in response_content
                                ],
                            }
                        else:
                            assistant_msg = {"role": "assistant", "content": content}
                        # Preserve provider metadata (provider-agnostic passthrough)
                        if hasattr(response, "metadata") and response.metadata:
                            assistant_msg["metadata"] = response.metadata
                        await context.add_message(assistant_msg)

            except LLMError as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                        "retryable": e.retryable,
                        "status_code": e.status_code,
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")
            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")

        await hooks.emit(
            PROMPT_COMPLETE,
            {
                "response_preview": (final_content or "")[:200],
                "length": len(final_content or ""),
            },
        )

        # Emit orchestrator complete event
        await hooks.emit(
            ORCHESTRATOR_COMPLETE,
            {
                "orchestrator": "loop-basic",
                "turn_count": iteration,
                "status": "success" if final_content else "incomplete",
            },
        )

        return final_content

    @staticmethod
    def _cancelled(coordinator):
        return coordinator is not None and (
            coordinator.cancellation.is_cancelled
            or coordinator.cancellation.is_immediate
        )

    async def _finish_cancelled(
        self, context, hooks, coordinator, iteration, messages, *, save=True,
    ):
        """Save paired evidence and closure; persistence failure never skips cleanup."""
        try:
            if hasattr(context, "add_message"):
                for message in messages:
                    await context.add_message(message)
                await context.add_message({
                    "role": "assistant",
                    "content": (
                        "The previous operation was cancelled. "
                        "Results from completed tools have been preserved."
                    ),
                })
            if save:
                await checkpoint(coordinator, messages)
        finally:
            if coordinator and not self._cancel_requested_emitted:
                self._cancel_requested_emitted = True
                try:
                    await hooks.emit(CANCEL_REQUESTED, {
                        "orchestrator": "loop-basic",
                        "state": str(coordinator.cancellation.state),
                        "turn_count": iteration,
                    })
                except Exception:  # noqa: BLE001 - notification cannot mask save failure
                    logger.warning("Cancellation request notification failed")
                try:
                    await coordinator.cancellation.trigger_callbacks()
                except Exception:  # noqa: BLE001 - keep the primary save/cancel outcome
                    logger.warning("Cancellation cleanup callback failed")
            if coordinator:
                try:
                    await hooks.emit(CANCEL_COMPLETED, {
                        "orchestrator": "loop-basic",
                        "was_immediate": coordinator.cancellation.is_immediate,
                        "turn_count": iteration,
                    })
                except Exception:  # noqa: BLE001 - best-effort observation
                    logger.warning("Cancellation completion notification failed")
            try:
                await hooks.emit(ORCHESTRATOR_COMPLETE, {
                    "orchestrator": "loop-basic", "turn_count": iteration,
                    "status": "cancelled",
                })
            except Exception:  # noqa: BLE001 - best-effort observation
                logger.warning("Orchestrator cancellation notification failed")

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """Select a provider based on priority."""
        if not providers:
            return None

        # Collect providers with their priority (default priority is 100)
        provider_list = []
        for name, provider in providers.items():
            # Try to get priority from provider's config or attributes
            priority = 100  # Default priority
            if hasattr(provider, "priority"):
                priority = provider.priority
            elif hasattr(provider, "config") and isinstance(provider.config, dict):
                priority = provider.config.get("priority", 100)

            provider_list.append((priority, name, provider))

        # Sort by priority (lower number = higher priority)
        provider_list.sort(key=lambda x: x[0])

        # Return the highest priority provider
        if provider_list:
            return provider_list[0][2]

        return None
