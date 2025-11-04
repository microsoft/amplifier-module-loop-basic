"""
Basic orchestrator with complete event emissions (desired state).
"""

import logging
from typing import Any
from typing import Optional

from amplifier_core import HookRegistry
from amplifier_core import HookResult
from amplifier_core import ModuleCoordinator
from amplifier_core.events import CONTENT_BLOCK_END
from amplifier_core.events import CONTENT_BLOCK_START
from amplifier_core.events import CONTEXT_POST_COMPACT
from amplifier_core.events import CONTEXT_PRE_COMPACT
from amplifier_core.events import PLAN_END
from amplifier_core.events import PLAN_START
from amplifier_core.events import PROMPT_COMPLETE
from amplifier_core.events import PROMPT_SUBMIT
from amplifier_core.events import PROVIDER_ERROR
from amplifier_core.events import PROVIDER_REQUEST
from amplifier_core.events import PROVIDER_RESPONSE
from amplifier_core.events import TOOL_ERROR
from amplifier_core.events import TOOL_POST
from amplifier_core.events import TOOL_PRE
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    config = config or {}
    orchestrator = BasicOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("Mounted BasicOrchestrator (desired-state)")
    return


class BasicOrchestrator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.max_iterations = int(config.get("max_iterations", 30))
        self.default_provider: str | None = config.get("default_provider")
        self.extended_thinking = config.get("extended_thinking", False)

    async def execute(
        self, prompt: str, context, providers: dict[str, Any], tools: dict[str, Any], hooks: HookRegistry
    ) -> str:
        await hooks.emit(PROMPT_SUBMIT, {"data": {"prompt": prompt}})

        # Add user message
        if hasattr(context, "add_message"):
            await context.add_message({"role": "user", "content": prompt})

        # Optionally compact before provider call
        if hasattr(context, "compact") and hasattr(context, "messages"):
            await hooks.emit(CONTEXT_PRE_COMPACT, {"data": {"messages": len(getattr(context, "messages", []))}})
            # simple heuristic: compact if more than 50 messages
            if len(getattr(context, "messages", [])) > 50:
                await context.compact()
            await hooks.emit(CONTEXT_POST_COMPACT, {"data": {"messages": len(getattr(context, "messages", []))}})

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

        while iteration < self.max_iterations:
            # Get messages from context
            message_dicts = getattr(context, "messages", [{"role": "user", "content": prompt}])

            # Convert to ChatRequest with Message objects
            try:
                messages_objects = [Message(**msg) for msg in message_dicts]
                chat_request = ChatRequest(messages=messages_objects)
                logger.debug(f"Created ChatRequest with {len(messages_objects)} messages")
                logger.debug(f"Message roles: {[m.role for m in chat_request.messages]}")
            except Exception as e:
                logger.error(f"Failed to create ChatRequest: {e}")
                logger.error(f"Message dicts: {message_dicts}")
                raise

            await hooks.emit(PROVIDER_REQUEST, {"data": {"provider": provider_name, "input_count": len(message_dicts)}})
            try:
                if hasattr(provider, "complete"):
                    # Pass tools and extended_thinking if configured
                    kwargs = {}
                    if tools:
                        kwargs["tools"] = list(tools.values())
                    # Pass extended_thinking if enabled in orchestrator config
                    if self.extended_thinking:
                        kwargs["extended_thinking"] = True
                    response = await provider.complete(chat_request, **kwargs)
                else:
                    raise RuntimeError(f"Provider {provider_name} missing 'complete'")

                usage = getattr(response, "usage", None)
                content = getattr(response, "content", None)
                tool_calls = getattr(response, "tool_calls", None)

                await hooks.emit(
                    PROVIDER_RESPONSE,
                    {"data": {"provider": provider_name, "usage": usage, "tool_calls": bool(tool_calls)}},
                )

                # Emit content block events if present
                content_blocks = getattr(response, "content_blocks", None)
                logger.info(
                    f"Response has content_blocks: {content_blocks is not None} - count: {len(content_blocks) if content_blocks else 0}"
                )
                if content_blocks:
                    logger.info(f"Emitting events for {len(content_blocks)} content blocks")
                    for idx, block in enumerate(content_blocks):
                        logger.info(f"Emitting CONTENT_BLOCK_START for block {idx}, type: {block.type.value}")
                        # Emit block start (without non-serializable raw object)
                        await hooks.emit(
                            CONTENT_BLOCK_START,
                            {
                                "data": {
                                    "block_type": block.type.value,
                                    "block_index": idx,
                                    # Don't include raw metadata as it may not be JSON serializable
                                }
                            },
                        )

                        # Emit block end with complete block
                        await hooks.emit(CONTENT_BLOCK_END, {"data": {"block_index": idx, "block": block.to_dict()}})

                # Handle tool calls (parallel execution)
                if tool_calls:
                    # Add assistant message with tool calls BEFORE executing them
                    if hasattr(context, "add_message"):
                        assistant_msg = {
                            "role": "assistant",
                            "content": content if content else "",
                            "tool_calls": [
                                {
                                    "id": getattr(tc, "id", None) or tc.get("id"),
                                    "tool": getattr(tc, "tool", None) or tc.get("tool"),
                                    "arguments": getattr(tc, "arguments", None) or tc.get("arguments") or {},
                                }
                                for tc in tool_calls
                            ],
                        }
                        await context.add_message(assistant_msg)

                    # Execute tools in parallel (user guidance: assume parallel intent when multiple tool calls)
                    import asyncio
                    import uuid

                    # Generate parallel group ID for event correlation
                    parallel_group_id = str(uuid.uuid4())

                    # Create tasks for parallel execution
                    async def execute_single_tool(tc, group_id: str):
                        """Execute one tool, handling all errors gracefully.

                        Always returns (tool_call_id, result_or_error) tuple.
                        Never raises - errors become error results.
                        """
                        tool_name = getattr(tc, "tool", None) or tc.get("tool")
                        tool_call_id = getattr(tc, "id", None) or tc.get("id")
                        args = getattr(tc, "arguments", None) or tc.get("arguments") or {}
                        tool = tools.get(tool_name)

                        try:
                            await hooks.emit(
                                TOOL_PRE,
                                {"data": {"tool": tool_name, "args": args, "parallel_group_id": group_id}},
                            )

                            if not tool:
                                error_msg = f"Error: Tool '{tool_name}' not found"
                                await hooks.emit(
                                    TOOL_ERROR,
                                    {
                                        "data": {
                                            "tool": tool_name,
                                            "error": {"type": "RuntimeError", "msg": error_msg},
                                            "parallel_group_id": group_id,
                                        }
                                    },
                                )
                                return (tool_call_id, error_msg)

                            result = await tool.execute(args)

                            # Serialize result for logging
                            result_data = result
                            if hasattr(result, "to_dict"):
                                result_data = result.to_dict()

                            await hooks.emit(
                                TOOL_POST,
                                {
                                    "data": {
                                        "tool": tool_name,
                                        "result": result_data,
                                        "parallel_group_id": group_id,
                                    }
                                },
                            )

                            # Return success with result content
                            result_content = str(
                                getattr(result, "data", None) or getattr(result, "text", None) or result
                            )
                            return (tool_call_id, result_content)

                        except Exception as te:
                            # Emit error event
                            await hooks.emit(
                                TOOL_ERROR,
                                {
                                    "data": {
                                        "tool": tool_name,
                                        "error": {"type": type(te).__name__, "msg": str(te)},
                                        "parallel_group_id": group_id,
                                    }
                                },
                            )

                            # Return failure with error message (don't raise!)
                            error_msg = f"Error executing tool: {str(te)}"
                            logger.error(f"Tool {tool_name} failed: {te}")
                            return (tool_call_id, error_msg)

                    # Execute all tools in parallel with asyncio.gather
                    # return_exceptions=False because we handle exceptions inside execute_single_tool
                    tool_results = await asyncio.gather(
                        *[execute_single_tool(tc, parallel_group_id) for tc in tool_calls]
                    )

                    # Add all tool results to context in original order (deterministic)
                    for tool_call_id, content in tool_results:
                        if hasattr(context, "add_message"):
                            await context.add_message(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": content,
                                }
                            )

                    # After executing tools, continue loop to get final response
                    iteration += 1
                    continue

                # If we have content (no tool calls), we're done
                if content:
                    final_content = content
                    if hasattr(context, "add_message"):
                        await context.add_message({"role": "assistant", "content": content})
                    break

                # No content and no tool calls - this shouldn't happen but handle it
                logger.warning("Provider returned neither content nor tool calls")
                iteration += 1

            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {"data": {"provider": provider_name, "error": {"type": type(e).__name__, "msg": str(e)}}},
                )
                raise

        # Check if we exceeded max iterations
        if iteration >= self.max_iterations and not final_content:
            logger.warning(f"Max iterations ({self.max_iterations}) reached without final response")

        await hooks.emit(
            PROMPT_COMPLETE,
            {"data": {"response_preview": (final_content or "")[:200], "length": len(final_content or "")}},
        )
        return final_content

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
