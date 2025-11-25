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
from amplifier_core.events import ORCHESTRATOR_COMPLETE
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
from amplifier_core.message_models import ToolSpec

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
        # -1 means unlimited iterations (default)
        max_iter_config = config.get("max_iterations", -1)
        self.max_iterations = int(max_iter_config) if max_iter_config != -1 else -1
        self.default_provider: str | None = config.get("default_provider")
        self.extended_thinking = config.get("extended_thinking", False)

    async def execute(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> str:
        # Emit and process prompt submit (allows hooks to inject context on session start)
        result = await hooks.emit(PROMPT_SUBMIT, {"prompt": prompt})
        if coordinator:
            result = await coordinator.process_hook_result(result, "prompt:submit", "orchestrator")
            if result.action == "deny":
                return f"Operation denied: {result.reason}"

        # Add user message
        if hasattr(context, "add_message"):
            await context.add_message({"role": "user", "content": prompt})

        # Optionally compact before provider call
        if hasattr(context, "compact") and hasattr(context, "messages"):
            await hooks.emit(CONTEXT_PRE_COMPACT, {"messages": len(getattr(context, "messages", []))})
            # simple heuristic: compact if more than 50 messages
            if len(getattr(context, "messages", [])) > 50:
                await context.compact()
            await hooks.emit(CONTEXT_POST_COMPACT, {"messages": len(getattr(context, "messages", []))})

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
            # Emit provider request BEFORE getting messages (allows hook injections)
            result = await hooks.emit(PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration})
            if coordinator:
                result = await coordinator.process_hook_result(result, "provider:request", "orchestrator")
                if result.action == "deny":
                    return f"Operation denied: {result.reason}"

            # Get messages from context (includes permanent injections)
            message_dicts = getattr(context, "messages", [{"role": "user", "content": prompt}])

            # Append ephemeral injection if present (temporary, not stored)
            if result.action == "inject_context" and result.ephemeral and result.context_injection:
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
                        logger.debug("Appended ephemeral injection to last tool result message")
                    else:
                        # Fall back to new message if last message isn't a tool result
                        message_dicts.append(
                            {"role": result.context_injection_role, "content": result.context_injection}
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    message_dicts.append({"role": result.context_injection_role, "content": result.context_injection})

            # Convert to ChatRequest with Message objects
            try:
                messages_objects = [Message(**msg) for msg in message_dicts]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(name=t.name, description=t.description, parameters=t.input_schema)
                        for t in tools.values()
                    ]

                chat_request = ChatRequest(messages=messages_objects, tools=tools_list)
                logger.debug(f"Created ChatRequest with {len(messages_objects)} messages")
                logger.debug(f"Message roles: {[m.role for m in chat_request.messages]}")
            except Exception as e:
                logger.error(f"Failed to create ChatRequest: {e}")
                logger.error(f"Message dicts: {message_dicts}")
                raise
            try:
                if hasattr(provider, "complete"):
                    # Pass extended_thinking if enabled in orchestrator config
                    kwargs = {}
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
                    {"provider": provider_name, "usage": usage, "tool_calls": bool(tool_calls)},
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
                        logger.info(f"Emitting CONTENT_BLOCK_START for block {idx}, type: {block.type.value}")
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
                            event_data["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
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
                                    block.model_dump() if hasattr(block, "model_dump") else block
                                    for block in response_content
                                ],
                                "tool_calls": [
                                    {
                                        "id": getattr(tc, "id", None) or tc.get("id"),
                                        "tool": getattr(tc, "name", None) or tc.get("tool"),
                                        "arguments": getattr(tc, "arguments", None) or tc.get("arguments") or {},
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
                                        "tool": getattr(tc, "name", None) or tc.get("tool"),
                                        "arguments": getattr(tc, "arguments", None) or tc.get("arguments") or {},
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
                    async def execute_single_tool(tc: Any, group_id: str) -> tuple[str, str]:
                        """Execute one tool, handling all errors gracefully.

                        Always returns (tool_call_id, result_or_error) tuple.
                        Never raises - errors become error results.
                        """
                        tool_name = getattr(tc, "name", None) or tc.get("tool")
                        tool_call_id = getattr(tc, "id", None) or tc.get("id")
                        args = getattr(tc, "arguments", None) or tc.get("arguments") or {}
                        tool = tools.get(tool_name)

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
                                pre_result = await coordinator.process_hook_result(pre_result, "tool:pre", tool_name)
                                if pre_result.action == "deny":
                                    return (tool_call_id, f"Denied by hook: {pre_result.reason}")

                            if not tool:
                                error_msg = f"Error: Tool '{tool_name}' not found"
                                await hooks.emit(
                                    TOOL_ERROR,
                                    {
                                        "tool_name": tool_name,
                                        "error": {"type": "RuntimeError", "msg": error_msg},
                                        "parallel_group_id": group_id,
                                    },
                                )
                                return (tool_call_id, error_msg)

                            result = await tool.execute(args)

                            # Serialize result for logging
                            result_data = result
                            if hasattr(result, "to_dict"):
                                result_data = result.to_dict()

                            # Emit and process tool post (allows hooks to inject feedback)
                            post_result = await hooks.emit(
                                TOOL_POST,
                                {
                                    "tool_name": tool_name,
                                    "tool_input": args,
                                    "result": result_data,
                                    "parallel_group_id": group_id,
                                },
                            )
                            if coordinator:
                                await coordinator.process_hook_result(post_result, "tool:post", tool_name)

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
                                    "tool_name": tool_name,
                                    "error": {"type": type(te).__name__, "msg": str(te)},
                                    "parallel_group_id": group_id,
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
                    # Extract text from content blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if hasattr(block, "text"):
                                text_parts.append(block.text)
                            elif isinstance(block, dict) and "text" in block:
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
                                    block.model_dump() if hasattr(block, "model_dump") else block
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

            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {"provider": provider_name, "error": {"type": type(e).__name__, "msg": str(e)}},
                )
                raise

        # Check if we exceeded max iterations (only if not unlimited)
        if self.max_iterations != -1 and iteration >= self.max_iterations and not final_content:
            logger.warning(f"Max iterations ({self.max_iterations}) reached without final response")

            # Inject system reminder to agent before final response
            await hooks.emit(PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration, "max_reached": True})
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

            # Get one final response with the reminder
            message_dicts = getattr(context, "messages", [{"role": "user", "content": prompt}])
            message_dicts = list(message_dicts)
            message_dicts.append(
                {
                    "role": "system",
                    "content": """<system-reminder>
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.
</system-reminder>""",
                }
            )

            try:
                messages_objects = [Message(**msg) for msg in message_dicts]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(name=t.name, description=t.description, parameters=t.input_schema)
                        for t in tools.values()
                    ]

                chat_request = ChatRequest(messages=messages_objects, tools=tools_list)

                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True

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
                                    block.model_dump() if hasattr(block, "model_dump") else block
                                    for block in response_content
                                ],
                            }
                        else:
                            assistant_msg = {"role": "assistant", "content": content}
                        # Preserve provider metadata (provider-agnostic passthrough)
                        if hasattr(response, "metadata") and response.metadata:
                            assistant_msg["metadata"] = response.metadata
                        await context.add_message(assistant_msg)

            except Exception as e:
                logger.error(f"Error getting final response after max iterations: {e}")

        await hooks.emit(
            PROMPT_COMPLETE,
            {"response_preview": (final_content or "")[:200], "length": len(final_content or "")},
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
