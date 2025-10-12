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

        messages = getattr(context, "messages", [{"role": "user", "content": prompt}])

        await hooks.emit(PROVIDER_REQUEST, {"data": {"provider": provider_name, "input_count": len(messages)}})
        try:
            if hasattr(provider, "complete"):
                # Pass tools and extended_thinking if configured
                kwargs = {}
                if tools:
                    kwargs["tools"] = list(tools.values())
                # Pass extended_thinking if enabled in orchestrator config
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True
                response = await provider.complete(messages, **kwargs)
            else:
                raise RuntimeError(f"Provider {provider_name} missing 'complete'")

            usage = getattr(response, "usage", None)
            content = getattr(response, "content", None)
            tool_calls = getattr(response, "tool_calls", None)

            await hooks.emit(
                PROVIDER_RESPONSE, {"data": {"provider": provider_name, "usage": usage, "tool_calls": bool(tool_calls)}}
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

            # Handle tool calls (simple sequential)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = getattr(tc, "tool", None) or tc.get("tool")
                    args = getattr(tc, "arguments", None) or tc.get("arguments") or {}
                    tool = tools.get(tool_name)
                    await hooks.emit(TOOL_PRE, {"data": {"tool": tool_name, "args": args}})
                    try:
                        if not tool:
                            raise RuntimeError(f"Tool '{tool_name}' not found")
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
                                }
                            },
                        )
                        # Attach tool result to context
                        if hasattr(context, "add_message"):
                            await context.add_message(
                                {
                                    "role": "tool",
                                    "tool_call_id": args.get("id") if isinstance(args, dict) else None,
                                    "content": str(
                                        getattr(result, "data", None) or getattr(result, "text", None) or result
                                    ),
                                }
                            )
                    except Exception as te:
                        await hooks.emit(
                            TOOL_ERROR,
                            {"data": {"tool": tool_name, "error": {"type": type(te).__name__, "msg": str(te)}}},
                        )
                        raise

            # Add assistant message
            if content and hasattr(context, "add_message"):
                await context.add_message({"role": "assistant", "content": content})

            await hooks.emit(
                PROMPT_COMPLETE, {"data": {"response_preview": (content or "")[:200], "length": len(content or "")}}
            )
            return content or ""

        except Exception as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {"data": {"provider": provider_name, "error": {"type": type(e).__name__, "msg": str(e)}}},
            )
            raise

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
