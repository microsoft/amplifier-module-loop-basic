"""
Basic orchestrator with complete event emissions (desired state).
"""

import logging
from typing import Any
from typing import Optional

from amplifier_core import HookRegistry
from amplifier_core import HookResult
from amplifier_core import ModuleCoordinator
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

        # Pick provider
        provider_name = self.default_provider or (list(providers.keys())[0] if providers else None)
        if not provider_name:
            raise RuntimeError("No provider available")
        provider = providers[provider_name]

        messages = getattr(context, "messages", [{"role": "user", "content": prompt}])

        await hooks.emit(PROVIDER_REQUEST, {"data": {"provider": provider_name, "input_count": len(messages)}})
        try:
            if hasattr(provider, "complete"):
                response = await provider.complete(messages)
            else:
                raise RuntimeError(f"Provider {provider_name} missing 'complete'")

            usage = getattr(response, "usage", None)
            content = getattr(response, "content", None)
            tool_calls = getattr(response, "tool_calls", None)

            await hooks.emit(
                PROVIDER_RESPONSE, {"data": {"provider": provider_name, "usage": usage, "tool_calls": bool(tool_calls)}}
            )

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
