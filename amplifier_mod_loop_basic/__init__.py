"""
Basic agent loop orchestrator module.
Reference implementation following Claude Code's proven pattern.
"""

import logging
from typing import Any
from typing import Optional

from amplifier_core import HookRegistry
from amplifier_core import HookResult
from amplifier_core import ModuleCoordinator
from amplifier_core import ToolResult

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the basic orchestrator module.

    Args:
        coordinator: Module coordinator
        config: Optional configuration

    Returns:
        Optional cleanup function
    """
    config = config or {}
    orchestrator = BasicOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    logger.info("Mounted BasicOrchestrator")
    return


class BasicOrchestrator:
    """
    Reference implementation of the agent loop.
    Follows Claude Code's pattern: while(tool_calls) -> execute -> feed results -> repeat
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.max_iterations = config.get("max_iterations", 50)
        self.default_provider = config.get("default_provider")

    async def execute(
        self, prompt: str, context, providers: dict[str, Any], tools: dict[str, Any], hooks: HookRegistry
    ) -> str:
        """
        Execute the agent loop with given prompt.

        Args:
            prompt: User input prompt
            context: Context manager
            providers: Available providers
            tools: Available tools
            hooks: Hook registry

        Returns:
            Final response string
        """
        # Emit session start
        await hooks.emit("session:start", {"prompt": prompt})

        # Add user message to context
        await context.add_message({"role": "user", "content": prompt})

        # Select provider
        provider = self._select_provider(providers)
        if not provider:
            return "Error: No providers available"

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            # Get messages from context
            messages = await context.get_messages()

            # Get completion from provider
            try:
                # Pass tools to provider so LLM can use them
                response = await provider.complete(messages, tools=list(tools.values()))
            except Exception as e:
                logger.error(f"Provider error: {e}")
                final_response = f"Error getting response: {e}"
                break

            # Check for tool calls
            tool_calls = provider.parse_tool_calls(response)

            if not tool_calls:
                # No tool calls - we're done
                final_response = response.content
                await context.add_message({"role": "assistant", "content": final_response})
                break

            # Add assistant message with tool calls to context
            # This is required so Anthropic sees the tool_use blocks before tool_result blocks
            await context.add_message(
                {
                    "role": "assistant",
                    "content": response.content if response.content else "",
                    "tool_calls": [{"tool": tc.tool, "arguments": tc.arguments, "id": tc.id} for tc in tool_calls],
                }
            )

            # Execute tool calls
            for tool_call in tool_calls:
                # Get tool object first to pass to hook
                tool = tools.get(tool_call.tool)

                # Pre-tool hook (now includes tool object for metadata checking)
                hook_data = {"tool": tool_call.tool, "arguments": tool_call.arguments}
                if tool:
                    hook_data["tool_obj"] = tool
                hook_result = await hooks.emit("tool:pre", hook_data)

                if hook_result.action == "deny":
                    # Tool denied by hook - MUST add tool_result for API compliance
                    reason = hook_result.reason or "Tool execution denied"
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.tool,
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {reason}",
                        }
                    )
                    continue

                # Check if tool exists (we already got it earlier for the hook)
                if not tool:
                    # Tool not found - MUST add tool_result for API compliance
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.tool,
                            "tool_call_id": tool_call.id,
                            "content": f"Error: Tool {tool_call.tool} not found",
                        }
                    )
                    continue

                # Execute tool
                try:
                    result = await tool.execute(tool_call.arguments)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result = ToolResult(success=False, error={"message": str(e)})

                # Post-tool hook
                await hooks.emit(
                    "tool:post",
                    {
                        "tool": tool_call.tool,
                        "result": result.model_dump() if hasattr(result, "model_dump") else str(result),
                    },
                )

                # Add tool result to context
                await context.add_message(
                    {
                        "role": "tool",
                        "name": tool_call.tool,
                        "tool_call_id": tool_call.id,
                        "content": str(result.output) if result.success else f"Error: {result.error}",
                    }
                )

            # Check if we should compact context
            if await context.should_compact():
                await hooks.emit("context:pre-compact", {})
                await context.compact()

        # Emit session end
        await hooks.emit("session:end", {"response": final_response})

        return final_response

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """
        Select a provider to use.

        Args:
            providers: Available providers

        Returns:
            Selected provider or None
        """
        if not providers:
            return None

        # Use configured default if available
        if self.default_provider and self.default_provider in providers:
            return providers[self.default_provider]

        # Otherwise use first available
        return next(iter(providers.values()))
