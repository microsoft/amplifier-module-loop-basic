"""Tests for reasoning_effort on ChatRequest construction.

Verifies that the orchestrator sets reasoning_effort on ChatRequest
from config, and that extended_thinking kwargs are still passed for
backward compatibility.
"""

import pytest

from amplifier_core.message_models import ChatRequest, ChatResponse, TextBlock
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator


def _make_text_response(text="Mock response"):
    """Create a simple text ChatResponse."""
    return ChatResponse(content=[TextBlock(text=text)])


class MockProvider:
    """Mock provider that captures the ChatRequest it receives."""

    name = "mock"

    def __init__(self):
        self.last_request = None
        self.last_kwargs = None

    async def complete(self, request, **kwargs):
        self.last_request = request
        self.last_kwargs = kwargs
        return _make_text_response()


@pytest.mark.asyncio
async def test_reasoning_effort_high_on_chat_request():
    """Config reasoning_effort='high' flows to ChatRequest."""
    orchestrator = BasicOrchestrator({"reasoning_effort": "high"})
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request is not None
    assert isinstance(provider.last_request, ChatRequest)
    assert provider.last_request.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_reasoning_effort_low_on_chat_request():
    """Config reasoning_effort='low' flows to ChatRequest."""
    orchestrator = BasicOrchestrator({"reasoning_effort": "low"})
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort == "low"


@pytest.mark.asyncio
async def test_reasoning_effort_medium_on_chat_request():
    """Config reasoning_effort='medium' flows to ChatRequest."""
    orchestrator = BasicOrchestrator({"reasoning_effort": "medium"})
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_no_reasoning_effort_defaults_to_none():
    """Without reasoning_effort in config, ChatRequest has None."""
    orchestrator = BasicOrchestrator({})
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort is None


@pytest.mark.asyncio
async def test_extended_thinking_kwarg_still_passed():
    """extended_thinking kwarg is still passed for backward compat."""
    orchestrator = BasicOrchestrator({"extended_thinking": True})
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_kwargs.get("extended_thinking") is True


@pytest.mark.asyncio
async def test_reasoning_effort_and_extended_thinking_coexist():
    """Both reasoning_effort and extended_thinking can be set together."""
    orchestrator = BasicOrchestrator(
        {
            "reasoning_effort": "high",
            "extended_thinking": True,
        }
    )
    provider = MockProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort == "high"
    assert provider.last_kwargs.get("extended_thinking") is True


@pytest.mark.asyncio
async def test_reasoning_effort_on_max_iterations_fallback():
    """reasoning_effort is also set on the max-iterations fallback ChatRequest."""
    orchestrator = BasicOrchestrator(
        {
            "reasoning_effort": "high",
            "max_iterations": 1,
        }
    )
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolCallThenTextProvider:
        """First call returns tool_calls (consumes the iteration),
        fallback call returns text."""

        name = "mock"

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Return tool calls to consume the single iteration
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[
                        ToolCall(id="tc1", name="test_tool", arguments={"x": 1})
                    ],
                )
            # Second call is the max-iterations fallback - capture the request
            self.last_request = request
            self.last_kwargs = kwargs
            return _make_text_response("Final response")

    provider = ToolCallThenTextProvider()

    # Need a tool that the orchestrator can find
    class SimpleTool:
        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            from amplifier_core import ToolResult

            return ToolResult(success=True, output="done")

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    # The fallback path should have been triggered (call_count > 1)
    assert call_count == 2
    assert hasattr(provider, "last_request")
    assert provider.last_request.reasoning_effort == "high"
