"""Tests for CancelledError handler with dict-based tool_calls.

Regression test for unsafe tc.id / tc.name access at lines 536-537.
The CancelledError handler used bare attribute access on tool_call objects
that may be plain dicts. Every other access site (9 of them) uses the safe
dual-access pattern: getattr(tc, "id", None) or tc.get("id").
"""

import asyncio

import pytest

from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator


class DictToolCallProvider:
    """Provider that returns tool_calls as plain dicts (not ToolCall objects).

    Some providers return tool_calls as dicts rather than objects.
    The orchestrator explicitly accommodates this with a dual-access pattern.
    """

    name = "dict-provider"

    async def complete(self, request, **kwargs):
        return type(
            "Response",
            (),
            {
                "content": "Calling tool",
                "tool_calls": [
                    {"id": "tc1", "tool": "cancel_tool", "arguments": {}}
                ],
                "usage": None,
                "content_blocks": None,
                "metadata": None,
            },
        )()


class CancellingTool:
    """Tool that raises CancelledError to simulate immediate cancellation."""

    name = "cancel_tool"
    description = "tool that simulates cancellation"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, args):
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_cancelled_error_handler_with_dict_tool_calls():
    """CancelledError handler must not crash when tool_calls are plain dicts.

    Without the fix, line 536 (tc.id) raises:
        AttributeError: 'dict' object has no attribute 'id'

    With the fix, CancelledError propagates cleanly after synthesizing
    cancelled tool results into the context.
    """
    orchestrator = BasicOrchestrator({})
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": DictToolCallProvider()},
            tools={"cancel_tool": CancellingTool()},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_cancelled_error_synthesizes_messages_for_dict_tool_calls():
    """After fix, cancelled tool results are properly added to context.

    Verifies the synthesized cancellation messages contain the correct
    tool_call_id and tool name extracted via the safe dual-access pattern.
    """
    orchestrator = BasicOrchestrator({})
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": DictToolCallProvider()},
            tools={"cancel_tool": CancellingTool()},
            hooks=hooks,
        )

    # Find the synthesized cancellation message in context
    tool_messages = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1, "Expected at least one synthesized tool message"

    cancel_msg = tool_messages[-1]
    assert cancel_msg["tool_call_id"] == "tc1"
    assert "cancelled" in cancel_msg["content"]
    assert "cancel_tool" in cancel_msg["content"]
