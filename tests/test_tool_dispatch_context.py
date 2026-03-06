"""Tests for _tool_dispatch_context set on coordinator during tool.execute().

Verifies that BasicOrchestrator sets coordinator._tool_dispatch_context
with the correct tool_call_id and parallel_group_id immediately before
calling tool.execute(), and clears it in a finally block afterward.

Covers:
- execute_single_tool (inner path): context set with tool_call_id and parallel_group_id
- context cleared after tool completes normally
- context cleared even when tool raises an exception
- Integration: full execute() path sets dispatch context during tool call
"""

import pytest
from amplifier_core import ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator


# ---------------------------------------------------------------------------
# Helpers (reuse pattern from test_lifecycle_events.py)
# ---------------------------------------------------------------------------


class MockCancellation:
    """Minimal cancellation token stub."""

    is_cancelled: bool = False
    is_immediate: bool = False

    def register_tool_start(self, tool_call_id: str, tool_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass


class MockCoordinator:
    """Minimal coordinator stub that supports _tool_dispatch_context."""

    def __init__(self, is_cancelled: bool = False, is_immediate: bool = False) -> None:
        self.cancellation = MockCancellation()
        self.cancellation.is_cancelled = is_cancelled
        self.cancellation.is_immediate = is_immediate
        self._contributions: dict = {}

    async def process_hook_result(self, result: object, event_name: str, source: str) -> object:
        return result

    def register_contributor(self, channel: str, name: str, callback: object) -> None:
        self._contributions[(channel, name)] = callback


# ---------------------------------------------------------------------------
# Integration tests via full execute() path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sets_tool_call_id_in_dispatch_context_during_tool_execution() -> None:
    """BasicOrchestrator sets tool_call_id in coordinator._tool_dispatch_context
    before calling tool.execute().
    """
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context during execution"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(
                            id="dispatch-call-id-001",
                            name="capture_tool",
                            arguments={"_": 1},
                        )
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"capture_tool": CapturingTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("tool_call_id") == "dispatch-call-id-001", (
        "coordinator._tool_dispatch_context must have tool_call_id set during tool.execute()"
    )


@pytest.mark.asyncio
async def test_execute_sets_parallel_group_id_in_dispatch_context() -> None:
    """BasicOrchestrator sets parallel_group_id in coordinator._tool_dispatch_context."""
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="group-test-call-id", name="capture_tool", arguments={"_": 1})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"capture_tool": CapturingTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    # parallel_group_id is a UUID generated per-batch — verify it's a non-empty string
    pgid = captured.get("parallel_group_id")
    assert isinstance(pgid, str) and pgid, (
        "_tool_dispatch_context must have a non-empty parallel_group_id string"
    )


@pytest.mark.asyncio
async def test_execute_clears_dispatch_context_after_tool_completes() -> None:
    """BasicOrchestrator clears coordinator._tool_dispatch_context after tool.execute()."""
    coordinator = MockCoordinator()

    class SimpleTool:
        name = "simple_tool"
        description = "Returns a successful result"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="clear-test-id", name="simple_tool", arguments={"_": 1})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"simple_tool": SimpleTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared to {} after tool execution completes"
    )


@pytest.mark.asyncio
async def test_execute_clears_dispatch_context_after_tool_raises() -> None:
    """BasicOrchestrator clears coordinator._tool_dispatch_context even when tool raises."""
    coordinator = MockCoordinator()

    class RaisingTool:
        name = "raising_tool"
        description = "Always raises an exception"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            raise ValueError("tool exploded")

    class ToolThenTextProvider:
        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="raise-test-id", name="raising_tool", arguments={"_": 1})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done despite error!")])

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    # BasicOrchestrator handles tool exceptions gracefully (no raise propagation)
    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"raising_tool": RaisingTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared even when tool.execute() raises"
    )


@pytest.mark.asyncio
async def test_execute_no_coordinator_does_not_set_dispatch_context() -> None:
    """Without a coordinator, execute() still runs tools normally (no dispatch context set)."""

    class SimpleTool:
        name = "simple_tool"
        description = "Returns a successful result"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="no-coord-id", name="simple_tool", arguments={"_": 1})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    # Should complete without error even when coordinator=None
    result = await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"simple_tool": SimpleTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
    )

    assert result is not None
