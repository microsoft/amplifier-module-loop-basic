"""Tests for CP-6: execution lifecycle events, tool_call_id in tool events,
and observability.events registration.

Verifies that:
- execution:start fires with {prompt} after prompt:submit succeeds
- execution:end fires with {response, status="completed"} on normal exit
- execution:end fires with {response, status="cancelled"} on cancellation paths
- execution:end fires with {response, status="error"} on exception paths
- execution:start is NOT fired when prompt:submit returns "deny"
- tool_call_id appears in TOOL_PRE, TOOL_POST, and TOOL_ERROR event payloads
- mount() registers observability.events contributions
"""

import pytest

from amplifier_core import events as amp_events
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.models import HookResult
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator

# Import event constants via the module to avoid pyright false-positives on PyO3 re-exports
EXECUTION_START = amp_events.EXECUTION_START  # type: ignore[attr-defined]
EXECUTION_END = amp_events.EXECUTION_END  # type: ignore[attr-defined]
TOOL_PRE = amp_events.TOOL_PRE  # type: ignore[attr-defined]
TOOL_POST = amp_events.TOOL_POST  # type: ignore[attr-defined]
TOOL_ERROR = amp_events.TOOL_ERROR  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


class SimpleTextProvider:
    """Provider that always returns a plain text response."""

    async def complete(self, request, **kwargs):
        return ChatResponse(
            content=[TextBlock(text="Hello, world!")],
            tool_calls=None,
        )


class ToolThenTextProvider:
    """Provider returns one tool call, then a plain text response."""

    def __init__(self, tool_name="test_tool", tool_id="tc-abc123"):
        self._tool_name = tool_name
        self._tool_id = tool_id
        self._call_count = 0

    async def complete(self, request, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content=[TextBlock(text="Using tool")],
                tool_calls=[
                    ToolCall(id=self._tool_id, name=self._tool_name, arguments={"x": 1})
                ],
            )
        return ChatResponse(
            content=[TextBlock(text="Done!")],
            tool_calls=None,
        )


class ErrorProvider:
    """Provider that always raises an exception."""

    def __init__(self, error=None):
        self._error = error or RuntimeError("provider failed")

    async def complete(self, request, **kwargs):
        raise self._error


class SimpleTool:
    """Tool that returns a successful result."""

    def __init__(self, name="test_tool"):
        self.name = name
        self.description = "A test tool"
        self.input_schema = {"type": "object", "properties": {}}

    async def execute(self, args):
        from amplifier_core import ToolResult

        return ToolResult(success=True, output="tool result")


class MockCancellation:
    """Minimal cancellation token stub."""

    def __init__(self, is_cancelled=False, is_immediate=False):
        self.is_cancelled = is_cancelled
        self.is_immediate = is_immediate

    def register_tool_start(self, tool_call_id, tool_name):
        pass

    def register_tool_complete(self, tool_call_id):
        pass


class MockCoordinator:
    """Minimal coordinator stub for cancellation tests."""

    def __init__(self, is_cancelled=False, is_immediate=False):
        self.cancellation = MockCancellation(
            is_cancelled=is_cancelled, is_immediate=is_immediate
        )
        self._contributions: dict = {}

    async def process_hook_result(self, result, event_name, source):
        return result

    def register_contributor(self, channel, name, callback):
        self._contributions[(channel, name)] = callback


# ---------------------------------------------------------------------------
# execution:start and execution:end on normal completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_start_fires_with_prompt():
    """execution:start event is emitted with the prompt payload."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="Say hello",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
    )

    events = hooks.get_events(EXECUTION_START)
    assert len(events) == 1, f"Expected 1 execution:start, got {len(events)}"
    _, data = events[0]
    assert data["prompt"] == "Say hello"


@pytest.mark.asyncio
async def test_execution_end_fires_on_normal_completion():
    """execution:end fires with status='completed' after a normal response."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="Say hello",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
    )

    events = hooks.get_events(EXECUTION_END)
    assert len(events) == 1, f"Expected 1 execution:end, got {len(events)}"
    _, data = events[0]
    assert data["status"] == "completed"
    assert "response" in data


@pytest.mark.asyncio
async def test_execution_end_response_matches_return_value():
    """execution:end payload 'response' matches the value returned by execute()."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    return_val = await orchestrator.execute(
        prompt="Say hello",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
    )

    _, data = hooks.get_events(EXECUTION_END)[0]
    # The response in the event should equal the returned string
    assert data["response"] == return_val


# ---------------------------------------------------------------------------
# execution:start NOT fired when prompt:submit is denied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_start_not_fired_when_prompt_submit_denied():
    """execution:start must NOT fire when the coordinator denies prompt:submit."""

    class DenyingCoordinator(MockCoordinator):
        async def process_hook_result(self, result, event_name, source):
            if event_name == "prompt:submit":
                return HookResult(action="deny", reason="blocked by policy")
            return result

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    coordinator = DenyingCoordinator()

    result = await orchestrator.execute(
        prompt="blocked",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
        coordinator=coordinator,
    )

    assert "denied" in result.lower()
    assert len(hooks.get_events(EXECUTION_START)) == 0, (
        "execution:start should not fire when prompt:submit is denied"
    )
    assert len(hooks.get_events(EXECUTION_END)) == 0, (
        "execution:end should not fire when prompt:submit is denied (no execution began)"
    )


# ---------------------------------------------------------------------------
# execution:end on cancellation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_cancelled_at_loop_start():
    """execution:end fires with status='cancelled' when cancelled at loop start."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    coordinator = MockCoordinator(is_cancelled=True)

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
        coordinator=coordinator,
    )

    events = hooks.get_events(EXECUTION_END)
    assert len(events) == 1
    _, data = events[0]
    assert data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_execution_end_fires_cancelled_after_provider():
    """execution:end fires with status='cancelled' on immediate cancellation
    after the provider returns (is_immediate path)."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    # Cancellation that takes effect after first provider call
    class ImmediateCancellationCoordinator(MockCoordinator):
        def __init__(self):
            super().__init__()
            self.cancellation = MockCancellation(is_cancelled=False, is_immediate=True)

    coordinator = ImmediateCancellationCoordinator()

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
        coordinator=coordinator,
    )

    events = hooks.get_events(EXECUTION_END)
    assert len(events) == 1
    _, data = events[0]
    assert data["status"] == "cancelled"


# ---------------------------------------------------------------------------
# execution:end on error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_on_provider_exception():
    """execution:end fires with status='error' when provider raises."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    with pytest.raises(RuntimeError, match="provider failed"):
        await orchestrator.execute(
            prompt="test",
            context=context,
            providers={"default": ErrorProvider()},
            tools={},
            hooks=hooks,
        )

    events = hooks.get_events(EXECUTION_END)
    assert len(events) == 1
    _, data = events[0]
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_execution_end_fires_exactly_once():
    """execution:end is emitted exactly once per execute() call on the happy path."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": SimpleTextProvider()},
        tools={},
        hooks=hooks,
    )

    assert len(hooks.get_events(EXECUTION_START)) == 1
    assert len(hooks.get_events(EXECUTION_END)) == 1


# ---------------------------------------------------------------------------
# tool_call_id in TOOL_PRE and TOOL_POST payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_pre_includes_tool_call_id():
    """TOOL_PRE event payload must include tool_call_id."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    provider = ToolThenTextProvider(tool_id="my-call-id-123")
    tool = SimpleTool("test_tool")

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": provider},
        tools={"test_tool": tool},
        hooks=hooks,
    )

    pre_events = hooks.get_events(TOOL_PRE)
    assert len(pre_events) == 1
    _, data = pre_events[0]
    assert "tool_call_id" in data, "TOOL_PRE must include tool_call_id"
    assert data["tool_call_id"] == "my-call-id-123"


@pytest.mark.asyncio
async def test_tool_post_includes_tool_call_id():
    """TOOL_POST event payload must include tool_call_id."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    provider = ToolThenTextProvider(tool_id="post-call-id-456")
    tool = SimpleTool("test_tool")

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": provider},
        tools={"test_tool": tool},
        hooks=hooks,
    )

    post_events = hooks.get_events(TOOL_POST)
    assert len(post_events) == 1
    _, data = post_events[0]
    assert "tool_call_id" in data, "TOOL_POST must include tool_call_id"
    assert data["tool_call_id"] == "post-call-id-456"


@pytest.mark.asyncio
async def test_tool_error_not_found_includes_tool_call_id():
    """TOOL_ERROR (tool not found) event payload must include tool_call_id."""
    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    # Provider requests a tool that doesn't exist in tools dict
    provider = ToolThenTextProvider(
        tool_name="missing_tool", tool_id="error-call-id-789"
    )

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": provider},
        tools={},  # empty — 'missing_tool' won't be found
        hooks=hooks,
    )

    error_events = hooks.get_events(TOOL_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]
    assert "tool_call_id" in data, "TOOL_ERROR (not found) must include tool_call_id"
    assert data["tool_call_id"] == "error-call-id-789"


@pytest.mark.asyncio
async def test_tool_error_exception_includes_tool_call_id():
    """TOOL_ERROR (exception in tool.execute) event payload must include tool_call_id."""

    class FailingTool:
        name = "failing_tool"
        description = "fails"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            raise ValueError("tool exploded")

    orchestrator = BasicOrchestrator({})
    hooks = EventRecorder()
    context = MockContextManager()
    provider = ToolThenTextProvider(tool_name="failing_tool", tool_id="exc-call-id-000")

    await orchestrator.execute(
        prompt="test",
        context=context,
        providers={"default": provider},
        tools={"failing_tool": FailingTool()},
        hooks=hooks,
    )

    error_events = hooks.get_events(TOOL_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]
    assert "tool_call_id" in data, "TOOL_ERROR (exception) must include tool_call_id"
    assert data["tool_call_id"] == "exc-call-id-000"


# ---------------------------------------------------------------------------
# observability.events registration in mount()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_registers_observability_events():
    """mount() must call coordinator.register_contributor('observability.events', ...)."""
    from amplifier_module_loop_basic import mount

    contributions = {}

    class CapturingCoordinator:
        async def mount(self, role, instance):
            pass

        def register_contributor(self, channel, name, callback):
            contributions[(channel, name)] = callback

    coordinator = CapturingCoordinator()
    await mount(coordinator)

    assert ("observability.events", "loop-basic") in contributions, (
        "mount() must register 'loop-basic' as a contributor to 'observability.events'"
    )

    # Verify the callback returns the expected events
    callback = contributions[("observability.events", "loop-basic")]
    events_list = callback()
    assert "execution:start" in events_list
    assert "execution:end" in events_list
