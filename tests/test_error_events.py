"""Tests for enriched PROVIDER_ERROR events with LLMError fields.

Verifies that:
- LLMError exceptions produce PROVIDER_ERROR events with retryable and status_code
- Generic exceptions produce PROVIDER_ERROR events without those fields
- Both paths still re-raise the exception
"""

import pytest

from amplifier_core.events import PROVIDER_ERROR
from amplifier_core.llm_errors import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderUnavailableError,
)
from amplifier_core.message_models import ChatResponse, TextBlock
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator


class ErrorProvider:
    """Provider that raises a configurable exception on complete()."""

    name = "error-provider"

    def __init__(self, error):
        self._error = error

    async def complete(self, request, **kwargs):
        raise self._error


@pytest.mark.asyncio
async def test_llm_error_enriches_provider_error_event():
    """LLMError populates retryable and status_code on PROVIDER_ERROR event."""
    error = RateLimitError(
        "Rate limit exceeded",
        provider="openai",
        status_code=429,
        retryable=True,
    )
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RateLimitError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    # Find the PROVIDER_ERROR event
    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["provider"] == "default"
    assert data["error"]["type"] == "RateLimitError"
    assert "Rate limit exceeded" in data["error"]["msg"]
    assert data["retryable"] is True
    assert data["status_code"] == 429


@pytest.mark.asyncio
async def test_auth_error_not_retryable():
    """AuthenticationError has retryable=False in event data."""
    error = AuthenticationError(
        "Invalid API key",
        provider="anthropic",
        status_code=401,
        retryable=False,
    )
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(AuthenticationError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["retryable"] is False
    assert data["status_code"] == 401
    assert data["error"]["type"] == "AuthenticationError"


@pytest.mark.asyncio
async def test_provider_unavailable_retryable():
    """ProviderUnavailableError has retryable=True (default) in event data."""
    error = ProviderUnavailableError(
        "Service unavailable",
        provider="gemini",
        status_code=503,
    )
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(ProviderUnavailableError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["retryable"] is True
    assert data["status_code"] == 503


@pytest.mark.asyncio
async def test_llm_error_with_none_status_code():
    """LLMError with no status_code still includes the field (as None)."""
    error = LLMError("Unknown error", provider="vllm", retryable=True)
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(LLMError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["retryable"] is True
    assert data["status_code"] is None
    assert data["error"]["type"] == "LLMError"


@pytest.mark.asyncio
async def test_generic_exception_no_retryable_field():
    """Generic Exception produces PROVIDER_ERROR without retryable/status_code."""
    error = RuntimeError("Something broke")
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RuntimeError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["provider"] == "default"
    assert data["error"]["type"] == "RuntimeError"
    assert "Something broke" in data["error"]["msg"]
    # Generic exceptions should NOT have retryable or status_code
    assert "retryable" not in data
    assert "status_code" not in data


@pytest.mark.asyncio
async def test_llm_error_still_reraises():
    """LLMError is re-raised after event emission (not swallowed)."""
    error = RateLimitError("Rate limit", provider="openai", status_code=429)
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RateLimitError, match="Rate limit"):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_generic_exception_still_reraises():
    """Generic Exception is re-raised after event emission (not swallowed)."""
    error = ValueError("Bad value")
    orchestrator = BasicOrchestrator({})
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(ValueError, match="Bad value"):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_max_iterations_fallback_llm_error_event():
    """LLMError in max-iterations fallback path emits enriched PROVIDER_ERROR."""
    orchestrator = BasicOrchestrator({"max_iterations": 1})
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolThenErrorProvider:
        """First call returns tool_calls, second call (fallback) raises LLMError."""

        name = "fallback-error"

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[
                        ToolCall(id="tc1", name="test_tool", arguments={"x": 1})
                    ],
                )
            # Fallback path raises LLMError
            raise ProviderUnavailableError(
                "Server down",
                provider="gemini",
                status_code=503,
            )

    class SimpleTool:
        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            from amplifier_core import ToolResult

            return ToolResult(success=True, output="done")

    # The fallback path catches exceptions without re-raising, so no pytest.raises
    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": ToolThenErrorProvider()},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    # The fallback path should have emitted a PROVIDER_ERROR with LLMError fields
    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) >= 1

    # Find the enriched event (with retryable field)
    enriched = [e for e in error_events if "retryable" in e[1]]
    assert len(enriched) == 1
    _, data = enriched[0]

    assert data["retryable"] is True
    assert data["status_code"] == 503
    assert data["error"]["type"] == "ProviderUnavailableError"


@pytest.mark.asyncio
async def test_max_iterations_fallback_generic_error_event():
    """Generic Exception in max-iterations fallback emits basic PROVIDER_ERROR."""
    orchestrator = BasicOrchestrator({"max_iterations": 1})
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolThenGenericErrorProvider:
        """First call returns tool_calls, second call raises generic error."""

        name = "fallback-generic"

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={"x": 1})],
                )
            raise RuntimeError("Unexpected failure")

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
        providers={"default": ToolThenGenericErrorProvider()},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) >= 1

    # Find the generic error event (without retryable field)
    generic = [e for e in error_events if "retryable" not in e[1]]
    assert len(generic) == 1
    _, data = generic[0]

    assert data["error"]["type"] == "RuntimeError"
    assert "retryable" not in data
    assert "status_code" not in data
