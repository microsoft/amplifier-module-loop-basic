"""Regression tests for thinking block content leaking into text extraction.

Root cause:
    There are TWO content block model systems:
      - content_models.ThinkingContent  → has .text  (was the bug hazard)
      - message_models.ThinkingBlock    → has .thinking (already safe)

    The old execute() inline text extraction used ``hasattr(block, "text")``
    which allowed ThinkingContent objects through unchanged, leaking thinking
    text into final_content and ultimately into downstream parse_json calls
    (causing the "Cannot access 'task_id' on str, not dict" failure mode).

Fix:
    Use an explicit ``block.type == "text"`` guard so only text blocks are
    included, regardless of which model system the block comes from.

RED / GREEN verification:
    Run against the unfixed code to see test_thinking_content_does_not_leak
    FAIL (thinking text present in result).
    Run after the fix to see all tests PASS.

Cross-ecosystem:
    Same fix pattern as amplifier-module-loop-streaming PR #25 (df5c0e1).
"""

import pytest

from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orch():
    return BasicOrchestrator({})


def _make_fake_response(blocks):
    """Create a fake provider response with the given content blocks."""

    class FakeResponse:
        content = blocks
        tool_calls = None
        usage = None
        content_blocks = None
        metadata = None

    return FakeResponse()


class FakeProvider:
    """Mock provider that returns a pre-configured response."""

    name = "mock-thinking"

    def __init__(self, response):
        self._response = response

    async def complete(self, request, **kwargs):
        return self._response


# ---------------------------------------------------------------------------
# Primary regression test: ThinkingContent (content_models) must be filtered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_content_does_not_leak_into_final_content():
    """content_models.ThinkingContent has .text — must be filtered by type check.

    This is the primary regression test.  Before the fix, the
    hasattr(block, "text") guard would include ThinkingContent blocks because
    they *do* have a .text attribute, just with type="thinking".

    RED (before fix):  result contains "internal reasoning" — thinking text leaked.
    GREEN (after fix): result is ONLY "real response".
    """
    from amplifier_core.content_models import TextContent, ThinkingContent

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            ThinkingContent(text="internal reasoning"),
            TextContent(text="real response"),
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    # Thinking text must NOT appear in the final content
    assert "internal reasoning" not in result, (
        f"Thinking text leaked into final_content: {result!r}"
    )
    # Only the TextContent payload should be present
    assert "real response" in result, (
        f"Expected 'real response' in result but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Complementary test: ThinkingBlock (message_models) was already safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_block_does_not_leak_into_final_content():
    """message_models.ThinkingBlock has .thinking (not .text) — already safe.

    ThinkingBlock was not affected by the original bug (no .text attribute),
    but this test documents that it remains excluded after the type-check
    refactor.
    """
    from amplifier_core.message_models import TextBlock, ThinkingBlock

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            ThinkingBlock(thinking="internal reasoning", signature="sig"),
            TextBlock(text="real response"),
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert "internal reasoning" not in result, (
        f"ThinkingBlock content leaked into final_content: {result!r}"
    )
    assert "real response" in result, (
        f"Expected 'real response' in result but got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Smoke test: normal TextContent and TextBlock pass through correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_content_passes_through():
    """content_models.TextContent is included in final_content as expected."""
    from amplifier_core.content_models import TextContent

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([TextContent(text="hello from TextContent")])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "hello from TextContent", f"Unexpected result: {result!r}"


@pytest.mark.asyncio
async def test_text_block_passes_through():
    """message_models.TextBlock is included in final_content as expected."""
    from amplifier_core.message_models import TextBlock

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([TextBlock(text="hello from TextBlock")])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "hello from TextBlock", f"Unexpected result: {result!r}"


@pytest.mark.asyncio
async def test_dict_text_block_passes_through():
    """Dict blocks with type='text' pass through correctly."""
    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response([{"type": "text", "text": "dict text block"}])
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "dict text block", f"Unexpected result: {result!r}"


@pytest.mark.asyncio
async def test_dict_thinking_block_is_filtered():
    """Dict blocks with type='thinking' are filtered out (consistency fix)."""
    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    response = _make_fake_response(
        [
            {"type": "thinking", "text": "dict thinking block"},
            {"type": "text", "text": "dict real response"},
        ]
    )
    provider = FakeProvider(response)

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert "dict thinking block" not in result, (
        f"Dict thinking block leaked into final_content: {result!r}"
    )
    assert "dict real response" in result, (
        f"Expected 'dict real response' in result but got: {result!r}"
    )
