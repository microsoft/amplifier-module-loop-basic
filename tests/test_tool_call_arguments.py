"""Tool-call argument regression tests through the normal, local agent loop."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from amplifier_core import ToolResult
from amplifier_core import events as core_events
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager
from pydantic import ValidationError

from amplifier_module_loop_basic import BasicOrchestrator

# Core exposes these event constants at runtime but omits them from its stubs.
events: Any = core_events
PROVIDER_ERROR = events.PROVIDER_ERROR
TOOL_ERROR = events.TOOL_ERROR

PUBLIC_PAYLOAD = (
    "def public_fixture():\n    return {'items': [1, 2], 'label': 'open'}\n"
)
FINAL_ANSWER = "The local tool completed."
CONTENTS = ["structured", "text", "empty-list", "none"]


def tool_response(content_kind, calls):
    content = {
        "structured": [TextBlock(text="Calling the local tool.")],
        "text": "Calling the local tool.",
        "empty-list": [],
        "none": None,
    }[content_kind]
    if content_kind in ("structured", "empty-list") and all(
        isinstance(call, ToolCall) for call in calls
    ):
        return ChatResponse(content=content, tool_calls=calls)
    # Preserve representations that ChatResponse would reject or coerce.
    return SimpleNamespace(content=content, tool_calls=calls)


class TwoCompletionProvider:
    name = "local-script"

    def __init__(self, first_response, final_answer=FINAL_ANSWER):
        self.responses = [
            first_response,
            ChatResponse(content=[TextBlock(text=final_answer)]),
        ]
        self.requests = []

    async def complete(self, request, **kwargs):
        assert len(self.requests) < 2, "Unexpected third completion"
        self.requests.append(deepcopy(request))
        return self.responses[len(self.requests) - 1]


class RecordingTool:
    name = "local_tool"
    description = "Return a fixed public fixture."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, output=PUBLIC_PAYLOAD):
        self.output = output
        self.arguments = []

    async def execute(self, arguments):
        self.arguments.append(arguments)
        return ToolResult(success=True, output=self.output)


async def run_loop(context, provider, tool, hooks=None):
    event_hooks: Any = hooks if hooks is not None else EventRecorder()
    return await BasicOrchestrator({}).execute(
        prompt="Run the local fixture.",
        context=context,
        providers={"default": provider},
        tools={tool.name: tool},
        hooks=event_hooks,
    )


def assert_second_request(provider, expected_calls, outputs, final=FINAL_ANSWER):
    """Check the actual request after dispatch, not just the stored context."""
    assert len(provider.requests) == 2
    messages = provider.requests[1].messages
    assistants = [
        message for message in messages if getattr(message, "tool_calls", None)
    ]
    assert [
        call for assistant in assistants for call in assistant.tool_calls
    ] == expected_calls
    results = [message for message in messages if message.role == "tool"]
    assert [(message.tool_call_id, message.content) for message in results] == [
        (call["id"], output)
        for call, output in zip(expected_calls, outputs, strict=True)
    ]
    assert provider.responses[-1].content[0].text == final


@pytest.mark.asyncio
@pytest.mark.parametrize("content_kind", CONTENTS)
@pytest.mark.parametrize(
    "arguments", [{}, {"items": [1, {"value": "open"}]}], ids=["empty", "nonempty"]
)
async def test_typed_arguments_reach_tool_and_second_request(content_kind, arguments):
    call = ToolCall(id="typed-call", name="local_tool", arguments=arguments)
    before = deepcopy(call.model_dump())
    provider = TwoCompletionProvider(tool_response(content_kind, [call]))
    context = MockContextManager()
    tool = RecordingTool()
    hooks = EventRecorder()

    assert await run_loop(context, provider, tool, hooks) == FINAL_ANSWER

    assert tool.arguments == [arguments]
    if arguments:
        assert tool.arguments[0] is call.arguments
    assert call.model_dump() == before
    assert_second_request(
        provider,
        [{"id": call.id, "tool": call.name, "arguments": arguments}],
        [PUBLIC_PAYLOAD],
    )
    assert context.messages[-1]["content"] == [
        TextBlock(text=FINAL_ANSWER).model_dump()
    ]
    assert not hooks.get_events(PROVIDER_ERROR)
    assert not hooks.get_events(TOOL_ERROR)


@pytest.mark.asyncio
@pytest.mark.parametrize("content_kind", CONTENTS)
@pytest.mark.parametrize("argument_kind", ["empty", "nonempty", "missing", "none"])
async def test_raw_dict_argument_compatibility(content_kind, argument_kind):
    call = {"id": "dict-call", "tool": "local_tool"}
    if argument_kind != "missing":
        call["arguments"] = {
            "empty": {},
            "nonempty": {"items": [1, {"value": "open"}]},
            "none": None,
        }[argument_kind]
    before = deepcopy(call)
    expected = call.get("arguments") or {}
    provider = TwoCompletionProvider(tool_response(content_kind, [call]))
    context = MockContextManager()
    tool = RecordingTool()

    assert await run_loop(context, provider, tool) == FINAL_ANSWER

    assert tool.arguments == [expected]
    if expected:
        assert tool.arguments[0] is call["arguments"]
    assert call == before
    assert_second_request(
        provider,
        [{"id": call["id"], "tool": call["tool"], "arguments": expected}],
        [PUBLIC_PAYLOAD],
    )
    assert context.messages[-1]["content"] == [
        TextBlock(text=FINAL_ANSWER).model_dump()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argument_fields", [{}, {"arguments": None}], ids=["missing", "none"]
)
async def test_legacy_attribute_arguments_default_to_empty(argument_fields):
    call = SimpleNamespace(id="legacy-call", name="local_tool", **argument_fields)
    before = deepcopy(vars(call))
    provider = TwoCompletionProvider(tool_response("text", [call]))
    tool = RecordingTool()

    assert await run_loop(MockContextManager(), provider, tool) == FINAL_ANSWER

    assert tool.arguments == [{}]
    assert vars(call) == before
    assert_second_request(
        provider,
        [{"id": call.id, "tool": call.name, "arguments": {}}],
        [PUBLIC_PAYLOAD],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{}, {"value": 1}], ids=["empty", "nonempty"])
async def test_dispatch_without_history_api(arguments):
    """The supported simple-context fallback isolates dispatch from history."""
    context = SimpleNamespace(messages=[{"role": "user", "content": "Local fixture"}])
    call = ToolCall(id="dispatch-call", name="local_tool", arguments=arguments)
    before = deepcopy(call.model_dump())
    provider = TwoCompletionProvider(tool_response("structured", [call]))
    tool = RecordingTool()
    hooks = EventRecorder()

    assert await run_loop(context, provider, tool, hooks) == FINAL_ANSWER

    assert tool.arguments == [arguments]
    assert call.model_dump() == before
    assert len(provider.requests) == 2
    assert len(hooks.get_events("tool:post")) == 1
    assert not hooks.get_events(TOOL_ERROR)
    assert not hasattr(context, "add_message")


@pytest.mark.parametrize(
    "argument_fields", [{}, {"arguments": None}], ids=["missing", "none"]
)
def test_core_still_rejects_invalid_arguments(argument_fields):
    with pytest.raises(ValidationError):
        ToolCall(id="invalid-call", name="local_tool", **argument_fields)


@pytest.mark.asyncio
async def test_mixed_parallel_calls_keep_provider_order_and_ids():
    first_finished = asyncio.Event()
    finished = []

    class OrderedTool(RecordingTool):
        async def execute(self, arguments):
            self.arguments.append(arguments)
            if not arguments:
                await asyncio.wait_for(first_finished.wait(), timeout=1)
                finished.append("typed-empty")
                return ToolResult(success=True, output=PUBLIC_PAYLOAD)
            finished.append("dict-nonempty")
            first_finished.set()
            return ToolResult(success=True, output="Second public result")

    calls = [
        ToolCall(id="typed-empty", name="local_tool", arguments={}),
        {"id": "dict-nonempty", "tool": "local_tool", "arguments": {"value": 2}},
    ]
    before = deepcopy([calls[0].model_dump(), calls[1]])
    provider = TwoCompletionProvider(tool_response("structured", calls))
    context = MockContextManager()
    tool = OrderedTool()

    assert await run_loop(context, provider, tool) == FINAL_ANSWER

    assert tool.arguments == [{}, {"value": 2}]
    assert finished == ["dict-nonempty", "typed-empty"]
    assert [calls[0].model_dump(), calls[1]] == before
    assert_second_request(
        provider,
        [
            {"id": "typed-empty", "tool": "local_tool", "arguments": {}},
            {"id": "dict-nonempty", "tool": "local_tool", "arguments": {"value": 2}},
        ],
        [PUBLIC_PAYLOAD, "Second public result"],
    )


@pytest.mark.asyncio
async def test_local_child_execution_and_transcript_continuation():
    """LOCAL_TRANSCRIPT_CONTINUATION, not installed CLI save/resume."""
    child_context = MockContextManager()
    child_call = ToolCall(id="child-call", name="local_tool", arguments={})
    child_provider = TwoCompletionProvider(
        tool_response("structured", [child_call]), final_answer=PUBLIC_PAYLOAD
    )
    child_tool = RecordingTool()

    class ChildTool(RecordingTool):
        async def execute(self, arguments):
            self.arguments.append(arguments)
            child_answer = await asyncio.wait_for(
                run_loop(child_context, child_provider, child_tool), timeout=2
            )
            return ToolResult(success=True, output=child_answer)

    root_context = MockContextManager()
    root_call = ToolCall(id="root-call", name="local_tool", arguments={})
    root_provider = TwoCompletionProvider(tool_response("text", [root_call]))
    root_tool = ChildTool()

    assert (
        await asyncio.wait_for(
            run_loop(root_context, root_provider, root_tool), timeout=3
        )
        == FINAL_ANSWER
    )
    assert root_tool.arguments == [{}]
    assert child_tool.arguments == [{}]
    for label, context, provider, call, final in [
        ("root", root_context, root_provider, root_call, FINAL_ANSWER),
        ("child", child_context, child_provider, child_call, PUBLIC_PAYLOAD),
    ]:
        expected_call = {"id": call.id, "tool": call.name, "arguments": {}}
        assert_second_request(provider, [expected_call], [PUBLIC_PAYLOAD], final)
        assert context.messages[-1]["content"] == [TextBlock(text=final).model_dump()]
        snapshot = deepcopy(context.messages)
        fresh_context = MockContextManager(json.loads(json.dumps(snapshot)))
        next_call = ToolCall(id=f"{label}-continued", name="local_tool", arguments={})
        fresh_provider = TwoCompletionProvider(tool_response("structured", [next_call]))
        fresh_tool = RecordingTool()

        assert await run_loop(fresh_context, fresh_provider, fresh_tool) == FINAL_ANSWER

        assert fresh_tool.arguments == [{}]
        assert fresh_context.messages[: len(snapshot)] == snapshot
        assert context.messages == snapshot
        assert_second_request(
            fresh_provider,
            [
                expected_call,
                {"id": next_call.id, "tool": next_call.name, "arguments": {}},
            ],
            [PUBLIC_PAYLOAD, PUBLIC_PAYLOAD],
        )
