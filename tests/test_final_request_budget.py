"""Final assembled canonical request counting, not context-only fit claims."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import ClassVar

import pytest
from amplifier_core import HookResult
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ToolCall,
)
from amplifier_core.testing import MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator
from amplifier_module_loop_basic._media import MEDIA
from amplifier_module_loop_basic._request_budget import (
    RequestBudgetError,
    check_final_request,
)


def measured(estimate=7, limit=100):
    return {
        "estimated_input_tokens": estimate,
        "input_limit_tokens": limit,
        "context_token_budget": limit,
        "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": estimate},
    }


class Hooks:
    async def emit(self, event, data):
        if event == "provider:request":
            return HookResult(
                action="inject_context", context_injection="request ephemeral",
                ephemeral=True,
            )
        if event == "tool:post":
            return HookResult(
                data=data, action="inject_context", context_injection="tool ephemeral",
                ephemeral=True,
            )
        return HookResult()


class Tool:
    name = "fixture_tool"
    description = "Tool schema must be counted"
    input_schema: ClassVar[dict] = {"type": "object", "properties": {"long_schema": {"type": "string"}}}

    async def execute(self, args):
        from amplifier_core import ToolResult
        return ToolResult(output="tool text")


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [-1, 1])
async def test_final_assembled_request_and_options_counted_then_sent_same_object(max_iterations):
    class Provider:
        def __init__(self):
            self.counts, self.sent = [], []

        async def request_budget(self, request, *, request_options=None):
            self.counts.append((request, request.model_dump(), deepcopy(request_options)))
            assert request.tools[0].name == "fixture_tool"
            assert request.reasoning_effort == "high"
            assert request_options == {"extended_thinking": True}
            return measured()

        async def complete(self, request, **kwargs):
            assert self.counts, "final assembled request was never counted"
            counted, snapshot, options = self.counts[-1]
            assert counted is request
            assert request.model_dump() == snapshot
            assert options == kwargs
            self.sent.append(snapshot)
            return ChatResponse(
                content=[TextBlock(text="observe" if len(self.sent) == 1 else "done")],
                tool_calls=[ToolCall(id="call", name="fixture_tool", arguments={})]
                if len(self.sent) == 1 else None,
            )

    provider = Provider()
    # Models the context's earlier, incomplete messages-only count: it is not
    # the count the loop may rely upon after adding schemas/options/injections.
    class CountingContext(MockContextManager):
        async def _get_messages_for_request(self, token_budget=None, provider=None):
            from amplifier_core.message_models import ChatRequest, Message
            before = ChatRequest(messages=[Message(**m) for m in self.messages])
            provider.counts.append((before, before.model_dump(), {}))
            return deepcopy(self.messages)

    context = CountingContext()
    await BasicOrchestrator({
        "max_iterations": max_iterations, "reasoning_effort": "high", "extended_thinking": True,
    }).execute("test", context, {"fixture": provider}, {"fixture_tool": Tool()}, Hooks())
    assert len(provider.sent) == 2
    assert "request ephemeral" in str(provider.sent[0])
    if max_iterations == -1:
        assert "tool ephemeral" in str(provider.sent[1])
    else:
        assert "orchestrator-loop-limit" in str(provider.sent[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [-1, 1])
async def test_final_hard_budget_refusal_never_dispatches(max_iterations):
    class Provider:
        calls = 0

        async def request_budget(self, request, *, request_options=None):
            return measured(101, 100) if self.calls else measured()

        async def complete(self, request, **kwargs):
            self.calls += 1
            return ChatResponse(
                content=[TextBlock(text="observe" if self.calls == 1 else "uncounted")],
                tool_calls=[ToolCall(id="call", name="fixture_tool", arguments={})]
                if self.calls == 1 else None,
            )

    provider = Provider()
    with pytest.raises(RuntimeError, match="budget|limit"):
        await BasicOrchestrator({"max_iterations": max_iterations}).execute(
            "test", MockContextManager(), {"p": provider}, {"fixture_tool": Tool()}, Hooks(),
        )
    assert provider.calls == 1


def request(media=False):
    return ChatRequest(messages=[Message(
        role="user", content="fixture",
        metadata={MEDIA: {"refs": [{"fixture": True}]}} if media else None,
    )])


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [
    True, "7",
    {"estimated_input_tokens": True},
    {"estimated_input_tokens": -1},
    {"estimated_input_tokens": float("inf")},
    {"estimated_input_tokens": 1, "input_limit_tokens": 0},
    {**measured(), "measurement": {"kind": "provider_count", "source": "", "input_tokens": 1}},
    {**measured(), "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": True}},
    {**measured(), "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}},
])
async def test_malformed_final_budget_refuses(decision):
    provider = SimpleNamespace(request_budget=lambda req, *, request_options=None: decision)
    with pytest.raises(RequestBudgetError):
        await check_final_request(provider, request(), {})


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [None, {}, {"estimated_input_tokens": 3}])
async def test_legacy_text_unknown_or_estimate_is_not_falsely_measured(decision):
    provider = SimpleNamespace(request_budget=lambda req: decision)
    await check_final_request(provider, request(), {})
    await check_final_request(SimpleNamespace(), request(), {})


@pytest.mark.asyncio
async def test_strict_options_signature_no_kwargs_guess_and_no_retry():
    calls = []

    def kwargs_only(req, **kwargs):
        calls.append(kwargs)
        return measured()

    with pytest.raises(RequestBudgetError, match="effective request options"):
        await check_final_request(
            SimpleNamespace(request_budget=kwargs_only), request(), {"extended_thinking": True},
        )
    assert calls == []

    def strict(req, *, request_options):
        calls.append(request_options)
        raise TypeError("inside body")

    with pytest.raises(RequestBudgetError, match="operation refused"):
        await check_final_request(SimpleNamespace(request_budget=strict), request(), {})
    assert calls == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["request", "options"])
async def test_mutation_during_count_never_reaches_witness(change):
    canonical, options, witnesses = request(True), {"extended_thinking": True}, []

    async def count(req, *, request_options):
        if change == "request":
            req.messages.append(Message(role="user", content="mutated"))
        else:
            options["extended_thinking"] = False
        return measured()

    provider = SimpleNamespace(
        request_budget=count,
        media_budget_witness=lambda *a, **kw: witnesses.append(True),
    )
    with pytest.raises(RequestBudgetError, match="changed"):
        await check_final_request(provider, canonical, options)
    assert witnesses == []


@pytest.mark.asyncio
async def test_media_missing_count_fails_closed_but_warning_only_witness_can_proceed():
    canonical = request(True)
    with pytest.raises(RequestBudgetError, match="unavailable"):
        await check_final_request(SimpleNamespace(), canonical, {})
    seen = []

    async def fallback(req, *, request_options=None):
        assert req is canonical and request_options == {}
        seen.append(req)
        return {
            "v": 1, "validated": True, "projection_digest": "a" * 64,
            "messages": [{"role": "user", "content": "[Image unavailable; no pixels supplied]"}],
        }

    provider = SimpleNamespace(media_text_fallback_witness=fallback)
    await check_final_request(provider, canonical, {})
    assert seen == [canonical]
    assert canonical.messages[0].metadata[MEDIA]["refs"]


@pytest.mark.asyncio
async def test_measured_media_requires_same_candidate_witness():
    canonical = request(True)
    with pytest.raises(RequestBudgetError, match="projection witness"):
        await check_final_request(
            SimpleNamespace(request_budget=lambda req: measured()), canonical, {},
        )
    seen = []

    def witness(req, *, request_options=None):
        seen.append((req, request_options))
        return {"v": 1, "validated": True, "projection_digest": "b" * 64}

    await check_final_request(
        SimpleNamespace(request_budget=lambda req: measured(), media_budget_witness=witness),
        canonical, {},
    )
    assert seen == [(canonical, {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [-1, 0])
async def test_cancel_during_final_count_never_dispatches(max_iterations):
    callbacks, saved, sent = [], [], []
    cancellation = SimpleNamespace(
        is_cancelled=False, is_immediate=False, state="cancelled",
    )

    async def cleanup():
        callbacks.append(True)

    cancellation.trigger_callbacks = cleanup

    async def checkpoint():
        saved.append(True)
        return True

    class Coordinator:
        def __init__(self):
            self.cancellation = cancellation

        def get_capability(self, name):
            return checkpoint if name == "session.durable_checkpoint" else None

        async def process_hook_result(self, result, *args):
            return result

    class Provider:
        async def request_budget(self, req, *, request_options=None):
            cancellation.is_cancelled = True
            return measured()

        async def complete(self, req, **kwargs):
            sent.append(True)
            raise AssertionError("cancelled request sent")

    await BasicOrchestrator({"max_iterations": max_iterations}).execute(
        "test", MockContextManager(), {"p": Provider()}, {}, Hooks(), Coordinator(),
    )
    assert sent == [] and callbacks == [True] and saved == [True]


@pytest.mark.asyncio
async def test_task_cancelled_count_is_not_retried_or_dispatched():
    calls = []

    async def count(req, *, request_options=None):
        calls.append(req)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await check_final_request(SimpleNamespace(request_budget=count), request(), {})
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count", "measured-witness", "text-witness"])
async def test_forwarded_options_mutation_refuses(operation):
    canonical = request(operation != "count")

    def mutate(req, *, request_options=None):
        request_options["extended_thinking"] = False
        return measured()

    provider = SimpleNamespace(
        request_budget=mutate if operation == "count" else
        lambda req, *, request_options=None: measured() if operation == "measured-witness" else None,
        media_budget_witness=mutate,
        media_text_fallback_witness=mutate,
    )
    with pytest.raises(RequestBudgetError, match="forwarded options changed"):
        await check_final_request(provider, canonical, {"extended_thinking": True})


@pytest.mark.asyncio
async def test_context_target_is_not_a_second_hard_input_limit():
    # Real provider decisions may budget more than the local rebuild target.
    decision = measured(100000, 267904)
    decision["context_token_budget"] = 80000
    await check_final_request(
        SimpleNamespace(request_budget=lambda req: decision), request(), {},
    )
