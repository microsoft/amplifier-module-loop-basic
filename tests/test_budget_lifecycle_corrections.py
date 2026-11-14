"""Required provider estimate, pre-count cancel and ordered settlement guards."""

import asyncio
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
from amplifier_core import HookResult
from amplifier_core.cancellation import CancellationToken
from amplifier_core.message_models import ChatResponse, TextBlock
from amplifier_core.testing import MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator
from amplifier_module_loop_basic._media import MEDIA, checkpoint
from amplifier_module_loop_basic._request_budget import (
    RequestBudgetError,
    check_final_request,
)


class Hooks:
    async def emit(self, event, data):
        return HookResult(data=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [-1, 0])
@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "gemini"])
async def test_required_provider_context_estimate_is_message_scale(limit, provider_kind):
    counts, sends, estimates = [], [], []

    class Context(MockContextManager):
        def estimate_tokens(self, messages):
            estimates.append(deepcopy(messages))
            return 37

    class Provider:
        async def request_budget(self, request, *, context_estimate, request_options=None):
            assert context_estimate == 37
            assert request_options == {"extended_thinking": True}
            counts.append(request)
            return {
                "estimated_input_tokens": 100000, "input_limit_tokens": 267904,
                "context_token_budget": 80000,
                "measurement": {"kind": "provider_count", "source": provider_kind, "input_tokens": 100000},
            }

        async def complete(self, request, **kwargs):
            assert counts[-1] is request
            sends.append(request)
            return ChatResponse(content=[TextBlock(text="done")])

    context = Context()
    await BasicOrchestrator({"max_iterations": limit, "extended_thinking": True}).execute(
        "test", context, {"p": Provider()}, {}, Hooks(),
    )
    assert len(counts) == len(sends) == len(estimates) == 1
    if limit == 0:
        assert "orchestrator-loop-limit" in str(estimates[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [-1, 0], ids=["main", "final"])
@pytest.mark.parametrize("with_options", [True, False], ids=["options", "legacy"])
@pytest.mark.parametrize("estimator_kind", ["async-cancel", "sync-control"])
async def test_estimator_cancellation_fences_provider_budget(limit, with_options, estimator_kind):
    cancellation = CancellationToken()
    estimates, counts, sends, released, callbacks, saves, events = [], [], [], [], [], [], []
    options = {"extended_thinking": True} if with_options else {}

    async def cleanup():
        callbacks.append(True)

    cancellation.on_cancel(cleanup)

    def save():
        saves.append(deepcopy(context.messages))
        return True

    def estimate(messages):
        estimates.append(deepcopy(messages))
        return 37

    async def cancel_estimate(messages):
        await asyncio.sleep(0)
        cancellation.request_cancellation()
        assert cancellation.is_cancelled and asyncio.current_task().cancelling() == 0
        return estimate(messages)

    class Context(MockContextManager):
        estimate_tokens = staticmethod(
            cancel_estimate if estimator_kind == "async-cancel" else estimate
        )

    class Coordinator:
        def __init__(self):
            self.cancellation = cancellation

        def get_capability(self, name):
            return save if name == "session.durable_checkpoint" else None

        async def process_hook_result(self, result, *args):
            return result

    def count(request, context_estimate, request_options):
        assert context_estimate == 37 and request_options == options
        counts.append((request, request.model_dump(), deepcopy(request_options)))
        return {"estimated_input_tokens": 1, "input_limit_tokens": 100}

    def count_with_options(request, *, context_estimate, request_options):
        return count(request, context_estimate, request_options)

    def count_without_options(request, *, context_estimate):
        return count(request, context_estimate, {})

    class Provider:
        request_budget = staticmethod(count_with_options if with_options else count_without_options)

        def release_media_request(self, request, *, request_options):
            assert cancellation.is_cancelled and request_options == options
            released.append(request)
            return True

        def close(self):
            raise AssertionError("shared provider must not be closed")

        async def complete(self, request, **kwargs):
            assert not cancellation.is_cancelled
            counted, snapshot, counted_options = counts[-1]
            assert counted is request and snapshot == request.model_dump()
            assert kwargs == counted_options == options
            sends.append(request)
            return ChatResponse(content=[TextBlock(text="done")])

    class Observer:
        async def emit(self, event, data):
            events.append((event, data))
            return HookResult()

    context = Context()
    result = await BasicOrchestrator({
        "max_iterations": limit, "extended_thinking": with_options,
        "reasoning_effort": "high",
    }).execute(
        "test", context, {"p": Provider()},
        {"fixture": SimpleNamespace(name="fixture", description="schema", input_schema={"type": "object"})},
        Observer(), Coordinator(),
    )
    assert len(estimates) == 1 and asyncio.current_task().cancelling() == 0
    if limit == 0:
        assert "orchestrator-loop-limit" in str(estimates[0])
    event_counts = Counter(event for event, _ in events)
    completions = [data["status"] for event, data in events if event == "orchestrator:complete"]
    if estimator_kind == "async-cancel":
        assert counts == sends == []  # No provider budget may begin after token cancellation.
        assert len(released) == len(saves) == 1 and callbacks == [True]
        assert result == "" and completions == ["cancelled"]
        assert event_counts["cancel:requested"] == event_counts["cancel:completed"] == 1
        assert event_counts["provider:error"] == event_counts["prompt:complete"] == 0
        assert saves[0] == context.messages and context.messages[-1]["role"] == "assistant"
        assert "cancelled" in context.messages[-1]["content"]
        candidate = released[0]
    else:
        assert len(counts) == len(sends) == 1
        assert released == saves == callbacks == [] and result == "done"
        assert completions == ["success"] and event_counts["cancel:requested"] == 0
        candidate = sends[0]
    assert candidate.tools[0].name == "fixture" and candidate.reasoning_effort == "high"
    assert ("orchestrator-loop-limit" in str(candidate.model_dump())) == (limit == 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [-1, 0])
async def test_context_preparation_cancel_never_starts_final_count(limit):
    counts, sends = [], []
    cancellation = SimpleNamespace(is_cancelled=False, is_immediate=False, state="cancelled")

    async def callbacks():
        pass

    cancellation.trigger_callbacks = callbacks

    class Context(MockContextManager):
        async def _get_messages_for_request(self, **kwargs):
            cancellation.is_cancelled = True
            return deepcopy(self.messages)

    class Coordinator:
        def __init__(self):
            self.cancellation = cancellation

        def get_capability(self, name):
            return None

        async def process_hook_result(self, result, *args):
            return result

    class Provider:
        async def request_budget(self, request, *, request_options=None):
            counts.append(request)

        async def complete(self, request, **kwargs):
            sends.append(request)
            return ChatResponse(content=[TextBlock(text="wrong")])

    await BasicOrchestrator({"max_iterations": limit}).execute(
        "test", Context(), {"p": Provider()}, {}, Hooks(), Coordinator(),
    )
    assert counts == sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["reverse", "interleave", "move-batch"])
async def test_settled_media_rows_ordered_across_checkpoint(mutation):
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "A", "metadata": {MEDIA: {"refs": [{"fixture": "a"}]}}},
        {"role": "tool", "tool_call_id": "b", "content": "B", "metadata": {MEDIA: {"refs": [{"fixture": "b"}]}}},
    ]
    context = SimpleNamespace(messages=deepcopy(messages))

    def save():
        if mutation == "reverse":
            context.messages.reverse()
        elif mutation == "move-batch":
            context.messages.insert(0, {"role": "assistant", "content": "moved batch"})
        else:
            context.messages.insert(1, {"role": "assistant", "content": "moved boundary"})
        return True

    coordinator = SimpleNamespace(get_capability=lambda name: save)
    with pytest.raises(RuntimeError, match="Ordered settled media"):
        await checkpoint(coordinator, messages, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["overflow", "count-exception", "witness", "cancel"])
async def test_exact_candidate_released_on_local_refusal_not_shared_close(failure):
    import asyncio

    from amplifier_core.message_models import ChatRequest, Message

    request = ChatRequest(messages=[Message(
        role="user", content="fixture", metadata={MEDIA: {"refs": [{"fixture": True}]}},
    )])
    options, released = {"extended_thinking": True}, []

    class Provider:
        active = None

        async def request_budget(self, candidate, *, request_options=None):
            self.active = candidate
            if failure == "count-exception":
                raise TypeError("inside implementation")
            if failure == "cancel":
                raise asyncio.CancelledError()
            return {
                "estimated_input_tokens": 11, "input_limit_tokens": 10 if failure == "overflow" else 100,
                "context_token_budget": 80,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 11},
            }

        def release_media_request(self, candidate, *, request_options=None):
            assert candidate is request and request_options == options
            released.append(candidate)
            self.active = None
            return True

        def close(self):
            raise AssertionError("shared provider must not be closed")

    provider = Provider()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RequestBudgetError):
        await check_final_request(provider, request, options)
    assert provider.active is None and released == [request]


@pytest.mark.asyncio
async def test_local_message_estimate_strips_image_transport_not_schema_cost():
    from amplifier_core.message_models import ChatRequest, ImageBlock, Message

    canonical = ChatRequest(messages=[Message(
        role="user", content=[ImageBlock(source={"type": "base64", "data": "A" * 100000})],
    )])
    seen = []

    class Provider:
        def request_budget(self, candidate, *, context_estimate, request_options=None):
            seen.append(context_estimate)

        async def media_text_fallback_witness(self, candidate, *, request_options=None):
            return {
                "v": 1, "validated": True, "projection_digest": "a" * 64,
                "messages": [{"role": "user", "content": "Image unavailable; no pixels supplied"}],
            }

    await check_final_request(Provider(), canonical, {})
    assert 0 < seen[0] < 1000
    assert len(canonical.messages[0].content[0].source["data"]) == 100000


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [-1, 0])
async def test_release_failure_after_cancel_is_scrubbed_and_closure_runs(limit, caplog):
    sentinel = "SECRET_IMAGE_TRANSPORT_SENTINEL"
    cancellation = SimpleNamespace(is_cancelled=False, is_immediate=False, state="cancelled")
    callbacks, saves, released, events, sends = [], [], [], [], []

    async def cleanup():
        callbacks.append(True)

    cancellation.trigger_callbacks = cleanup

    def save():
        saves.append(True)
        return True

    class Coordinator:
        def __init__(self):
            self.cancellation = cancellation

        def get_capability(self, name):
            return save if name == "session.durable_checkpoint" else None

        async def process_hook_result(self, result, *args):
            return result

    class Provider:
        async def request_budget(self, candidate, *, request_options=None):
            cancellation.is_cancelled = True
            return {"estimated_input_tokens": 1, "input_limit_tokens": 100}

        def release_media_request(self, candidate, *, request_options=None):
            released.append(True)
            raise ValueError(sentinel)

        async def complete(self, request, **kwargs):
            sends.append(True)

    class Observer:
        async def emit(self, event, data):
            events.append((event, data))
            return HookResult()

    context = MockContextManager()
    await BasicOrchestrator({"max_iterations": limit}).execute(
        "test", context, {"p": Provider()}, {}, Observer(), Coordinator(),
    )
    assert sends == [] and released == callbacks == saves == [True]
    assert context.messages[-1]["role"] == "assistant"
    assert any(event == "cancel:completed" for event, _ in events)
    assert sentinel not in str(events) and sentinel not in caplog.text
