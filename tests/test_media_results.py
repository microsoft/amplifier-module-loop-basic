"""Offline structural media-v1 fixtures; no store, transport or vision qualification."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import ClassVar

import pytest
from amplifier_core import HookResult, ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator
from amplifier_module_loop_basic._media import (
    MEDIA,
    UNAVAILABLE,
    capability,
    dispatch_context,
)

PIXELS = "U0VFRF9QSVhFTF9TRU5USU5FTA=="


def ref(identity):
    return {
        "v": 1, "session_id": "test-session", "sha256": identity * 64,
        "mime": "image/png", "bytes": 19, "width": 1, "height": 1,
        "provenance": {"kind": "tool", "session_id": "test-session", "id": identity,
                       "created_at": "2026-01-01T00:00:00Z", "expires_at": None},
    }


A, B, C = ref("a"), ref("b"), ref("c")


def rich():
    return ToolResult(output="useful label", content=[
        {"type": "text", "text": "image description"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXELS}},
    ])


def referenced(refs):
    return ToolResult(output={"label": "useful label", MEDIA: {"refs": refs}})


class Media:
    version = 1
    session_id = "test-session"

    def __init__(self):
        self.ingested = []
        self.published = []
        self.staged = {}

    async def ingest(self, value, *, source_format, origin, policy):
        assert source_format == "core-tool-result"
        assert origin["session_id"] == self.session_id
        assert origin["kind"] == "tool"
        assert policy is None
        self.ingested.append(deepcopy(origin))
        refs = value.output.get(MEDIA, {}).get("refs", []) if isinstance(value.output, dict) else [A, B]
        self.staged[origin["id"]] = deepcopy(refs)
        return {"value": referenced(refs), "staged_refs": refs}

    async def publish(self, refs, *, origin, disposition):
        assert disposition == ("retain" if refs else "drop")
        assert all(item in self.staged[origin["id"]] for item in refs)
        self.published.append((deepcopy(refs), deepcopy(origin), disposition))
        return {"refs": deepcopy(refs), "publication": {"state": "published", "durability": "assets-only"}}

    def validate(self, value, **kwargs):
        return True

    def resolve(self, *args, **kwargs):
        raise AssertionError("loop must never resolve pixels")

    def prepare_request(self, *args, **kwargs):
        raise AssertionError("loop must never hydrate its canonical request")

    def report(self, *args, **kwargs):
        return {}


class Coordinator:
    def __init__(self, media=None, save=True, process=None):
        self.media = media
        self.save = save
        self.process = process
        self.checkpoints = 0
        self.cancellation = SimpleNamespace(
            is_cancelled=False, is_immediate=False, state="running",
            register_tool_start=lambda *a: None, register_tool_complete=lambda *a: None,
            trigger_callbacks=self.callbacks,
        )

    async def callbacks(self):
        pass

    async def checkpoint(self):
        self.checkpoints += 1
        if isinstance(self.save, Exception):
            raise self.save
        return self.save

    def get_capability(self, name):
        return {MEDIA: self.media, "session.durable_checkpoint": self.checkpoint if self.save != "absent" else None}.get(name)

    async def process_hook_result(self, result, event, name):
        return self.process(result, event) if self.process else result


class Tool:
    name = "observe"
    description = "synthetic observation"
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}}

    def __init__(self, result=None, execute=None):
        self.result = result if result is not None else rich()
        self.run = execute
        self.calls = 0

    async def execute(self, args):
        self.calls += 1
        return await self.run(args) if self.run else self.result


class Provider:
    def __init__(self, calls=None, coordinator=None):
        self.calls = calls or [ToolCall(id="call-a", name="observe", arguments={})]
        self.requests = []
        self.coordinator = coordinator

    async def complete(self, request, **kwargs):
        saved = request.model_dump()
        assert PIXELS not in json.dumps(saved)
        self.requests.append(saved)
        if len(self.requests) == 1:
            return ChatResponse(content=[TextBlock(text="observing")], tool_calls=self.calls)
        if self.coordinator:
            assert self.coordinator.checkpoints == 1
        return ChatResponse(content=[TextBlock(text="finished")])


class Hooks(EventRecorder):
    def __init__(self, change=None):
        super().__init__()
        self.change = change

    async def emit(self, event, data):
        assert PIXELS not in json.dumps(data, default=str)
        self.events.append((event, deepcopy(data)))
        if event == "tool:post" and self.change:
            return self.change(data)
        return HookResult()


async def run(*, change=None, media=True, save=True, max_iterations=-1, tool=None, process=None):
    cap = Media() if media is True else media
    coord = Coordinator(cap, save, process)
    provider = Provider(coordinator=coord)
    context = MockContextManager()
    hooks = Hooks(change)
    await BasicOrchestrator({"max_iterations": max_iterations}).execute(
        "test", context, {"fixture": provider}, {"observe": tool or Tool()}, hooks, coord,
    )
    return context, provider, cap, hooks, coord


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [-1, 1])
async def test_preobserver_staging_and_main_final_refs_only(max_iterations):
    context, provider, media, hooks, coord = await run(max_iterations=max_iterations)
    assert media.ingested and media.published[0][0] == [A, B]
    tool = next(m for m in context.messages if m["role"] == "tool")
    assert tool["metadata"][MEDIA]["refs"] == [A, B]
    assert provider.requests[1]["messages"][-1 if max_iterations == -1 else -2]["metadata"][MEDIA]["refs"] == [A, B]
    assert PIXELS not in json.dumps([context.messages, hooks.events], default=str)
    assert coord.checkpoints == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["null", "missing", "output-removed", "string", "list", "wrong-envelope", "deny", "inplace", "reorder", "copied"])
async def test_only_effective_structured_output_is_published(case):
    def change(data):
        if case == "null":
            return HookResult(action="modify", data={"result": None})
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "output-removed":
            return HookResult(action="modify", data={"result": {"success": True}})
        if case == "string":
            return HookResult(action="modify", data={"result": json.dumps(data["result"])})
        if case == "list":
            return HookResult(action="modify", data={"result": [data["result"]]})
        if case == "wrong-envelope":
            return HookResult(action="modify", data={"result": {"output": {MEDIA: {"refs": [A], "unknown": True}}}})
        if case == "deny":
            return HookResult(action="deny", reason="redacted")
        if case == "inplace":
            data["result"]["output"].pop(MEDIA)
            return HookResult()
        if case == "reorder":
            data["result"]["output"][MEDIA]["refs"] = [B, A, B]
            return HookResult(action="modify", data=data)
        return HookResult(action="modify", data=deepcopy(data))

    context, _, media, _, _ = await run(change=change)
    row = next(m for m in context.messages if m["role"] == "tool")
    expected = [B, A, B] if case == "reorder" else [A, B] if case == "copied" else []
    assert row.get("metadata", {}).get(MEDIA, {}).get("refs", []) == expected
    assert [refs for refs, _, _ in media.published if refs] == ([expected] if expected else [])
    if case in ("null", "missing"):
        assert row["content"] == "None"


@pytest.mark.asyncio
async def test_processed_hook_result_is_authoritative():
    def process(result, event):
        return HookResult(action="modify", data={"result": None}) if event == "tool:post" else result
    context, _, media, _, _ = await run(process=process)
    assert not next(m for m in context.messages if m["role"] == "tool").get("metadata")
    assert media.published[0][0] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [None, SimpleNamespace(version=1), SimpleNamespace(version=True)])
async def test_absent_incompatible_capability_never_serializes_pixels(media):
    context, provider, _, hooks, _ = await run(media=media)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert UNAVAILABLE in row["content"]
    assert not row.get("metadata")
    assert PIXELS not in json.dumps([context.messages, provider.requests, hooks.events])


@pytest.mark.asyncio
@pytest.mark.parametrize("save", [False, None, RuntimeError("save failed"), "absent"])
async def test_checkpoint_failure_stops_subsequent_provider(save):
    cap, context = Media(), MockContextManager()
    coord = Coordinator(cap, save)
    provider = Provider()
    with pytest.raises(RuntimeError):
        await BasicOrchestrator({}).execute("test", context, {"p": provider}, {"observe": Tool()}, Hooks(), coord)
    assert len(provider.requests) == 1
    assert next(m for m in context.messages if m["role"] == "tool")["metadata"][MEDIA]["refs"] == [A, B]


@pytest.mark.asyncio
async def test_parallel_reverse_completion_preserves_order_and_multiplicity():
    release = asyncio.Event()

    async def execute(args):
        if args["n"] == 0:
            await release.wait()
            return referenced([A, B, A])
        release.set()
        return referenced([C])

    tool, cap, context = Tool(execute=execute), Media(), MockContextManager()
    coord = Coordinator(cap)
    provider = Provider([ToolCall(id=f"call-{n}", name="observe", arguments={"n": n}) for n in range(2)])
    await BasicOrchestrator({}).execute("test", context, {"p": provider}, {"observe": tool}, Hooks(), coord)
    rows = [m for m in context.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in rows] == ["call-0", "call-1"]
    assert [m["metadata"][MEDIA]["refs"] for m in rows] == [[A, B, A], [C]]


@pytest.mark.asyncio
async def test_cancelled_gather_preserves_completed_sibling():
    completed, blocked = asyncio.Event(), asyncio.Event()

    async def execute(args):
        if args["n"] == 0:
            completed.set()
            return referenced([A])
        await blocked.wait()
        return referenced([B])

    tool, cap, context = Tool(execute=execute), Media(), MockContextManager()
    coord = Coordinator(cap)
    provider = Provider([ToolCall(id=f"call-{n}", name="observe", arguments={"n": n}) for n in range(2)])
    task = asyncio.create_task(BasicOrchestrator({}).execute("test", context, {"p": provider}, {"observe": tool}, Hooks(), coord))
    await completed.wait()
    # Wait for the first sibling's post-hook/publication to settle, not wall time.
    while not cap.published:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = [m for m in context.messages if m["role"] == "tool"]
    assert rows[0]["metadata"][MEDIA]["refs"] == [A]
    assert "cancelled" in rows[1]["content"] and "metadata" not in rows[1]
    assert coord.checkpoints == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_immediate_cancel_after_tools_retains_refs_without_dispatch():
    cap, context = Media(), MockContextManager()
    coord = Coordinator(cap)

    async def execute(args):
        coord.cancellation.is_immediate = True
        return referenced([A])

    provider = Provider()
    await BasicOrchestrator({}).execute("test", context, {"p": provider}, {"observe": Tool(execute=execute)}, Hooks(), coord)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert row["metadata"][MEDIA]["refs"] == [A]
    assert coord.checkpoints == 1 and len(provider.requests) == 1


@pytest.mark.asyncio
async def test_hook_failure_has_no_promotable_refs():
    def fail(data):
        raise RuntimeError("hook failed")
    context, _, media, _, _ = await run(change=fail)
    assert media.published == []
    row = next(m for m in context.messages if m["role"] == "tool")
    assert "hook failed" in row["content"] and "metadata" not in row


def test_capability_requires_all_six_operations_and_integer_version():
    assert capability(Coordinator(Media())) is not None
    for method in ("ingest", "publish", "validate", "resolve", "prepare_request", "report"):
        media = Media()
        setattr(media, method, None)
        assert capability(Coordinator(media)) is None


@pytest.mark.asyncio
async def test_producer_context_is_task_local_and_cleaned_after_error():
    coord = Coordinator(Media())
    entered = asyncio.Event()
    observed = {}

    async def execute(args):
        current = asyncio.current_task()
        observed[args["n"]] = deepcopy(coord._tool_dispatch_contexts[current])
        if args["n"] == 0:
            entered.set()
            await asyncio.sleep(0)
            raise RuntimeError("synthetic tool failure")
        await entered.wait()
        assert coord._tool_dispatch_contexts[current] == observed[args["n"]]
        return referenced([C])

    provider = Provider([ToolCall(id=f"call-{n}", name="observe", arguments={"n": n}) for n in range(2)])
    await BasicOrchestrator({}).execute(
        "test", MockContextManager(), {"p": provider}, {"observe": Tool(execute=execute)}, Hooks(), coord,
    )
    assert [observed[n]["tool_call_id"] for n in range(2)] == ["call-0", "call-1"]
    assert observed[0]["parallel_group_id"] == observed[1]["parallel_group_id"]
    assert coord._tool_dispatch_contexts == {}


@pytest.mark.asyncio
async def test_nested_producer_context_restored_even_on_cancellation():
    coord, task = Coordinator(), asyncio.current_task()
    previous = {"tool_call_id": "outer"}
    coord._tool_dispatch_contexts = {task: previous}
    with (
        pytest.raises(asyncio.CancelledError),
        dispatch_context(coord, task, "inner", "group"),
    ):
        assert coord._tool_dispatch_contexts[task]["tool_call_id"] == "inner"
        raise asyncio.CancelledError()
    assert coord._tool_dispatch_contexts[task] is previous


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [-1, 0])
async def test_restored_refs_without_capability_are_explicit_and_history_unmodified(max_iterations):
    row = {"role": "tool", "tool_call_id": "old", "content": "old observation",
           "metadata": {MEDIA: {"refs": [A]}, "other-owner": {"opaque": "unchanged"}}}
    saved = deepcopy(row)
    context = MockContextManager([row])
    seen = []

    class FinalProvider:
        async def complete(self, request, **kwargs):
            seen.append(request.model_dump())
            return ChatResponse(content=[TextBlock(text="finished")])

    await BasicOrchestrator({"max_iterations": max_iterations}).execute(
        "test", context, {"p": FinalProvider()}, {}, Hooks(),
    )
    assert context.messages[0] == saved
    selected = seen[0]["messages"][0]
    assert UNAVAILABLE in selected["content"]
    assert MEDIA not in selected["metadata"]
    assert selected["metadata"]["other-owner"] == {"opaque": "unchanged"}


@pytest.mark.asyncio
async def test_staging_failure_diagnostics_do_not_include_pixels():
    class BrokenMedia(Media):
        async def ingest(self, *args, **kwargs):
            raise RuntimeError(PIXELS)
    context, provider, _, hooks, _ = await run(media=BrokenMedia())
    assert PIXELS not in json.dumps([context.messages, provider.requests, hooks.events], default=str)
    assert not next(m for m in context.messages if m["role"] == "tool").get("metadata")


@pytest.mark.asyncio
async def test_inplace_result_null_and_rich_hook_replacement_never_resurrect():
    def null(data):
        data["result"] = None
        return HookResult()
    context, _, media, _, _ = await run(change=null)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert row["content"] == "None" and "metadata" not in row
    assert media.published[0][0] == []

    def image(data):
        return HookResult(action="modify", data={"result": rich().model_dump()})
    context, _, media, _, _ = await run(change=image)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert "must be staged" in row["content"] and "metadata" not in row
    assert media.published == []


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["ordinary text", {"stdout": "text", "returncode": 0}, [1, "text"]])
async def test_non_media_serialization_unchanged(output):
    tool = Tool(ToolResult(output=output))
    context, _, media, _, _ = await run(tool=tool)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert row["content"] == tool.result.get_serialized_output()
    assert "metadata" not in row and not media.ingested and not media.published


@pytest.mark.asyncio
async def test_prehook_denial_never_executes_or_invents_refs():
    def deny(result, event):
        return HookResult(action="deny", reason="fixture") if event == "tool:pre" else result
    tool = Tool()
    context, _, media, _, _ = await run(tool=tool, process=deny)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert row["content"] == "Denied by hook: fixture" and "metadata" not in row
    assert tool.calls == 0 and not media.ingested and not media.published


@pytest.mark.asyncio
async def test_forged_posthook_ref_is_rejected_by_owner():
    def forge(data):
        return HookResult(action="modify", data={"result": referenced([C]).model_dump()})
    context, _, media, _, _ = await run(change=forge)
    row = next(m for m in context.messages if m["role"] == "tool")
    assert "metadata" not in row
    assert not media.published


@pytest.mark.asyncio
async def test_sync_checkpoint_supported_after_ordered_context_append():
    context, cap = MockContextManager(), Media()
    coord = Coordinator(cap)
    saved = []

    def save():
        saved.append(deepcopy(context.messages))
        coord.checkpoints += 1
        return True

    coord.checkpoint = save
    provider = Provider(coordinator=coord)
    await BasicOrchestrator({}).execute(
        "test", context, {"p": provider}, {"observe": Tool()}, Hooks(), coord,
    )
    assert saved[0][-1]["role"] == "tool"
    assert saved[0][-1]["metadata"][MEDIA]["refs"] == [A, B]


@pytest.mark.asyncio
async def test_cancellation_during_checkpoint_never_reaches_final_request():
    context, cap = MockContextManager(), Media()
    coord = Coordinator(cap)

    async def save():
        coord.checkpoints += 1
        coord.cancellation.is_cancelled = True
        coord.cancellation.is_immediate = True
        return True

    coord.checkpoint = save
    provider = Provider()
    await BasicOrchestrator({"max_iterations": 1}).execute(
        "test", context, {"p": provider}, {"observe": Tool()}, Hooks(), coord,
    )
    assert len(provider.requests) == 1
    assert context.messages[-1]["role"] == "assistant"
    assert "cancelled" in context.messages[-1]["content"]


@pytest.mark.asyncio
async def test_cancel_checkpoint_includes_closure_and_failure_still_cleans_up():
    context, cap = MockContextManager(), Media()
    coord = Coordinator(cap)
    callbacks, saved = [], []

    async def cleanup():
        callbacks.append(True)

    async def save():
        saved.append(deepcopy(context.messages))
        return False

    async def execute(args):
        coord.cancellation.is_immediate = True
        return referenced([A])

    coord.checkpoint = save
    coord.cancellation.trigger_callbacks = cleanup
    provider, hooks = Provider(), Hooks()
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await BasicOrchestrator({}).execute(
            "test", context, {"p": provider}, {"observe": Tool(execute=execute)}, hooks, coord,
        )
    assert callbacks == [True]
    assert saved[0][-1]["role"] == "assistant"
    assert saved[0][-2]["metadata"][MEDIA]["refs"] == [A]
    assert hooks.get_events("cancel:completed")
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_fallback_preserves_content_only_text_order():
    result = rich()
    result.output = None
    context, _, _, _, _ = await run(media=None, tool=Tool(result))
    row = next(m for m in context.messages if m["role"] == "tool")
    assert row["content"] == f"image description\n\n{UNAVAILABLE}"


@pytest.mark.asyncio
async def test_mount_reports_readiness_without_granting_execution_authority():
    from amplifier_module_loop_basic import mount

    cap, reported = Media(), []
    coord = Coordinator(cap)

    def report(origin, **kwargs):
        reported.append((origin, kwargs))

    async def mount_module(name, module):
        assert name == "orchestrator"

    cap.report = report
    coord.mount = mount_module
    await mount(coord)
    assert reported == [(None, {
        "event": "consumer-ready", "outcome": "ready",
        "details": {"contract": 1, "role": "loop"},
    })]


@pytest.mark.asyncio
async def test_task_cancellation_during_checkpoint_never_replays_unknown_save():
    context, cap = MockContextManager(), Media()
    coord = Coordinator(cap)
    entered, blocked, cleaned = asyncio.Event(), asyncio.Event(), []

    async def save():
        coord.checkpoints += 1
        entered.set()
        await blocked.wait()

    async def cleanup():
        cleaned.append(True)

    coord.checkpoint = save
    coord.cancellation.trigger_callbacks = cleanup
    provider = Provider()
    task = asyncio.create_task(BasicOrchestrator({"max_iterations": 1}).execute(
        "test", context, {"p": provider}, {"observe": Tool()}, Hooks(), coord,
    ))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == [True]
    assert coord.checkpoints == 1 and len(provider.requests) == 1
    assert context.messages[-1]["role"] == "assistant"
    assert context.messages[-2]["metadata"][MEDIA]["refs"] == [A, B]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_checkpoint_failure_or_notifications():
    coord = Coordinator(Media(), save=False)
    context, hooks = MockContextManager(), Hooks()

    async def cleanup():
        raise RuntimeError("cleanup failed")

    async def execute(args):
        coord.cancellation.is_immediate = True
        return referenced([A])

    coord.cancellation.trigger_callbacks = cleanup
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        await BasicOrchestrator({}).execute(
            "test", context, {"p": Provider()}, {"observe": Tool(execute=execute)}, hooks, coord,
        )
    assert hooks.get_events("cancel:completed")
    assert hooks.get_events("orchestrator:complete")
