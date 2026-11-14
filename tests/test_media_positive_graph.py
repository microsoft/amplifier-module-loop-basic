"""Actual public cooperative bridge -> F1 -> final private count/send fixtures.

Opt-in exact exported Foundation union only. No private hook witness, fabricated
complete result, model call, CLI/Unified host parity or malicious-Python claim.
"""

import asyncio
import base64
import hashlib
import inspect
import io
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ToolCall,
)

from amplifier_module_loop_basic import BasicOrchestrator
from amplifier_module_loop_basic._media import MEDIA

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("AMPLIFIER_MEDIA_POSITIVE_SOURCE"),
        reason="exact cooperative positive Foundation candidate not selected",
    ),
]


class GraphLoader:
    def __init__(self, modules, policy):
        self.modules, self.policy, self.queue = modules, policy, []

    async def load(self, module_id, config, *, source_hint=None, coordinator=None):
        async def mount(current):
            if module_id == "fixture-hooks":
                current.hooks.register("tool:post", self.policy, name="policy", priority=10)
            else:
                slot, module, name = self.modules[module_id]
                await current.mount(slot, module, name=name)
        return mount

    def enqueue_on_session_ready(self, name, callback):
        self.queue.append((name, callback))

    def get_on_session_ready_queue(self):
        return self.queue

    def clear_on_session_ready_queue(self):
        self.queue.clear()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("limit", [-1, 1], ids=["main", "final"])
@pytest.mark.parametrize("case", [
    "keep", "partial", "null", "missing", "inplace", "string", "deny", "error",
    "outer-redaction", "unknown-observed", "checkpoint-failure", "alter-after-ack", "cancel",
    "alter-on-append", "cancel-alter-on-append",
    "continue-null-echo", "special-null-echo",
])
async def test_actual_public_graph_final_count_send(tmp_path, fmt, limit, case):
    await run_graph(tmp_path, fmt, limit, case)


async def run_graph(tmp_path, fmt, limit, case, *, parent_id=None, resumed=False,
                    restored=None, existing_store=False):
    from amplifier_foundation.session.checkpoint import (
        acknowledge_legacy_checkpoint,
        verified_checkpoint_refs,
    )
    from amplifier_foundation.session.media import (
        MediaPolicy,
        SessionMediaStore,
        bind_session_media,
    )
    from amplifier_foundation.session.media_hook_bridge import install_media_hook_bridge
    from amplifier_foundation.session.media_provider import MediaPreparedProvider
    from PIL import Image

    source = Path(os.environ["AMPLIFIER_MEDIA_POSITIVE_SOURCE"]).resolve()
    for module in (SessionMediaStore, MediaPreparedProvider, install_media_hook_bridge):
        assert Path(inspect.getfile(module)).resolve().is_relative_to(source)
    session = AmplifierSession({
        "session": {"orchestrator": "loop-basic", "context": "fixture-context"},
        "providers": [{"module": "fixture-provider"}],
        "tools": [{"module": "fixture-images"}],
        "hooks": [{"module": "fixture-hooks"}],
    }, session_id="basic-positive", parent_id=parent_id, is_resumed=resumed)
    core = session.coordinator
    images = []
    for color in ("red", "blue"):
        stream = io.BytesIO()
        Image.new("RGB", (3, 2), color).save(stream, format=fmt)
        images.append(stream.getvalue())
    sentinels = [base64.b64encode(image).decode() for image in images]
    hashes = [hashlib.sha256(image).hexdigest() for image in images]
    order, posts, publications, counted, sent, canonical, acknowledgments = [], [], [], [], [], [], []

    class Context:
        def __init__(self):
            self.messages = deepcopy(restored or [])

        async def add_message(self, message):
            if case in ("alter-on-append", "cancel-alter-on-append") and MEDIA in (
                message.get("metadata") or {}
            ):
                message["metadata"][MEDIA]["refs"].reverse()
            self.messages.append(deepcopy(message))
            if message["role"] == "tool":
                order.append("append")

        async def get_messages(self):
            return self.messages

        async def get_messages_for_request(self, provider=None):
            assert provider is adapter
            # Real earlier context preparation intentionally fits a DIFFERENT
            # messages-only candidate, proving the loop must refit its final one.
            preliminary = ChatRequest(messages=[Message(**row) for row in self.messages])
            await provider.request_budget(preliminary, request_options={})
            return deepcopy(self.messages)

    context = Context()
    store = SessionMediaStore(
        tmp_path / "media", core.session_id,
        policy=MediaPolicy(1024 * 1024, 10000, 10), checkpoint_verifier=verified_checkpoint_refs,
    )
    cap = bind_session_media(core, store=store, enabled=True)
    if existing_store:
        receipt = acknowledge_legacy_checkpoint(
            tmp_path, session_id=core.session_id, expected_messages=context.messages,
        )
        store.acknowledge_checkpoint(receipt)
    original_publish = cap.publish

    async def publish(refs, *, origin, disposition):
        publications.append((deepcopy(refs), deepcopy(origin), disposition))
        return await original_publish(refs, origin=origin, disposition=disposition)

    cap.publish = publish

    class Wire:
        name, config, default_model = "fixture", {}, "fixture"

        def get_media_request_support(self, *, model, request_options=None):
            return {"v": 1, "positions": ["tool"], "mime_types": ["image/png", "image/jpeg"]}

        async def request_budget(self, request, *, request_options=None):
            cap.validate(request, purpose="provider-count", request_options=request_options)
            counted.append((request, request.model_dump(), deepcopy(request_options)))
            order.append("count")
            return {
                "estimated_input_tokens": 100, "input_limit_tokens": 1000,
                "context_token_budget": 80,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100},
            }

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="provider-send", request_options=kwargs)
            assert counted[-1][0] is request
            assert counted[-1][1] == request.model_dump()
            assert counted[-1][2] == kwargs == {"extended_thinking": True}
            assert request.tools[0].name == "images" and request.reasoning_effort == "high"
            sent.append(request.model_dump())
            order.append("send")
            return ChatResponse(
                content=[TextBlock(text="done" if resumed or len(sent) == 2 else "observe")],
                tool_calls=None if resumed or len(sent) == 2 else
                [ToolCall(id="actual-call", name="images", arguments={})],
            )

    adapter = MediaPreparedProvider(
        Wire(), cap, provider_identity={"provider": "fixture", "account": "fixture", "generation": 1},
        model="fixture",
    )
    original_complete = adapter.complete

    async def capture(request, **kwargs):
        canonical.append(request.model_dump())
        return await original_complete(request, **kwargs)

    adapter.complete = capture
    executions = []

    class ImageTool:
        name, description = "images", "actual synthetic PNG/JPEG"

        def __init__(self):
            self.input_schema = {"type": "object", "properties": {"detail": {"type": "string"}}}

        async def execute(self, args):
            executions.append(True)
            assert core._tool_dispatch_contexts[asyncio.current_task()]["tool_call_id"] == "actual-call"
            return ToolResult(output="useful image result", content=[
                {"type": "image", "source": {"type": "base64", "media_type": "image/png" if fmt == "PNG" else "image/jpeg", "data": text}}
                for text in sentinels
            ])

    def policy(event, data):
        posts.append(deepcopy(data))
        order.append("post")
        if case == "partial":
            value = deepcopy(data["result"])
            value["output"][MEDIA]["refs"] = value["output"][MEDIA]["refs"][:1]
            return HookResult(action="modify", data={"result": value})
        if case == "null":
            return HookResult(action="modify", data={"result": None})
        if case == "continue-null-echo":
            return HookResult(data={"result": None})
        if case == "special-null-echo":
            return HookResult(
                action="inject_context", data={"result": None},
                context_injection="special redaction", ephemeral=True,
            )
        if case == "missing":
            return HookResult(action="modify", data={})
        if case == "inplace":
            data["result"]["output"].pop(MEDIA)
        if case == "string":
            return HookResult(action="modify", data={"result": "redacted text"})
        if case == "deny":
            return HookResult(action="deny", reason="fixture redaction")
        if case == "error":
            raise RuntimeError("fixture callback failure")
        if case in ("cancel", "cancel-alter-on-append"):
            core.cancellation.request_cancellation()
        return HookResult()

    orchestrator = BasicOrchestrator({
        "max_iterations": limit, "reasoning_effort": "high", "extended_thinking": True,
    })
    modules = {
        "loop-basic": ("orchestrator", orchestrator, None),
        "fixture-context": ("context", context, None),
        "fixture-provider": ("providers", adapter, "fixture"),
        "fixture-images": ("tools", ImageTool(), "images"),
    }
    core.loader = GraphLoader(modules, policy)
    bridge = install_media_hook_bridge(core)
    graph = bridge.begin_participating_graph(
        session, owner="basic.controlled", participants=(*modules, "fixture-hooks", "host.policy"),
    )
    await session.initialize()

    def outer(event, data):
        if case == "outer-redaction":
            return HookResult(action="modify", data={"result": None})
        if case == "null":
            return HookResult(action="inject_context", context_injection="late qualified feedback", ephemeral=True)
        if case in ("continue-null-echo", "special-null-echo"):
            return HookResult(action="modify", data=data)
        return HookResult()

    graph.participant("host.policy").hooks.register("tool:post", outer, priority=20, name="outer")
    receipt = graph.seal()
    assert bridge.validate_receipt(receipt)
    if case == "unknown-observed":
        core.hooks.register("tool:post", lambda e, d: HookResult(), name="unknown")

    def save():
        if case == "checkpoint-failure":
            raise RuntimeError("fixture checkpoint failed")
        saved = deepcopy(context.messages)
        (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in saved))
        ack = acknowledge_legacy_checkpoint(tmp_path, session_id=core.session_id, expected_messages=saved)
        store.acknowledge_checkpoint(ack)
        acknowledgments.append(saved)
        order.append("checkpoint")
        if case == "alter-after-ack":
            row = next(m for m in context.messages if m["role"] == "tool")
            row["metadata"][MEDIA]["refs"].reverse()
        return True

    core.register_capability("session.durable_checkpoint", save)
    try:
        if case in ("checkpoint-failure", "alter-after-ack", "alter-on-append", "cancel-alter-on-append"):
            with pytest.raises(RuntimeError, match="checkpoint|Settled media"):
                await session.execute("Inspect synthetic media")
            assert len(sent) == 1
            return None
        await session.execute("Inspect synthetic media")
        if case == "cancel":
            assert len(sent) == len(executions) == len(posts) == len(publications) == 1
            assert publications[0][2] == "retain"
            row = next(row for row in context.messages if row["role"] == "tool")
            ack_row = next(row for row in acknowledgments[-1] if row["role"] == "tool")
            assert row["metadata"][MEDIA] == ack_row["metadata"][MEDIA]
            assert [r["sha256"] for r in row["metadata"][MEDIA]["refs"]] == hashes
            assert context.messages[-1]["role"] == "assistant"
            return None
        expected_count = 2 if case == "keep" else 1 if case == "partial" else 0
        if resumed:
            expected_count = 2
            assert executions == posts == publications == []
        else:
            assert len(executions) == len(posts) == len(publications) == 1
            assert publications[0][2] == ("retain" if expected_count else "drop")
            tool = next(row for row in context.messages if row["role"] == "tool")
            refs = tool.get("metadata", {}).get(MEDIA, {}).get("refs", [])
            ack_tool = next(row for row in acknowledgments[-1] if row["role"] == "tool")
            assert refs == ack_tool.get("metadata", {}).get(MEDIA, {}).get("refs", [])
            assert [ref["sha256"] for ref in refs] == hashes[:expected_count]
            assert order.index("post") < order.index("append") < order.index("checkpoint")
        private = [
            base64.b64decode(block["source"]["data"])
            for message in sent[-1]["messages"] if isinstance(message["content"], list)
            for block in message["content"] if block["type"] == "image"
        ]
        assert [hashlib.sha256(image).hexdigest() for image in private] == hashes[:expected_count]
        assert not any(text in json.dumps([context.messages, canonical, posts]) for text in sentinels)
        for path in (tmp_path / "media").rglob("*.json"):
            assert not any(text in path.read_text() for text in sentinels)
        if limit == 1 and not resumed:
            assert "orchestrator-loop-limit" in str(canonical[-1])
        if case == "keep" and not resumed:
            # Model the owning host's end-turn save, not only the loop's
            # post-settlement pre-dispatch checkpoint used above.
            save()
        return deepcopy(context.messages)
    finally:
        adapter.close()
        await session.cleanup()
        store.close()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("lifecycle", ["finite-child", "resume"])
async def test_public_graph_fresh_child_or_resume(tmp_path, fmt, lifecycle):
    if lifecycle == "finite-child":
        await run_graph(tmp_path, fmt, -1, "keep", parent_id="actual-parent")
    else:
        saved = await run_graph(tmp_path, fmt, -1, "keep")
        await run_graph(tmp_path, fmt, -1, "keep", resumed=True, restored=saved, existing_store=True)
