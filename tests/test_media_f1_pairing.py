"""Opt-in exact Foundation candidate pairing, never an installed/live host claim.

Set AMPLIFIER_MEDIA_F1_SOURCE to a mechanically exported F1 candidate and put
that root first on PYTHONPATH. The original store/checkpoint/private-wrapper
cases use an explicitly synthetic hook witness. The safe-* cases additionally
load the exact public helper at AMPLIFIER_MEDIA_HOOK_BRIDGE_SOURCE, with actual
Core HookRegistry; they prove only fail-closed behavior, never positive delivery.
"""

import base64
import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from amplifier_core import HookRegistry, HookResult, ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_basic import BasicOrchestrator
from amplifier_module_loop_basic._media import MEDIA, TOOL_POST_BOUNDARY

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("AMPLIFIER_MEDIA_F1_SOURCE"),
        reason="opt-in exact Foundation candidate pairing, not installed graph coverage",
    ),
]


@pytest.mark.parametrize("max_iterations", [-1, 1])
@pytest.mark.parametrize("disposition", [
    "preserve", "partial", "null", "incomplete",
    "safe-keep", "safe-remove", "safe-null-inject", "safe-inplace", "safe-deny", "safe-error",
    "safe-text-success", "safe-text-failure", "safe-text-null", "safe-text-remove",
    "safe-text-addref", "safe-content-text",
])
async def test_actual_f1_store_checkpoint_and_private_request(tmp_path, max_iterations, disposition):
    from amplifier_foundation.session.checkpoint import (
        acknowledge_legacy_checkpoint,
        verified_checkpoint_refs,
    )
    from amplifier_foundation.session.media import (
        MediaPolicy,
        SessionMediaCapability,
        SessionMediaStore,
    )
    from amplifier_foundation.session.media_provider import MediaPreparedProvider
    from PIL import Image

    source = Path(os.environ["AMPLIFIER_MEDIA_F1_SOURCE"]).resolve()
    assert Path(inspect.getfile(SessionMediaStore)).resolve() == source / "amplifier_foundation/session/media.py"
    images = []
    for color in ("red", "blue"):
        stream = io.BytesIO()
        Image.new("RGB", (3, 2), color).save(stream, format="PNG")
        images.append(stream.getvalue())
    sentinels = [base64.b64encode(image).decode() for image in images]
    original = ToolResult(output="real store observation", content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": text}}
        for text in sentinels
    ])
    if disposition == "safe-content-text":
        original.output = None
        original.content = [TextBlock(text="useful content-only text"), *original.content]
    text_only = disposition.startswith("safe-text-")
    if text_only:
        original = ToolResult(
            success=disposition != "safe-text-failure",
            output="ordinary failure detail" if disposition == "safe-text-failure" else "ordinary success detail",
            error={"message": "ordinary failure detail"} if disposition == "safe-text-failure" else None,
        )
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    safe_bridge = disposition.startswith("safe-")
    context = MockContextManager()
    hooks = HookRegistry() if safe_bridge else EventRecorder()
    counts, observed = {"emit": 0, "process": 0, "checkpoint": 0}, []
    store = SessionMediaStore(
        session_dir / "media", "paired-session",
        policy=MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=verified_checkpoint_refs,
    )
    cap = SessionMediaCapability(store, enabled=True)
    publications, canonical, private_images = [], [], []
    publish = cap.publish

    async def counted_publish(refs, **kwargs):
        publications.append((deepcopy(refs), kwargs["disposition"]))
        return await publish(refs, **kwargs)

    cap.publish = counted_publish

    class Coordinator:
        session_id = "paired-session"
        loader = object()
        cancellation = SimpleNamespace(
            is_cancelled=False, is_immediate=False,
            register_tool_start=lambda *a: None,
            register_tool_complete=lambda *a: None,
        )

        def __init__(self):
            self.hooks = hooks
            self.capabilities = {MEDIA: cap, "session.durable_checkpoint": self.save}
            if not safe_bridge:
                self.capabilities[TOOL_POST_BOUNDARY] = self.boundary

        def register_capability(self, name, value):
            self.capabilities[name] = value

        def get_capability(self, name):
            return self.capabilities.get(name)

        async def process_hook_result(self, result, event, name):
            if event == "tool:post":
                counts["process"] += 1
            return result

        async def boundary(self, event, *, hooks=None, coordinator=None):
            post = await hooks.emit("tool:post", event)
            post = await coordinator.process_hook_result(post, "tool:post", event["tool_name"])
            value = deepcopy(event["result"])
            if disposition == "partial":
                value["output"][MEDIA]["refs"] = value["output"][MEDIA]["refs"][1:]
            elif disposition == "null":
                value = None
            return {"hook_result": post, "result": value, "complete": disposition != "incomplete"}

        async def save(self):
            counts["checkpoint"] += 1
            encoded = "".join(json.dumps(row) + "\n" for row in context.messages)
            assert not any(text in encoded for text in sentinels)
            with (session_dir / "transcript.jsonl").open("w") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            receipt = acknowledge_legacy_checkpoint(
                session_dir, session_id=self.session_id, expected_messages=context.messages,
            )
            store.acknowledge_checkpoint(receipt)
            return True

    class Tool:
        name, description, input_schema = "observe", "synthetic images", {}

        async def execute(self, args):
            return original

    class Transport:
        name, config, default_model = "fixture", {}, "fixture-model"
        calls = 0

        def get_media_request_support(self, *, model, request_options=None):
            return {"v": 1, "positions": ["tool"], "mime_types": ["image/png"]}

        async def complete(self, request, **kwargs):
            cap.validate(request, purpose="transport", request_options=kwargs)
            self.calls += 1
            for message in request.messages:
                if isinstance(message.content, list):
                    private_images.extend(
                        base64.b64decode(block.source["data"])
                        for block in message.content if getattr(block, "type", None) == "image"
                    )
            return ChatResponse(
                content=[TextBlock(text="observe" if self.calls == 1 else "finished")],
                tool_calls=[ToolCall(id="actual-call", name="observe", arguments={})] if self.calls == 1 else None,
            )

    wrapped = MediaPreparedProvider(
        Transport(), cap, provider_identity={"provider": "fixture", "account": "test", "generation": "1"},
        model="fixture-model",
    )
    complete = wrapped.complete

    async def capture(request, **kwargs):
        canonical.append(request.model_dump())
        return await complete(request, **kwargs)

    wrapped.complete = capture
    coordinator = Coordinator()
    if safe_bridge:
        bridge_source = os.environ.get("AMPLIFIER_MEDIA_HOOK_BRIDGE_SOURCE")
        if not bridge_source:
            wrapped.close()
            store.close()
            pytest.skip("opt-in safe bridge candidate not supplied; no positive bridge claim")
        path = Path(bridge_source).resolve() / "amplifier_foundation/session/media_hook_bridge.py"
        # Import this exact helper alongside the separately pinned F1 package;
        # no copied/edited product files and no synthetic private witness calls.
        spec = importlib.util.spec_from_file_location("_basic_safe_bridge_candidate", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        bridge = module.install_media_hook_bridge(coordinator)
        assert coordinator.get_capability(TOOL_POST_BOUNDARY) is bridge

        async def first(event, data):
            # One first callback proves one actual native chain emission.
            counts["emit"] += 1
            observed.append(deepcopy(data))
            if disposition in ("safe-remove", "safe-text-remove"):
                return HookResult(action="modify", data={})
            if disposition in ("safe-null-inject", "safe-text-null"):
                return HookResult(action="modify", data={"result": None})
            if disposition == "safe-text-addref":
                value = deepcopy(data["result"])
                value["output"] = {
                    "label": "replacement useful text",
                    MEDIA: {"refs": [{"unowned": True}]},
                }
                return HookResult(action="modify", data={"result": value})
            if disposition == "safe-inplace":
                data["result"]["output"].pop(MEDIA)
            if disposition == "safe-deny":
                return HookResult(action="deny", reason="fixture denial")
            if disposition == "safe-error":
                raise RuntimeError("synthetic callback failure")
            return HookResult()

        async def injection(event, data):
            return HookResult(
                action="inject_context", context_injection="fixture feedback",
                ephemeral=True,
            )

        bridge.hooks.register("tool:post", first, name="counted-first")
        if disposition == "safe-null-inject":
            bridge.hooks.register("tool:post", injection, priority=10, name="injection")
    try:
        await BasicOrchestrator({"max_iterations": max_iterations}).execute(
            "test", context, {"fixture": wrapped}, {"observe": Tool()}, hooks, coordinator,
        )
        row = next(m for m in context.messages if m["role"] == "tool")
        expected = images if disposition == "preserve" else images[1:] if disposition == "partial" else []
        assert private_images == expected
        assert counts["checkpoint"] == 1
        assert len(canonical) == 2
        assert len(publications) == (0 if text_only else 1)
        if publications:
            published_refs, action = publications[0]
            assert action == ("retain" if expected else "drop")
            assert len(published_refs) == (len(expected) if expected else 2)
        refs = row.get("metadata", {}).get(MEDIA, {}).get("refs", [])
        assert [r["sha256"] for r in refs] == [hashlib.sha256(image).hexdigest() for image in expected]
        for ref in refs:
            with cap.resolve(ref, purpose="model") as handle:
                assert handle.read() in expected
        if safe_bridge:
            assert counts == {"emit": 1, "process": 1, "checkpoint": 1}
            assert "metadata" not in row
            if disposition in ("safe-text-null", "safe-text-remove"):
                assert row["content"] == "None"
            elif disposition in ("safe-text-success", "safe-text-failure"):
                assert row["content"] == original.get_serialized_output()
            elif disposition == "safe-text-addref":
                assert "replacement useful text" in row["content"]
                assert MEDIA not in row["content"]
            elif disposition == "safe-deny":
                assert row["content"] == "Denied by hook: fixture denial"
            else:
                assert "final hook result was not established" in row["content"]
                if disposition in ("safe-keep", "safe-inplace", "safe-error"):
                    assert "real store observation" in row["content"]
                if disposition == "safe-content-text":
                    assert "useful content-only text" in row["content"]
            readiness = cap.report("inspector", event="readiness", outcome="inspect")
            assert readiness["consumers"] == {}
            event_data = observed
        else:
            event_data = hooks.events
        if disposition != "incomplete" and not safe_bridge:
            readiness = cap.report("inspector", event="readiness", outcome="inspect")
            assert readiness["consumers"]["loop-basic"] == {"contract": 1, "role": "loop"}
        serialized = json.dumps([context.messages, canonical, event_data], default=str)
        assert not any(text in serialized for text in sentinels)
        for path in (session_dir / "media").rglob("*.json"):
            assert not any(text in path.read_text() for text in sentinels)
    finally:
        wrapped.close()
        store.close()


@pytest.mark.parametrize("max_iterations", [-1, 1])
@pytest.mark.parametrize("kind", ["rich", "refs", "text"])
async def test_actual_disabled_f1_capability_uses_text_fallback_without_store_io(
    max_iterations, kind,
):
    from amplifier_foundation.session.media import SessionMediaCapability

    from amplifier_module_loop_basic._media import UNAVAILABLE

    class NoStoreAccess:
        session_id = "disabled-session"

        def __getattr__(self, name):
            raise AssertionError(f"disabled store operation: {name}")

    cap = SessionMediaCapability(NoStoreAccess(), enabled=False)
    forbidden, messages = [], MockContextManager()
    cap.ingest = cap.publish = cap.report = lambda *a, **kw: forbidden.append(True)
    sentinel = "U0VFRF9QSVhFTF9TRU5USU5FTA=="
    result = ToolResult(output="useful output")
    if kind == "rich":
        result.content = [
            TextBlock(text="useful rich text"),
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": sentinel,
            }},
        ]
    elif kind == "refs":
        result.output = {"label": "useful ref text", MEDIA: {"refs": [{"unowned": True}]}}

    class Coordinator:
        cancellation = SimpleNamespace(
            is_cancelled=False, is_immediate=False,
            register_tool_start=lambda *a: None, register_tool_complete=lambda *a: None,
        )

        def get_capability(self, name):
            return cap if name == MEDIA else None

        async def process_hook_result(self, result, *args):
            return result

    class Tool:
        name, description, input_schema = "observe", "fixture", {}

        async def execute(self, args):
            return result

    class Provider:
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            assert sentinel not in request.model_dump_json()
            return ChatResponse(
                content=[TextBlock(text="observe" if self.calls == 1 else "done")],
                tool_calls=[ToolCall(id="actual", name="observe", arguments={})]
                if self.calls == 1 else None,
            )

    hooks = HookRegistry()
    observed = []

    async def observe(event, data):
        observed.append(deepcopy(data))
        return HookResult()

    hooks.register("tool:post", observe, name="observe")
    await BasicOrchestrator({"max_iterations": max_iterations}).execute(
        "test", messages, {"p": Provider()}, {"observe": Tool()}, hooks, Coordinator(),
    )
    row = next(m for m in messages.messages if m["role"] == "tool")
    assert "metadata" not in row
    assert not forbidden
    assert sentinel not in json.dumps([messages.messages, observed], default=str)
    if kind == "text":
        assert row["content"] == "useful output"
    else:
        assert UNAVAILABLE in row["content"] and "useful" in row["content"]
        assert MEDIA not in row["content"]
        if kind == "rich":
            assert "useful rich text" in row["content"]
