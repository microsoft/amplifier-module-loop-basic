"""Opt-in exact Foundation candidate pairing, never an installed/live host claim.

Set AMPLIFIER_MEDIA_F1_SOURCE to a mechanically exported F1 candidate and put
that root first on PYTHONPATH. This fixture uses the actual store, source-sealed
checkpoint and private provider wrapper; its hook witness is synthetic until the
independently owned real-registry bridge is supplied.
"""

import base64
import hashlib
import inspect
import io
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from amplifier_core import ToolResult
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
@pytest.mark.parametrize("disposition", ["preserve", "partial", "null", "incomplete"])
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
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    context, hooks = MockContextManager(), EventRecorder()
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
        cancellation = SimpleNamespace(
            is_cancelled=False, is_immediate=False,
            register_tool_start=lambda *a: None,
            register_tool_complete=lambda *a: None,
        )

        def get_capability(self, name):
            return {MEDIA: cap, TOOL_POST_BOUNDARY: self.boundary,
                    "session.durable_checkpoint": self.save}.get(name)

        async def process_hook_result(self, result, event, name):
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
    try:
        await BasicOrchestrator({"max_iterations": max_iterations}).execute(
            "test", context, {"fixture": wrapped}, {"observe": Tool()}, hooks, Coordinator(),
        )
        row = next(m for m in context.messages if m["role"] == "tool")
        expected = images if disposition == "preserve" else images[1:] if disposition == "partial" else []
        assert private_images == expected
        assert len(publications) == 1
        published_refs, action = publications[0]
        assert action == ("retain" if expected else "drop")
        assert len(published_refs) == (len(expected) if expected else 2)
        refs = row.get("metadata", {}).get(MEDIA, {}).get("refs", [])
        assert [r["sha256"] for r in refs] == [hashlib.sha256(image).hexdigest() for image in expected]
        for ref in refs:
            with cap.resolve(ref, purpose="model") as handle:
                assert handle.read() in expected
        if disposition != "incomplete":
            readiness = cap.report("inspector", event="readiness", outcome="inspect")
            assert readiness["consumers"]["loop-basic"] == {"contract": 1, "role": "loop"}
        serialized = json.dumps([context.messages, canonical, hooks.events], default=str)
        assert not any(text in serialized for text in sentinels)
        for path in (session_dir / "media").rglob("*.json"):
            assert not any(text in path.read_text() for text in sentinels)
    finally:
        wrapped.close()
        store.close()
