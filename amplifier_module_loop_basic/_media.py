"""Structural media-v1 boundary. Storage and request hydration belong to the host."""

import inspect
import json
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from amplifier_core import ToolResult

MEDIA = "experimental.media.v1"
UNAVAILABLE = (
    "Image unavailable to this session: compatible media handling is not installed; "
    "no pixels were sent."
)


@contextmanager
def dispatch_context(coordinator, task, tool_call_id, group_id):
    """Expose actual concurrent producer identity only during tool execution."""
    if coordinator is None:
        yield
        return
    contexts = getattr(coordinator, "_tool_dispatch_contexts", None)
    if contexts is None:
        contexts = {}
        coordinator._tool_dispatch_contexts = contexts
    missing = object()
    previous = contexts.get(task, missing)
    contexts[task] = {"tool_call_id": tool_call_id, "parallel_group_id": group_id}
    try:
        yield
    finally:
        if previous is missing:
            contexts.pop(task, None)
        else:
            contexts[task] = previous


async def invoke(method, *args, **kwargs):
    value = method(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


async def media_call(method, *args, **kwargs):
    # Media adapters may include payloads in exception strings; never log them.
    try:
        return await invoke(method, *args, **kwargs)
    except Exception:  # noqa: BLE001 - deliberately scrub untrusted adapter errors
        raise RuntimeError("Media operation failed; no image delivery was confirmed") from None


def capability(coordinator):
    get = getattr(coordinator, "get_capability", None)
    cap = get(MEDIA) if callable(get) else None
    if (
        cap is not None
        and type(getattr(cap, "version", None)) is int
        and cap.version == 1
        and all(
            callable(getattr(cap, name, None))
            for name in ("ingest", "publish", "validate", "resolve", "prepare_request", "report")
        )
    ):
        return cap
    return None


def refs_from(value):
    """Never promote JSON-looking strings or search recursively."""
    if not isinstance(value, dict) or not isinstance(value.get("output"), dict):
        return []
    envelope = value["output"].get(MEDIA)
    if envelope is None:
        return []
    if not isinstance(envelope, dict) or set(envelope) != {"refs"}:
        raise ValueError("Invalid media-v1 output envelope")
    if not isinstance(envelope["refs"], list):
        raise TypeError("Invalid media-v1 refs")
    return deepcopy(envelope["refs"])


def has_images(result):
    blocks = result.content if isinstance(result, ToolResult) else (
        result.get("content") if isinstance(result, dict) else None
    )
    return isinstance(blocks, list) and any(
        (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "image"
        for block in blocks
    )


def without_images(result):
    """Remove canonical rich images without ever dumping their source bytes."""
    content = [
        block for block in (result.content or [])
        if getattr(block, "type", None) != "image"
    ]
    output = deepcopy(result.output)
    if isinstance(output, dict):
        output.pop(MEDIA, None)
    text = ToolResult(success=result.success, output=output, error=result.error)
    parts = [text.get_serialized_output()] if output is not None or result.error else []
    parts.extend(block.text for block in content if getattr(block, "type", None) == "text")
    return ToolResult(
        success=result.success,
        output="\n\n".join([*parts, UNAVAILABLE]),
        error=result.error,
        content=content or None,
    )


async def stage(result, cap, origin):
    if not isinstance(result, ToolResult):
        return result
    media = has_images(result) or (
        isinstance(result.output, dict) and MEDIA in result.output
    )
    if not media:
        return result
    if cap is None:
        return without_images(result)
    staged = await media_call(
        cap.ingest, result, source_format="core-tool-result", origin=origin, policy=None
    )
    value = staged["value"]
    if not isinstance(value, ToolResult) or has_images(value):
        raise ValueError("Media staging did not return a refs-only ToolResult")
    return value


def presentation(result):
    if isinstance(result, ToolResult):
        return result.model_dump()
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


async def settle(original, sent, post, cap, origin, tool_call_id, current):
    """Select final effective output by field presence, never object identity."""
    if post.action == "deny":
        effective = None
        text = f"Denied by hook: {post.reason}"
    elif isinstance(post.data, dict) and "result" in post.data:
        effective = post.data["result"]
        text = None
    elif post.action == "modify":
        effective = None
        text = "None"
    else:
        effective = current
        text = None

    if has_images(effective):
        raise ValueError("Post-hook images must be staged by their producing owner")
    if isinstance(effective, ToolResult):
        effective = effective.model_dump()
    refs = refs_from(effective)
    if cap is not None and (refs or refs_from(sent)):
        published = await media_call(
            cap.publish, refs, origin=origin, disposition="retain" if refs else "drop"
        )
        if published["refs"] != refs:
            raise ValueError("Media publication changed the effective refs")
    elif refs:
        # Hooks cannot bypass absence of a compatible capability.
        effective = deepcopy(effective)
        effective["output"].pop(MEDIA, None)
        refs = []
        text = f"{json.dumps(effective)}\n\n{UNAVAILABLE}"

    if text is None:
        if effective == sent:
            text = original.get_serialized_output()
        elif isinstance(effective, (dict, list)):
            text = json.dumps(effective)
        else:
            text = str(effective)
    message: dict[str, Any] = {
        "role": "tool", "tool_call_id": tool_call_id, "content": text,
    }
    if refs:
        message["metadata"] = {MEDIA: {"refs": refs}}
    return message


async def checkpoint(coordinator, messages):
    """Existing no-argument host checkpoint owns the exact canonical save ACK."""
    get = getattr(coordinator, "get_capability", None)
    save = get("session.durable_checkpoint") if callable(get) else None
    media = any(MEDIA in (message.get("metadata") or {}) for message in messages)
    if not callable(save):
        if media:
            raise RuntimeError("Media delivery requires a durable settled checkpoint")
        return
    acknowledged = await invoke(save)
    if acknowledged is False or (media and acknowledged is None):
        raise RuntimeError("Durable settled checkpoint failed")


def request_messages(messages, cap):
    """Unsupported restored history is explicit, without rewriting saved history."""
    if cap is not None:
        return messages
    selected = []
    for message in messages:
        if MEDIA not in (message.get("metadata") or {}):
            selected.append(message)
            continue
        copy = deepcopy(message)
        copy["metadata"].pop(MEDIA)
        copy["content"] = f"{copy.get('content', '')}\n\n{UNAVAILABLE}"
        selected.append(copy)
    return selected