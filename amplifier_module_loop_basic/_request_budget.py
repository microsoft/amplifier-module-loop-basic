"""Final-candidate budget admission; no request hydration or provider imports."""

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping
from copy import deepcopy

from ._media import MEDIA, invoke

logger = logging.getLogger(__name__)


class RequestBudgetError(RuntimeError):
    """Local refusal: no provider completion was attempted for this candidate."""


def _named(method, name):
    try:
        parameter = inspect.signature(method).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )


def _accepts_context_estimate(method):
    return _named(method, "context_estimate") or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(method).parameters.values()
    )


def _text_blocks(content):
    result = []
    for block in content:
        if not isinstance(block, dict):
            result.append(block)
            continue
        if block.get("type") in ("image", "image_url", "input_image"):
            block = {k: v for k, v in block.items() if k not in (
                "source", "image_url", "url", "data", "file_id",
            )}
        elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            block = {**block, "content": _text_blocks(block["content"])}
        result.append(block)
    return result


async def _context_estimate(context, request):
    # Existing context-simple coordinate is len(str(message))//4, excluding
    # image transport. It is a text scale for compaction, NEVER a pixel count.
    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    safe = [
        {**message, "content": _text_blocks(message["content"])}
        if isinstance(message.get("content"), list) else message
        for message in messages
    ]
    estimator = getattr(context, "estimate_tokens", None)
    estimate = await invoke(estimator, safe) if callable(estimator) else sum(
        len(str(message)) // 4 for message in safe
    )
    if type(estimate) is not int or estimate < 0:
        raise RequestBudgetError("Context returned an invalid token estimate")
    return estimate


def _fingerprint(request, options):
    return json.dumps(
        [request.model_dump(mode="json"), options],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


async def _call_with_options(method, request, options, **extra):
    forwarded = deepcopy(options)
    before = _fingerprint(request, forwarded)
    value = await invoke(method, request, request_options=forwarded, **extra)
    if _fingerprint(request, forwarded) != before:
        raise RequestBudgetError("Final request or forwarded options changed during budgeting")
    return value


def _media_input(request):
    return any(
        MEDIA in (message.metadata or {}) or (
            isinstance(message.content, list)
            and any(getattr(block, "type", None) == "image" for block in message.content)
        )
        for message in request.messages
    )


def _witness(value):
    return (
        isinstance(value, Mapping) and set(value) == {"v", "validated", "projection_digest"}
        and type(value["v"]) is int and value["v"] == 1
        and value["validated"] is True
        and isinstance(value["projection_digest"], str)
        and len(value["projection_digest"]) == 64
        and all(c in "0123456789abcdef" for c in value["projection_digest"])
    )


def _decision(value):
    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise RequestBudgetError("Malformed final request budget")
    for name in ("estimated_input_tokens", "input_limit_tokens", "context_token_budget"):
        if name in value and (type(value[name]) is not int or value[name] < 0):
            raise RequestBudgetError("Malformed final request budget integer")
    estimate = value.get("estimated_input_tokens")
    limit = value.get("input_limit_tokens")
    if limit is not None and (
        limit <= 0 or estimate is None or estimate > limit
    ):
        raise RequestBudgetError("Final request exceeds the input budget limit")
    measurement = value.get("measurement")
    if measurement is None:
        return False  # A legacy local estimate is not a measured image count.
    if not isinstance(measurement, Mapping):
        raise RequestBudgetError("Malformed final request budget measurement")
    if measurement.get("kind") != "provider_count":
        return False
    count = measurement.get("input_tokens")
    if (
        any(name not in value for name in (
            "estimated_input_tokens", "input_limit_tokens", "context_token_budget",
        ))
        or type(count) is not int or count < 0 or count > estimate
        or not isinstance(measurement.get("source"), str) or not measurement["source"]
    ):
        raise RequestBudgetError("Malformed measured final request budget")
    return True


async def _text_fallback(provider, request, options):
    method = getattr(provider, "media_text_fallback_witness", None)
    if not callable(method) or not _named(method, "request_options"):
        raise RequestBudgetError("Final media request budget is unavailable")
    value = await _call_with_options(method, request, options)
    if not isinstance(value, Mapping) or set(value) != {
        "v", "validated", "projection_digest", "messages",
    } or not _witness({k: v for k, v in value.items() if k != "messages"}):
        raise RequestBudgetError("Malformed media text fallback witness")
    messages = value["messages"]
    if not isinstance(messages, list) or not messages:
        raise RequestBudgetError("Malformed media text fallback messages")
    # This detached view is only proof of warning-only projection, not a count.
    for message in messages:
        if not isinstance(message, dict) or MEDIA in (message.get("metadata") or {}):
            raise RequestBudgetError("Non-text media fallback")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(block, dict) or block.get("type") != "text"
            or not isinstance(block.get("text"), str) for block in content
        ):
            raise RequestBudgetError("Non-text media fallback")


async def release_final_request(provider, request, options):
    try:
        release = getattr(provider, "release_media_request", None)
        if callable(release) and _named(release, "request_options"):
            return await invoke(release, request, request_options=deepcopy(options))
    except Exception:  # noqa: BLE001 - preserve cancellation and scrub private cleanup errors
        logger.warning("Refused media candidate release failed")
    return False


async def check_final_request(provider, request, options, context=None):
    original_options = deepcopy(options)
    try:
        await _check_final_request(provider, request, options, context)
    except asyncio.CancelledError:
        await _release_refused(provider, request, original_options)
        raise
    except RequestBudgetError:
        await _release_refused(provider, request, original_options)
        raise
    except Exception:  # noqa: BLE001 - provider count errors may contain image transport
        await _release_refused(provider, request, original_options)
        raise RequestBudgetError("Final request budget operation refused") from None


async def _release_refused(provider, request, options):
    await release_final_request(provider, request, options)


async def _check_final_request(provider, request, options, context):
    """Count the fully assembled object, then dispatch that unchanged object.

    Earlier context counts exclude loop overlays/tools/options. No signature
    guessing or TypeError retry is allowed after a budget operation was attempted.
    """
    before = _fingerprint(request, options)
    has_media = _media_input(request)
    method = getattr(provider, "request_budget", None)
    decision = None
    if callable(method):
        extra = {}
        if _accepts_context_estimate(method):
            extra["context_estimate"] = await _context_estimate(context, request)
        if _named(method, "request_options"):
            decision = await _call_with_options(method, request, options, **extra)
        elif options:
            raise RequestBudgetError("Provider cannot count effective request options")
        else:
            decision = await invoke(method, request, **extra)
    if _fingerprint(request, options) != before:
        raise RequestBudgetError("Final request changed while counting its budget")
    measured = _decision(decision)
    if has_media:
        if measured:
            method = getattr(provider, "media_budget_witness", None)
            if callable(method) and _named(method, "request_options"):
                witness = await _call_with_options(method, request, options)
            else:
                witness = decision.get(MEDIA)
            if not _witness(witness):
                raise RequestBudgetError("Final media request has no measured projection witness")
        else:
            await _text_fallback(provider, request, options)
    if _fingerprint(request, options) != before:
        raise RequestBudgetError("Final request changed while counting its budget")