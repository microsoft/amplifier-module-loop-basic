"""Final-candidate budget admission; no request hydration or provider imports."""

import inspect
import json
from collections.abc import Mapping
from copy import deepcopy

from ._media import MEDIA, invoke


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


def _fingerprint(request, options):
    return json.dumps(
        [request.model_dump(mode="json"), options],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


async def _call_with_options(method, request, options):
    forwarded = deepcopy(options)
    before = _fingerprint(request, forwarded)
    value = await invoke(method, request, request_options=forwarded)
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


async def check_final_request(provider, request, options):
    try:
        await _check_final_request(provider, request, options)
    except RequestBudgetError:
        raise
    except Exception:  # noqa: BLE001 - provider count errors may contain image transport
        raise RequestBudgetError("Final request budget operation refused") from None


async def _check_final_request(provider, request, options):
    """Count the fully assembled object, then dispatch that unchanged object.

    Earlier context counts exclude loop overlays/tools/options. No signature
    guessing or TypeError retry is allowed after a budget operation was attempted.
    """
    before = _fingerprint(request, options)
    has_media = _media_input(request)
    method = getattr(provider, "request_budget", None)
    decision = None
    if callable(method):
        if _named(method, "request_options"):
            decision = await _call_with_options(method, request, options)
        elif options:
            raise RequestBudgetError("Provider cannot count effective request options")
        else:
            decision = await invoke(method, request)
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