"""Command Code failure classification for the ``commandcode-alpha`` provider plugin.

``ProviderProfile.classify_api_error`` is consulted (for this provider's failures only)
by ``agent.error_classifier`` before the built-in pipeline, so returning a reason here is
how the retry/failover machinery learns what a Command Code failure means. The cases that
matter are the ones the community bridge documents as non-obvious on this endpoint:

* ``402 Insufficient Balance`` arrives as an **error event inside a 200 stream** — our
  transport turns it into a ``CommandCodeAPIError`` with the status attached, and this hook
  maps it to ``billing`` so the pool parks the credential instead of retrying forever;
* throttling is a plain ``429`` (``rate_limit``), and the model-scoped "this account cannot
  use that model" case is left to the generic pipeline;
* a stream that ends without any visible output is raised as a 502 by the transport
  (fail closed), which lands on ``server_error`` → retry, not a silent empty answer.

Return ``{"reason": <FailoverReason name>}`` to override, ``None`` to decline and let the
generic classifier decide.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _status_of(error: Any, status_code: Any) -> Optional[int]:
    for candidate in (getattr(error, "status_code", None), status_code):
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def classify_api_error(
    error: Any,
    *,
    status_code: Any = None,
    error_code: Any = None,
    message: Any = None,
    body: Any = None,
    model: Any = None,
) -> Optional[Mapping[str, Any]]:
    """Map a Command Code failure onto a Hermes ``FailoverReason`` (or decline with None)."""
    code = _status_of(error, status_code)
    text = f"{message or ''} {body or ''} {error or ''}".lower()

    if code == 402 or "insufficient balance" in text or "insufficient_balance" in text:
        return {"reason": "billing"}
    if code == 401:
        return {"reason": "auth"}
    if code == 403:
        # The alpha endpoint answers 403 when a credential has no access to the requested
        # model/route; the key itself may be fine, but from here it is an auth-shaped stop.
        return {"reason": "auth"}
    if code == 429:
        return {"reason": "rate_limit"}
    if code in (503, 529):
        return {"reason": "overloaded"}
    if code in (500, 502):
        return {"reason": "server_error"}
    if code in (504, 408):
        return {"reason": "timeout"}
    if code == 404:
        return {"reason": "model_not_found"}
    if code == 413:
        return {"reason": "payload_too_large"}
    if code == 400:
        if "tool_choice" in text or "tool choice" in text:
            return {"reason": "format_error"}
        if "context" in text or "too long" in text or "maximum" in text:
            return {"reason": "context_overflow"}
    return None
