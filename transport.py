"""Command Code ``/alpha/generate`` transport for the ``commandcode-alpha`` provider plugin.

This provider reaches models through an NDJSON endpoint — the same one the Command Code CLI uses
(``POST https://api.commandcode.ai/alpha/generate``) rather than the OpenAI-compatible
Provider API under ``/provider/v1``; the official CLI and the community bridges all speak
it, and it is what ``~/.commandcode/auth.json`` credentials are valid for.

This module is the whole transport: the request shape, the NDJSON parser, and an
OpenAI-client-shaped facade (``chat.completions.create``) so Hermes' agent loop and its
auxiliary tasks can drive it like any other provider — **from outside the tree**, using
``ProviderProfile.create_client`` (see ``__init__.py``), with no core edits.

Behavioural notes that come from the community bridge's field notes and our own smoke
runs, each one a bug we would otherwise ship:

* The relay can answer ``HTTP 200`` with an ``{"type": "error"}`` event inside the stream
  (``Insufficient Balance`` / 402 is the common one). That is a failure, not an empty
  completion; it is raised as :class:`CommandCodeAPIError` carrying the status code so the
  caller can fail over or surface it.
* A stream that ends after ``start`` with no ``text-delta``/``tool-call``/``finish`` is
  **fail-closed**: raising beats handing the agent a silently empty turn.
* ``stream=True`` yields OpenAI-style chunks (the agent loop streams tokens); ``stream=False``
  returns a complete response. Both are also awaitable so the async auxiliary path (the only
  async one is ``vision_analyze``) can consume them without the core rebuilding a real
  ``AsyncOpenAI`` against the wrong endpoint — see ``HERMES_SKIP_ASYNC_WRAP`` below.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import mimetypes
import re
import logging
import os
import platform
import urllib.error
import urllib.parse
import urllib.request
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

if __package__:
    from .credentials import env_token, allow_cli_fallback
    from .http_support import open_credentialed
else:  # Standalone fixtures also load this file outside its package.
    from credentials import env_token, allow_cli_fallback
    from http_support import open_credentialed

try:
    from .endpoints import api_origin
except ImportError:  # Standalone transport tests and consumers.
    from endpoints import api_origin

logger = logging.getLogger("plugins.commandcode_oauth.transport")

ALPHA_ORIGIN = "https://api.commandcode.ai"
ALPHA_PATH = "/alpha/generate"
ALPHA_GENERATE_URL = f"{ALPHA_ORIGIN}{ALPHA_PATH}"
DEFAULT_TIMEOUT_S = 180.0
CLI_AUTH_PATH = "~/.commandcode/auth.json"
# The CLI's own wire version: the relay uses it for capability gating.
CLI_WIRE_VERSION = "1.87.1"
DEFAULT_MAX_TOKENS = 64_000

# Dashed slugs the picker may hand us → canonical wire ids the endpoint accepts.
MODEL_ALIASES: Dict[str, str] = {
    "deepseek-deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-deepseek-v4-flash-vision-exp": "deepseek/deepseek-v4-flash-vision-exp",
    "deepseek-deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "meituan-LongCat-2.0:free": "meituan/LongCat-2.0:free",
    "meta-muse-spark-1.3-contributor": "meta/muse-spark-1.3-contributor",
    "MiniMaxAI-MiniMax-M3": "MiniMaxAI/MiniMax-M3",
    "moonshotai-Kimi-K3": "moonshotai/Kimi-K3",
    "poolside-laguna-s-2.1-free": "poolside/laguna-s-2.1-free",
    "Qwen-Qwen3.8-Max-0902": "Qwen/Qwen3.8-Max-0902",
    "xai-grok-4.5": "xai/grok-4.5",
    "xiaomi-mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "z-ai-glm-5.3-flash": "z-ai/glm-5.3-flash",
}


class CommandCodeAPIError(RuntimeError):
    """A failure from the relay, carrying enough for failover classification.

    ``status_code`` is the upstream status when known (HTTP status, or the ``statusCode``
    on an in-stream ``error`` event — 402 balance / 429 rate limit / 401 auth). The
    profile's ``classify_api_error`` maps it to Hermes' failover reasons.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """True when trying again (or another credential) can plausibly succeed."""
        code = self.status_code or 0
        return code in (0, 402, 408, 425, 429, 500, 502, 503, 504)


def canonical_model_id(model_id: str) -> str:
    """Map dashed slugs / prefixed names onto the endpoint's canonical ids."""
    clean = (model_id or "").strip()
    for prefix in ("command-code/", "commandcode-oauth/", "commandcode-alpha/", "commandcode/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    return MODEL_ALIASES.get(clean, clean)


def cli_token() -> str:
    """Read the access token the official Command Code CLI keeps in its auth file.

    Self-contained on purpose: an out-of-tree plugin must not import Hermes internals for
    this, and the file is exactly what the CLI itself writes (``apiKey``).
    """
    try:
        with open(os.path.expanduser(CLI_AUTH_PATH), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    for key in ("apiKey", "api_key", "accessToken", "access_token", "token"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# The vendor registry marks these text-only: pixels sent to them are dropped server-side, so an
# attached image becomes a text placeholder instead of vanishing. Unknown ids stay image-capable
# (the registry's own fallback), which is why this is a deny-list and not an allow-list.
_TEXT_ONLY_MODELS = frozenset({
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-fast",
    "zai-org/GLM-5.3", "zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast", "zai-org/GLM-5.1", "zai-org/GLM-5",
    "MiniMaxAI/MiniMax-M2.7", "minimax/minimax-m2.7-free", "MiniMaxAI/MiniMax-M2.5",
    "xiaomi/mimo-v2.5-pro", "Qwen/Qwen3.6-Max-Preview", "Qwen/Qwen3.7-Max",
    "meituan/LongCat-2.0:free", "stepfun/Step-3.5-Flash", "tencent/hy4-preview", "tencent/Hy3",
    "tencent/hy3-paid", "nvidia/nemotron-3-ultra-550b-a55b", "poolside/laguna-s-2.1-free",
    "inclusionai/ling-3.0-flash-free", "inclusionai/ling-3.0-flash-sante:free",
})


def _image_mime(url: str) -> str:
    """``image/png`` out of ``data:image/png;base64,…``; empty when it is not a data uri."""
    match = re.match(r"data:([^;,]+)", url or "")
    return match.group(1) if match else ""


def _media_url(part: Dict[str, Any]) -> tuple[str, str]:
    """``(kind, url)`` for any shape a caller sends — flat ``image``, OpenAI's nested
    ``image_url``, or an Anthropic ``source`` block."""
    part_type = str(part.get("type", ""))
    if part_type not in ("image", "image_url", "video", "video_url"):
        return part_type or "attachment", ""
    kind = "video" if "video" in part_type else "image"
    url = part.get(kind)
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        nested = part.get(f"{kind}_url")
        url = nested.get("url") if isinstance(nested, dict) else nested if isinstance(nested, str) else ""
    source = part.get("source")
    if not url and isinstance(source, dict):
        if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
            url = f"data:{source['media_type']};base64,{source['data']}"
        elif source.get("type") == "url":
            url = source.get("url")
    return kind, url if isinstance(url, str) else ""


def _image_placeholder(part: Dict[str, Any]) -> str:
    """What a text-only model (or an unusable part) sees instead of the pixels."""
    kind, url = _media_url(part)
    return f"[{kind}: {_image_mime(url) or 'attached'}]"



def _normalize_media_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One non-text content part → the wire shape (``{"type": "image", "image": url}``).

    Flatten OpenAI URLs and Anthropic source blocks with an explicit MIME type.
    """
    kind, url = _media_url(part)
    if isinstance(url, str) and url:
        # ``mimeType`` is what makes the endpoint actually *process* the pixels: the same part
        # without it is accepted and silently ignored (verified against a solid-colour image).
        source = part.get("source") or {}
        nested = part.get(f"{kind}_url") or {}
        mime = (_image_mime(url) or part.get("mimeType") or part.get("mime_type")
                or (source.get("media_type") if isinstance(source, dict) else "")
                or (nested.get("mimeType") if isinstance(nested, dict) else "")
                or mimetypes.guess_type(urllib.parse.urlsplit(url).path)[0])
        return {"type": kind, kind: url, "mimeType": mime} if mime else None
    return None


def format_messages(
    messages: List[Dict[str, Any]], model: str = ""
) -> tuple[str, List[Dict[str, Any]]]:
    """Split out the system prompt and convert messages to the wire format.

    ``model`` gates attachments: a text-only model gets a placeholder so the caller can still
    see that an image was attached, rather than a part the endpoint would drop.
    """
    images_ok = (model or "").strip().lower() not in {m.lower() for m in _TEXT_ONLY_MODELS}
    system_parts: List[str] = []
    wire_msgs: List[Dict[str, Any]] = []
    call_id_to_name: Dict[str, str] = {}
    pending_tool_images: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # All results in a parallel tool batch must precede the synthetic user images.
        if role != "tool" and pending_tool_images:
            wire_msgs.append({"role": "user", "content": pending_tool_images})
            pending_tool_images = []

        if role in ("system", "developer"):
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        elif role == "user":
            parts: List[Dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    else:
                        media = _normalize_media_part(part) if images_ok else None
                        parts.append(media or {"type": "text", "text": _image_placeholder(part)})
            wire_msgs.append({"role": "user", "content": parts})
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                parts.append({"type": "reasoning", "text": reasoning})
            if isinstance(content, list):
                parts.extend(part for part in content if isinstance(part, dict)
                             and part.get("type") in ("text", "reasoning"))
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for call in msg.get("tool_calls") or []:
                call_id = call.get("id") or str(uuid.uuid4())
                func = call.get("function") or {}
                name = func.get("name", "tool")
                call_id_to_name[call_id] = name
                args: Any = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                parts.append({"type": "tool-call", "toolCallId": call_id, "toolName": name, "input": args})
            wire_msgs.append({"role": "assistant", "content": parts or [{"type": "text", "text": ""}]})
        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            if isinstance(content, list):
                texts = []
                for part in content:
                    if not isinstance(part, dict):
                        texts.append(str(part))
                        continue
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    else:
                        media = _normalize_media_part(part) if images_ok else None
                        if media is not None:
                            pending_tool_images.extend([
                                {"type": "text", "text": f"Image returned by tool call {call_id}:"}, media])
                            texts.append("[Tool image follows in the next user message]")
                        else:
                            texts.append(_image_placeholder(part))
                val = "\n".join(texts)
            else:
                val = content if isinstance(content, str) else json.dumps(content)
            wire_msgs.append({
                "role": "tool",
                "content": [{
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "toolName": call_id_to_name.get(call_id, "tool"),
                    "output": {"type": "text", "value": val},
                }],
            })

    if pending_tool_images:
        wire_msgs.append({"role": "user", "content": pending_tool_images})
    return "\n\n".join(system_parts), wire_msgs





def format_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """OpenAI tool definitions → the wire ``input_schema`` form."""
    wire: List[Dict[str, Any]] = []
    for tool in tools or []:
        func = tool.get("function", {}) if isinstance(tool, dict) and tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        wire.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": func.get("parameters") or {"type": "object", "properties": {}},
        })
    return wire


def workspace_config(cwd: Optional[str] = None) -> Dict[str, Any]:
    """The ``config`` block the endpoint expects (a light workspace snapshot)."""
    return {
        "workingDir": cwd or os.getcwd(),
        "isGitRepo": False,
        "currentBranch": "",
        "mainBranch": "",
        "gitStatus": "",
        "recentCommits": [],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        # The official client reports the real host; a hardcoded "linux" is wrong on macOS
        # and can steer the model's shell assumptions.
        "environment": f"{platform.system().lower()}-{platform.machine()}, Python {platform.python_version()}",
        "structure": [],
    }


def _token_from_kwargs(api_key: Optional[str]) -> str:
    token = (api_key or "").strip() or env_token()
    return token or (cli_token() if allow_cli_fallback() else "")


def _request_body(api_kwargs: Dict[str, Any], model: str) -> Dict[str, Any]:
    system_prompt, wire_msgs = format_messages(api_kwargs.get("messages") or [], model)
    tools = api_kwargs.get("tools")
    tool_choice = api_kwargs.get("tool_choice")
    if tool_choice == "none":
        # Honour "no tools": forwarding them anyway lets the model call something the
        # caller explicitly ruled out.
        tools = None
    elif isinstance(tool_choice, dict) or tool_choice == "required":
        # The wire protocol has no forced-tool form; forward the tools and let the model
        # choose rather than failing the turn.
        logger.debug("commandcode: tool_choice=%r is not expressible on /alpha/generate; forwarding tools", tool_choice)

    params: Dict[str, Any] = {
        "model": model,
        "messages": wire_msgs,
        "tools": format_tools(tools),
        "system": system_prompt,
        "max_tokens": api_kwargs.get("max_tokens") or api_kwargs.get("max_completion_tokens") or DEFAULT_MAX_TOKENS,
        "stream": True,
    }
    temperature = api_kwargs.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        params["temperature"] = temperature

    if api_kwargs.get("reasoning_effort") is not None:
        params["reasoning_effort"] = api_kwargs["reasoning_effort"]

    return {
        "config": workspace_config(),
        "memory": None,
        "taste": None,
        "skills": None,
        "permissionMode": "standard",
        "mode": "agent",
        "params": params,
    }


def generate_url(base_url: Optional[str] = None) -> str:
    """``/alpha/generate`` on the account's API origin.

    ``base_url`` may arrive as the Provider-API form (``…/provider/v1``) that
    ``resolve_runtime_provider`` hands out, as a bare origin, or as a self-hosted proxy, so
    reduce it to its origin before appending the route.
    """
    return f"{api_origin(base_url)}{ALPHA_PATH}"


def coerce_timeout(value: Any, default: float = DEFAULT_TIMEOUT_S) -> float:
    """Accept every shape a caller may hand us as ``timeout``.

    The agent passes whatever ``httpx``/``openai`` would — a float, an ``httpx.Timeout`` or the
    OpenAI SDK's ``Timeout`` — so take the read (then total) component and fall back to the
    default instead of dying mid-generation with ``float() argument must be ... not 'Timeout'``.
    """
    for attr in ("read", "total", "timeout"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    if isinstance(value, (int, float)):
        return float(value)
    return default


@contextlib.contextmanager
def _open_stream(url: str, body: Dict[str, Any], token: str, timeout: float, *, headers=None, http_client=None, session_id=None):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "cli")
    req.add_header("x-command-code-version", CLI_WIRE_VERSION)
    req.add_header("x-cli-environment", "production")
    req.add_header("x-taste-learning", "false")
    req.add_header("x-co-flag", "false")
    req.add_header("x-session-id", session_id or str(uuid.uuid4()))
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    req.data = json.dumps(body).encode("utf-8")
    if http_client is not None:
        import httpx

        try:
            # Never inherit follow_redirects=True for a credential-bearing custom client.
            with http_client.stream("POST", url, content=req.data, headers=dict(req.header_items()),
                                    timeout=timeout, follow_redirects=False) as response:
                if response.status_code >= 300:
                    detail = response.read().decode("utf-8", "replace")[:500]
                    raise CommandCodeAPIError(f"Command Code HTTP {response.status_code}: {detail}",
                                              status_code=response.status_code)
                yield response.iter_lines()
        except httpx.HTTPError as exc:
            raise CommandCodeAPIError(f"Command Code connection failed: {exc}") from exc
        return
    try:
        with open_credentialed(req, timeout=timeout) as response:
            yield response
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")[:500]
        raise CommandCodeAPIError(
            f"Command Code HTTP {exc.code}: {detail or exc.reason}", status_code=int(exc.code)
        ) from exc
    except OSError as exc:  # DNS/TLS/reset: retryable, no status
        raise CommandCodeAPIError(f"Command Code connection failed: {exc}") from exc


def _iter_events(resp) -> Iterator[Dict[str, Any]]:
    """Yield decoded NDJSON events, raising on in-stream ``error`` events."""
    for raw_line in resp:
        line = (raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line).strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            logger.debug("commandcode: skipping non-JSON stream line: %.200s", line)
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "abort":
            raise CommandCodeAPIError("Command Code generation aborted", status_code=502)
        if event.get("type") == "error":
            err = event.get("error")
            # The wire puts the status at the top level and ``error`` is usually a plain
            # string: {"type":"error","statusCode":402,"error":"Insufficient Balance"}.
            # Reading only the nested dict dropped the status, so a 402 looked like a
            # generic failure and lost its failover classification.
            status = event.get("statusCode") or event.get("status_code")
            if isinstance(err, dict):
                message = err.get("message") or err.get("type") or "generation error"
                status = status or err.get("statusCode") or err.get("status_code")
            else:
                message = str(err or event.get("message") or "generation error")
            code = int(status) if str(status).isdigit() else None
            raise CommandCodeAPIError(f"Command Code stream error: {message}", status_code=code)
        yield event


def _finish_reason(event: Dict[str, Any], has_tools: bool) -> str:
    raw = event.get("rawFinishReason") or event.get("finishReason") or "stop"
    if raw in ("refusal", "content_filter", "content-filter"):
        return "content_filter"
    if has_tools:
        return "tool_calls"
    return {"tool-calls": "tool_calls", "tool_use": "tool_calls",
            "max_tokens": "length", "end_turn": "stop"}.get(raw, raw)



def _usage_from_event(event: Dict[str, Any], previous: Optional[SimpleNamespace] = None) -> Optional[SimpleNamespace]:
    """Map the wire's ``totalUsage`` onto an OpenAI-shaped usage object (cache included).

    ``inputTokens`` is the whole prompt; the cached share rides in
    ``inputTokenDetails.cacheReadTokens`` and is billed (and reported here) separately.
    """
    raw = event.get("totalUsage") or event.get("usage") or {}
    if not isinstance(raw, dict) or not raw:
        return previous
    in_tok = raw.get("inputTokens", raw.get("prompt_tokens", getattr(previous, "prompt_tokens", 0)))
    out_tok = raw.get("outputTokens", raw.get("completion_tokens", getattr(previous, "completion_tokens", 0)))
    usage = SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok, total_tokens=in_tok + out_tok)
    details = raw.get("inputTokenDetails") or {}
    old_details = getattr(previous, "prompt_tokens_details", None)
    cache_read = details.get("cacheReadTokens", raw.get("cacheReadTokens", raw.get(
        "cachedInputTokens", getattr(old_details, "cached_tokens", 0))))
    cache_write = details.get("cacheWriteTokens", raw.get("cacheWriteTokens", getattr(old_details, "cache_write_tokens", 0)))
    if cache_read or cache_write:
        usage.prompt_tokens_details = SimpleNamespace(cached_tokens=cache_read, cache_write_tokens=cache_write)
    return usage


def _chunk(response_id: str, model: str, *, content=None, reasoning=None, tool_calls=None,
           finish_reason=None, usage=None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id=response_id, object="chat.completion.chunk", model=model,
                           choices=[choice], usage=usage)


def _response(response_id: str, model: str, content, reasoning, tool_calls, finish_reason, usage) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None,
                              reasoning_content=reasoning)
    return SimpleNamespace(id=response_id, object="chat.completion", model=model,
                           choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
                           usage=usage)


class _Stream:
    """``stream=True`` result: a sync chunk iterator that is also awaitable.

    Iterating yields OpenAI-style chunks (the agent loop streams tokens). ``await`` — the
    auxiliary path does ``await client.chat.completions.create(...)`` — resolves to the
    aggregated complete response, off the event loop, so no core rewrite is needed.
    Applying ``hasattr(value, "choices")`` to *this* object must stay False, or the agent
    loop would treat the stream as already-complete; it is therefore a plain iterator.
    """

    def __init__(self, run_stream, run_complete, kwargs) -> None:
        self._run_stream = run_stream
        self._run_complete = run_complete
        self._kwargs = kwargs

    def __iter__(self) -> Iterator[Any]:
        return self._run_stream(self._kwargs)

    def __await__(self):
        async def _go():
            import asyncio

            return await asyncio.to_thread(self._run_complete, self._kwargs)

        return _go().__await__()


class _Call:
    """``stream=False`` result: lazily resolved, attribute-accessible and awaitable."""

    def __init__(self, run_complete, kwargs) -> None:
        self._run_complete = run_complete
        self._kwargs = kwargs
        self._value: Any = None

    def _resolve(self) -> Any:
        if self._value is None:
            self._value = self._run_complete(self._kwargs)
        return self._value

    def __await__(self):
        async def _go():
            import asyncio

            return await asyncio.to_thread(self._resolve)

        return _go().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


class CommandCodeAlphaClient:
    """OpenAI-client-shaped facade over ``/alpha/generate``.

    Declares the two opt-outs core looks for so it never re-dispatches this client through
    a wire adapter (``HERMES_SKIP_TRANSPORT_WRAP``) nor rebuilds an ``AsyncOpenAI`` against
    the wrong endpoint for async auxiliary tasks (``HERMES_SKIP_ASYNC_WRAP``) — the async
    path gets this same object and awaits it.
    """

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 default_headers: dict | None = None, timeout: float | None = None,
                 http_client: Any = None, **_: Any) -> None:
        self.api_key = _token_from_kwargs(api_key)
        self.base_url = str(base_url or ALPHA_ORIGIN)
        # Honour a user-set base_url (proxy / self-hosted) instead of pinning the public origin.
        self._generate_url = generate_url(self.base_url)
        self._default_headers = dict(default_headers or {})
        self._http_client = http_client
        self._session_id = str(uuid.uuid4())
        self._timeout = coerce_timeout(timeout)
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    # ── OpenAI-compatible surface ────────────────────────────────────────────
    def _create_chat_completion(self, *, stream: bool = False, **kwargs: Any) -> Any:
        if stream:
            return _Stream(self._iter_stream, self._complete, kwargs)
        return _Call(self._complete, kwargs)

    def list_models(self, **_: Any) -> list[str] | None:
        return None

    def close(self) -> None:
        self.is_closed = True

    # ── transport ────────────────────────────────────────────────────────────
    def _prepare(self, api_kwargs: Dict[str, Any], token: Optional[str] = None):
        model = canonical_model_id(api_kwargs.get("model") or "")
        if not model:
            raise CommandCodeAPIError("Command Code request is missing a model")
        token = (token or self.api_key or "").strip()
        if not token:
            raise CommandCodeAPIError(
                "Command Code CLI token missing — run `commandcode` once or `hermes auth add commandcode-oauth`",
                status_code=401,
            )
        timeout = coerce_timeout(api_kwargs.get("timeout"), self._timeout)
        response_id = f"cmdcode-{uuid.uuid4().hex[:12]}"
        return model, token, timeout, response_id

    def _iter_stream(self, api_kwargs: Dict[str, Any]) -> Iterator[Any]:
        model, token, timeout, response_id = self._prepare(api_kwargs)
        body = _request_body(api_kwargs, model)
        session = (api_kwargs.get("extra_body") or {}).get("commandcode_session_id")
        if session:
            try:
                session = str(uuid.UUID(str(session)))
            except ValueError:
                session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-commandcode:{session}"))
        else:
            session = self._session_id
        body["threadId"] = session
        headers = {**self._default_headers, **(api_kwargs.get("extra_headers") or {})}
        started = False
        finished = False
        finish_event: Dict[str, Any] = {}
        cache_write_tokens = 0
        usage = None
        tool_calls: List[Any] = []
        with _open_stream(self._generate_url, body, token, timeout,
                          session_id=session, headers=headers, http_client=self._http_client) as resp:
            for event in _iter_events(resp):
                ev_type = event.get("type")
                if ev_type == "text-delta":
                    text = event.get("text", "")
                    if text:
                        started = True
                        yield _chunk(response_id, model, content=text)
                elif ev_type == "reasoning-delta":
                    text = event.get("text", "")
                    if text:
                        started = True
                        yield _chunk(response_id, model, reasoning=text)
                elif ev_type == "tool-call":
                    started = True
                    call_id = event.get("toolCallId") or str(uuid.uuid4())
                    raw_input = event.get("input", event.get("args", {}))
                    args = raw_input if isinstance(raw_input, str) else json.dumps(raw_input)
                    tool_calls.append(SimpleNamespace(
                        index=len(tool_calls), id=call_id, type="function",
                        function=SimpleNamespace(name=event.get("toolName", "tool"), arguments=args),
                    ))
                    yield _chunk(response_id, model, tool_calls=[tool_calls[-1]])
                elif ev_type in ("finish", "finish-step"):
                    usage = _usage_from_event(event, usage)
                    if ev_type == "finish":
                        finished = True
                        finish_event = event
                elif ev_type == "cache-write-tokens":
                    value = event.get("cacheWriteTokens")
                    if isinstance(value, (int, float)) and value >= 0:
                        cache_write_tokens = value
        if not started or not finished:
            # "200 OK" whose stream stopped early: an empty turn is worse than an error.
            raise CommandCodeAPIError(
                "Command Code returned an empty/incomplete stream"
                f" (started={started}, finished={finished}, tools={len(tool_calls)})",
                status_code=502,
            )

        if usage is not None and cache_write_tokens:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is None:
                details = SimpleNamespace(cached_tokens=0, cache_write_tokens=0)
                usage.prompt_tokens_details = details
            if not details.cache_write_tokens:
                details.cache_write_tokens = cache_write_tokens
        yield _chunk(response_id, model, finish_reason=_finish_reason(finish_event, bool(tool_calls)),
                     usage=usage)


    def _complete(self, api_kwargs: Dict[str, Any]) -> SimpleNamespace:
        """Run the stream to completion and aggregate into one response object."""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Any] = []
        usage = None
        finish_reason = "stop"
        for chunk in self._iter_stream(api_kwargs):
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            tool_calls.extend(delta.tool_calls or [])
            usage = chunk.usage or usage
            finish_reason = choice.finish_reason or finish_reason
        return _response(chunk.id, chunk.model, "".join(content_parts) or None,
                         "".join(reasoning_parts) or None, tool_calls, finish_reason, usage)
