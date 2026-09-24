"""Tests for the ``commandcode-alpha`` provider plugin.

Hermetic: no network. The transport is pointed at a local NDJSON server through
``base_url`` (which is why ``generate_url()`` honours an override), and every request the
client makes is captured so the body can be asserted.

Run either way, from the plugin directory::

    python -m unittest discover -s tests -v
    pytest tests -q

``pool_provider`` needs a Hermes tree on ``sys.path`` (it consults the provider registry to
canonicalise an alias); point ``HERMES_TREE`` at one to exercise that case, otherwise it is
skipped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

import auth  # noqa: E402
import errors  # noqa: E402
import transport  # noqa: E402

HERMES_TREE = os.environ.get("HERMES_TREE", str(Path.home() / ".hermes" / "hermes-agent"))
if HERMES_TREE and Path(HERMES_TREE).is_dir() and HERMES_TREE not in sys.path:
    sys.path.append(HERMES_TREE)
try:  # pragma: no cover - depends on the environment
    import providers  # noqa: F401

    HAS_HERMES_TREE = True
except Exception:  # pragma: no cover
    HAS_HERMES_TREE = False


def _line(**event) -> bytes:
    return (json.dumps(event) + "\n").encode()


def _text(text: str) -> bytes:
    return _line(type="text-delta", text=text)


FINISH_OK = _line(
    type="finish",
    finishReason="stop",
    totalUsage={
        "inputTokens": 7596,
        "inputTokenDetails": {"noCacheTokens": 44, "cacheReadTokens": 7552},
        "outputTokens": 2,
        "totalTokens": 7598,
    },
)


class _Handler(BaseHTTPRequestHandler):
    """Serves one canned NDJSON payload and records the request it received."""

    payload = b""
    captured: dict = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        type(self).captured = {
            "path": self.path,
            "body": json.loads(raw.decode("utf-8") or "{}"),
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, *args):  # keep the test output clean
        return


class PluginTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def client(self, **kw):
        # api_key is passed explicitly so the tests never read the real CLI credential file.
        return transport.CommandCodeAlphaClient(api_key="test-token", base_url=self.origin, **kw)

    def call(self, payload: bytes, **kwargs):
        _Handler.payload = payload
        _Handler.captured = {}
        result = self.client().chat.completions.create(
            model="deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            **kwargs,
        )
        if not kwargs.get("stream"):
            result.choices  # the facade is lazy: reading it is what sends the request
        return result

    # ── transport ────────────────────────────────────────────────────────────
    def test_complete_response_maps_cache_usage(self):
        response = self.call(_text("ok") + FINISH_OK)
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(response.usage.prompt_tokens, 7596)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 7552)  # ← the whole point
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_stream_yields_chunks_then_finish_with_usage(self):
        chunks = list(self.call(_text("he") + _text("llo") + FINISH_OK, stream=True))
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks), "hello")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[-1].usage.prompt_tokens_details.cached_tokens, 7552)

    def test_empty_stream_fails_closed(self):
        """A stream that stops after ``start`` must raise, not look like an empty answer."""
        with self.assertRaises(transport.CommandCodeAPIError):
            self.call(_line(type="start") + _line(type="finish", finishReason="stop"))

    def test_in_stream_error_keeps_status_code(self):
        payload = _line(type="error", statusCode=402, error="Insufficient Balance")
        with self.assertRaises(transport.CommandCodeAPIError) as ctx:
            self.call(payload)
        self.assertEqual(ctx.exception.status_code, 402)
        self.assertTrue(ctx.exception.retryable)

    def test_accepts_httpx_style_timeout(self):
        """The agent hands us an httpx/OpenAI ``Timeout``; that must not kill the call.

        Regression: ``float(timeout)`` raised ``TypeError`` on the official core, which passes
        a ``Timeout`` object rather than a number.
        """

        class _FakeTimeout:  # httpx.Timeout / openai.Timeout shape
            read = 42.0
            connect = 5.0

        self.assertEqual(transport.coerce_timeout(_FakeTimeout()), 42.0)
        self.assertEqual(transport.coerce_timeout(7.5), 7.5)
        self.assertEqual(transport.coerce_timeout(None), transport.DEFAULT_TIMEOUT_S)
        self.assertEqual(transport.coerce_timeout(object()), transport.DEFAULT_TIMEOUT_S)
        # …and the client survives being constructed and called with one.
        _Handler.payload = _text("ok") + FINISH_OK
        client = transport.CommandCodeAlphaClient(
            api_key="test-token", base_url=self.origin, timeout=_FakeTimeout()
        )
        response = client.chat.completions.create(
            model="deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            timeout=_FakeTimeout(),
        )
        self.assertEqual(response.choices[0].message.content, "ok")

    def test_awaitable_response_for_async_aux(self):
        """The async aux path does ``await client.chat.completions.create(...)``.

        We declare ``HERMES_SKIP_ASYNC_WRAP`` (the core would otherwise rebuild a real
        ``AsyncOpenAI`` against ``/provider/v1``, which 403s for these credentials), so the sync
        facade has to be awaitable *and* satisfy the core's completed-response predicate
        (``hasattr(value, "choices")``) — including for a ``stream=True`` request.
        """
        import asyncio

        async def _await(call):
            # Exactly what the core does: ``await client.chat.completions.create(...)`` inside a
            # coroutine (``asyncio.run`` alone would demand a coroutine object, not an awaitable).
            return await call

        client = self.client()
        _Handler.payload = _text("ok") + FINISH_OK
        response = asyncio.run(
            _await(
                client.chat.completions.create(
                    model="deepseek/deepseek-v4.1-flash",
                    messages=[{"role": "user", "content": "hi"}],
                )
            )
        )
        self.assertEqual(response.choices[0].message.content, "ok")

        _Handler.payload = _text("ok") + FINISH_OK
        streamed = asyncio.run(
            _await(
                client.chat.completions.create(
                    model="deepseek/deepseek-v4.1-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
            )
        )
        self.assertEqual(streamed.choices[0].message.content, "ok")

    def test_tool_choice_none_drops_tools(self):
        """``tool_choice="none"`` must empty the tool list.

        We send ``tools: []`` rather than omitting the key: the relay accepts an empty array
        (every no-tool call we have made proves it), and keeping the key avoids depending on
        how it treats a missing one.
        """
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        self.call(_text("ok") + FINISH_OK, tools=tools, tool_choice="none")
        self.assertEqual(_Handler.captured["body"]["params"]["tools"], [])

    def test_tools_are_forwarded_without_tool_choice(self):
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        self.call(_text("ok") + FINISH_OK, tools=tools)
        self.assertEqual(_Handler.captured["body"]["params"]["tools"][0]["name"], "t")

    def test_request_reports_real_environment(self):
        self.call(_text("ok") + FINISH_OK)
        environment = _Handler.captured["body"]["config"]["environment"]
        self.assertIn(sys.platform.replace("darwin", "darwin"), environment.lower())
        self.assertNotEqual(environment, "linux")
        self.assertIn("Python", environment)

    def test_request_carries_the_cli_headers(self):
        self.call(_text("ok") + FINISH_OK)
        headers = _Handler.captured["headers"]
        self.assertEqual(headers["user-agent"], "cli")
        self.assertEqual(headers["authorization"], "Bearer test-token")
        self.assertTrue(headers["x-session-id"])

    def test_generate_url_honours_base_url(self):
        self.assertEqual(
            transport.generate_url("https://api.commandcode.ai/provider/v1"),
            "https://api.commandcode.ai/alpha/generate",
        )
        self.assertEqual(transport.generate_url("http://127.0.0.1:9999"), "http://127.0.0.1:9999/alpha/generate")
        self.assertEqual(transport.generate_url(""), transport.ALPHA_GENERATE_URL)

    def test_canonical_model_id_accepts_prefixed_and_bare_ids(self):
        self.assertEqual(
            transport.canonical_model_id("command-code/deepseek-deepseek-v4-flash"),
            "deepseek/deepseek-v4-flash",
        )
        self.assertEqual(transport.canonical_model_id("deepseek/deepseek-v4.1-flash"), "deepseek/deepseek-v4.1-flash")


class ErrorClassificationTest(unittest.TestCase):
    def classify(self, status=None, message=""):
        return errors.classify_api_error(
            transport.CommandCodeAPIError(message, status_code=status),
            status_code=status,
            message=message,
        )

    def test_billing_covers_the_in_stream_402(self):
        self.assertEqual(self.classify(402, "Insufficient Balance"), {"reason": "billing"})

    def test_status_table(self):
        self.assertEqual(self.classify(401)['reason'], "auth")
        self.assertEqual(self.classify(403)['reason'], "auth")
        self.assertEqual(self.classify(429)['reason'], "rate_limit")
        self.assertEqual(self.classify(500)['reason'], "server_error")
        self.assertEqual(self.classify(503)['reason'], "overloaded")
        self.assertEqual(self.classify(504)['reason'], "timeout")

    def test_unknown_is_declined(self):
        self.assertIsNone(self.classify(None, "something else"))

    def test_context_overflow_wording(self):
        self.assertEqual(self.classify(400, "maximum context length exceeded")['reason'], "context_overflow")


class AuthTest(unittest.TestCase):
    @unittest.skipUnless(HAS_HERMES_TREE, "needs a Hermes tree on sys.path (providers)")
    def test_pool_provider_canonicalises_aliases(self):
        self.assertEqual(auth.pool_provider(SimpleNamespace(provider="commandcode-oauth")), "commandcode-alpha")
        self.assertEqual(auth.pool_provider(SimpleNamespace(provider="commandcode-alpha")), "commandcode-alpha")

    def test_pool_provider_falls_back_to_the_requested_name(self):
        self.assertEqual(auth.pool_provider(SimpleNamespace(provider="")), "commandcode-alpha")

    def test_read_cli_token_raises_when_absent(self):
        original = auth.CLI_AUTH_PATH
        auth.CLI_AUTH_PATH = Path("/nonexistent/commandcode/auth.json")
        try:
            with self.assertRaises(auth.CommandCodeAuthError):
                auth.read_cli_token()
        finally:
            auth.CLI_AUTH_PATH = original

    def test_refresh_credential_reports_missing_grant(self):
        original = auth.CLI_AUTH_PATH
        auth.CLI_AUTH_PATH = Path("/nonexistent/commandcode/auth.json")
        try:
            with self.assertRaises(auth.CommandCodeAuthError):
                auth.refresh_credential(SimpleNamespace(provider="commandcode-alpha"))
        finally:
            auth.CLI_AUTH_PATH = original


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ImageWireFormatTests(unittest.TestCase):
    """``/alpha`` needs ``mimeType`` on an image part, and a placeholder for text-only models.

    Without ``mimeType`` the endpoint accepts the part and silently ignores the pixels — a solid
    blue 64x64 answered "White" until the field was added.
    """

    DATA_URL = "data:image/png;base64,iVBORw0KGgo="

    def _body(self, model: str):
        return transport._request_body({
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "colour?"},
                {"type": "image_url", "image_url": {"url": self.DATA_URL}}]}],
        }, model)

    def test_image_part_carries_mime_type(self):
        part = self._body("Qwen/Qwen3.8-Omni-Flash")["params"]["messages"][0]["content"][1]
        self.assertEqual(part["type"], "image")
        self.assertEqual(part["mimeType"], "image/png")

    def test_text_only_model_gets_a_placeholder(self):
        content = self._body("deepseek/deepseek-v4-pro")["params"]["messages"][0]["content"]
        self.assertNotIn("image", [p.get("type") for p in content])
        self.assertTrue(any("image" in str(p.get("text", "")) for p in content))
