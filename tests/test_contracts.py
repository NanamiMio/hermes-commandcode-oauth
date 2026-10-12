"""Provider regressions through real HTTP and Hermes consumers; fixture keys only."""
import contextlib
import io
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import test_plugin as fixtures
import auth
import transport


class TransportContractTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.PluginTestCase.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.PluginTestCase.tearDownClass.__func__)
    client = fixtures.PluginTestCase.client

    def test_custom_headers_reach_the_relay(self):
        fixtures._Handler.payload = fixtures._text("ok") + fixtures.FINISH_OK
        response = self.client(default_headers={
            "X-Proxy-Key": "fixture-proxy-key", "User-Agent": "fixture-client"}).chat.completions.create(
                model="vision/model", messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(response.choices[0].message.content, "ok")
        headers = fixtures._Handler.captured["headers"]
        self.assertEqual(headers.get("x-proxy-key"), "fixture-proxy-key")
        self.assertEqual(headers["user-agent"], "fixture-client")

    @unittest.skipUnless(fixtures.HAS_HERMES_TREE, "httpx comes from Hermes")
    def test_supplied_http_client_is_used_for_sync_stream_and_async(self):
        import asyncio
        import httpx
        requests = []

        def relay(request):
            requests.append(request)
            return httpx.Response(200, content=fixtures._text("ok") + fixtures.FINISH_OK)

        with httpx.Client(transport=httpx.MockTransport(relay)) as http:
            client = self.client(http_client=http, default_headers={"X-Proxy-Key": "fixture-key"})
            self.assertEqual(client.chat.completions.create(model="vision/model", messages=[]).choices[0].message.content, "ok")
            self.assertEqual(list(client.chat.completions.create(model="vision/model", messages=[], stream=True))[-1].choices[0].finish_reason, "stop")

            async def call():
                return await client.chat.completions.create(model="vision/model", messages=[])

            self.assertEqual(asyncio.run(call()).choices[0].message.content, "ok")
            client.close()
            self.assertFalse(http.is_closed)  # Caller retains ownership of its client.
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(req.headers["x-proxy-key"] == "fixture-key" for req in requests))

    def test_tool_screenshot_stays_media_and_keeps_call_identity(self):
        _, wire = transport.format_messages([
            {"role": "assistant", "tool_calls": [{"id": "shot-1", "function": {
                "name": "screenshot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "shot-1", "content": [
                {"type": "text", "text": "Screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        ], "Qwen/Qwen3.8-Omni-Flash")
        result = wire[-1]["content"][0]
        self.assertEqual((result["toolCallId"], result["toolName"]), ("shot-1", "screenshot"))
        self.assertEqual(result["output"], {"type": "content", "value": [
            {"type": "text", "text": "Screenshot"},
            {"type": "media", "data": "AAAA", "mediaType": "image/png"}]})

    def test_text_only_tool_result_has_a_placeholder_without_base64(self):
        _, wire = transport.format_messages([{"role": "tool", "tool_call_id": "shot-1", "content": [
            {"type": "text", "text": "Screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
            "deepseek/deepseek-v4-pro")
        output = wire[0]["content"][0]["output"]
        self.assertEqual(output["type"], "text")
        self.assertIn("Screenshot", output["value"])
        self.assertIn("image", output["value"])
        self.assertNotIn("AAAA", output["value"])

    @unittest.skipUnless(fixtures.HAS_HERMES_TREE, "httpx comes from Hermes")
    def test_supplied_client_does_not_follow_credential_redirects(self):
        import httpx
        requests = []

        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={"Location": "https://another-origin.test/result"})

        with httpx.Client(transport=httpx.MockTransport(redirect), follow_redirects=True) as http:
            response = self.client(http_client=http).chat.completions.create(model="vision/model", messages=[])
            with self.assertRaises(transport.CommandCodeAPIError) as error:
                response.choices
            self.assertEqual(error.exception.status_code, 302)
        self.assertEqual(len(requests), 1)

    def test_remote_tool_image_and_completion_budget_are_preserved(self):
        body = transport._request_body({"max_completion_tokens": 8192, "messages": [
            {"role": "tool", "tool_call_id": "shot", "content": [
                {"type": "image_url", "image_url": {"url": "https://images.test/screenshot.png"}}]}]}, "vision/model")
        self.assertEqual(body["params"]["max_tokens"], 8192)
        self.assertEqual(body["params"]["messages"][0]["content"][0]["output"], {
            "type": "content", "value": [{"type": "image-url", "url": "https://images.test/screenshot.png"}]})


class RedirectContractTest(unittest.TestCase):
    def test_bearer_is_retained_only_on_the_original_origin(self):
        received = []

        class Sink(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')
            def log_message(self, *args):
                pass

        sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
        target = f"http://127.0.0.1:{sink.server_port}/result"

        class Source(Sink):
            def do_GET(self):
                if self.path == "/result":
                    return super().do_GET()
                self.send_response(302)
                self.send_header("Location", "/result" if self.path == "/same" else target)
                self.end_headers()

        source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
        for server in (source, sink):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            module = fixtures.load_profile() if fixtures.HAS_HERMES_TREE else None
            origin = f"http://127.0.0.1:{source.server_port}"
            methods = [lambda url: auth.validate("fixture-token", timeout=2)]
            if module:
                methods.append(lambda url: module._http_json(url, "fixture-token", timeout=2))
            for method in methods:
                for path, expected in (("/same", "Bearer fixture-token"), ("/other", None)):
                    with patch.object(auth, "WHOAMI_URL", origin + path):
                        self.assertIsNotNone(method(origin + path))
                    headers = {k.lower(): v for k, v in received[-1].items()}
                    self.assertEqual(headers.get("authorization"), expected)
        finally:
            for server in (source, sink):
                server.shutdown()
                server.server_close()

    def test_generation_redirect_drops_bearer_and_custom_credentials(self):
        received = []

        class Sink(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append({k.lower(): v for k, v in self.headers.items()})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(fixtures._text("ok") + fixtures.FINISH_OK)
            def log_message(self, *args):
                pass

        sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)

        class Redirect(Sink):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/result")
                self.end_headers()

        source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        for server in (source, sink):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = transport.CommandCodeAlphaClient(api_key="fixture-token",
                base_url=f"http://127.0.0.1:{source.server_port}",
                default_headers={"X-Proxy-Key": "fixture-proxy-key"})
            response = client.chat.completions.create(model="vision/model", messages=[])
            self.assertEqual(response.choices[0].message.content, "ok")
            self.assertNotIn("authorization", received[0])
            self.assertNotIn("x-proxy-key", received[0])
        finally:
            for server in (source, sink):
                server.shutdown()
                server.server_close()


@unittest.skipUnless(fixtures.HAS_HERMES_TREE, "requires Hermes consumers")
class HermesContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = fixtures.load_profile()
        cls.profile = cls.module.commandcode_oauth

    def test_usage_snapshot_round_trips_through_picker_cache(self):
        from agent.account_usage import fetch_account_usage
        from hermes_cli.inventory import _apply_usage
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HERMES_HOME": home}), \
                patch.object(self.module, "_http_json", side_effect=[{}, {
                    "windowLimits": {"weekly": {"used": 2, "cap": 10}}}, {}]):
            snapshot = fetch_account_usage(self.profile.name, api_key="fixture-token")
            self.assertIsNotNone(snapshot)
            row = {"slug": self.profile.name, "authenticated": True}
            with patch("agent.account_usage_cache.refresh_account_usage_async"), \
                    patch("agent.account_usage_cache.refresh_account_usage_entries_async"):
                _apply_usage([row])
            self.assertEqual(row["usage"]["windows"], [{
                "label": "Weekly", "used_percent": 20.0, "resets_at": None, "scope": "account"}])

    def test_core_receives_terminal_and_compression_hints(self):
        from agent.error_classifier import classify_api_error
        for code, message in ((401, "Unauthorized"), (402, "Insufficient Balance"),
                              (403, "The free model tier has been retired"), (404, "Model not found")):
            with self.subTest(code=code):
                result = classify_api_error(transport.CommandCodeAPIError(message, status_code=code),
                                            provider=self.profile.name)
                self.assertFalse(result.retryable)
                self.assertTrue(result.should_fallback)
        for code, message in ((400, "maximum context length exceeded"), (413, "payload too large")):
            with self.subTest(code=code):
                result = classify_api_error(transport.CommandCodeAPIError(message, status_code=code),
                                            provider=self.profile.name)
                self.assertTrue(result.should_compress)

    def test_active_profile_token_wins_over_process_environment(self):
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        scope = set_secret_scope({"COMMANDCODE_CLI_TOKEN": "profile-token"})
        try:
            with patch.dict(os.environ, {"COMMANDCODE_CLI_TOKEN": "process-token"}), \
                    patch.object(auth, "validate", return_value={"user_id": "fixture-user"}):
                self.assertEqual(self.profile._credential(), "profile-token")
                self.assertEqual(transport.CommandCodeAlphaClient().api_key, "profile-token")
                self.assertEqual(auth.login()["token"], "profile-token")
        finally:
            reset_secret_scope(scope)

    def test_routed_profile_does_not_inherit_the_host_cli_account(self):
        from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
        scope = set_secret_scope({})
        routed = set_multiplex_context(True)
        try:
            with patch("hermes_cli.auth.read_credential_pool", return_value=[]), \
                    patch.object(self.module, "read_cli_token", side_effect=AssertionError("Host account leaked")), \
                    patch.object(transport, "cli_token", side_effect=AssertionError("Host account leaked")):
                self.assertEqual(self.profile._credential(), "")
                self.assertEqual(transport.CommandCodeAlphaClient().api_key, "")
        finally:
            reset_multiplex_context(routed)
            reset_secret_scope(scope)

    def test_static_key_refresh_refuses_an_account_switch(self):
        entry = SimpleNamespace(access_token="old-token", extra={"commandcode": {
            "source": "commandcode-cli", "user_id": "original-account"}})
        with patch.object(auth, "read_cli_token", return_value="new-token"), \
                patch.object(auth, "validate", return_value={"user_id": "other-account"}):
            with self.assertRaises(auth.CommandCodeAuthError):
                auth.refresh_credential(entry)

    def test_refresh_command_preserves_browser_account_and_other_rows(self):
        from hermes_cli.auth_commands import auth_refresh_command
        from hermes_cli.auth import read_credential_pool
        package_auth = self.module.auth
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {
            "HERMES_HOME": home, "COMMANDCODE_CLI_TOKEN": ""}), \
                patch.object(package_auth, "login", side_effect=[
                    {"token": "browser-one", "source": "commandcode-studio", "user_id": "one"},
                    {"token": "browser-two", "source": "commandcode-studio", "user_id": "two"}]), \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                self.profile.auth_handler("add", SimpleNamespace(provider=self.profile.name))
            before = read_credential_pool(self.profile.name)
            with patch.object(package_auth, "validate", return_value={"user_id": "one"}), \
                    patch.object(package_auth, "read_cli_token", side_effect=AssertionError("Do not switch account")):
                auth_refresh_command(SimpleNamespace(provider=self.profile.name, target=before[0]["id"]))
            after = read_credential_pool(self.profile.name)
            self.assertEqual([row["id"] for row in after], [row["id"] for row in before])
            self.assertEqual([row["access_token"] for row in after], ["browser-one", "browser-two"])
            self.assertFalse(after[0].get("refresh_token"))
            self.assertEqual(after[1], before[1])


if __name__ == "__main__":
    unittest.main()
