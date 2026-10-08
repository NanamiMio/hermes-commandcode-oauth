"""Review regressions, using the real transport and loopback callback."""

import asyncio
import contextlib
import io
import json
import socket
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_plugin as fixtures
from test_plugin import FINISH_OK, HAS_HERMES_TREE, _Handler, _line, _text
import auth
import errors
import transport


class TransportRegressionTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.PluginTestCase.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.PluginTestCase.tearDownClass.__func__)
    client = fixtures.PluginTestCase.client
    call = fixtures.PluginTestCase.call

    def test_partial_output_requires_terminal_finish(self):
        for payload in (_text("partial"), _text("partial") + _line(type="finish-step")):
            for streaming in (False, True):
                with self.subTest(payload=payload, streaming=streaming):
                    with self.assertRaises(transport.CommandCodeAPIError) as ctx:
                        result = self.call(payload, stream=streaming)
                        if streaming:
                            list(result)
                    self.assertEqual(ctx.exception.status_code, 502)
                    self.assertEqual(errors.classify_api_error(ctx.exception)["reason"], "server_error")

    def test_final_usage_keeps_step_cache_details(self):
        payload = _text("ok") + _line(
            type="finish-step", usage={"inputTokens": 100, "outputTokens": 2,
                                       "inputTokenDetails": {"cacheReadTokens": 80, "cacheWriteTokens": 5}},
        ) + _line(type="finish", totalUsage={"inputTokens": 100, "outputTokens": 3})
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                result = self.call(payload, stream=streaming)
                usage = list(result)[-1].usage if streaming else result.usage
                self.assertEqual(usage.prompt_tokens, 100)
                self.assertEqual(usage.completion_tokens, 3)
                self.assertEqual(usage.total_tokens, 103)
                self.assertEqual(usage.prompt_tokens_details.cached_tokens, 80)
                self.assertEqual(usage.prompt_tokens_details.cache_write_tokens, 5)

    def test_sparse_final_usage_preserves_counts_and_accepts_explicit_zero(self):
        step = {"inputTokens": 100, "outputTokens": 2,
                "inputTokenDetails": {"cacheReadTokens": 80, "cacheWriteTokens": 5}}
        payload = _text("ok") + _line(type="finish-step", usage=step) + _line(
            type="finish", totalUsage={"outputTokens": 0, "inputTokenDetails": {"cacheWriteTokens": 9}})
        response = self.call(payload)
        self.assertEqual(response.usage.prompt_tokens, 100)
        self.assertEqual(response.usage.completion_tokens, 0)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 80)
        self.assertEqual(response.usage.prompt_tokens_details.cache_write_tokens, 9)

    def test_empty_stream_raises_before_emitting_successful_finish(self):
        stream = self.call(_line(type="finish", finishReason="stop"), stream=True)
        with self.assertRaises(transport.CommandCodeAPIError):
            next(iter(stream))

    def test_temperature_only_forwards_real_numbers(self):
        for value in (0.0, 0.7, "warm", True, False):
            with self.subTest(value=value):
                self.call(_text("ok") + FINISH_OK, temperature=value)
                params = _Handler.captured["body"]["params"]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.assertEqual(params["temperature"], value)
                else:
                    self.assertNotIn("temperature", params)

    def test_image_shapes_keep_pixels_and_mime(self):
        cases = (
            ({"type": "image_url", "image_url": {"url": "https://example.test/blue.png?size=64"}},
             "https://example.test/blue.png?size=64", "image/png"),
            ({"type": "image_url", "image_url": {"url": "https://example.test/opaque"}, "mimeType": "image/jpeg"},
             "https://example.test/opaque", "image/jpeg"),
            ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
             "data:image/png;base64,AAAA", "image/png"),
        )
        for part, url, mime in cases:
            with self.subTest(part=part):
                _, messages = transport.format_messages([{"role": "user", "content": [part]}], "vision/model")
                image = messages[0]["content"][1]
                self.assertEqual(image, {"type": "image", "image": url, "mimeType": mime})

    def test_unknown_parts_are_not_labelled_as_images(self):
        _, messages = transport.format_messages([{"role": "user", "content": [
            {"type": "file", "file": {"name": "notes.pdf"}}]}], "vision/model")
        text = messages[0]["content"][0]["text"]
        self.assertIn("file", text)
        self.assertNotIn("image", text)

    @unittest.skipUnless(HAS_HERMES_TREE, "requires a Hermes tree")
    def test_real_async_dispatch_preserves_alpha_transport(self):
        from agent.auxiliary_client import _to_async_client
        client = self.client()
        async_client, model = _to_async_client(client, "vision/model")
        self.assertIs(async_client, client)
        _Handler.payload = _text("blue") + FINISH_OK

        async def run():
            return await async_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "colour?"}])

        self.assertEqual(asyncio.run(run()).choices[0].message.content, "blue")
        self.assertEqual(_Handler.captured["path"], "/alpha/generate")
        self.assertEqual(_Handler.captured["headers"]["authorization"], "Bearer test-token")


class AuthRegressionTest(unittest.TestCase):
    def setUp(self):
        guard = patch.dict("os.environ", {"COMMANDCODE_CLI_TOKEN": ""})
        guard.start()
        self.addCleanup(guard.stop)

    def test_browser_handoff_uses_bound_address_and_does_not_write_cli_store(self):
        original_start = auth._start_callback_server

        def start(state):
            return original_start(state, port=0)

        def browser(url):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            callback = query["callback"][0]
            self.assertNotEqual(urllib.parse.urlsplit(callback).port, auth.CALLBACK_PORT)
            req = urllib.request.Request(callback, data=json.dumps({
                "state": query["state"][0], "apiKey": "browser-token"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as response:
                self.assertEqual(response.status, 200)

        with tempfile.TemporaryDirectory() as root:
            cli_path = Path(root) / "auth.json"
            cli_path.write_text('{"apiKey":"old", "vendorField":"preserve"}')
            original = cli_path.read_bytes()
            with patch.object(auth, "CLI_AUTH_PATH", cli_path), patch.object(auth, "validate", return_value=None), \
                    patch.object(auth, "_start_callback_server", side_effect=start), \
                    patch("webbrowser.open", side_effect=browser), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(auth.login(timeout=0.2)["token"], "browser-token")
            self.assertEqual(cli_path.read_bytes(), original)

    def test_cancelled_browser_flow_releases_listener(self):
        original_start = auth._start_callback_server
        address = []

        def start(state):
            server, thread, result = original_start(state, port=0)
            address.append(server.server_address)
            return server, thread, result

        with patch.object(auth, "read_cli_token", side_effect=auth.CommandCodeAuthError("absent")), \
                patch.object(auth, "_start_callback_server", side_effect=start), \
                patch("webbrowser.open", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                auth.login()
        with socket.socket() as listener:
            listener.bind(address[0])

    def test_studio_pool_credential_refresh_does_not_require_cli_file(self):
        entry = SimpleNamespace(access_token="studio-token", extra={"commandcode": {"source": "commandcode-studio"}})
        with patch.object(auth, "read_cli_token", side_effect=auth.CommandCodeAuthError("absent")), \
                patch.object(auth, "validate", return_value={"user_id": "studio-user"}):
            self.assertEqual(auth.refresh_credential(entry)["access_token"], "studio-token")

    def test_env_key_can_be_imported_without_a_cli_file_or_browser(self):
        with patch.dict("os.environ", {"COMMANDCODE_CLI_TOKEN": "env-token"}), \
                patch.object(auth, "validate", return_value={"user_id": "env-user"}), \
                patch.object(auth, "read_cli_token", side_effect=AssertionError("CLI file must not be read")), \
                patch("webbrowser.open", side_effect=AssertionError("browser must not open")):
            self.assertEqual(auth.login(), {"token": "env-token", "source": "commandcode-env", "user_id": "env-user"})


@unittest.skipUnless(HAS_HERMES_TREE, "requires a Hermes tree")
class ProfileRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = fixtures.load_profile()
        cls.profile = cls.module.commandcode_oauth

    def test_setup_and_discovery_work_with_pool_only_credentials(self):
        with patch.dict("os.environ", {"COMMANDCODE_CLI_TOKEN": ""}), \
                patch("hermes_cli.auth.read_credential_pool", return_value=[{"access_token": "pool-token"}]), \
                patch.object(self.module, "read_cli_token", side_effect=self.module.CommandCodeAuthError("absent")), \
                patch.object(self.module, "validate", return_value={"user_name": "pool-user"}) as validate, \
                patch.object(self.module, "_http_json", return_value={"data": [{"id": "vision/model"}]}):
            status = self.profile.setup_status()
            self.assertTrue(status["logged_in"])
            self.assertEqual(status["login_command"], ["hermes", "auth", "add", "commandcode-oauth"])
            validate.assert_called_once_with("pool-token")
            self.assertIn("vision/model", [row["id"] for row in self.profile.discover_models()])

    def test_logged_out_setup_has_actionable_login_command(self):
        with patch.dict("os.environ", {"COMMANDCODE_CLI_TOKEN": ""}), \
                patch("hermes_cli.auth.read_credential_pool", return_value=[]), \
                patch.object(self.module, "read_cli_token", side_effect=self.module.CommandCodeAuthError("absent")):
            status = self.profile.setup_status()
            self.assertFalse(status["logged_in"])
            self.assertIn("hermes auth add commandcode-oauth", status["detail"])

    def test_account_usage_calls_share_one_deadline(self):
        credits = {"credits": {"monthlyCredits": 10}, "windowLimits": {"fiveHour": {"used": 2, "cap": 10}}}
        with patch.object(self.module.time, "monotonic", side_effect=[100, 100, 104, 108]), \
                patch.object(self.module, "_http_json", side_effect=[{"orgId": "org 1"}, credits, {"totalCost": 3}]) as get:
            snapshot = self.profile.fetch_account_usage(api_key="fixture-token")
        self.assertEqual([call.kwargs["timeout"] for call in get.call_args_list], [4, 4, 2])
        self.assertEqual(snapshot.windows[0].used_percent, 20)
        self.assertIn("orgId=org%201", get.call_args_list[1].args[0])

    def test_real_pool_and_model_setup_keep_browser_credentials(self):
        from hermes_cli.auth import read_credential_pool, resolve_provider
        from hermes_cli.model_setup_flows import _plugin_flow_oauth, _plugin_flow_live_rows

        package_auth = fixtures.load_profile().auth
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {
            "HERMES_HOME": root, "COMMANDCODE_CLI_TOKEN": ""}), \
                patch.object(package_auth, "login", return_value={"token": "pool-token", "source": "commandcode-studio"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.profile.auth_handler("add", SimpleNamespace(provider="commandcode-alpha"))
            self.assertEqual(resolve_provider("commandcode-alpha"), "commandcode-oauth")
            self.assertEqual(read_credential_pool("commandcode-oauth")[0]["access_token"], "pool-token")
            base_url, token = _plugin_flow_oauth("commandcode-oauth", self.profile)
            self.assertEqual(token, "pool-token")
            with patch.object(self.module, "_http_json", return_value={"data": [{"id": "vision/model"}]}):
                models, _notes = _plugin_flow_live_rows(self.profile, token, base_url)
                self.assertIn("vision/model", models)

    def test_real_oauth_aux_resolution_uses_registered_transport(self):
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("commandcode-oauth", model="vision/model",
                                               explicit_api_key="fixture-token", explicit_base_url="http://127.0.0.1:9999")
        self.assertIsInstance(client, self.module.CommandCodeAlphaClient)
        self.assertEqual(model, "vision/model")
        self.assertEqual(client._generate_url, "http://127.0.0.1:9999/alpha/generate")

    def test_models_and_usage_honour_proxy_provider_api_origin(self):
        with patch.object(self.module, "_http_json", return_value={"data": [{"id": "vision/model"}]}) as get:
            self.profile.fetch_models(api_key="fixture-token", base_url="http://127.0.0.1:9999/provider/v1")
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:9999/provider/v1/models")
        self.assertEqual(self.module._api_origin("http://127.0.0.1:9999/provider/v1"), "http://127.0.0.1:9999")
