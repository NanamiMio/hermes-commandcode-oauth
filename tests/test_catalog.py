"""Temporary OAuth catalog bridge through real Hermes picker entry points."""
import contextlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import test_plugin as fixtures


@unittest.skipUnless(fixtures.HAS_HERMES_TREE, "requires Hermes consumers")
class OAuthCatalogTest(unittest.TestCase):
    def setUp(self):
        self.module = fixtures.load_profile()
        self.profile = self.module.CommandCodeOAuthProfile(
            name="commandcode-oauth", aliases=("commandcode-alpha",), auth_type="oauth_external",
            base_url=self.module.ALPHA_ORIGIN, fallback_models=self.module.FALLBACK_MODELS)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        home = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.home = Path(home)
        self.stack.enter_context(patch.dict(os.environ, {
            "HERMES_HOME": home, "COMMANDCODE_CLI_TOKEN": "fixture-token"}))
        self.stack.enter_context(patch("providers.get_provider_profile", return_value=self.profile))

    def test_oauth_picker_reads_live_models_without_reinserting_static_ids(self):
        from hermes_cli.models import _profile_live_catalog
        self.assertEqual(self.module.commandcode_oauth.auth_type, "oauth_external")
        with patch.object(self.module, "_http_json", return_value={"data": [{"id": "fixture/live"}]}) as get:
            self.assertEqual(_profile_live_catalog(self.profile.name), ["fixture/live"])
            self.assertEqual(self.profile.fallback_models, ("fixture/live",))
        get.assert_called_once()
        self.assertEqual(get.call_args.args[1], "fixture-token")
        self.assertLessEqual(get.call_args.kwargs["timeout"], 8)

    def test_dataclass_constructor_and_registration_do_not_request_models(self):
        from providers import register_provider
        with patch.object(self.module, "_http_json", side_effect=AssertionError("Import must not request models")):
            profile = self.module.CommandCodeOAuthProfile(name="fixture-constructor", auth_type="oauth_external",
                fallback_models=("fixture/offline",))
            with patch.dict("providers._REGISTRY", {}):
                register_provider(profile)

    def test_cache_expiry_retains_last_good_list_and_backs_off_after_failure(self):
        # The clock is patched at the plugin's cache seam, without changing Hermes' clock.
        with patch.object(self.module.catalog, "monotonic", return_value=100) as clock, \
                patch.object(self.module, "_http_json", side_effect=[
                    {"data": [{"id": "fixture/old"}]}, None, {"data": [{"id": "fixture/new"}]}]) as get:
            self.assertEqual(self.profile.fallback_models, ("fixture/old",))
            clock.return_value = 399
            self.assertEqual(self.profile.fallback_models, ("fixture/old",))
            self.assertEqual(get.call_count, 1)
            clock.return_value = 401
            self.assertEqual(self.profile.fallback_models, ("fixture/old",))
            clock.return_value = 420
            self.assertEqual(self.profile.fallback_models, ("fixture/old",))
            self.assertEqual(get.call_count, 2)
            clock.return_value = 462
            self.assertEqual(self.profile.fallback_models, ("fixture/new",))
            self.assertEqual(get.call_count, 3)

    def test_logout_does_not_return_the_previous_accounts_models(self):
        with patch.object(self.module, "_http_json", return_value={"data": [{"id": "fixture/private"}]}):
            self.assertEqual(self.profile.fallback_models, ("fixture/private",))
        with patch.dict(os.environ, {"COMMANDCODE_CLI_TOKEN": ""}), \
                patch("hermes_cli.auth.read_credential_pool", return_value=[]), \
                patch.object(self.module, "read_cli_token", side_effect=AssertionError("No implicit host login")), \
                patch.object(self.module, "_http_json", side_effect=AssertionError("No logged-out probe")):
            self.assertEqual(self.profile.fallback_models, self.module.FALLBACK_MODELS)

    def test_profile_home_token_and_origin_partition_the_cache(self):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        observed = []

        def get(url, token, **kwargs):
            observed.append((url, token))
            return {"data": [{"id": f"fixture/{len(observed)}"}]}

        with patch.object(self.module, "_http_json", side_effect=get):
            for home_name, token, origin in (("a", "token-a", "https://relay-a.test"),
                                             ("b", "token-a", "https://relay-a.test"),
                                             ("b", "token-b", "https://relay-a.test"),
                                             ("b", "token-b", "https://relay-b.test")):
                home = self.home / home_name
                home.mkdir(exist_ok=True)
                (home / "config.yaml").write_text(
                    f"model:\n  provider: commandcode-alpha\n  base_url: {origin}/provider/v1\n")
                bound = set_hermes_home_override(home)
                scoped = set_secret_scope({"COMMANDCODE_CLI_TOKEN": token})
                try:
                    expected = (f"fixture/{len(observed) + 1}",)
                    self.assertEqual(self.profile.fallback_models, expected)
                finally:
                    reset_secret_scope(scoped)
                    reset_hermes_home_override(bound)
        self.assertEqual(observed, [
            ("https://relay-a.test/provider/v1/models", "token-a"),
            ("https://relay-a.test/provider/v1/models", "token-a"),
            ("https://relay-a.test/provider/v1/models", "token-b"),
            ("https://relay-b.test/provider/v1/models", "token-b")])

    def test_native_fetch_then_merge_uses_the_same_live_list_without_second_request(self):
        from hermes_cli.models import merge_profile_catalog
        with patch.object(self.module, "_http_json", return_value={"data": [{"id": "fixture/live"}]}) as get:
            live = self.profile.fetch_models(api_key="fixture-token")
            self.assertEqual(merge_profile_catalog(self.profile.name, self.profile, live), ["fixture/live"])
        get.assert_called_once()

    def test_cold_failure_returns_offline_models_without_repeated_requests(self):
        with patch.object(self.module, "_http_json", return_value=None) as get:
            self.assertEqual(self.profile.fallback_models, self.module.FALLBACK_MODELS)
            self.assertEqual(self.profile.fallback_models, self.module.FALLBACK_MODELS)
        get.assert_called_once()

    def test_desktop_curated_catalog_uses_configured_proxy_and_dynamic_rows(self):
        from hermes_cli.model_switch_providers import _build_curated_lists
        (self.home / "config.yaml").write_text(
            "model:\n  provider: commandcode-oauth\n  base_url: https://relay.test/provider/v1\n")
        with patch("providers.list_providers", return_value=[self.profile]), \
                patch("hermes_cli.models.get_curated_nous_model_ids", return_value=[]), \
                patch.object(self.module, "_http_json", return_value={"data": [{"id": "fixture/relay"}]}) as get:
            rows = _build_curated_lists(self.profile.name, "https://relay.test/provider/v1", "", non_blocking=True)
        self.assertEqual(rows[self.profile.name], ["fixture/relay"])
        get.assert_called_once_with("https://relay.test/provider/v1/models", "fixture-token", timeout=3.0)

    def test_concurrent_reads_do_not_duplicate_the_inflight_request(self):
        started, release = threading.Event(), threading.Event()

        def get(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(5))
            return {"data": [{"id": "fixture/live"}]}

        with patch.object(self.module, "_http_json", side_effect=get) as request, ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(lambda: self.profile.fallback_models)
            try:
                self.assertTrue(started.wait(5))
                second = workers.submit(lambda: self.profile.fallback_models)
                self.assertEqual(second.result(timeout=2), self.module.FALLBACK_MODELS)
                self.assertEqual(request.call_count, 1)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5), ("fixture/live",))

    def test_property_uses_real_http_get_against_configured_origin(self):
        received = []

        class Relay(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append((self.path, self.headers.get("Authorization")))
                body = json.dumps({"data": [{"id": "fixture/relay-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        relay = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        try:
            (self.home / "config.yaml").write_text(
                f"model:\n  provider: commandcode-oauth\n  base_url: http://127.0.0.1:{relay.server_port}/provider/v1\n")
            self.assertEqual(self.profile.fallback_models, ("fixture/relay-model",))
            self.assertEqual(self.profile.fallback_models, ("fixture/relay-model",))
            self.assertEqual(received, [("/provider/v1/models", "Bearer fixture-token")])
        finally:
            relay.shutdown()
            relay.server_close()
            thread.join(timeout=2)
