"""Client-view catalog and account locks through real local HTTP, with no inference."""
import contextlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

import test_plugin as fixtures
from model_access import access_note, model_rows, organization_id, fetch_access_context


class AccessRuleTest(unittest.TestCase):
    def test_expired_deadline_does_not_claim_verified_access(self):
        with patch('model_access.time.monotonic', return_value=10), \
                patch('model_access.organization_id') as organization:
            self.assertIsNone(fetch_access_context('https://fixture.test', 'token',
                lambda *args, **kwargs: self.fail('expired budget must not send requests'), deadline=9))
            organization.assert_not_called()

    def test_plan_rank_extra_credits_unknowns_and_hidden_rows(self):
        pro = {"id": "fixture/pro", "category": "premium", "minimumPlan": "individual-pro"}
        go = {"plan": "individual-go", "extra": False}
        self.assertTrue(access_note(pro, go))
        self.assertFalse(access_note(pro, {"plan": "individual-pro", "extra": False}))
        self.assertFalse(access_note(pro, {**go, "extra": True}))
        self.assertFalse(access_note(pro, None))
        self.assertTrue(access_note({**pro, "minimumPlan": None}, go))
        self.assertFalse(access_note({**pro, "category": "unknown"}, go))
        self.assertFalse(access_note({**pro, "category": {"future": "shape"}}, go))
        self.assertEqual(model_rows({"data": [{"id": "hidden", "hidden": True}, pro]}, go), [
            {"id": "fixture/pro", "label": "fixture/pro", "available": False,
             "note": "Locked: requires individual-pro or extra usage credits"}])

    def test_nested_organization_and_empty_catalog_have_explicit_meanings(self):
        self.assertEqual(organization_id({"org": {"id": "org/fixture"}, "orgId": "old"}), "org/fixture")
        self.assertEqual(organization_id({"orgId": "old"}), "old")
        self.assertEqual(model_rows({"data": []}), [])
        self.assertIsNone(model_rows({"data": [{"unexpected": "schema"}]}))


@unittest.skipUnless(fixtures.HAS_HERMES_TREE, "requires Hermes consumers")
class AccountCatalogTest(unittest.TestCase):
    def setUp(self):
        self.module = fixtures.load_profile()
        self.profile = self.module.CommandCodeOAuthProfile(
            name="commandcode-oauth", auth_type="oauth_external", base_url="https://api.commandcode.ai",
            fallback_models=("fixture/offline",))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.home = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict(os.environ, {"HERMES_HOME": self.home, "COMMANDCODE_CLI_TOKEN": "fixture-token"}))
        self.rows = [
            {"id": "fixture/go", "name": "Go model", "category": "opensource", "minimumPlan": "individual-go"},
            {"id": "fixture/pro", "category": "premium", "minimumPlan": "individual-pro"},
            {"id": "fixture/hidden", "hidden": True},
        ]

    def response(self, url, token, **kwargs):
        self.assertEqual(token, "fixture-token")
        route = urlsplit(url)
        if route.path == "/provider/v1/models":
            self.assertEqual(parse_qs(route.query), {"view": ["client"]})
            return {"schemaVersion": 1, "data": self.rows}
        if route.path == "/alpha/whoami":
            return {"org": {"id": "org/fixture"}}
        self.assertEqual(parse_qs(route.query), {"orgId": ["org/fixture"]})
        if route.path == "/alpha/billing/subscriptions":
            return {"success": True, "data": {"planId": "individual-go", "status": "active"}}
        if route.path == "/alpha/billing/credits":
            return {"credits": {"purchasedCredits": 0, "freeCredits": 0, "monthlyCredits": 99}}
        return None

    def test_local_http_filters_locked_models_and_preserves_lock_notes(self):
        received = []
        owner = self

        class Relay(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append((self.path, self.headers.get("Authorization")))
                value = owner.response("http://relay" + self.path, "fixture-token")
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        relay = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(relay.server_close)
        self.addCleanup(relay.shutdown)
        origin = f"http://127.0.0.1:{relay.server_port}"
        self.assertEqual(self.profile.fetch_models(api_key="fixture-token", base_url=origin), ["fixture/go"])
        rows = self.profile.discover_models(api_key="fixture-token", base_url=origin)
        self.assertEqual([row["id"] for row in rows], ["fixture/go", "fixture/pro"])
        self.assertTrue(rows[0]["available"])
        self.assertFalse(rows[1]["available"])
        self.assertIn("Locked", rows[1]["note"])
        self.assertEqual(len(received), 4)  # cached discovery does not repeat the billing queries
        self.assertTrue(all(auth == "Bearer fixture-token" for _, auth in received))

    def test_all_locked_is_cached_empty_without_restoring_offline_models(self):
        self.rows = self.rows[1:2]
        with patch.object(self.module, "_http_json", side_effect=self.response) as request:
            self.assertEqual(self.profile.fallback_models, ())
            self.assertEqual(self.profile.fallback_models, ())
            self.assertEqual(self.profile.discover_models()[0]["available"], False)
            self.assertEqual(request.call_count, 4)

    def test_billing_outage_and_unknown_plan_do_not_publish_unverified_catalog(self):
        for plan in (None, "future-plan"):
            def get(url, token, **kwargs):
                if "/subscriptions" in url:
                    return None if plan is None else {"success": True, "data": {"planId": plan, "status": "active"}}
                return self.response(url, token, **kwargs)
            with self.subTest(plan=plan), patch.object(self.module, "_http_json", side_effect=get):
                self.assertIsNone(self.profile.fetch_models(api_key="fixture-token"))
                self.assertEqual(self.profile.fallback_models, ())
                self.assertIsNone(self.profile.discover_models())

    def test_failed_refresh_retains_verified_models_and_lock_notes_then_recovers(self):
        with patch.object(self.module.catalog, "monotonic", return_value=100) as clock:
            with patch.object(self.module, "_http_json", side_effect=self.response):
                self.assertEqual(self.profile.fallback_models, ("fixture/go",))
                verified = self.profile.discover_models()
            clock.return_value = 401
            def outage(url, token, **kwargs):
                return None if "/subscriptions" in url else self.response(url, token, **kwargs)
            with patch.object(self.module, "_http_json", side_effect=outage) as request:
                self.assertEqual(self.profile.fallback_models, ("fixture/go",))
                self.assertEqual(self.profile.discover_models(), verified)
                calls = request.call_count
                clock.return_value = 450
                self.assertEqual(self.profile.fallback_models, ("fixture/go",))
                self.assertEqual(request.call_count, calls)
            clock.return_value = 462
            def unlocked(url, token, **kwargs):
                value = self.response(url, token, **kwargs)
                if "/credits" in url:
                    value["credits"]["purchasedCredits"] = 1
                return value
            with patch.object(self.module, "_http_json", side_effect=unlocked):
                self.assertEqual(self.profile.fallback_models, ("fixture/go", "fixture/pro"))

    def test_timeout_does_not_publish_public_catalog_or_static_models(self):
        with patch.object(self.module, "_http_json", side_effect=self.response), \
                patch.object(self.module, "fetch_access_context", return_value=None):
            self.assertEqual(self.profile.fallback_models, ())
            self.assertIsNone(self.profile.discover_models())

    def test_failed_new_account_does_not_reuse_previous_accounts_access(self):
        with patch.object(self.module, "_http_json", side_effect=self.response):
            self.assertEqual(self.profile.fallback_models, ("fixture/go",))
        with patch.dict(os.environ, {"COMMANDCODE_CLI_TOKEN": "other-token"}), \
                patch.object(self.module, "_http_json", return_value={"data": self.rows}), \
                patch.object(self.module, "fetch_access_context", return_value=None):
            self.assertEqual(self.profile.fallback_models, ())

    def test_hermes_disk_cache_does_not_expand_on_failed_access_refresh(self):
        from hermes_cli.models import cached_provider_model_ids
        from pathlib import Path
        with patch('providers.get_provider_profile', return_value=self.profile), \
                patch.object(self.module.catalog, 'monotonic', return_value=100) as clock:
            with patch.object(self.module, '_http_json', side_effect=self.response):
                self.assertEqual(cached_provider_model_ids(self.profile.name, force_refresh=True), ['fixture/go'])
            clock.return_value = 401
            with patch.object(self.module, '_http_json', side_effect=self.response), \
                    patch.object(self.module, 'fetch_access_context', return_value=None):
                self.assertEqual(cached_provider_model_ids(self.profile.name, force_refresh=True), ['fixture/go'])
            saved = json.loads((Path(self.home) / 'provider_models_cache.json').read_text())
            self.assertEqual(saved[self.profile.name]['models'], ['fixture/go'])

    def test_purchased_or_free_credits_unlock_models_but_monthly_credits_do_not(self):
        for field in ("purchasedCredits", "freeCredits"):
            def get(url, token, **kwargs):
                value = self.response(url, token, **kwargs)
                if "/credits" in url:
                    value["credits"][field] = 1
                return value
            with self.subTest(field=field), patch.object(self.module, "_http_json", side_effect=get):
                self.assertEqual(self.profile.fetch_models(api_key="fixture-token"), ["fixture/go", "fixture/pro"])

    def test_account_usage_uses_the_same_nested_organization(self):
        with patch.object(self.module, "_http_json", side_effect=self.response) as request:
            self.assertIsNotNone(self.profile.fetch_account_usage(api_key="fixture-token"))
            urls = [call.args[0] for call in request.call_args_list]
            self.assertTrue(any("/credits?orgId=org/fixture" in url or "/credits?orgId=org%2Ffixture" in url for url in urls))
