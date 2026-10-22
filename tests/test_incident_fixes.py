"""The RPM → exhausted pool → foreign endpoint incident, using fixture credentials."""
import json
import unittest
from unittest.mock import patch

import test_plugin as fixtures
import errors
import transport


class IncidentFixTest(unittest.TestCase):
    def test_rpm_402_is_model_rate_limit_but_real_credit_failure_is_billing(self):
        message = "Failed after 3 attempts. Last error: rate limit exceeded(RPM) (1002)"
        with self.assertRaises(transport.CommandCodeAPIError) as caught:
            list(transport._iter_events([json.dumps({"type": "error", "statusCode": 402, "error": message})]))
        verdict = errors.classify_api_error(caught.exception)
        self.assertEqual(verdict["reason"], "rate_limit")
        self.assertEqual(verdict["error_context"]["rate_limit_scope"], "model")
        self.assertTrue(verdict["should_rotate_credential"])
        self.assertEqual(errors.classify_api_error(transport.CommandCodeAPIError(
            "Insufficient Balance", status_code=402))["reason"], "billing")

    def test_foreign_endpoint_is_rejected_before_credentials_can_be_sent(self):
        for url in ("https://openrouter.ai/api/v1", "https://OPENROUTER.AI:443/api/v1"):
            with self.subTest(url=url), patch.object(transport, "open_credentialed") as open_request:
                with self.assertRaises(ValueError) as caught:
                    transport.CommandCodeAlphaClient(api_key="fixture-key", base_url=url)
                self.assertIn("endpoint", str(caught.exception).lower())
                self.assertEqual(errors.classify_api_error(caught.exception)["reason"], "format_error")
                open_request.assert_not_called()
        self.assertEqual(transport.generate_url("http://127.0.0.1:9999"),
                         "http://127.0.0.1:9999/alpha/generate")

    def test_html_404_is_endpoint_failure_while_structured_404_stays_model_scoped(self):
        error = transport.CommandCodeAPIError("HTTP 404: <!DOCTYPE html><html>missing route</html>", status_code=404)
        self.assertEqual(errors.classify_api_error(error)["reason"], "format_error")
        error = transport.CommandCodeAPIError('HTTP 404: {"error":"model_not_found"}', status_code=404)
        self.assertEqual(errors.classify_api_error(error)["reason"], "model_not_found")

    @unittest.skipUnless(fixtures.HAS_HERMES_TREE, "requires Hermes consumers")
    def test_catalog_and_usage_reject_foreign_origin_without_sending_a_key(self):
        module = fixtures.load_profile()
        with patch.object(module, "open_credentialed") as open_request:
            with self.assertRaises(ValueError):
                module.commandcode_oauth.fetch_models(api_key="fixture-key", base_url="https://openrouter.ai/api/v1")
            with self.assertRaises(ValueError):
                module.commandcode_oauth.fetch_account_usage(api_key="fixture-key", base_url="https://openrouter.ai/api/v1")
            open_request.assert_not_called()
