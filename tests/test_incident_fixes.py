"""The RPM → exhausted pool → foreign endpoint incident, using fixture credentials."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_plugin as fixtures
import errors
import transport


class IncidentFixTest(unittest.TestCase):
    def test_free_window_ended_preserves_account_and_requests_fallback(self):
        message = ('The free MiniMax M3.1 Flash window has ended. Run /model and pick '
                   'MiniMax M3.1 Flash to keep going.\nhttps://commandcode.ai/models')
        for stream in (False, True):
            with self.subTest(stream=stream):
                error = transport.CommandCodeAPIError(message, status_code=403)
                if stream:
                    with self.assertRaises(transport.CommandCodeAPIError) as caught:
                        list(transport._iter_events([json.dumps(
                            {"type": "error", "statusCode": 403, "error": message})]))
                    error = caught.exception
                verdict = errors.classify_api_error(error)
                self.assertEqual(verdict['reason'], 'model_not_found')
                self.assertFalse(verdict['retryable'])
                self.assertFalse(verdict['should_rotate_credential'])
                self.assertTrue(verdict['should_fallback'])
        for code, message in ((401, 'Invalid token'), (403, 'Access denied'),
                              (403, 'Your authentication window has ended')):
            self.assertEqual(errors.classify_api_error(transport.CommandCodeAPIError(
                message, status_code=code))['reason'], 'auth')

    @unittest.skipUnless(fixtures.HAS_HERMES_TREE, 'requires Hermes consumers')
    def test_real_hermes_pool_and_fallback_consume_free_window_verdict(self):
        profile = fixtures.load_profile().commandcode_oauth
        from agent.error_classifier import classify_api_error, FailoverReason
        from agent.agent_runtime_helpers import recover_with_credential_pool
        from agent.turn_api_error import settle_unrecovered_error
        error = transport.CommandCodeAPIError(
            'The free MiniMax M3.1 Flash window has ended. Run /model and pick MiniMax M3.1 Flash.',
            status_code=403)
        verdict = classify_api_error(error, provider=profile.name)
        pool = Mock(provider=profile.name)
        agent = Mock(provider=profile.name, model='minimax/minimax-m3.1-flash:free',
                     api_key='fixture-token', base_url='https://api.commandcode.ai',
                     requested_provider=profile.name, api_mode='chat_completions',
                     _credential_pool=pool)
        self.assertFalse(verdict.is_auth)
        self.assertEqual(recover_with_credential_pool(
            agent, status_code=403, has_retried_429=False, classified_reason=verdict.reason),
            (False, False))
        pool.mark_exhausted_and_rotate.assert_not_called()
        agent._try_activate_fallback.return_value = True
        with patch('agent.conversation_loop._arm_fallback_restart', return_value='prompt') as restart:
            result = settle_unrecovered_error(
                agent, api_error=error, classified=verdict,
                _retry=SimpleNamespace(copilot_stale_cred_retry_attempted=False),
                status_code=403, error_msg=str(error), is_context_length_error=False,
                is_rate_limited=False, _is_zai_coding_overload=False,
                _provider=profile.name, _base=agent.base_url, _model=agent.model,
                messages=[], api_messages=[], api_kwargs={}, active_system_prompt='prompt',
                conversation_history=[], approx_tokens=0, retry_count=1, max_retries=3,
                compression_attempts=0, api_call_count=1)
        agent._try_activate_fallback.assert_called_once_with(
            reason=FailoverReason.model_not_found, reset_at=None)
        restart.assert_called_once()
        self.assertEqual(result.action, 'break')
        self.assertEqual(result.retry_count, 0)

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
