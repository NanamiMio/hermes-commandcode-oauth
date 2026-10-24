"""Real Hermes contracts. Set HERMES_TREE and an isolated HERMES_HOME to run."""
import importlib.util
import io
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import test_plugin as support


@unittest.skipUnless(support.HAS_HERMES_TREE and os.environ.get('HERMES_HOME'),
                     'requires Hermes and an isolated HERMES_HOME')
class HermesIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = support.load_profile()
        cls.profile = cls.plugin.commandcode_oauth
        auth, transport = cls.plugin.auth, cls.plugin.transport
        from agent.transports.chat_completions import ChatCompletionsTransport
        cls.auth, cls.transport = auth, transport
        cls.core = ChatCompletionsTransport()

    def kwargs(self, **extra):
        params = dict(model='minimax/minimax-m3.1-flash:free', messages=[{'role': 'user', 'content': 'test'}],
                      provider_profile=self.profile, base_url=self.profile.base_url,
                      max_tokens_param_fn=lambda value: {'max_tokens': value})
        params.update(extra)
        return self.core.build_kwargs(**params)

    def test_reasoning_reaches_wire_through_core(self):
        for model, config, expected in [
            ('minimax/minimax-m3.1-flash:free', {'effort': 'high'}, 'high'),
            ('deepseek/deepseek-v4.1-flash', {'effort': 'medium'}, 'low'),
            ('deepseek/deepseek-v4.1-flash', {'enabled': False}, 'off'),
            ('poolside/laguna-s-2.1-free', {'effort': 'high'}, None),
        ]:
            with self.subTest(model=model, config=config):
                kw = self.kwargs(model=model, reasoning_config=config, supports_reasoning=True)
                self.assertEqual(self.transport._request_body(kw, model)['params'].get('reasoning_effort'), expected)
        self.assertEqual(self.kwargs()['max_tokens'], 64000)
        self.assertEqual(self.kwargs(max_tokens=321)['max_tokens'], 321)

    def test_real_model_capability_lookup(self):
        from agent.models_dev import get_model_capabilities
        from agent.image_routing import _lookup_supports_vision
        with patch('agent.models_dev._get_provider_models', return_value={}):
            for model, vision in [('minimax/minimax-m3.1-flash:free', True), ('deepseek/deepseek-v4-pro', False)]:
                caps = get_model_capabilities(self.profile.name, model, config={})
                self.assertEqual(caps.supports_vision, vision)
                self.assertTrue(caps.supports_tools)
                self.assertEqual(_lookup_supports_vision(self.profile.name, model, {}), vision)

    def test_tool_images_are_not_json_text_and_batch_order_is_preserved(self):
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fake'}}
        messages = [
            {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': {'name': 'screen', 'arguments': '{}'}},
                                                {'id': 'b', 'function': {'name': 'other', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': [{'type': 'text', 'text': 'screenshot'}, image]},
            {'role': 'tool', 'tool_call_id': 'b', 'content': 'done'},
        ]
        kw = self.kwargs(messages=messages)
        wire = self.transport._request_body(kw, kw['model'])['params']['messages']
        self.assertEqual([m['role'] for m in wire], ['assistant', 'tool', 'tool', 'user'])
        self.assertNotIn('base64', wire[1]['content'][0]['output']['value'])
        self.assertEqual(wire[-1]['content'][-1]['type'], 'image')
        self.assertEqual(wire[-1]['content'][-1]['mimeType'], 'image/png')

    def test_session_survives_client_rebuild_and_changes_between_conversations(self):
        from test_plugin import _Handler, _text, FINISH_OK
        server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        headers, threads = [], []
        try:
            _Handler.payload = _text('ok') + FINISH_OK
            for session in ('hermes-session-A', 'hermes-session-A', 'hermes-session-B'):
                client = self.profile.create_client(api_key='fake', base_url=f'http://127.0.0.1:{server.server_port}')
                kw = self.kwargs(session_id=session)
                kw['extra_headers'] = {'x-test-request': 'present'}
                result = client.chat.completions.create(**kw)
                self.assertEqual(self.core.normalize_response(result).content, 'ok')
                headers.append(_Handler.captured['headers']['x-session-id'])
                threads.append(_Handler.captured['body']['threadId'])
                self.assertEqual(_Handler.captured['headers']['x-test-request'], 'present')
            self.assertEqual(headers, threads)
            self.assertEqual(headers[0], headers[1])
            self.assertNotEqual(headers[0], headers[2])
        finally:
            server.shutdown(); server.server_close()

    def test_usage_snapshot_passes_real_inventory_serializer(self):
        from agent.account_usage import AccountUsageWindow
        from hermes_cli.inventory import _apply_usage
        credits = {'windowLimits': {'fiveHour': {'used': 1, 'cap': 2, 'resetAt': 1792700000000}}}
        with patch.object(self.plugin, '_http_json', side_effect=[{'orgId': 'test'}, credits, {}]):
            snapshot = self.profile.fetch_account_usage(api_key='fake')
        self.assertIsInstance(snapshot.windows[0], AccountUsageWindow)
        rows = [{'slug': self.profile.name, 'authenticated': True}]
        with patch('agent.account_usage_cache.has_account_usage', return_value=True), \
             patch('agent.account_usage_cache.cached_account_usage', return_value=snapshot), \
             patch('agent.account_usage_cache.snapshot_is_stale', return_value=False), \
             patch('hermes_cli.auth.read_credential_pool', return_value={}):
            _apply_usage(rows)
        self.assertIn('account', json.dumps(rows))

    def test_rejected_grant_is_terminal_but_network_error_is_not(self):
        from agent.credential_pool_plugin import is_terminal_plugin_refresh_error
        for code, terminal in [(401, True), (403, True), (429, False), (503, False)]:
            with self.subTest(code=code):
                error = urllib.error.HTTPError('http://localhost/whoami', code, 'test', {}, io.BytesIO())
                with patch.object(self.auth, 'read_cli_token', return_value='fake'), \
                     patch.object(self.auth, 'open_credentialed', side_effect=error):
                    with self.assertRaises(Exception) as caught:
                        self.auth.refresh_credential(None)
                self.assertEqual(is_terminal_plugin_refresh_error(caught.exception), terminal)
                error.close()

    def test_transient_validation_does_not_open_browser(self):
        with patch.object(self.auth, 'read_cli_token', return_value='fake'), \
             patch.object(self.auth, 'open_credentialed', side_effect=OSError('offline')), \
             patch('webbrowser.open') as browser:
            with self.assertRaises(self.auth.CommandCodeAuthError):
                self.auth.login()
            browser.assert_not_called()

    def test_auxiliary_projection_and_async_client(self):
        import asyncio
        from agent.auxiliary_client import _build_call_kwargs, _to_async_client
        kw = _build_call_kwargs(provider=self.profile.name, model='minimax/minimax-m3.1-flash:free',
            messages=[{'role': 'user', 'content': 'test'}], max_tokens=128,
            reasoning_config={'effort': 'low'}, base_url=self.profile.base_url)
        self.assertEqual(kw['reasoning_effort'], 'low')
        client = self.profile.create_client(api_key='fake')
        async_client, _ = _to_async_client(client, kw['model'], is_vision=True)
        self.assertIs(async_client, client)
        async def run():
            return await async_client.chat.completions.create(**kw)
        with patch.object(self.transport, 'open_credentialed',
                          return_value=io.BytesIO(support._text('ok') + support.FINISH_OK)):
            response = asyncio.run(run())
        self.assertEqual(self.core.normalize_response(response).content, 'ok')

    def test_pool_marks_rejected_grant_dead(self):
        from agent.credential_pool import CredentialPool, PooledCredential, AUTH_TYPE_OAUTH, STATUS_DEAD
        entry = PooledCredential(provider=self.profile.name, id='audit', auth_type=AUTH_TYPE_OAUTH,
                                 source='manual:audit', access_token='fake', label='audit', priority=0)
        pool = CredentialPool(self.profile.name, [entry])
        with patch.object(self.auth, 'read_cli_token', return_value='fake'), \
             patch.object(self.auth, 'validate', return_value=None), \
             patch.object(pool, '_sync_entry_from_pool_store', return_value=entry), \
             patch.object(pool, '_persist'):
            self.assertIsNone(pool._refresh_entry_impl(entry, force=True))
        self.assertEqual(pool.entries()[0].last_status, STATUS_DEAD)

    def test_cross_origin_redirect_strips_all_credentials_on_every_http_path(self):
        seen = []
        class Sink(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(dict(self.headers))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"user":{"id":"test"},"data":[]}')
            def log_message(self, *args): pass
        sink = ThreadingHTTPServer(('127.0.0.1', 0), Sink)
        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header('Location', f'http://127.0.0.1:{sink.server_port}/target')
                self.end_headers()
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.do_GET()
            def log_message(self, *args): pass
        redirect = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        for server in (sink, redirect):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f'http://127.0.0.1:{redirect.server_port}'
        try:
            self.profile.fetch_models(api_key='fake', base_url=origin + '/provider/v1')
            self.profile.fetch_account_usage(api_key='fake', base_url=origin)
            with patch.object(self.auth, 'WHOAMI_URL', origin + '/whoami'):
                self.auth.validate('fake')
            with self.transport._open_stream(origin, {}, 'fake', 3, headers={'x-secret': 'fake-secret'}) as r:
                r.read()
            self.assertEqual(len(seen), 6)
            for headers in seen:
                self.assertNotIn('authorization', {h.lower() for h in headers})
                self.assertNotIn('x-secret', {h.lower() for h in headers})
        finally:
            for server in (sink, redirect):
                server.shutdown(); server.server_close()
