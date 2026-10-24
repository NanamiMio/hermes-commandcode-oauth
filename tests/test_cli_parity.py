"""Regressions from Command Code 1.87.1 source and captured wire events."""
import importlib.util
import json
import socket
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import test_plugin as support
from test_plugin import _Handler, _line, _text, FINISH_OK
import auth
import transport


class StreamParityTests(unittest.TestCase):
    setUpClass = classmethod(support.PluginTestCase.setUpClass.__func__)
    tearDownClass = classmethod(support.PluginTestCase.tearDownClass.__func__)
    client = support.PluginTestCase.client
    call = support.PluginTestCase.call

    def test_step_is_not_terminal(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(transport.CommandCodeAPIError) as caught:
                    result = self.call(_text('partial') + _line(type='finish-step', finishReason='stop'), stream=stream)
                    if stream:
                        list(result)
                self.assertEqual(caught.exception.status_code, 502)

    def test_one_terminal_chunk_and_total_usage(self):
        payload = (_text('ok') + _line(type='finish-step', finishReason='stop', usage={'inputTokens': 10})
                   + FINISH_OK)
        chunks = list(self.call(payload, stream=True))
        self.assertEqual(sum(c.choices[0].finish_reason is not None for c in chunks), 1)
        self.assertEqual(chunks[-1].usage.prompt_tokens, 7596)

    def test_error_after_finish_does_not_emit_success(self):
        result = self.call(_text('ok') + FINISH_OK + _line(type='error', statusCode='429', error='rate limited'), stream=True)
        seen = []
        with self.assertRaises(transport.CommandCodeAPIError) as caught:
            for chunk in result:
                seen.append(chunk)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertFalse(any(c.choices[0].finish_reason for c in seen))

    def test_abort_even_after_output_is_failure(self):
        with self.assertRaises(transport.CommandCodeAPIError):
            self.call(_text('partial') + _line(type='abort'))

    def test_non_object_json_events_are_ignored(self):
        self.assertEqual(self.call(b'null\n[]\n' + _text('ok') + FINISH_OK).choices[0].message.content, 'ok')

    def test_args_alias_and_tool_roundtrip(self):
        result = self.call(_line(type='tool-call', toolCallId='call1', toolName='lookup', args={'city': 'Paris'}) + FINISH_OK)
        call = result.choices[0].message.tool_calls[0]
        self.assertEqual(json.loads(call.function.arguments), {'city': 'Paris'})
        self.assertEqual(result.choices[0].finish_reason, 'tool_calls')
        _, messages = transport.format_messages([
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call.id, 'function': vars(call.function)}]},
            {'role': 'tool', 'tool_call_id': call.id, 'content': 'sunny'},
        ])
        self.assertEqual(messages[1]['content'][0]['toolName'], 'lookup')

    def test_refusal_and_length_mapping(self):
        for raw, expected in [('max_tokens', 'length'), ('refusal', 'content_filter'), ('end_turn', 'stop')]:
            response = self.call(_text('result') + _line(type='finish', finishReason='other', rawFinishReason=raw))
            self.assertEqual(response.choices[0].finish_reason, expected)

    def test_trailing_cache_write_event(self):
        response = self.call(_text('ok') + FINISH_OK + _line(type='cache-write-tokens', cacheWriteTokens=123))
        self.assertEqual(response.usage.prompt_tokens_details.cache_write_tokens, 123)
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 7552)

    def test_session_is_stable_and_headers_are_honoured(self):
        client = self.client(default_headers={'x-custom': 'custom'})
        _Handler.payload = _text('ok') + FINISH_OK
        sessions = []
        for _ in range(2):
            client.chat.completions.create(model='model', messages=[]).choices
            sessions.append(_Handler.captured['headers']['x-session-id'])
        self.assertEqual(sessions[0], sessions[1])
        self.assertEqual(_Handler.captured['headers']['x-custom'], 'custom')


class MessageParityTests(unittest.TestCase):
    def test_reasoning_and_developer_messages(self):
        system, messages = transport.format_messages([
            {'role': 'developer', 'content': 'instruction'},
            {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'thinking'},
        ])
        self.assertEqual(system, 'instruction')
        self.assertEqual(messages[0]['content'], [{'type': 'reasoning', 'text': 'thinking'}, {'type': 'text', 'text': 'answer'}])

    def test_media_order_and_anthropic_shape(self):
        _, messages = transport.format_messages([{'role': 'user', 'content': [
            {'type': 'text', 'text': 'before'},
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'abc'}},
            {'type': 'text', 'text': 'after'},
        ]}])
        self.assertEqual(messages[0]['content'], [
            {'type': 'text', 'text': 'before'},
            {'type': 'image', 'image': 'data:image/png;base64,abc', 'mimeType': 'image/png'},
            {'type': 'text', 'text': 'after'},
        ])

    def test_case_insensitive_vision_gate(self):
        _, messages = transport.format_messages([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}], 'zai-org/glm-5.3')
        self.assertEqual(messages[0]['content'][0]['type'], 'text')

    def test_budget_and_effort(self):
        self.assertEqual(transport._request_body({}, 'm')['params']['max_tokens'], 64000)
        params = transport._request_body({'max_completion_tokens': 99, 'reasoning_effort': 'high'}, 'm')['params']
        self.assertEqual(params['max_tokens'], 99)
        self.assertEqual(params['reasoning_effort'], 'high')



class CallbackParityTests(unittest.TestCase):
    def receive(self, send):
        server, thread, result = auth._start_callback_server('expected', port=0)
        try:
            send(server.server_port)
            return auth._wait_for_callback(result, timeout=2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def post(self, port, payload, form=False):
        data = urllib.parse.urlencode(payload) if form else json.dumps(payload)
        req = urllib.request.Request(f'http://127.0.0.1:{port}/callback', data=data.encode(),
            headers={'Content-Type': 'application/x-www-form-urlencoded' if form else 'application/json'})
        return urllib.request.urlopen(req, timeout=5)

    def test_bad_state_then_valid_form(self):
        def send(port):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post(port, {'state': 'wrong', 'apiKey': 'fake'})
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()
            with self.post(port, {'state': 'expected', 'apiKey': 'fake'}, form=True) as response:
                self.assertEqual(response.headers['Content-Type'], 'text/html; charset=utf-8')
        payload = self.receive(send)
        self.assertEqual(payload['apiKey'], 'fake')

    def test_login_binds_before_browser_and_encodes_url(self):
        def open_browser(url):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            self.assertEqual(query['mode'], ['redirect'])
            callback = urllib.parse.urlparse(query['callback'][0])
            self.post(callback.port, {'state': query['state'][0], 'apiKey': 'fake'}, form=True).close()
        with patch.object(auth, 'read_cli_token', side_effect=auth.CommandCodeAuthError('absent')), \
             patch('webbrowser.open', side_effect=open_browser), patch('builtins.print'):
            self.assertEqual(auth.login()['token'], 'fake')

    def test_denial_ends_wait(self):
        def send(port):
            self.post(port, {'state': 'expected', 'error': 'access_denied'}).close()
        with self.assertRaisesRegex(auth.CommandCodeAuthError, 'access_denied'):
            self.receive(send)

