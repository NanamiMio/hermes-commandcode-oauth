"""Opt-in live checks. Uses the CLI grant and may consume account credits."""
import argparse
import asyncio
import base64
import importlib.util
import json
import struct
import sys
import uuid
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def blue_image():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress((b'\0' + b'\0\0\xff' * 64) * 64)) + chunk(b'IEND', b''))
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--model', default='minimax/minimax-m3.1-flash')
    parser.add_argument('--hermes-tree', type=Path)
    parser.add_argument('--vision', action='store_true')
    args = parser.parse_args()
    core = profile = None
    if args.hermes_tree:
        sys.path.insert(0, str(args.hermes_tree))
        spec = importlib.util.spec_from_file_location('cc_smoke', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
        from cc_smoke import auth, transport
        from agent.transports.chat_completions import ChatCompletionsTransport
        profile, core = plugin.commandcode_oauth, ChatCompletionsTransport()
        assert auth.validate(transport.cli_token()) is not None
        models = profile.fetch_models()
        assert args.model in models
        usage = profile.fetch_account_usage()
        assert usage is not None and usage.available
        assert all(w.scope == 'account' for w in usage.windows)
        print(json.dumps({'check': 'auth/catalog/usage', 'models': len(models), 'windows': len(usage.windows)}))
    else:
        import transport
    client = transport.CommandCodeAlphaClient(timeout=60)
    session_id = str(uuid.uuid4())
    observed = []
    original_open = transport._open_stream
    def observe(url, body, token, timeout, **kwargs):
        # Record only the fields under test, never credentials or prompt/image bodies.
        observed.append({'effort': body['params'].get('reasoning_effort'), 'thread': body.get('threadId')})
        return original_open(url, body, token, timeout, **kwargs)
    transport._open_stream = observe

    def request(messages, *, stream=False, effort=None, **kwargs):
        params = dict(model=args.model, messages=messages, max_tokens=2048, **kwargs)
        if core:
            params = core.build_kwargs(**params, provider_profile=profile, base_url=profile.base_url,
                reasoning_config={'effort': effort} if effort else None, supports_reasoning=True,
                session_id=session_id, max_tokens_param_fn=lambda value: {'max_tokens': value})
        elif effort:
            params['reasoning_effort'] = effort
        return client.chat.completions.create(**params, stream=stream)

    response = request([{'role': 'user', 'content': 'Reply only PARITY_OK. Do not use tools.'}], effort='high')
    assert response.choices[0].message.content.strip() == 'PARITY_OK'
    assert observed[-1]['effort'] == 'high'
    if core:
        assert core.normalize_response(response).content == 'PARITY_OK'
    print('text and reasoning wire through core: PASS' if core else 'text and reasoning wire: PASS')
    messages = [{'role': 'user', 'content': 'Call parity_lookup with key="sample". After its result, reply only with the value.'}]
    tools = [{'type': 'function', 'function': {'name': 'parity_lookup', 'description': 'Look up a test value.',
              'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}}]
    response = request(messages, tools=tools)
    calls = response.choices[0].message.tool_calls
    assert calls and response.choices[0].finish_reason == 'tool_calls'
    for call in calls:
        assert call.function.name == 'parity_lookup'
        assert json.loads(call.function.arguments) == {'key': 'sample'}
    messages.append({'role': 'assistant', 'content': response.choices[0].message.content,
                     'tool_calls': [{'id': c.id, 'type': 'function', 'function': vars(c.function)} for c in calls]})
    messages.extend({'role': 'tool', 'tool_call_id': c.id, 'content': 'PARITY_TOOL_OK'} for c in calls)
    # Main transport takes explicit request overrides; auxiliary/direct callers use top-level kwargs.
    kwargs = {'request_overrides': {'tool_choice': 'none'}} if core else {'tool_choice': 'none'}
    chunks = list(request(messages, tools=tools, stream=True, **kwargs))
    assert ''.join(c.choices[0].delta.content or '' for c in chunks).strip() == 'PARITY_TOOL_OK'
    assert sum(c.choices[0].finish_reason is not None for c in chunks) == 1
    print('tool round trip and streaming termination: PASS')

    if args.vision:
        image = blue_image()
        response = request([{'role': 'user', 'content': [{'type': 'text', 'text': 'What color is this square? Reply one English word.'}, image]}], effort='low')
        assert 'blue' in (response.choices[0].message.content or '').lower()
        print('user image: PASS')
        # A real tool call, with image content produced locally as the tool result.
        vision_tools = [{'type': 'function', 'function': {'name': 'capture_square', 'description': 'Capture a square image.',
                         'parameters': {'type': 'object', 'properties': {}}}}]
        history = [{'role': 'user', 'content': 'Use capture_square. What color is the square it returns? Answer one word after seeing the image.'}]
        response = request(history, tools=vision_tools)
        calls = response.choices[0].message.tool_calls
        assert calls and all(c.function.name == 'capture_square' for c in calls)
        history.append({'role': 'assistant', 'content': response.choices[0].message.content,
                        'tool_calls': [{'id': c.id, 'type': 'function', 'function': vars(c.function)} for c in calls]})
        history.extend({'role': 'tool', 'tool_call_id': c.id, 'content': [image]} for c in calls)
        response = request(history, tools=vision_tools, effort="low", **kwargs)
        assert 'blue' in (response.choices[0].message.content or '').lower()
        print('tool image round trip: PASS')

    if core:
        from agent.auxiliary_client import _to_async_client, _build_call_kwargs
        async_client, model = _to_async_client(client, args.model, is_vision=True)
        kw = _build_call_kwargs(provider=profile.name, model=model,
                                messages=[{'role': 'user', 'content': 'Reply only ASYNC_OK.'}],
                                max_tokens=512, base_url=profile.base_url,
                                reasoning_config={'effort': 'low'})
        async def run():
            return await async_client.chat.completions.create(**kw)
        response = asyncio.run(run())
        assert (response.choices[0].message.content or '').strip() == 'ASYNC_OK'
        assert observed[-1]['effort'] == 'low'
        assert all(o['thread'] == session_id for o in observed[:-1])
        print('Hermes async auxiliary route and main session identity: PASS')


if __name__ == '__main__':
    main()
