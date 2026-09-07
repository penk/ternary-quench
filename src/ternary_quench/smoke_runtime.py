#!/usr/bin/env python3
"""Exercise a real Ollama model: text, structured response, tool round trip."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:11434')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', choices=['ollama', 'openai'], default='ollama')
    a = p.parse_args()
    report = {'model': a.model, 'url': a.url, 'protocol': a.protocol, 'thinking': False,
              'temperature': 0, 'num_ctx': 8192, 'num_predict': 256, 'cases': []}

    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2)+'\n')

    def call(messages, **extra):
        payload = {'model': a.model, 'messages': messages, 'stream': False,
                   'think': False, 'keep_alive': '5m',
                   'options': {'temperature': 0, 'seed': 7, 'num_ctx': 8192, 'num_predict': 256}, **extra}
        if a.protocol == 'openai':
            payload = {'model': a.model, 'messages': messages, 'stream': False,
                       'temperature': 0, 'seed': 7, 'max_tokens': 256,
                       'repeat_penalty': 1.05, 'repeat_last_n': 256,
                       'chat_template_kwargs': {'enable_thinking': False}, **extra}
        endpoint = '/api/chat' if a.protocol == 'ollama' else '/v1/chat/completions'
        req = urllib.request.Request(a.url+endpoint, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as f:
            reply = json.load(f)
        if a.protocol == 'openai':
            reply['message'] = reply['choices'][0]['message']
        return {'request': payload, 'reply': reply, 'elapsed_s': time.monotonic()-start}

    def record(name, result, passed):
        report['cases'].append({'name': name, 'passed': bool(passed), **result})
        save()
        print(json.dumps({'name': name, 'passed': bool(passed),
                          'message': result['reply'].get('message'),
                          'elapsed_s': result['elapsed_s']}), flush=True)

    r = call([{'role': 'user', 'content': 'What is 17 + 25? Reply with just the number.'}])
    record('arithmetic', r, r['reply']['message'].get('content', '').strip() == '42')
    r = call([{'role': 'user', 'content': 'Return only a JSON object with key "names" containing the list ["Ada", "Linus"] and key "count" equal to 2.'}])
    try:
        parsed = json.loads(r['reply']['message'].get('content', ''))
        passed = parsed == {'names': ['Ada', 'Linus'], 'count': 2}
    except ValueError:
        passed = False
    record('json_without_grammar', r, passed)
    messages = [{'role': 'user', 'content': 'Use get_weather to check the current weather in Berlin, then report the temperature.'}]
    tools = [{'type': 'function', 'function': {'name': 'get_weather',
             'description': 'Look up current weather for a city.',
             'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    r = call(messages, tools=tools)
    calls = r['reply']['message'].get('tool_calls', [])
    arguments = calls[0]['function'].get('arguments') if calls else None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = None
    passed = len(calls) == 1 and calls[0]['function']['name'] == 'get_weather' and arguments == {'city': 'Berlin'}
    record('tool_call', r, passed)
    if passed:
        messages += [r['reply']['message'], {'role': 'tool', 'tool_name': 'get_weather',
                     'content': '{"city":"Berlin","temperature_c":19,"condition":"sunny"}'}]
        if a.protocol == 'openai':
            messages[-1]['tool_call_id'] = calls[0]['id']
        r = call(messages, tools=tools)
        msg = r['reply']['message']
        record('tool_result', r, '19' in msg.get('content', '') and not msg.get('tool_calls'))
    report['status'] = 'PASS' if len(report['cases']) == 4 and all(c['passed'] for c in report['cases']) else 'FAIL'
    save()
    print(report['status'], flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
