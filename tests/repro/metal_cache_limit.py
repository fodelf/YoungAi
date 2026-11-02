"""External V4.1 regression: max_tokens must bound the speculative KV checkpoint.

Start a server with DSpark enabled. This short prompt fits the baseline too;
without the fix, accepted but un-emitted tokens can prevent the end-cache match.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def fetch(base, route, payload=None):
    req = urllib.request.Request(base + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def idle(base):
    for _ in range(450):
        result = fetch(base, '/metrics')
        if result['live']['state'] == 'idle' and not result['live'].get('queued', 0):
            return result
        time.sleep(.2)
    raise TimeoutError('server did not return to idle')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limits', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31])
    args = ap.parse_args()
    base = args.url.removesuffix('/v1').rstrip('/')
    args.out.mkdir(parents=True, exist_ok=False)
    model = fetch(base, '/v1/models')['data'][0]['id']
    message = {'role': 'user', 'content': 'Write a Python function that returns the first non-repeating character in a string. Explain its complexity.'}
    common = dict(model=model, temperature=0, top_p=1, seed=42, frequency_penalty=0,
                  presence_penalty=0, dry_multiplier=0, reasoning_effort='none', stream=False)
    reports = []
    for limit in args.limits:
        idle(base)
        first = fetch(base, '/v1/chat/completions', common | {'messages': [message], 'max_tokens': limit})
        first_metric = max(idle(base)['requests'], key=lambda r: r['time'])
        choice = first['choices'][0]
        if choice['finish_reason'] != 'length':
            raise AssertionError('probe ended before max_tokens; use the fixed long-answer prompt')
        follow = [message, {'role': 'assistant', 'content': choice['message']['content']},
                  {'role': 'user', 'content': 'Continue with one concrete edge case.'}]
        second = fetch(base, '/v1/chat/completions', common | {'messages': follow, 'max_tokens': 1})
        second_metric = max(idle(base)['requests'], key=lambda r: r['time'])
        expected = first_metric['prompt_tokens'] + first_metric['output_tokens']
        report = dict(limit=limit, expected_reused=expected, first=first, second=second,
                      first_metric=first_metric, second_metric=second_metric)
        reports.append(report)
        (args.out / 'results.json').write_text(json.dumps(reports, indent=2) + '\n')
        print(f'max_tokens={limit}: reused={second_metric["reused"]}, expected={expected}', flush=True)
        if second_metric['reused'] != expected:
            raise AssertionError('the end checkpoint contains un-emitted tokens or does not match')


if __name__ == '__main__':
    main()
