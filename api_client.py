import json, os, time, urllib.request, urllib.error

def api(system, user, args, schema=None):
    body = {'model': args.model,
            'input': [{'role': 'system', 'content': system},
                      {'role': 'user', 'content': user}],
            'max_output_tokens': args.max_output_tokens}
    if schema is not None:
        body['text'] = {'format': {'type': 'json_schema', 'name': 'result', 'strict': True, 'schema': schema}}
    if args.effort != 'omit':
        body['reasoning'] = {'effort': args.effort}
    endpoint = args.base_url.rstrip('/') + '/responses'
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={
        'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json'})
    start = time.perf_counter()
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result = json.load(response)
            text = ''.join(part.get('text', '') for item in result.get('output', [])
                           if item.get('type') == 'message'
                           for part in item.get('content', [])
                           if part.get('type') == 'output_text').strip()
            if result.get('status') != 'completed' or not text:
                raise RuntimeError('Incomplete/empty API output; raise --max-output-tokens if needed')
            usage = result.get('usage', {})
            return {'text': text, 'seconds': time.perf_counter() - start,
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0), 'attempts': attempt + 1}
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == args.retries:
                raise RuntimeError(f'API HTTP {exc.code}: {exc.read().decode()[:1000]}') from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == args.retries:
                raise
        time.sleep(min(2 ** attempt, 16))

