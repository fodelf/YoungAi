"""Sequential external-API correctness/lifetime probes; start a guarded server first."""
import argparse
import ast
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request


def fetch(base, route, payload=None):
    req = urllib.request.Request(base+route, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=20) as r: return json.load(r)


def idle(base, timeout=90):
    start=time.monotonic()
    while time.monotonic()-start < timeout:
        m=fetch(base,'/metrics')
        if m['live']['state']=='idle' and not m['live'].get('queued',0): return m
        time.sleep(.2)
    raise TimeoutError('server did not return to idle')


def request(base, payload, out):
    idle(base); out.mkdir()
    (out/'request.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    u=urllib.parse.urlsplit(base); conn=http.client.HTTPConnection(u.hostname,u.port,timeout=180)
    start=time.monotonic(); text=[]; reasoning=[]; result={'completed':False}; done=False
    try:
        conn.request('POST','/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        r=conn.getresponse()
        if r.status!=200: raise RuntimeError(f'HTTP {r.status}: {r.read().decode()[:500]}')
        with (out/'response.sse').open('wb') as stream:
            while line:=r.readline():
                stream.write(line)
                if time.monotonic()-start>1800: raise TimeoutError('request deadline')
                if not line.startswith(b'data: '): continue
                value=line[6:].strip()
                if value==b'[DONE]': done=True; break
                event=json.loads(value)
                if event.get('error'): raise RuntimeError(str(event['error']))
                if event.get('usage'): result['usage']=event['usage']
                for c in event.get('choices',[]):
                    d=c.get('delta',{}); piece=d.get('content') or ''; thought=d.get('reasoning_content') or d.get('reasoning') or ''
                    if piece or thought: result.setdefault('first_token_seconds',time.monotonic()-start)
                    text.append(piece); reasoning.append(thought)
                    if c.get('finish_reason'): result['finish_reason']=c['finish_reason']
        result['completed']=done and result.get('finish_reason') in ['stop','length']
        if not result['completed']: raise RuntimeError('incomplete stream')
        result['metrics']=max(idle(base)['requests'],key=lambda r:r['time'])
    except BaseException as ex:
        result['error']=f'{type(ex).__name__}: {ex}'
        raise
    finally:
        conn.close(); result.update(elapsed_seconds=time.monotonic()-start,response=''.join(text),reasoning=''.join(reasoning))
        result['all_sha256']=hashlib.sha256((result['reasoning']+'\0'+result['response']).encode()).hexdigest()
        (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(out.name,json.dumps({k:v for k,v in result.items() if k not in ['response','reasoning']}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--suite',choices=['quick','questions','lifetime'],default='quick')
    a=ap.parse_args(); base=a.url.removesuffix('/v1').rstrip('/'); a.out.mkdir()
    root=Path(__file__).resolve().parents[2]
    fixtures=json.loads((root/'tests/repro/metal_api_fixtures.json').read_text())
    p=fixtures['code']; runs=[]
    def run(name,payload):
        r=request(base,payload,a.out/name);runs.append({'name':name,**r});return r
    if a.suite=='quick':
        run('warmup',p|{'max_tokens':32,'messages':[{'role':'user','content':'Explain a FIFO queue.'}]})
        for name,payload in fixtures.items():
            r=run(name+'-1',payload);r2=run(name+'-2',payload)
            if r['all_sha256']!=r2['all_sha256']: raise AssertionError(name+' is not deterministic')
    elif a.suite=='questions':
        code=(root/'src/eval/eval_cases_a.c').read_text()
        for qi,case in enumerate(re.findall(r'\{\s*\.source =.*?\n    \}',code,re.S)[:4],1):
            question=ast.literal_eval(re.search(r'\.question = ("(?:[^"\\]|\\.)*")',case)[1])+'\n'
            choices=[ast.literal_eval(x) for x in re.findall(r'\.choice\[\d+\] = ("(?:[^"\\]|\\.)*")',case)]
            if choices: question+='\nChoices:\n'+''.join(f'{chr(65+i)}. {v}\n' for i,v in enumerate(choices))
            question+='\nSolve the '+('question' if choices else 'problem')+'. At the end, write exactly one final line in this format and do not write anything after it:\nAnswer: <'+('letter' if choices else 'integer')+'>'
            run(f'q{qi}',p|{'seed':1,'max_tokens':2048,'reasoning_effort':'high','messages':[{'role':'system','content':'You are solving a hard benchmark question. Reason carefully. The final answer must follow the requested format exactly.'},{'role':'user','content':question}]})
    else:
        # Synthetic ~70K prompt; count is recorded by the server, never estimated as a result.
        body=p['messages'][-1]['content']
        messages=[{'role':'user','content':('\n'.join([body]*9))}]
        payload=p|{'max_tokens':16,'messages':messages}
        r=run('long-cold',payload)
        for i in range(12):
            messages.extend([{'role':'assistant','content':r['response']},{'role':'user','content':'Continue.'}])
            r=run(f'long-followup-{i+1:02}',payload)
            if r['metrics']['reused']<60000: raise AssertionError('long prefix was not reused')
        u=urllib.parse.urlsplit(base); conn=http.client.HTTPConnection(u.hostname,u.port,timeout=30)
        cancel=p|{'messages':[{'role':'user','content':'Cancellation probe.\n'+body*17}]}
        conn.request('POST','/v1/chat/completions',json.dumps(cancel).encode(),{'Content-Type':'application/json'})
        for _ in range(100):
            if fetch(base,'/metrics')['live']['state']!='idle': break
            time.sleep(.1)
        else: raise AssertionError('cancellation probe never became active')
        time.sleep(1); start=time.monotonic(); conn.close()
        m=idle(base); last=max(m['requests'],key=lambda r:r['time'])
        (a.out/'cancel.json').write_text(json.dumps({'seconds_to_idle':time.monotonic()-start,'request':last},indent=2)+'\n')
        if last['finish']!='disconnect': raise AssertionError('disconnect not observed')
        run('post-cancel',p|{'max_tokens':32,'messages':[{'role':'user','content':'Explain a FIFO queue.'}]})
    (a.out/'summary.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
