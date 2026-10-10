"""Prepare 100 authentic LongMemEval questions, without constructing new dialogue."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import urllib.request

URL='https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json'
QUOTAS={'multi-session':34,'temporal-reasoning':33,'knowledge-update':33}

def prepare(data,seed=42):
    rng=random.Random(seed);groups={}
    for kind,n in QUOTAS.items():
        eligible=[x for x in data if x['question_type']==kind and not x['question_id'].endswith('_abs')]
        rng.shuffle(eligible)
        if len(eligible)<n:raise ValueError(f'Not enough {kind} records')
        groups[kind]=eligible[:n]
    out=[]
    for i in range(max(QUOTAS.values())):
        for kind in QUOTAS:
            if i>=len(groups[kind]):continue
            x=groups[kind][i];history=[];evidence=[]
            sessions=sorted(zip(x['haystack_dates'],x['haystack_session_ids'],x['haystack_sessions']),key=lambda z:z[0])
            for date,sid,turns in sessions:
                for turn in turns:
                    idx=len(history)
                    history.append({'session_id':sid,'date':date,'role':turn['role'],'content':turn['content']})
                    if turn.get('has_answer'):evidence.append(idx)
            out.append({'id':x['question_id'],'kind':kind,'question':x['question'],
                        'question_date':x['question_date'],'references':[str(x['answer'])],
                        'history':history,'evidence_indices':evidence,
                        'answer_session_ids':x['answer_session_ids'],
                        'source':URL,'construction':'original_longmemeval_oracle'})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('longmemeval_oracle.json'))
    p.add_argument('--out',type=Path,default=Path('samples_100.json'));p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if not a.source.exists():
        with urllib.request.urlopen(URL,timeout=120)as r:raw=r.read()
        json.loads(raw);a.source.write_bytes(raw)
    xs=prepare(json.loads(a.source.read_text()),a.seed)
    a.out.write_text(json.dumps(xs,ensure_ascii=False,indent=2))
    manifest={'source':URL,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'seed':a.seed,'n':len(xs),'counts':dict(collections.Counter(x['kind']for x in xs)),
              'ids':[x['id']for x in xs]}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items()if k!='ids'},indent=2))

if __name__=='__main__':main()
