"""Paired typo correction/retrieval/oracle-context QA benchmark."""
import argparse, concurrent.futures as cf, hashlib, json, os, random, re, string, time
from pathlib import Path
import requests
import numpy as np


def norm(s):
    s = s.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def em(pred, answers):
    return float(any(norm(pred) == norm(a) for a in answers))


def corrupt(q, rng):
    words = list(re.finditer(r'\b[A-Za-z]{4,}\b', q))
    if not words: return None
    m = rng.choice(words); w = m.group(); op = rng.choice(['swap','delete','insert','replace'])
    i = rng.randrange(1, len(w)-1)
    if op == 'swap':
        candidates = [j for j in range(len(w)-1) if w[j] != w[j+1]]
        if not candidates: return corrupt(q, rng)
        i = rng.choice(candidates); bad = w[:i]+w[i+1]+w[i]+w[i+2:]
    elif op == 'delete': bad = w[:i]+w[i+1:]
    elif op == 'insert': bad = w[:i]+rng.choice(string.ascii_lowercase)+w[i:]
    else: bad = w[:i]+rng.choice(string.ascii_lowercase.replace(w[i].lower(),''))+w[i+1:]
    return q[:m.start()]+bad+q[m.end():], {'operation':op,'original':w,'corrupted':bad}


def fetch(url, path):
    if not path.exists():
        r = requests.get(url, timeout=120); r.raise_for_status(); path.write_bytes(r.content)
    return json.loads(path.read_text())


def validate_keyword(result, question):
    if not isinstance(result,dict): raise ValueError('Expected JSON object')
    word=result.get('keyword')
    if word is None:
        if 'keyword' not in result: raise ValueError('Missing keyword field')
        return None
    if result.get('kind') not in {'entity','topic','relation'}: raise ValueError('Invalid keyword kind')
    if not isinstance(word,str) or not word.strip(): raise ValueError('Invalid keyword type')
    word=word.strip()
    match=re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',question,re.I)
    if not match: raise ValueError('Keyword/phrase absent from final question: '+repr(word))
    candidates=list(re.finditer(r'\b[A-Za-z]{4,}\b',match.group()))
    if not candidates: raise ValueError('No eligible alphabetic word in keyword: '+repr(word))
    # A phrase is accepted, but exactly one longest eligible component is corrupted.
    token=max(candidates,key=lambda m:len(m.group()))
    start=match.start()+token.start(); end=match.start()+token.end()
    return {**result,'selected_phrase':match.group(),'keyword':question[start:end],
            'start':start,'end':end,'component_policy':'longest_word_first_tie'}


def choose_keyword(x, llm):
    messages=[
        {'role':'system','content':'Select the most important explicit keyword or named-entity phrase in the FINAL question for document retrieval. Use history only to understand intent. Prefer distinctive entities or technical topics, then requested relations. Do not choose question words, auxiliaries, generic verbs, or pronouns. The keyword must be an exact substring of the final question and contain an English alphabetic word of at least four letters. If none exists return null. Output ONLY JSON: {"keyword": string or null, "kind": "entity" or "topic" or "relation" or "none", "reason": string}.'},
        {'role':'user','content':json.dumps({'history':x['history'],'final_question':x['clean_query']})}]
    attempts=[]
    for attempt in range(3):
        # v2 avoids reusing invalid outputs from the previous selector prompt.
        response=llm.ask('keyword_selection_v2_'+str(attempt),messages)
        try:
            text=response['text'].strip()
            if text.startswith('```'):
                text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
            result=validate_keyword(json.loads(text),x['clean_query'])
            x['keyword_selection_attempts']=attempts
            return result
        except (ValueError,TypeError,KeyError) as error:
            attempts.append({'output':response['text'],'error':str(error)})
            messages += [{'role':'assistant','content':response['text']},
                         {'role':'user','content':'Validation failed: '+str(error)+'. Return valid JSON with an exact keyword from the FINAL question, or null if none qualifies.'}]
    x['keyword_selection_attempts']=attempts
    x['keyword_selection_failure']='invalid_output_after_3_attempts'
    return None


def targeted_corrupt(q, selection, rng):
    word=selection['keyword']; corrupted,meta=corrupt(word,rng)
    return q[:selection['start']]+corrupted+q[selection['end']:],meta


def prepare(a, out):
    rng = random.Random(a.seed); docs = {}; groups = {'single':[], 'multi':[]}
    if a.input:
        for line in Path(a.input).read_text().splitlines():
            x = json.loads(line); assert x['clean_query'] != x['typo_query']
            assert x['answers'] and x['gold_doc_id'] in x['documents']
            for k,v in x.pop('documents').items():
                if k in docs and docs[k] != v: raise ValueError('Conflicting document IDs')
                docs[k] = v
            groups[x['split']].append(x)
    else:
        sq = fetch('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json', out/'squad.json')
        cq = fetch('https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json', out/'coqa.json')
        for article in sq['data']:
            for p in article['paragraphs']:
                did = 'squad-'+hashlib.sha256(p['context'].encode()).hexdigest()[:20]; docs[did] = p['context']
                for x in p['qas']:
                    groups['single'].append(dict(id='squad-'+x['id'],split='single',clean_query=x['question'],history=[],answers=[z['text'] for z in x['answers']],gold_doc_id=did))
        for story in cq['data']:
            did = 'coqa-'+str(story['id']); docs[did] = story['story']; history=[]
            for q,ans in zip(story['questions'],story['answers']):
                answers = [ans['input_text']]
                for additional in story.get('additional_answers',{}).values():
                    answers.extend(z['input_text'] for z in additional if z['turn_id']==q['turn_id'])
                if history:
                    groups['multi'].append(dict(id=did+'-'+str(q['turn_id']),split='multi',clean_query=q['input_text'],history=list(history),answers=answers,gold_doc_id=did))
                history += [{'role':'user','content':q['input_text']},{'role':'assistant','content':ans['input_text']}]
    samples=[]; excluded=[]; selector=LLM(a,out) if not a.input else None
    for split, rows in groups.items():
        rng.shuffle(rows); selected=[]; stories=set()
        for x in rows:
            if not a.input and split=='multi' and x['gold_doc_id'] in stories: continue
            if not a.input:
                selection=choose_keyword(x,selector)
                if selection is None:
                    excluded.append({'id':x['id'],'split':split,'question':x['clean_query'],'reason':x.get('keyword_selection_failure','no explicit informative keyword'),'attempts':x.get('keyword_selection_attempts',[])}); continue
                x['keyword_selection']=selection
                x['typo_query'],x['typo_metadata']=targeted_corrupt(x['clean_query'],selection,rng)
                x['random_query'],x['random_typo_metadata']=corrupt(x['clean_query'],random.Random(str(a.seed)+x['id']))
                x['typo_source']='synthetic_important_keyword_noise'
            selected.append(x); stories.add(x['gold_doc_id'])
            if len(selected)==a.n: break
        if len(selected)<a.n: raise ValueError(f'{split}: only {len(selected)} eligible samples')
        samples.extend(selected)
    (out/'excluded.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in excluded))
    gold={x['gold_doc_id'] for x in samples}; distractors=list(set(docs)-gold); distractors.sort(); rng.shuffle(distractors)
    keep=gold|set(distractors[:a.distractors]); docs={k:docs[k] for k in sorted(keep)}
    return samples,docs


class LLM:
    def __init__(self,a,out):
        from openai import OpenAI
        self.client=OpenAI(); self.a=a; self.cache=out/'cache'; self.cache.mkdir(exist_ok=True)
    def ask(self,kind,messages):
        config={'model':self.a.model,'messages':messages,'kind':kind,'effort':self.a.effort}
        key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); p=self.cache/(key+'.json')
        if p.exists():
            cached=json.loads(p.read_text())
            if isinstance(cached.get('text'),str) and cached['text'].strip(): return cached
        t=time.perf_counter(); failures=[]
        for attempt,budget in enumerate([512,1024,2048],1):
            kwargs={'model':self.a.model,'input':messages,'max_output_tokens':budget}
            if self.a.effort != 'omit': kwargs['reasoning']={'effort':self.a.effort}
            r=self.client.responses.create(**kwargs)
            answer=(r.output_text or '').strip()
            if r.status=='completed' and answer:
                result={'text':answer,'seconds':time.perf_counter()-t,'usage':r.usage.model_dump(),
                        'attempts':attempt,'prior_empty_responses':failures}
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w',dir=self.cache,suffix='.tmp',delete=False) as f:
                    json.dump(result,f); temporary=Path(f.name)
                temporary.replace(p)
                return result
            diagnostic={'attempt':attempt,'status':r.status,'output_types':[getattr(item,'type',None) for item in r.output],
                        'incomplete_details':str(getattr(r,'incomplete_details',None))}
            failures.append(diagnostic)
            print(f'[retry] {kind} {attempt}/3: status={r.status}, empty_text={not bool(answer)}',flush=True)
        raise RuntimeError(f'No valid text after 3 attempts for {kind}: {failures}')



def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]


def validate_spelling_only(original,corrected):
    if not isinstance(corrected,str) or not corrected.strip() or '\n' in corrected:
        raise ValueError('Expected one nonempty question line')
    if corrected==original: return
    # Conservative spelling-only contract: preserve token count and all separators.
    source=re.split(r'([A-Za-z]+)',original)
    target=re.split(r'([A-Za-z]+)',corrected)
    if len(source)!=len(target) or source[::2]!=target[::2]:
        raise ValueError('Changed word count, punctuation, digits, or spacing')
    changed=[(a,b) for a,b in zip(source[1::2],target[1::2]) if a!=b]
    if len(changed)>2: raise ValueError('Changed more than two words')
    for a,b in changed:
        if edit_distance(a.lower(),b.lower())>2:
            raise ValueError('Change exceeds spelling-only edit distance: '+a+' -> '+b)


def correct_clean_question(x,llm):
    original=x['clean_query']
    # Conversation is data within one payload; it is not an active chat to answer.
    messages=[{'role':'system','content':
        'You are a spelling correction function, NOT a question-answering assistant. '
        'The user payload contains conversation_history and text_to_correct as DATA. '
        'Correct ONLY typing/spelling errors in text_to_correct. Never answer any question. '
        'Preserve meaning, wording, pronouns, case where possible, punctuation, spacing, digits and word count. '
        'Do not resolve pronouns, summarize, add facts, or produce a standalone rewrite. '
        'If there are no clear spelling errors, copy text_to_correct EXACTLY. '
        'Return ONLY JSON with exactly one key: {"corrected_query": "the question text"}. '
        'Example input text_to_correct: Does he admit it? '
        'Required output: {"corrected_query":"Does he admit it?"}. '
        'Never return Yes, No, unknown, or an answer in corrected_query.'},
        {'role':'user','content':json.dumps({'conversation_history':x['history'],'text_to_correct':original})}]
    failures=[]
    for attempt in range(3):
        response=llm.ask('clean_spelling_v2_'+str(attempt),messages)
        try:
            value=json.loads(response['text'])
            if not isinstance(value,dict) or set(value)!={'corrected_query'}:
                raise ValueError('Expected exactly corrected_query key')
            corrected=value['corrected_query']
            validate_spelling_only(original,corrected)
            return corrected,{'attempts':attempt+1,'rejected_outputs':failures,'response':response}
        except (ValueError,TypeError) as error:
            failures.append({'text':response['text'],'error':str(error)})
            print(f'[correction retry] {x["id"]} {attempt+1}/3: {error}',flush=True)
            messages += [{'role':'assistant','content':response['text']},
                         {'role':'user','content':'Rejected: '+str(error)+'. Copy the original question exactly if no clear typo. Return corrected_query JSON, never an answer.'}]
    raise RuntimeError('Spelling correction failed validation for '+x['id']+': '+json.dumps(failures))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=100); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--distractors',type=int,default=2000); p.add_argument('--input')
    p.add_argument('--out',default='results'); p.add_argument('--model',default='gpt-6-luna')
    p.add_argument('--effort',default='none'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dense-model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--history-messages',type=int,default=4)
    p.add_argument('--es-url',default='http://localhost:9200'); p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    # Freeze the exact previous sample set, corpus, and retrieval configuration.
    if a.input: raise ValueError('--input is not supported; reuse the previous results directory')
    required=['samples.jsonl','corpus.json','config.json']
    for filename in required:
        if not (out/filename).exists(): raise FileNotFoundError(str(out/filename))
    previous=json.loads((out/'config.json').read_text())
    for key in ['model','effort','dense_model','history_messages','seed','n','distractors']:
        if key in previous: setattr(a,key,previous[key])
    samples=[json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines() if line.strip()]
    docs=json.loads((out/'corpus.json').read_text())
    if len({x['id'] for x in samples})!=len(samples): raise ValueError('Duplicate sample IDs')
    print(f'Reusing {len(samples)} samples, {len(docs)} documents; model={a.model}; dense={a.dense_model}; history_messages={a.history_messages}',flush=True)
    (out/'clean_correction_v2_config.json').write_text(json.dumps(vars(a),indent=2))
    if a.prepare_only: return
    from elasticsearch import Elasticsearch, helpers
    from sentence_transformers import SentenceTransformer
    es=Elasticsearch(a.es_url,request_timeout=120,**({'api_key':os.environ['ES_API_KEY']} if os.getenv('ES_API_KEY') else {}))
    # A fresh isolated index avoids replacing the user's existing indexes.
    index='typo-bench-'+str(time.time_ns())
    es.indices.create(index=index,settings={'number_of_shards':1,'number_of_replicas':0},mappings={'properties':{'text':{'type':'text','analyzer':'standard'}}})
    helpers.bulk(es,({'_index':index,'_id':k,'_source':{'text':v}} for k,v in docs.items()))
    es.indices.refresh(index=index)
    dense=SentenceTransformer(a.dense_model); ids=list(docs)
    # Chunk long documents; retrieve documents using their maximum chunk score.
    chunks=[]; owners=[]
    for i,did in enumerate(ids):
        words=docs[did].split()
        for start in range(0,len(words),100):
            chunks.append(' '.join(words[start:start+150])); owners.append(i)
    matrix=dense.encode(chunks,normalize_embeddings=True,batch_size=64,show_progress_bar=True)
    llm=LLM(a,out)
    def correct_term(q):
        response=es.search(index=index,body={'size':0,'suggest':{'spell':{'text':q,'term':{'field':'text','suggest_mode':'missing','max_edits':2,'prefix_length':0,'min_word_length':3,'size':1}}}})
        corrected=q
        for token in reversed(response['suggest']['spell']):
            if token['options']:
                s=token['offset']; e=s+token['length']; corrected=corrected[:s]+token['options'][0]['text']+corrected[e:]
        return corrected
    def work(x):
        history=x['history']; q=x['typo_query']
        corrected,correction_audit=correct_clean_question(x,llm)
        queries={'clean':x['clean_query'],'clean_term':correct_term(x['clean_query']),'clean_llm':corrected}
        row={'id':x['id'],'split':x['split'],'queries':queries,'gold_doc_id':x['gold_doc_id'],'correction_audit':correction_audit,'variants':{}}
        for variant,query in queries.items():
            # Same history policy for all variants, no query rewriting.
            retrieval_history=history[-a.history_messages:] if a.history_messages>0 else []
            retrieval_query=' '.join(m['content'] for m in retrieval_history+[{'content':query}])
            hits=es.search(index=index,query={'match':{'text':retrieval_query}},size=10)['hits']['hits']; bm=[h['_id'] for h in hits]
            vec=dense.encode([retrieval_query],normalize_embeddings=True)[0]
            scores=matrix@vec; ds=np.full(len(ids),-np.inf); np.maximum.at(ds,owners,scores)
            dr=[ids[i] for i in np.argsort(-ds)[:10]]
            metrics={}
            for backend,ranking in [('bm25',bm),('dense',dr)]:
                rank=ranking.index(x['gold_doc_id'])+1 if x['gold_doc_id'] in ranking else None
                metrics[backend]={'mrr10':1/rank if rank else 0,**{f'recall{k}':float(rank is not None and rank<=k) for k in [1,3,5,10]}}
            generation=llm.ask('answer',[{'role':'system','content':'Answer the final question using the supplied passage and conversation. Return only the shortest answer text, without explanation. For yes/no questions return yes or no. If unanswerable return unknown.'},{'role':'user','content':'Passage:\n'+docs[x['gold_doc_id']]}]+history+[{'role':'user','content':query}])
            metrics.update(answer= generation['text'],answer_em=em(generation['text'],x['answers']),answer_seconds=generation['seconds'],correction_em=em(query,[x['clean_query']]),changed=float(query!=x['clean_query']))
            row['variants'][variant]=metrics
        return row
    rows=[]
    try:
        # Serialize encoder access to avoid concurrent model execution on GPU.
        # LLM requests run in parallel; dense encoding is guarded by a lock.
        import threading
        lock=threading.Lock(); original=dense.encode
        def encode(*args,**kwargs):
            with lock: return original(*args,**kwargs)
        dense.encode=encode
        checkpoint=out/'clean_correction_v2_details.jsonl'
        completed={}
        if checkpoint.exists():
            for number,line in enumerate(checkpoint.read_text().splitlines(),1):
                if not line.strip(): continue
                row=json.loads(line)
                if set(row.get('variants',{}))=={'clean','clean_term','clean_llm'}:
                    completed[row['id']]=row
        sample_ids={x['id'] for x in samples}
        if set(completed)-sample_ids: raise ValueError('Checkpoint contains IDs outside current samples')
        rows=[completed[x['id']] for x in samples if x['id'] in completed]
        pending=[x for x in samples if x['id'] not in completed]
        print(f'Resuming: {len(rows)}/{len(samples)} complete; {len(pending)} remaining',flush=True)
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool, checkpoint.open('a') as f:
            futures={pool.submit(work,x):x['id'] for x in pending}
            for future in cf.as_completed(futures):
                row=future.result()
                rows.append(row); f.write(json.dumps(row)+'\n'); f.flush(); print(f'{len(rows)}/{len(samples)}',flush=True)
        ordering={x['id']:i for i,x in enumerate(samples)}
        rows.sort(key=lambda row:ordering[row['id']])
    finally:
        es.indices.delete(index=index)
    rng=np.random.default_rng(a.seed); summary=[]
    for split in ['single','multi']:
        subset=[x for x in rows if x['split']==split]
        for variant in ['clean','clean_term','clean_llm']:
            entry={'split':split,'variant':variant,'n':len(subset)}
            for key in ['answer_em','correction_em','changed']:
                entry[key]=float(np.mean([x['variants'][variant][key] for x in subset]))
            for metric in ['answer_em','bm25','dense']:
                base_values=[x['variants']['clean']['answer_em'] if metric=='answer_em' else x['variants']['clean'][metric]['recall1'] for x in subset]
                values=[x['variants'][variant]['answer_em'] if metric=='answer_em' else x['variants'][variant][metric]['recall1'] for x in subset]
                losses=sum(b==1 and v==0 for b,v in zip(base_values,values))
                wins=sum(b==0 and v==1 for b,v in zip(base_values,values))
                successes=sum(base_values)
                entry[metric+'_wins']=wins; entry[metric+'_losses']=losses
                entry[metric+'_damage_rate']=losses/successes if successes else None
            for backend in ['bm25','dense']:
                for key in ['mrr10','recall1','recall3','recall5','recall10']:
                    values=np.array([x['variants'][variant][backend][key] for x in subset]); base=np.array([x['variants']['clean'][backend][key] for x in subset])
                    entry[backend+'_'+key]=float(values.mean())
                    diff=values-base; boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(axis=1)
                    entry[backend+'_'+key+'_delta']=float(diff.mean()); entry[backend+'_'+key+'_delta_ci95']=np.quantile(boot,[.025,.975]).tolist()
            diff=np.array([x['variants'][variant]['answer_em']-x['variants']['clean']['answer_em'] for x in subset]); boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(axis=1)
            entry['answer_em_delta']=float(diff.mean()); entry['answer_em_delta_ci95']=np.quantile(boot,[.025,.975]).tolist(); summary.append(entry)
    (out/'clean_correction_v2_summary.json').write_text(json.dumps(summary,indent=2))
    import csv
    with (out/'clean_correction_v2_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    lines=['| split | variant | BM25 R@1 | dense R@1 | gold-context EM | ΔEM vs clean | change rate |','|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['variant']} | {r['bm25_recall1']:.3f} | {r['dense_recall1']:.3f} | {r['answer_em']:.3f} | {r['answer_em_delta']:+.3f} | {r['changed']:.3f} |" for r in summary]
    (out/'clean_correction_v2_report.md').write_text('\n'.join(lines)); print('\n'.join(lines))

if __name__=='__main__': main()
