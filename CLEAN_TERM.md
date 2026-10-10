# 原문 clean → Elasticsearch term 교정 평가

기존 benchmark 결과 폴더를 그대로 사용합니다. 기존 samples.jsonl/corpus.json/config.json을 읽고 질문·corpus를 재생성하지 않습니다. API cache도 같은 폴더를 사용합니다.

```bash
python3 bench_clean_term.py --out results_keyword --workers 4
```

config.json에 저장된 model/effort/dense_model/history_messages/seed/n/distractors를 그대로 적용합니다. ES 주소만 필요에 따라 `--es-url http://localhost:9201`로 지정합니다. 기존 ES 서버를 재사용하므로 compose 재시작이 필요 없습니다.

비교는 clean과 clean_term 두 가지입니다. term missing 모드, corpus 사전, max_edits=2, 첫 후보 정책은 이전과 같습니다. 이전 history 및 gold passage 제공도 동일합니다. 답변 prompt와 cache key를 유지해 clean 응답을 이전 캐시에서 재사용하며 교정 후 질문이 원문과 동일하면 같은 응답을 재사용합니다. changed 질문만 추가 API 요청이 필요합니다. corpus embedding과 retrieval은 재계산합니다. API 캐시가 없는 경우 새 요청을 보냅니다.

새 파일: clean_term_config.json, clean_term_details.jsonl, clean_term_summary.json/csv, clean_term_report.md. 기존 summary/details/samples/corpus/config를 덮어쓰지 않습니다. 같은 clean_term 재실행은 clean_term 결과 파일을 갱신합니다.

- changed: 실제 문자열 변경률(clean 기준)
- correction_em: clean과 정규화 질문 일치율
- recall/mrr/answer_em의 delta와 paired CI: clean_term − clean
- wins/losses: clean 대비 실패→성공 / 성공→실패 샘플 수
- damage_rate: losses / clean 성공 샘플 수 (BM25/dense는 R@1, 답변은 EM)

LLM으로 정상성 판정하지 않습니다. clean은 오타 주입 이전 원문이지 사람이 모든 철자를 확인한 무오타 gold가 아닙니다. term 변경이 모두 잘못된 교정이라고 단정하지 마세요. details의 queries를 직접 확인해야 합니다.

사용자 환경의 기존 캐시 및 결과는 여기서 접근할 수 없으므로 실제 성능 측정은 미실행입니다. 문법 및 clean baseline 연결/결과파일 분리 검증 완료.
