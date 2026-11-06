"""Оценка Retrieval и RAG для локального туристического ассистента. Python 3.9+.

Корпус не изменяется; HTTP-проверки используют временную историю.
retrieve/score не вызывают LLM; compare/routes/reference могут использовать Ollama.
"""
import argparse
import csv
import json
import time
from pathlib import Path

OUT = Path('data/evaluation')
QUESTIONS = [
    ('R01', 'Какие основные достопримечательности Рима стоит посмотреть?', 'Rome'),
    ('R02', 'Что посмотреть в историческом центре Рима за два дня?', 'Rome'),
    ('R03', 'Что интересного есть в районе Ватикана?', 'Rome'),
    ('P01', 'Что посмотреть в Париже за один день?', 'Paris'),
    ('P02', 'Какие музеи стоит посетить в Париже?', 'Paris'),
    ('L01', 'Какие достопримечательности есть в Лондоне?', 'London'),
    ('B01', 'Что посмотреть в Берлине?', 'Berlin'),
    ('A01', 'Какие достопримечательности посетить в Ахене?', 'Aachen'),
]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def retrieve_cmd(args):
    from rag_pipeline import RagAssistant
    assistant = RagAssistant(db=Path('data/chroma'), articles=Path('data/processed/articles.jsonl'))
    rows, times = [], []
    for qid, question, city in QUESTIONS[:args.questions]:
        start = time.perf_counter()
        results, identified = assistant.retrieve(question, destination=city, top_k=args.pool)
        elapsed = time.perf_counter() - start
        times.append(dict(query_id=qid, question=question, destination=identified or '', seconds=round(elapsed, 3), returned=len(results)))
        for rank, p in enumerate(results, 1):
            rows.append(dict(query_id=qid, question=question, destination=identified or '', rank=rank,
                             chunk_id=p.chunk_id, title=p.title, url=p.url,
                             distance=round(p.distance, 5), excerpt=' '.join(p.text[:900].split()),
                             relevant=''))
        print('{}: {} результатов, {:.2f} с'.format(qid, len(results), elapsed))
    write_csv(OUT / 'retrieval_to_label.csv', ['query_id','question','destination','rank','chunk_id','title','url','distance','excerpt','relevant'], rows)
    write_csv(OUT / 'retrieval_timings.csv', ['query_id','question','destination','seconds','returned'], times)
    print('Сохранено:', OUT / 'retrieval_to_label.csv')
    print('Поставь 1 для релевантного чанка и 0 для нерелевантного; пустые значения недопустимы при score.')


def score_rows(rows, k):
    groups = {}
    for row in rows:
        qid = row['query_id'].strip()
        label = row['relevant'].strip()
        if label not in ('0', '1'):
            raise ValueError('Не размечен {} rank {} (нужно 0 или 1)'.format(qid, row['rank']))
        groups.setdefault(qid, []).append((int(row['rank']), int(label)))
    summary=[]
    for qid, pairs in sorted(groups.items()):
        ordered=sorted(pairs)
        ranks=[a for a,b in ordered]
        if len(ranks)!=len(set(ranks)) or min(ranks)!=1 or ranks!=list(range(1, max(ranks)+1)):
            raise ValueError('Ранги не последовательны в ' + qid)
        if len(ordered)<k:
            raise ValueError('Для {} меньше {} результатов; выбери меньшее --k'.format(qid,k))
        found=sum(label for rank,label in ordered if rank<=k)
        relevant_in_pool=sum(label for _,label in ordered)
        summary.append(dict(query_id=qid, precision_at_k=round(found/k,4),
                            pooled_recall_at_k=round(found/relevant_in_pool,4) if relevant_in_pool else '',
                            relevant_top_k=found, relevant_in_pool=relevant_in_pool,
                            pool_size=len(ordered)))
    return summary


def score_cmd(args):
    rows=read_csv(args.labels)
    summary=score_rows(rows,args.k)
    write_csv(OUT/'retrieval_metrics.csv', ['query_id','precision_at_k','pooled_recall_at_k','relevant_top_k','relevant_in_pool','pool_size'], summary)
    print('Precision@{} по размеченным документам: {:.3f}'.format(args.k,sum(float(r['precision_at_k']) for r in summary)/len(summary)))
    valid=[float(r['pooled_recall_at_k']) for r in summary if r['pooled_recall_at_k']!='']
    print('Recall@{} ВНУТРИ РАЗМЕЧЕННОГО ПУЛА: {}'.format(args.k, round(sum(valid)/len(valid),3) if valid else 'не определён'))
    print('Важно: pooled recall не равен полноте по всей ChromaDB. Для истинной Recall@k нужна полная релевантная разметка корпуса.')
    print('Сохранено:', OUT/'retrieval_metrics.csv')


def compare_cmd(args):
    from rag_pipeline import RagAssistant, answer_with_llm
    assistant=RagAssistant(db=Path('data/chroma'), articles=Path('data/processed/articles.jsonl'))
    selected=QUESTIONS[:args.questions]
    records=[]
    for qid, question, city in selected:
        print('Запрос:',qid,'—',question, flush=True)
        start=time.perf_counter()
        passages, _=assistant.retrieve(question,destination=city,top_k=args.k)
        search_seconds=round(time.perf_counter()-start,3)
        for mode in ['rag','without_rag']:
            genstart=time.perf_counter()
            try:
                # Для честного сравнения используется ОДНА функция генерации, меняется только контекст.
                answer=answer_with_llm(question,passages if mode=='rag' else [],[],without_rag=(mode=='without_rag'))
                error=''
            except Exception as exc:
                answer=''
                error='{}: {}'.format(type(exc).__name__,str(exc)[:250])
            generation_seconds=round(time.perf_counter()-genstart,3)
            records.append(dict(query_id=qid,question=question,destination=city,mode=mode,
                                retrieval_seconds=search_seconds if mode=='rag' else 0,
                                generation_seconds=generation_seconds,
                                answer=answer,error=error, factual_correctness='',source_support='',usefulness=''))
            print(' ', mode, generation_seconds, 'c', 'ошибка' if error else 'готово',flush=True)
            # Сохраняем после каждого ответа: прерывание не уничтожит предыдущие результаты.
            write_csv(OUT/'comparison_to_rate.csv', ['query_id','question','destination','mode','retrieval_seconds','generation_seconds','answer','error','factual_correctness','source_support','usefulness'], records)
    print('Оцени вручную factual_correctness, source_support, usefulness по шкале 0-2 (см. README).')
    print('Сохранено:',OUT/'comparison_to_rate.csv')


def routes_cmd(args):
    """Real local integration run; does not create users/conversations or vectors."""
    import json
    import re
    from types import SimpleNamespace
    from unittest.mock import patch
    from rag_pipeline import ArticleStore
    from structured_routes import answer_route, _ollama_json, evidence_options, name_key
    from route_enrichment import enrich_route_answer, STOP
    from build_vector_db import create_collection

    store = ArticleStore(Path('data/processed/articles.jsonl'))
    collection = create_collection(Path('data/chroma'))
    count_before = collection.count()
    examples = [
        ('Rome', 'Составь туристический маршрут по Риму на два дня с посещением главных исторических достопримечательностей', 2),
        ('Paris', 'Составь туристический маршрут по Парижу на один день для первого знакомства с городом. Включи самые известные достопримечательности и кратко расскажи о каждой', 1),
        ('London', 'Составь маршрут по Лондону на два дня для первого знакомства с городом. Включи основные исторические достопримечательности и кратко расскажи, чем интересно каждое место', 2),
        ('Istanbul', 'Составь туристический маршрут по Стамбулу на два дня для первого знакомства с городом.', 2),
        ('Tokyo', 'Составь туристический маршрут по Токио на два дня для первого знакомства с городом.', 2),
    ]
    # Test expectations only; production selection has no per-city landmark lists.
    expected = {
        'Rome': ['Colosseum', 'Pantheon'],
        'Paris': ['Eiffel Tower', 'Notre Dame', 'Arc de Triomphe'],
        'London': ['Tower Bridge', 'Buckingham Palace'],
        'Istanbul': ['Blue Mosque', 'Topkapı Palace', 'Basilica Cistern', 'Galata Tower', 'Hagia Sofia'],
        'Tokyo': ['Tokyo SkyTree', 'Sensō-ji', 'Meiji Shrine', 'Tokyo National Museum'],
    }
    report = []
    for city, question, days in examples:
        if args.city and args.city != city:
            continue
        calls = []
        def measured(system, user, model, tokens=700):
            start = time.perf_counter()
            result = _ollama_json(system, user, model, tokens)
            calls.append({'seconds': round(time.perf_counter() - start, 3),
                          'input_chars': len(system) + len(user), 'result': result})
            return result
        start = time.perf_counter()
        print('Проверка:', city, flush=True)
        passages = list(store.city(city))
        with patch('structured_routes._ollama_json', side_effect=measured):
            answer = answer_route(question, passages, city)
            answer, sources = enrich_route_answer(answer, passages, SimpleNamespace(articles=store), city)
        stops = [m for line in answer.splitlines() if (m := STOP.match(line))]
        names = [m.group(2) for m in stops]
        checks = {
            'days': re.findall(r'^День (\d+)$', answer, re.M) == [str(i) for i in range(1, days + 1)],
            'unique_stops': len(set(name.casefold() for name in names)) == len(names),
            'stop_count': len(stops) == days * 3,
            'russian_descriptions': bool(stops) and all(re.search('[А-Яа-яЁё]', m.group(4) or '') and
                    not any(marker in (m.group(4) or '') for marker in
                            ['Не удалось перевести', 'недостаточно проверенных', 'недостаточно однозначных']) for m in stops),
            'source_ids': all(1 <= int(m.group(3)) <= len(sources) for m in stops),
            'wiki_urls': all(p.url.startswith('https://en.wikivoyage.org/wiki/') for p in sources),
            'object_evidence': all(evidence_options(passages, name) for name in names),
            'required_landmarks': all(any(name_key(n) in name_key(name) for name in names) for n in expected[city]),
            'regression_exclusions': not set(names).intersection({
                'Tube', 'St Pancras', 'Catholic Church', 'St. Peter', 'Downstairs',
                'Park Hyatt Tokyo', 'Art Museum', 'Museums', 'Churches', 'Tokyo Dome', 'Edo', 'Sensō'}),
        }
        row = {'city': city, 'seconds': round(time.perf_counter() - start, 3),
               'article_count': len(passages), 'llm_calls': len(calls), 'checks': checks,
               'answer': answer, 'calls': calls}
        if args.http:
            import tempfile
            import web_app
            from fastapi.testclient import TestClient
            from rag_pipeline import RagAssistant
            assistant = RagAssistant()
            assistant.articles = store
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with patch.object(web_app, 'DATA', root), \
                     patch.object(web_app, 'HISTORY_DB', root / 'history.sqlite3'), \
                     patch.object(web_app, 'UPLOADS', root / 'uploads'), \
                     patch.object(web_app, 'LOG', root / 'requests.jsonl'), \
                     patch.object(web_app, 'get_assistant', return_value=assistant), \
                     patch.object(web_app, 'search_documents', return_value=[]), \
                     patch('structured_routes._ollama_json', side_effect=AssertionError('warm cache called Ollama')):
                    with TestClient(web_app.app) as api:
                        registered = api.post('/api/auth/register', json={
                            'username': 'route_validation', 'password': 'local-validation-password'})
                        registered.raise_for_status()
                        cid = api.post('/api/conversations', json={}).json()['id']
                        warm_start = time.perf_counter()
                        response = api.post('/api/conversations/' + cid + '/messages', json={
                            'question': question, 'destination': city, 'mode': 'route', 'top_k': 6})
                        response.raise_for_status()
                        row['warm_http_seconds'] = round(time.perf_counter() - warm_start, 3)
                        checks['http_answer'] = response.json()['answer'] == answer
                        checks['http_sources'] = len(response.json()['sources']) == len(sources)
                        checks['http_history'] = len(api.get('/api/conversations/' + cid + '/messages').json()) == 2
                        checks['encoder_unloaded'] = assistant._encoder is None
        if __import__('sys').platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ['peak', 'working', 'pool_peak', 'pool',
                                                        'nonpage_peak', 'nonpage', 'pagefile', 'pagefile_peak']]
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
            memory_info.restype = wintypes.BOOL
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if memory_info(handle, ctypes.byref(counters), counters.cb):
                row['python_peak_mb'] = round(counters.peak / 1024**2, 2)
                row['python_working_mb'] = round(counters.working / 1024**2, 2)
        report.append(row)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ('route_' + city.lower() + '.txt')).write_text(answer, encoding='utf-8')
        (OUT / 'route_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(city, row['seconds'], 'seconds;', len(calls), 'LLM calls;', checks, flush=True)
        print(answer, flush=True)
    if collection.count() != count_before:
        raise RuntimeError('Chroma count changed during read-only validation')
    if not all(all(row['checks'].values()) for row in report):
        raise SystemExit('Route validation failed; inspect data/evaluation/route_validation.json')


def reference_cmd(args):
    """Real reference-mode API run; corpus/history are not modified."""
    import re
    import tempfile
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import web_app
    import reference_rag
    from rag_pipeline import RagAssistant
    from contextlib import nullcontext

    cases = [
        ('berlin_gate', 'Berlin', 'Когда были построены Бранденбургские ворота в Берлине и для чего они предназначались?', None),
        ('berlin_dome', 'Berlin', 'Что такое Рейхстаг в Берлине и чем интересен его купол?', None),
        ('prague_bridge', 'Prague', 'Когда началось строительство Карлова моста в Праге и чем он известен?', None),
        ('prague_clock', 'Prague', 'Что такое Пражские астрономические часы и что они показывают?', None),
        ('tokyo_height', 'Tokyo', 'Какова высота башни Tokyo Skytree?', None),
        ('istanbul_date', 'Istanbul', 'Когда была построена Цистерна Базилика (Basilica Cistern) в Стамбуле?', None),
        ('pantheon', 'Rome', 'Что такое Пантеон в Риме и чем он известен?', 'pantheon'),
        ('pantheon_followup', 'Rome', 'Чем примечателен его купол?', 'pantheon'),
        ('transport', 'Rome', 'Как добраться из аэропорта Fiumicino в центр Рима? Перечисли виды транспорта без цен.', None),
        ('unknown', 'Rome', 'Каков номер телефона туристического бюро «Астра-917» в Риме?', None),
        ('current_price', 'Rome', 'Сколько стоит вход в Колизей сегодня?', None),
        ('reference_route_word', 'Paris', 'В справочном режиме объясни, что означает туристический маршрут. План по дням не составляй.', None),
    ]
    if args.city:
        cases = [case for case in cases if case[1] == args.city]
    document_mode = args.documents_only
    if document_mode:
        import chromadb
        import user_documents
        document_collection = chromadb.EphemeralClient().create_collection(
            name='reference_validation_docs', metadata={'hnsw:space': 'cosine'}, embedding_function=None)
        document_patcher = patch.object(user_documents, 'get_collection', return_value=document_collection)
        document_search = user_documents.search_documents
        cases = [
            ('document_date', None, 'Когда открыта обсерватория «Лунный маяк»?', 'document'),
            ('document_count', None, 'А сколько там метеоритов?', 'document'),
            ('document_unknown', None, 'Кто архитектор обсерватории «Лунный маяк»?', 'document'),
        ]
    else:
        document_patcher = nullcontext()
        document_search = lambda *a, **kw: []
    assistant = RagAssistant()
    before = assistant.collection.count()
    real_retrieve, real_answer = assistant.retrieve, web_app.answer_factual
    real_generate = reference_rag._generate
    retrieval, facts, calls, report = [], [], [], []

    def retrieve(*positional, **kwargs):
        found, city = real_retrieve(*positional, **kwargs)
        retrieval.append({'query': positional[0], 'city': city,
                          'passages': [dict(text=p.text, title=p.title, url=p.url, distance=p.distance) for p in found]})
        return found, city

    def answer(question, passages, history, mode='reference'):
        facts.extend(dict(text=p.text, title=p.title, url=p.url) for p in passages)
        return real_answer(question, passages, history, mode)

    def generate(system, user, tokens):
        start = time.perf_counter()
        value = real_generate(system, user, tokens)
        calls.append({'seconds': round(time.perf_counter() - start, 3), 'system': system,
                      'input': json.loads(user), 'result': value})
        return value

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        with document_patcher, \
             patch.object(web_app, 'DATA', root), \
             patch.object(web_app, 'HISTORY_DB', root / 'history.sqlite3'), \
             patch.object(web_app, 'UPLOADS', root / 'uploads'), \
             patch.object(web_app, 'LOG', root / 'requests.jsonl'), \
             patch.object(web_app, 'get_assistant', return_value=assistant), \
             patch.object(web_app, 'search_documents', side_effect=document_search), \
             patch.object(web_app, 'answer_factual', side_effect=answer), \
             patch.object(assistant, 'retrieve', side_effect=retrieve), \
             patch.object(reference_rag, '_generate', side_effect=generate):
            with TestClient(web_app.app) as api:
                api.post('/api/auth/register', json={'username': 'reference_validation',
                         'password': 'reference-validation-password'}).raise_for_status()
                conversations = {}
                for case_id, city, question, group in cases:
                    key = group or case_id
                    if key not in conversations:
                        conversations[key] = api.post('/api/conversations', json={}).json()['id']
                        if document_mode:
                            content = ('Обсерватория «Лунный маяк» открыта в 1987 году. '
                                       'В коллекции 42 метеорита.').encode('utf-8')
                            uploaded = api.post('/api/conversations/' + conversations[key] + '/documents',
                                                files={'file': ('reference_fixture.txt', content, 'text/plain')})
                            uploaded.raise_for_status()
                    retrieval.clear()
                    facts.clear()
                    calls.clear()
                    print('Проверка:', case_id, flush=True)
                    start = time.perf_counter()
                    response = api.post('/api/conversations/' + conversations[key] + '/messages', json={
                        'question': question, 'destination': city, 'mode': 'reference', 'top_k': 6})
                    result = response.json()
                    text = result.get('answer', '')
                    sources = result.get('sources', [])
                    cited = [int(n) for n in re.findall(r'\[Источник (\d+)\]', text)]
                    checks = {
                        'http_ok': response.status_code == 200,
                        'russian': bool(re.search('[А-Яа-яЁё]', text)),
                        'no_day_plan': not bool(re.search(r'^\s*(?:#{1,3}\s*)?День\s+\d+', text, re.M)),
                        'citation_ids': all(1 <= n <= len(sources) for n in cited),
                        'unique_source_urls': len({s['url'] for s in sources}) == len(sources),
                        'no_truncated_ending': bool(text) and bool(re.search(r'[.!?…\]]$', text)),
                    }
                    if case_id in {'unknown', 'reference_route_word', 'document_unknown'}:
                        checks['honest_absence_or_definition'] = case_id == 'reference_route_word' or text == reference_rag.NO_ANSWER
                    elif case_id == 'current_price':
                        checks['no_unverified_current_price'] = text == reference_rag.CURRENT_NOTICE and not sources
                    else:
                        checks['has_verified_facts'] = bool(cited) and text != reference_rag.NO_ANSWER
                    expected = {'berlin_gate': '1791', 'berlin_dome': '1999', 'prague_bridge': '1357',
                                'tokyo_height': '634', 'istanbul_date': '532',
                                'document_date': '1987', 'document_count': '42'}
                    if case_id in expected:
                        checks['expected_number'] = expected[case_id] in text
                    if case_id == 'berlin_gate':
                        checks['design_not_construction'] = '1757' not in text and 'проект' in text.lower()
                    if case_id == 'berlin_dome':
                        checks['parliament_and_dome'] = bool(re.search(r'парламент|бундестаг', text, re.I)) and '1913' not in text
                        checks['glass_dome_not_ball'] = bool(re.search('стеклянн.*купол|купол.*стеклянн', text, re.I)) and not re.search(r'\bшар[а-я]*', text, re.I)
                    if case_id == 'prague_clock':
                        checks['sun_and_moon'] = bool(re.search('солнц', text, re.I)) and bool(re.search('лун', text, re.I))
                        checks['no_cuckoo'] = 'кукуш' not in text.lower()
                    if case_id in {'pantheon', 'pantheon_followup'}:
                        checks['dome_not_arch'] = bool(re.search('купол', text, re.I)) and not bool(re.search(r'\bарк[а-я]*', text, re.I))
                    if case_id == 'transport':
                        checks['no_prices'] = not bool(re.search(r'[€£$¥]', text))
                        checks['transport_modes'] = bool(re.search('поезд|экспресс', text, re.I)) and bool(re.search('такси|автобус', text, re.I))
                    history = api.get('/api/conversations/' + conversations[key] + '/messages').json()
                    checks['history'] = bool(history) and history[-1]['content'] == text
                    row = dict(id=case_id, city=city, question=question, seconds=round(time.perf_counter() - start, 3),
                               status=response.status_code, response=result, checks=checks, facts=list(facts),
                               retrieval=list(retrieval), calls=list(calls))
                    report.append(row)
                    filename = 'reference_documents_validation.json' if document_mode else 'reference_validation.json'
                    (OUT / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(case_id, row['seconds'], 's;', checks, flush=True)
                    print(text, flush=True)
    if assistant.collection.count() != before:
        raise RuntimeError('Chroma count changed during reference validation')
    if not all(all(row['checks'].values()) for row in report):
        raise SystemExit('Reference validation failed; inspect data/evaluation/reference_validation.json')


def recommendations_cmd(args):
    """Real recommendation API checks with traceable candidate evidence."""
    import re
    import tempfile
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from rag_pipeline import RagAssistant
    import recommendation_rag as rec
    import web_app
    from contextlib import nullcontext
    cases = json.loads((Path(__file__).parent / 'tests/fixtures/recommendations.json').read_text(encoding='utf-8'))
    if args.city:
        cases = [case for case in cases if case['city'] == args.city]
    document_mode = args.documents_only
    if document_mode:
        import chromadb
        import user_documents
        collection = chromadb.EphemeralClient().create_collection(
            name='recommendation_validation_docs', metadata={'hnsw:space': 'cosine'}, embedding_function=None)
        document_patcher = patch.object(user_documents, 'get_collection', return_value=collection)
        document_search = user_documents.search_documents
        cases = [
            dict(id='document_museum', city=None, kind='museum', count=1, requested_count=2,
                 expected=['Северная палитра'], question='Порекомендуй два художественных музея из документа.'),
            dict(id='document_park', city=None, kind='park', count=1, expected=['Тихий луг'],
                 question='А теперь порекомендуй один парк из документа. Музеи не включай.'),
            dict(id='document_followup', city=None, kind='park', count=0,
                 question='Посоветуй ещё один такой парк из документа.'),
        ]
    else:
        document_patcher = nullcontext()
        document_search = lambda *a, **kw: []
    assistant = RagAssistant()
    before = assistant.collection.count()
    real_retrieve, real_select, real_generate = assistant.retrieve, rec.select_recommendations, rec._generate
    retrieval, candidates, calls, report = [], [], [], []

    def retrieve(*positional, **kwargs):
        encoded = []
        encoder = assistant.encoder
        real_encode = encoder.encode
        def encode(texts, *a, **kw):
            encoded.extend(texts)
            return real_encode(texts, *a, **kw)
        with patch.object(encoder, 'encode', side_effect=encode):
            found, city = real_retrieve(*positional, **kwargs)
        retrieval.append(dict(query=positional[0], city=city, embedded_queries=list(encoded),
                         passages=[dict(title=p.title, url=p.url, text=p.text, chunk_id=p.chunk_id) for p in found]))
        return found, city

    def select(*positional, **kwargs):
        found = real_select(*positional, **kwargs)
        candidates.extend(dict(name=p.name, kind=p.kind, evidence=p.evidence,
                               title=p.passage.title, url=p.passage.url) for p in found)
        return found

    def generate(system, user, tokens):
        start = time.perf_counter()
        value = real_generate(system, user, tokens)
        calls.append(dict(seconds=round(time.perf_counter() - start, 3), system=system,
                          input=json.loads(user), result=value))
        return value

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        with document_patcher, patch.object(web_app, 'DATA', root), \
             patch.object(web_app, 'HISTORY_DB', root / 'history.sqlite3'), \
             patch.object(web_app, 'UPLOADS', root / 'uploads'), \
             patch.object(web_app, 'LOG', root / 'requests.jsonl'), \
             patch.object(web_app, 'get_assistant', return_value=assistant), \
             patch.object(web_app, 'search_documents', side_effect=document_search), \
             patch.object(assistant, 'retrieve', side_effect=retrieve), \
             patch.object(rec, 'select_recommendations', side_effect=select), \
             patch.object(rec, '_generate', side_effect=generate):
            with TestClient(web_app.app) as api:
                api.post('/api/auth/register', json={'username': 'recommendations_validation',
                         'password': 'recommendations-validation-password'}).raise_for_status()
                document_cid = None
                for case in cases:
                    cid = document_cid or api.post('/api/conversations', json={}).json()['id']
                    if document_mode and document_cid is None:
                        document_cid = cid
                        content = ('Музей «Северная палитра» — это художественный музей. '
                                   'В коллекции представлены картины и рисунки местных художников.\n'
                                   'Парк «Тихий луг» — это парк с озером и прогулочными дорожками.').encode('utf-8')
                        uploaded = api.post('/api/conversations/' + cid + '/documents',
                                            files={'file': ('recommendations_fixture.txt', content, 'text/plain')})
                        uploaded.raise_for_status()
                    retrieval.clear()
                    candidates.clear()
                    calls.clear()
                    print('Проверка:', case['id'], flush=True)
                    start = time.perf_counter()
                    response = api.post('/api/conversations/' + cid + '/messages', json={
                        'question': case['question'], 'destination': case['city'], 'mode': 'recommendations', 'top_k': 6})
                    result = response.json()
                    text, sources = result.get('answer', ''), result.get('sources', [])
                    items = re.findall(r'^\d+\. (.+?) — (.+?) \[Источник (\d+)\]$', text, re.M)
                    indexed = {p['name']: p for p in candidates}
                    checks = {
                        'http_ok': response.status_code == 200,
                        'russian': bool(re.search('[А-Яа-яЁё]', text)),
                        'no_day_plan': not bool(re.search(r'^\s*(?:#{1,3}\s*)?День\s+\d+', text, re.M)),
                        'count': len(items) == case['count'],
                        'unique_names': len({n.casefold() for n, _, _ in items}) == len(items),
                        'source_ids': all(1 <= int(i) <= len(sources) for _, _, i in items),
                        'source_urls_unique': len({s['url'] for s in sources}) == len(sources),
                        'verified_names_and_types': all(n in indexed and indexed[n]['kind'] == case['kind'] for n, _, _ in items),
                        'exclusions': all(not any(rec.name_key(ex) in rec.name_key(n) for ex in case.get('excluded', [])) for n, _, _ in items),
                        'no_unverified_park_or_museum': not any(n in {'Zuid', 'Spui', 'Kalverstraat', 'Kassias Museum', 'Kastanienallee Museum', 'Adults', 'This Baroque', 'Bus X33'} for n, _, _ in items),
                        'one_specific_search_query': not retrieval if document_mode else bool(retrieval) and len(retrieval[0]['embedded_queries']) == 1,
                    }
                    if document_mode:
                        checks['document_sources'] = all('recommendations_fixture.txt' in s['title'] for s in sources)
                        checks['expected_names'] = {n for n, _, _ in items} == set(case.get('expected', []))
                        if case.get('requested_count'):
                            checks['honest_shortage'] = '{} из {}'.format(case['count'], case['requested_count']) in text
                    if case['count'] == 0:
                        checks['honest_absence'] = text == rec.NO_RECOMMENDATIONS and not sources
                    history = api.get('/api/conversations/' + cid + '/messages').json()
                    checks['history'] = bool(history) and history[-1]['content'] == text
                    row = dict(case, seconds=round(time.perf_counter() - start, 3), response=result,
                               checks=checks, candidates=list(candidates), retrieval=list(retrieval), calls=list(calls))
                    report.append(row)
                    filename = 'recommendations_documents_validation.json' if document_mode else 'recommendations_validation.json'
                    (OUT / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(case['id'], row['seconds'], 's;', checks, flush=True)
                    print(text, flush=True)
    if assistant.collection.count() != before:
        raise RuntimeError('Chroma count changed during recommendation validation')
    if not all(all(row['checks'].values()) for row in report):
        filename = 'recommendations_documents_validation.json' if document_mode else 'recommendations_validation.json'
        raise SystemExit('Recommendations validation failed; inspect ' + str(OUT / filename))


def main():
    parser=argparse.ArgumentParser(description='Локальная оценка туристического RAG (Python 3.9+)')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('retrieve',help='Получить кандидатов для ручной разметки; Ollama не нужна')
    p.add_argument('--questions',type=int,default=5,choices=range(1,len(QUESTIONS)+1))
    p.add_argument('--pool',type=int,default=10,choices=range(5,31))
    p.set_defaults(run=retrieve_cmd)
    p=sub.add_parser('score',help='Оценить размеченный поисковый пул')
    p.add_argument('--labels',type=Path,default=OUT/'retrieval_to_label.csv')
    p.add_argument('--k',type=int,default=5,choices=range(1,31))
    p.set_defaults(run=score_cmd)
    p=sub.add_parser('compare',help='Запустить генерацию с RAG и без RAG; медленно')
    p.add_argument('--questions',type=int,default=2,choices=range(1,len(QUESTIONS)+1))
    p.add_argument('--k',type=int,default=5,choices=range(1,13))
    p.set_defaults(run=compare_cmd)
    p=sub.add_parser('routes', help='Проверить реальные маршруты по пяти городам с локальной Ollama')
    p.add_argument('--city', choices=['Rome', 'Paris', 'London', 'Istanbul', 'Tokyo'])
    p.add_argument('--http', action='store_true', help='Также проверить HTTP API и историю во временной SQLite')
    p.set_defaults(run=routes_cmd)
    p=sub.add_parser('reference', help='Проверить справочные ответы через API с реальными Chroma и Ollama')
    p.add_argument('--city', choices=['Berlin', 'Prague', 'Tokyo', 'Istanbul', 'Rome', 'Paris'])
    p.add_argument('--documents-only', action='store_true', help='Реальная загрузка TXT и RAG в изолированной коллекции Chroma в памяти')
    p.set_defaults(run=reference_cmd)
    p=sub.add_parser('recommendations', help='Проверить рекомендации с интересами, исключениями и подтверждёнными объектами')
    p.add_argument('--city', choices=['Amsterdam', 'Vienna', 'Berlin', 'Madrid', 'Barcelona'])
    p.add_argument('--documents-only', action='store_true', help='Загрузка TXT и рекомендации в отдельной Chroma в памяти')
    p.set_defaults(run=recommendations_cmd)
    args=parser.parse_args()
    args.run(args)

if __name__=='__main__':
    main()
