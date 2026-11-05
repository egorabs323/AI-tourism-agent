"""Оценка Retrieval и RAG для локального туристического ассистента. Python 3.9+.

Не изменяет ChromaDB, SQLite и файлы сайта. Не запускает модель до команды compare.
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
    args=parser.parse_args()
    args.run(args)

if __name__=='__main__':
    main()
