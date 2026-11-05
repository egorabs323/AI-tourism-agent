"""Диагностика маршрута по Парижу. Читает базу, ничего не изменяет.

Python 3.9+. Запуск из корня проекта: python diagnose_paris.py
Создаёт paris_route_diagnostic.txt. Не обращается к Ollama/интернету.
"""
from pathlib import Path
import json
import re
from collections import Counter

CITY = "Paris"
DB = Path('data/chroma')
ARTICLES = Path('data/processed/articles.jsonl')
COLLECTION = 'wikivoyage_e5_small_v1'
OUTPUT = Path('paris_route_diagnostic.txt')
QUESTION = ('Составь туристический маршрут по Парижу на один день для первого '
            'знакомства с городом. Включи самые известные достопримечательности '
            'и кратко расскажи о каждой.')

# Названия здесь нужны только для проверки наличия данных, они НЕ используются
# для выбора объектов на сайте и не являются новой базой знаний.
TARGETS = {
    'Эйфелева башня': [r'\bEiffel Tower\b', r'\bTour Eiffel\b'],
    'Лувр': [r'\bLouvre\b', r'\bMusée du Louvre\b'],
    'Нотр-Дам': [r'\bNotre[- ]Dame\b'],
    'Триумфальная арка': [r'\bArc de Triomphe\b'],
    'Мост Искусств': [r'\bPont des Arts\b'],
    'Музей Клюни': [r'\bMusée national du Moyen Âge\b', r'\bCluny Museum\b'],
}


def snippet(text, pattern, before=100, after=270):
    match = re.search(pattern, text, re.I)
    if match is None:
        return ''
    return ' '.join(text[max(0, match.start()-before):match.end()+after].split())


def main():
    if not ARTICLES.is_file() or not DB.is_dir():
        raise SystemExit('Не найдена data/chroma или data/processed/articles.jsonl. Запусти из корня проекта.')
    titles = set()
    with ARTICLES.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                title = json.loads(line).get('title', '')
            except (ValueError, TypeError):
                continue
            if title == CITY or title.startswith(CITY + '/'):
                titles.add(title)
    if not titles:
        raise SystemExit('В articles.jsonl не найдены статьи Парижа.')

    import chromadb
    db = chromadb.PersistentClient(path=str(DB))
    collection = db.get_collection(name=COLLECTION)
    report = ['ДИАГНОСТИКА ПАРИЖА (только чтение)',
              'Город: ' + CITY,
              'Статей по городу: ' + str(len(titles)),
              'Задача: выяснить, содержатся ли названия объектов в городской части ChromaDB.',
              '']

    matches = {key: [] for key in TARGETS}
    counts = Counter()
    total = 0
    limit = 250
    offset = 0
    where = {'title': {'$in': sorted(titles)}} if len(titles) > 1 else {'title': next(iter(titles))}
    while True:
        data = collection.get(where=where, include=['documents', 'metadatas'], limit=limit, offset=offset)
        ids = data.get('ids') or []
        if not ids:
            break
        for chunk_id, doc, meta in zip(ids, data['documents'], data['metadatas']):
            total += 1
            title = str(meta.get('title', ''))
            counts[title] += 1
            for label, patterns in TARGETS.items():
                for pattern in patterns:
                    if re.search(pattern, doc, re.I):
                        matches[label].append((title, chunk_id, snippet(doc, pattern)))
                        break
        offset += len(ids)
        if len(ids) < limit:
            break
    report.extend(['Всего чанков городских статей: ' + str(total),
                   'Статей с наибольшим числом чанков: ' + repr(counts.most_common(12)), ''])
    for label, rows in matches.items():
        report.extend(['=' * 72, label + ': чанков с названием — ' + str(len(rows))])
        # Предпочитаем тематические статьи, а не случайные упоминания в общей статье.
        rows.sort(key=lambda r: (r[0] == CITY, len(r[2])))
        for title, chunk_id, excerpt in rows[:4]:
            report.extend([f'  {title} | {chunk_id}', '  ' + excerpt[:440]])
        report.append('')

    # Сравним с фактической выдачей приложения. Этот этап требует установленной
    # модели эмбеддингов, но не Ollama. Ошибка не мешает получить основной отчёт.
    report.extend(['=' * 72, 'ВЫДАЧА RagAssistant.retrieve (модель эмбеддингов, без Qwen)'])
    try:
        from rag_pipeline import RagAssistant
        assistant = RagAssistant(db=DB, articles=ARTICLES)
        for top_k in (6, 12):
            results, identified = assistant.retrieve(QUESTION, destination=CITY, top_k=top_k)
            report.append(f'\nTOP-K {top_k}: найден город {identified}, чанков {len(results)}')
            for i, passage in enumerate(results, 1):
                head = passage.text[:850]
                names_in_head = [label for label, patterns in TARGETS.items()
                                 if any(re.search(p, head, re.I) for p in patterns)]
                report.append(f'{i}. {passage.title} | {passage.chunk_id} | имена в первых 850 символах: {names_in_head}')
                report.append('   ' + ' '.join(head[:280].split()))
    except Exception as exc:
        report.append('Не удалось получить выдачу: ' + type(exc).__name__ + ': ' + str(exc)[:350])
    OUTPUT.write_text('\n'.join(report), encoding='utf-8')
    print('Отчёт сохранён:', OUTPUT.resolve())
    print('Городских чанков:', total)
    for label, rows in matches.items():
        print('  {}: {}'.format(label, len(rows)))


if __name__ == '__main__':
    main()
