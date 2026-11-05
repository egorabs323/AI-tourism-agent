"""Additional descriptions for RAG-generated tourist routes (Python 3.9).

The route and its stop order are not regenerated. Descriptions are grounded
in text about each exact named place. Citations are numbered by unique URL,
matching the frontend's link list.
"""

import hashlib
import json
import re
from pathlib import Path

from rag_pipeline import Passage, destination_titles

STOP = re.compile(r'^(\s*\d+[.)]\s+)(.+?)\s+\[Источник\s*(\d+)\](?:\s*[—–-]\s*(.*))?\s*$')
SOURCE = re.compile(r'^Источники:\s*$')
NOTE = re.compile(r'^(?:Группировка по координатам|Географическая группировка|Порядок остановок)')
INVALID_DESCRIPTION = re.compile(
    r'^(?:одно предложение|\.\.\.|…|описание|пример|текст|short description|'
    r'one sentence|description|не указано|нет данных)[.!\s]*$', re.I
)
CACHE_VERSION = 'descriptions-v2'
FALLBACK = 'Описание в найденных материалах недостаточно подробное.'


def _local_excerpt(text, name, max_chars=430):
    """Use evidence containing the *exact* selected attraction name."""
    hit = re.search(re.escape(name), text, re.I)
    if not hit:
        return ''
    start = max(0, hit.start() - 80)
    end = min(len(text), hit.end() + max_chars - 80)
    return ' '.join(text[start:end].split())[:max_chars]


def _stops(answer):
    output = []
    for line in answer.splitlines():
        m = STOP.match(line)
        if m:
            output.append({'name': m.group(2).strip(), 'source': int(m.group(3)),
                           'original': (m.group(4) or '').strip()})
    return output


def _search_evidence(assistant, destination, names, max_candidates=5):
    """Batch retrieval; no extra LLM calls. Retain passage with exact name."""
    if not names:
        return {}
    titles = destination_titles(destination, assistant.titles)
    if not titles:
        return {}
    where = {'title': {'$in': titles}} if len(titles) > 1 else {'title': titles[0]}
    queries = ['query: {} {} attraction description history'.format(destination, name)
               for name in names]
    embeddings = assistant.encoder.encode(queries, normalize_embeddings=True)
    results = assistant.collection.query(
        query_embeddings=[vector.tolist() for vector in embeddings],
        n_results=max_candidates,
        where=where,
        include=['documents', 'metadatas', 'distances'],
    )
    selected = {}
    for i, name in enumerate(names):
        for doc, meta, cid, distance in zip(
            results['documents'][i], results['metadatas'][i],
            results['ids'][i], results['distances'][i]
        ):
            excerpt = _local_excerpt(doc, name)
            if not excerpt:
                continue
            selected[name] = (Passage(text=doc, title=meta.get('title', ''),
                                      url=meta.get('url', ''), distance=float(distance),
                                      chunk_id=cid), excerpt)
            break
    return selected


def _valid_description(value):
    if not isinstance(value, str):
        return ''
    value = ' '.join(value.strip().split())
    if not value or len(value) > 175 or INVALID_DESCRIPTION.fullmatch(value):
        return ''
    # Reject prompt-template echoes (even in otherwise longer descriptions).
    low = value.casefold()
    if any(marker in low for marker in (
        'одно предложение', 'для каждого id', 'только json',
        'на основании evidence', 'впишите описание', 'пример ответа',
        'description here', 'one sentence',
    )):
        return ''
    if re.search(r'[A-Za-z]{12,}', value):
        return ''
    return value


def _batch_translate(excerpts, model=None, generator=None):
    """One local Ollama request for all places, with versioned persistent cache."""
    if not excerpts:
        return {}
    items = [dict(id=str(i), name=name, evidence=excerpt[:360])
             for i, (name, excerpt) in enumerate(excerpts.items(), 1)]
    system = (
        'Ты переводишь факты об объектах для русского туристического гида. '
        'Ответ строго JSON: объект с ключом descriptions, внутри пары '
        'идентификатор -> короткая строка на русском языке. '
        'Каждый текст должен содержать только факты из evidence соответствующего объекта. '
        'Максимум 140 символов на описание. Никаких примеров, шаблонных фраз, '
        'метаинструкций, предположений, цен и вымышленных сведений. '
        'Если в evidence недостаточно сведений, значение должно быть пустой строкой.'
    )
    cache_file = None
    if generator is None:
        from structured_routes import _ollama_json
        from rag_pipeline import ollama_configuration
        generator = _ollama_json
        model_name = ollama_configuration(model)[1]
        digest = hashlib.sha256(json.dumps(
            [CACHE_VERSION, model_name, items], ensure_ascii=False,
            sort_keys=True).encode('utf-8')).hexdigest()
        cache_file = Path('data/cache/route_descriptions') / (digest + '.json')

    response = None
    if cache_file is not None and cache_file.is_file():
        try:
            response = json.loads(cache_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    if not isinstance(response, dict):
        try:
            response = generator(system, json.dumps(items, ensure_ascii=False), model,
                                 tokens=700)
        except Exception:
            return {}
        if cache_file is not None and isinstance(response, dict):
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(response, ensure_ascii=False), encoding='utf-8')
            except OSError:
                pass
    descriptions = response.get('descriptions', {}) if isinstance(response, dict) else {}
    if not isinstance(descriptions, dict):
        return {}
    clean = {}
    for i, name in enumerate(excerpts, 1):
        valid = _valid_description(descriptions.get(str(i)))
        if valid:
            clean[name] = valid
    return clean


def enrich_route_answer(answer, passages, assistant, destination, model=None,
                        max_stops=7, generator=None):
    """Return (updated_answer, cited_passages), deduplicating citations by URL."""
    stops = _stops(answer)
    if not stops or not destination:
        return answer, passages
    names = list(dict.fromkeys(s['name'] for s in stops[:max_stops]))
    try:
        evidence = _search_evidence(assistant, destination, names)
    except Exception:
        evidence = {}
    descriptions = _batch_translate(
        {name: item[1] for name, item in evidence.items()},
        model=model, generator=generator,
    )

    # One citation number per distinct URL: the interface deduplicates URLs too.
    source_numbers = {}
    curated = []

    def number_for(passage):
        url = (passage.url or '').strip()
        key = url or ('chunk', passage.chunk_id or passage.title)
        if key not in source_numbers:
            source_numbers[key] = len(curated) + 1
            curated.append(passage)
        return source_numbers[key]

    lines = []
    in_old_sources = False
    for line in answer.splitlines():
        if SOURCE.match(line):
            in_old_sources = True
            continue
        if in_old_sources:
            if NOTE.match(line):
                in_old_sources = False
            else:
                continue
        match = STOP.match(line)
        if not match:
            lines.append(line)
            continue
        name = match.group(2).strip()
        old_number = int(match.group(3))
        chosen = evidence.get(name)
        original_passage = passages[old_number - 1] if 1 <= old_number <= len(passages) else None
        passage = chosen[0] if chosen else original_passage
        if passage is None:
            # Omit any dangling marker; do not fabricate a source.
            lines.append('{}{} — {}'.format(match.group(1), name, FALLBACK))
            continue
        citation = number_for(passage)
        description = descriptions.get(name) or FALLBACK
        lines.append('{}{} [Источник {}] — {}'.format(match.group(1), name, citation, description))

    if not curated:
        return '\n'.join(lines).strip(), []
    insert_at = next((i for i, text in enumerate(lines) if NOTE.match(text)), len(lines))
    source_lines = ['Источники:']
    source_lines.extend('[Источник {}] {}: {}'.format(i, p.title, p.url)
                        for i, p in enumerate(curated, 1))
    lines[insert_at:insert_at] = source_lines
    return '\n'.join(lines).strip(), curated
