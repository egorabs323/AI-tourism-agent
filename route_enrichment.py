"""Extra source-grounded descriptions for structured sightseeing routes.

Python 3.9 compatible. No network except the already configured local Ollama.
Does not alter ChromaDB. All attractions and day assignments come from the
existing structured route; enrichment only replaces the source and description.
"""
import json
import re
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

from rag_pipeline import Passage, destination_titles

STOP = re.compile(r'^(\s*\d+[.)]\s+)(.+?)\s+\[Источник\s*(\d+)\](?:\s*[—–-]\s*(.*))?\s*$')
SOURCE = re.compile(r'^Источники:\s*$')
NOTE = re.compile(r'^(?:Группировка по координатам|Географическая группировка|Порядок остановок)')


def _local_excerpt(text, name, max_chars=400):
    """Extract text near the EXACT attraction name (not other city snippets)."""
    found = re.search(re.escape(name), text, flags=re.I)
    if not found:
        return ''
    start = max(0, found.start() - 75)
    end = min(len(text), found.end() + max_chars - 75)
    return ' '.join(text[start:end].split())[:max_chars]


def _stops(answer):
    result = []
    for line in answer.splitlines():
        match = STOP.match(line)
        if match:
            result.append({'name': match.group(2).strip(), 'source': int(match.group(3)),
                           'original': (match.group(4) or '').strip()})
    return result


def _search_evidence(assistant, destination, names, max_candidates=4):
    """ONE batched encoder call, ONE Chroma query across all stops."""
    if not names:
        return {}
    titles = destination_titles(destination, assistant.titles)
    where = {'title': {'$in': titles}} if len(titles) > 1 else {'title': titles[0]}
    queries = ['query: {} {} sightseeing attraction history description'.format(destination, name)
               for name in names]
    vectors = assistant.encoder.encode(queries, normalize_embeddings=True)
    query_result = assistant.collection.query(
        query_embeddings=[vector.tolist() for vector in vectors],
        n_results=max_candidates, where=where,
        include=['documents', 'metadatas', 'distances'])
    selected = {}
    for i, name in enumerate(names):
        best = None
        for doc, meta, cid, distance in zip(
                query_result['documents'][i], query_result['metadatas'][i],
                query_result['ids'][i], query_result['distances'][i]):
            excerpt = _local_excerpt(doc, name)
            if not excerpt:
                continue
            # First exact matching chunk by semantic rank. Never use a chunk
            # that does not contain the selected object's literal name.
            best = (Passage(text=doc, title=meta.get('title', ''),
                            url=meta.get('url', ''), distance=float(distance),
                            chunk_id=cid), excerpt)
            break
        if best:
            selected[name] = best
    return selected


def _batch_translate(excerpts, model=None, generator=None):
    """At most one short Ollama request. Returns a dict keyed by attraction name.

    Failed generation is not allowed to break an otherwise working route.
    """
    if not excerpts:
        return {}
    items = [{'id': str(i), 'name': name, 'evidence': excerpt[:340]}
             for i, (name, excerpt) in enumerate(excerpts.items(), 1)]
    system = (
        'Ты переводчик туристических справок. Верни ТОЛЬКО JSON формата '
        '{"descriptions":{"1":"одно предложение","2":"..."}}. '
        'Для каждого id — одно короткое предложение НА РУССКОМ (до 130 символов) '
        'только на основании evidence именно этого объекта. '
        'Не выдумывай даты, авторов, адреса, стоимость или архитектурные детали. '
        'Если в evidence нет достоверной описательной информации, верни пустую строку. '
        'Не изменяй и не выбирай названия достопримечательностей.'
    )
    cache_file = None
    if generator is None:
        from structured_routes import _ollama_json
        from rag_pipeline import ollama_configuration
        generator = _ollama_json
        selected_model = ollama_configuration(model)[1]
        signature = hashlib.sha256(
            json.dumps([selected_model, items], ensure_ascii=False).encode('utf-8')
        ).hexdigest()
        cache_file = Path('data/cache/route_descriptions') / (signature + '.json')
        if cache_file.is_file():
            try:
                response = json.loads(cache_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                response = None
        else:
            response = None
    else:
        response = None
    if response is None:
        try:
            response = generator(system, json.dumps(items, ensure_ascii=False), model, tokens=650)
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
    for i, (name, _) in enumerate(excerpts.items(), 1):
        raw = descriptions.get(str(i), '')
        if not isinstance(raw, str):
            continue
        text = ' '.join(raw.split())
        if text and len(text) <= 150 and not re.search(r'[A-Za-z]{12,}', text):
            clean[name] = text
    return clean


def enrich_route_answer(answer, passages, assistant, destination, model=None,
                        max_stops=7, generator=None):
    """Returns (enriched_answer, passages_used_for_citations).

    If retrieval fails, returns original answer and its passages unchanged.
    Stop selection and order NEVER change; citations use actual returned chunks.
    """
    stops = _stops(answer)
    if not stops or not destination:
        return answer, passages
    names = list(dict.fromkeys(stop['name'] for stop in stops[:max_stops]))
    try:
        evidence = _search_evidence(assistant, destination, names)
    except Exception:
        return answer, passages
    descriptions = _batch_translate(
        {name: value[1] for name, value in evidence.items()}, model=model,
        generator=generator)
    lines = []
    source_map = {}
    curated = []

    def source_id(passage):
        key = passage.chunk_id or (passage.title, passage.url, passage.text[:90])
        if key not in source_map:
            source_map[key] = len(curated) + 1
            curated.append(passage)
        return source_map[key]

    in_sources = False
    for line in answer.splitlines():
        if SOURCE.match(line):
            in_sources = True
            continue
        if in_sources:
            if NOTE.match(line):
                in_sources = False
            else:
                continue
        match = STOP.match(line)
        if not match:
            lines.append(line)
            continue
        name, old_source = match.group(2).strip(), int(match.group(3))
        old_passage = passages[old_source - 1] if 1 <= old_source <= len(passages) else None
        selected = evidence.get(name)
        passage = selected[0] if selected else old_passage
        if passage is None:
            lines.append(line)
            continue
        number = source_id(passage)
        description = descriptions.get(name) or (match.group(4) or '').strip()
        prefix = '{}{} [Источник {}]'.format(match.group(1), name, number)
        lines.append(prefix + (' — ' + description if description else ''))

    if curated:
        # The HTML renderer hides this list and displays links from its API response.
        # The source numbers in the text now match the returned passage list.
        note_index = next((i for i, ln in enumerate(lines) if NOTE.match(ln)), len(lines))
        block = ['Источники:'] + [
            '[Источник {}] {}: {}'.format(i, p.title, p.url)
            for i, p in enumerate(curated, 1)]
        lines[note_index:note_index] = block
    return '\n'.join(lines).strip(), (curated or passages)
