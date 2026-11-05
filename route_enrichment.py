"""Evidence-first descriptions for tourist route stops (Python 3.9).

Does not alter route selection. Rejects snippets about neighboring objects, stale
opening hours and unsupported translations. Requires the named landmark to be
an explicit grammatical subject, otherwise shows a conservative fallback.
"""
import hashlib
import json
import re
from pathlib import Path

from rag_pipeline import Passage, destination_titles

STOP = re.compile(r'^(\s*\d+[.)]\s+)(.+?)\s+\[Источник\s*(\d+)\](?:\s*[—–-]\s*(.*))?\s*$')
SOURCE = re.compile(r'^Источники:\s*$')
NOTE = re.compile(r'^(?:Группировка по координатам|Географическая группировка|Порядок остановок)')
FALLBACK = 'В найденных фрагментах нет достаточно точного описания этого объекта.'
CACHE_VERSION = 'grounded-v6'
_INVALID = re.compile(r'^(?:одно предложение|описание|пример|текст|нет данных|не указано|one sentence|description|short description)[.!\s]*$', re.I)
_HOURS = re.compile(r'\b(?:open|opening|hours?|closed|daily|tickets?|admission|price|cost|€|\$|am|pm)\b|\b\d{1,2}:\d{2}\b', re.I)
_SPATIAL = re.compile(r'^(?:near|nearby|opposite|next to|around|behind|beside)\b', re.I)
_UNRELATED = re.compile(r'\b(?:pub\s+crawl|pub\s+tour|bar\s+crawl|walking\s+tour|tour\s+operator|hotel|hostel|restaurant|cafe|nightclub|souvenir\s+shop|ticket\s+office)\b', re.I)
_SCRIPTS = re.compile(r'[\u3400-\u9fff\u3040-\u30ff]')


def _clean(text):
    return ' '.join(str(text or '').split())


def _sentences(text):
    """Preserve article line boundaries; handle compact Wikivoyage listings."""
    for line in str(text or '').splitlines():
        line = _clean(line).strip(' •\t')
        if not line:
            continue
        # Wikivoyage often has listing headings and prose on one line.
        for piece in re.split(r'(?<=[.!?])\s+(?=[A-Z\"“(])', line):
            piece = piece.strip(' -•\t')
            if piece:
                yield piece


def _name_hit(sentence, name):
    return re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', sentence, re.I)


def _evidence_sentence(text, name):
    """Select a source clause actually describing the named place.

    Permit prose and Wikivoyage list entries; reject incidental mentions,
    opening hours, and clauses that describe another attraction.
    """
    ranked = []
    for clause in _sentences(text):
        m = _name_hit(clause, name)
        if not m:
            continue
        before = clause[:m.start()].strip(' \"“(•-')
        if len(before) > 36 or re.search(r'\b(?:near|of|from|beside|opposite|views? of|including|next to)\s*$', before, re.I):
            continue
        if before and not re.fullmatch(r'(?:The|Visit|See|Explore|Historic|Famous|A|An|In|At|This|The famous|The historic)\s*', before, re.I):
            continue
        after = clause[m.end():].strip(' *,:;()—–-')
        if not after or _HOURS.search(clause) or _UNRELATED.search(clause) or len(after) < 18:
            continue
        # A plain list entry may have an em dash or asterisks instead of 'is'.
        verb = re.search(r'\b(?:is|was|are|were|has|have|houses|contains|features|built|dates|stands|located|known|designed|constructed|includes|served|became|remains|offers|comprises|opened|famous)\b', after[:190], re.I)
        listing = bool(re.match(r'\s*(?:[—–:*]|\*)', clause[m.end():]))
        if not verb and not listing:
            continue
        if len(clause) > 370:
            # Long flat articles can join two listings. Don't assign the entire
            # following attraction to the current one.
            clause = clause[:370].rsplit(' ', 1)[0]
        if len(clause) < len(name) + 24:
            continue
        # Favor an explicit definition, then a meaningful list description.
        definition = bool(re.match(r'^\s*(?:is|was|are|were)\b', after, re.I))
        # A list description without a verb is less reliable.
        score = 0 if definition else (1 if verb else 3)
        ranked.append((score, len(clause), clause))
    if not ranked:
        return ''
    ranked.sort()
    return ranked[0][2]


def _local_excerpt(text, name, max_chars=430):
    return _evidence_sentence(text, name)[:max_chars]


def _stops(answer):
    out = []
    for line in answer.splitlines():
        m = STOP.match(line)
        if m:
            out.append({'name': m.group(2).strip(), 'source': int(m.group(3))})
    return out


def _quality_score(sentence, name):
    """Rank attributable facts above generic city statements / listing noise."""
    m = _name_hit(sentence, name)
    tail = sentence[m.end():].lstrip(' *,:;()—–-') if m else sentence
    low = tail.casefold()
    # These phrases usually discuss the city as a whole, not the named POI.
    generic = ('historic centre of the city', 'world heritage site',
               'tourist attractions in', 'many attractions', 'city is famous')
    if _UNRELATED.search(sentence) or any(x in low for x in generic):
        return -100
    # The passage must have meaningful facts beyond a title and category.
    score = 0
    if re.match(r'^(?:is|was|has|houses|contains|features|built|dates|stands|located|known|designed|constructed|includes|served|became)\b',tail,re.I):
        score += 7
    if re.search(r'\b(?:chapel|church|basilica|fountain|palace|museum|amphitheat(?:re|er)|temple|dome|ceiling|gallery|embassy|architecture|artist|painting|sculpture|century|built|designed|constructed|fresco)\b',tail,re.I):
        score += 4
    if 45 <= len(sentence) <= 240:
        score += 2
    if len(tail.split()) <= 5:
        score -= 5
    return score


def _lexical_candidates(collection, where, name, limit=90):
    """Retrieve city-constrained chunks actually containing the POI name.

    Unlike vector top-k this does not omit an exact mention simply because it
    is not semantically near the embedding query. No new index or API required.
    """
    seen = set()
    rows = []
    for variant in dict.fromkeys((name, name.lower())):
        try:
            result = collection.get(where=where,
                                    where_document={'$contains': variant},
                                    limit=limit,
                                    include=['documents', 'metadatas'])
        except (AttributeError, TypeError, ValueError):
            return rows  # Older Chroma releases: vector-search fallback.
        for cid, doc, meta in zip(result.get('ids') or [],
                                  result.get('documents') or [],
                                  result.get('metadatas') or []):
            if cid not in seen:
                seen.add(cid)
                rows.append((cid, doc, meta or {}, 0.0))
    return rows


def _search_evidence(assistant, destination, names, max_candidates=32):
    """Exact phrase lookup first, then vector search for remaining stops."""
    if not names:
        return {}
    titles = destination_titles(destination, assistant.titles)
    if not titles:
        return {}
    where = {'title': {'$in': titles}} if len(titles) > 1 else {'title': titles[0]}
    selected = {}

    def best_from_rows(name, rows):
        matches = []
        for cid, doc, meta, distance in rows:
            if not isinstance(doc, str):
                continue
            sentence = _evidence_sentence(doc, name)
            if not sentence:
                continue
            quality = _quality_score(sentence, name)
            if quality < 0:
                continue
            passage = Passage(text=doc, title=meta.get('title', ''),
                              url=meta.get('url', ''), distance=float(distance),
                              chunk_id=cid)
            matches.append((-quality, len(sentence), float(distance), passage, sentence))
        matches.sort(key=lambda row: row[:3])
        return (matches[0][3], matches[0][4]) if matches else None

    for name in names:
        rows = _lexical_candidates(assistant.collection, where, name)
        match = best_from_rows(name, rows)
        if match:
            selected[name] = match

    missing = [n for n in names if n not in selected]
    if not missing:
        return selected
    queries = ['query: {} {} landmark history architecture description'.format(destination, name)
               for name in missing]
    embeds = assistant.encoder.encode(queries, normalize_embeddings=True)
    results = assistant.collection.query(
        query_embeddings=[v.tolist() for v in embeds],
        n_results=max_candidates, where=where,
        include=['documents', 'metadatas', 'distances'],
    )
    for i, name in enumerate(missing):
        rows = list(zip(results['ids'][i], results['documents'][i],
                        results['metadatas'][i], results['distances'][i]))
        match = best_from_rows(name, rows)
        if match:
            selected[name] = match
    return selected

def _translation_valid(value, evidence, name):
    if not isinstance(value, str):
        return ''
    value = _clean(value)
    if not value or _INVALID.fullmatch(value) or _SCRIPTS.search(value) or len(value) > 260:
        return ''
    low = value.casefold()
    if any(t in low for t in ('одно предложение', 'источник', 'evidence', 'только json',
                             'фрагмент текста', 'туристический гид', 'описание в найденных')):
        return ''
    if not re.search('[А-Яа-яЁё]', value) or re.search(r'[A-Za-z]{13,}', value):
        return ''
    # Prevent invented dates and numbers (source must actually contain each number).
    numbers_in_source = set(re.findall(r'(?<!\w)\d+(?!\w)', evidence))
    if any(number not in numbers_in_source for number in re.findall(r'(?<!\w)\d+(?!\w)', value)):
        return ''
    # Basic semantic type consistency for high-impact class changes.
    e = evidence.lower()
    if re.search(r'\bchapel\b', e) and re.search(r'\bсобор\b', low) and not re.search(r'\bcathedral\b', e):
        return ''
    if re.search(r'\btemple\b', e) and re.search(r'\bбассейн\b', low) and not re.search(r'\bpool\b', e):
        return ''
    if re.search(r'\bfountain\b', e) and re.search(r'\bсистема\b', low) and 'system' not in e:
        return ''
    if re.search(r'\bamphitheat(?:re|er)\b', e) and re.search(r'\bсобор\b', low):
        return ''
    if _HOURS.search(value):
        return ''
    return value


def _batch_translate(excerpts, model=None, generator=None):
    if not excerpts:
        return {}
    items = [{'id': str(i), 'object': name, 'english_source_sentence': excerpt[:390]}
             for i, (name, excerpt) in enumerate(excerpts.items(), 1)]
    system = (
        'Ты переводчик английских предложений Wikivoyage на русский. '
        'Переводи КАЖДОЕ english_source_sentence точно, без пересказа. '
        'Не меняй тип сооружения, людей, века, числа, названия и факты. '
        'Не добавляй ничего от себя. Не описывай другой объект. '
        'Если английское предложение неполное или непонятное, верни пустую строку. '
        'Формат ответа: один JSON-объект с ключом translations, его значение — объект '
        'с ключами id из запроса и строками перевода. Без объяснений.'
    )
    cache_file = None
    if generator is None:
        from structured_routes import _ollama_json
        from rag_pipeline import ollama_configuration
        generator = _ollama_json
        model_name = ollama_configuration(model)[1]
        digest = hashlib.sha256(json.dumps([CACHE_VERSION, model_name, items], ensure_ascii=False,
                                           sort_keys=True).encode('utf-8')).hexdigest()
        cache_file = Path('data/cache/route_descriptions') / (digest + '.json')
    response = None
    if cache_file and cache_file.exists():
        try:
            response = json.loads(cache_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    if not isinstance(response, dict):
        try:
            response = generator(system, json.dumps(items, ensure_ascii=False), model, tokens=900)
        except Exception:
            return {}
        if cache_file and isinstance(response, dict) and isinstance(response.get('translations'), dict):
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(response, ensure_ascii=False), encoding='utf-8')
            except OSError:
                pass
    translations = response.get('translations', {}) if isinstance(response, dict) else {}
    if not isinstance(translations, dict):
        return {}
    good = {}
    for i, (name, evidence) in enumerate(excerpts.items(), 1):
        value = _translation_valid(translations.get(str(i)), evidence, name)
        if value:
            good[name] = value
    return good


def enrich_route_answer(answer, passages, assistant, destination, model=None,
                        max_stops=10, generator=None):
    """Return updated route and URL-unique citations. No invented fallback facts."""
    stops = _stops(answer)
    if not stops or not destination:
        return answer, passages
    names = list(dict.fromkeys(s['name'] for s in stops[:max_stops]))
    try:
        evidence = _search_evidence(assistant, destination, names)
    except Exception:
        evidence = {}
    translations = _batch_translate({n: p[1] for n, p in evidence.items()},
                                    model=model, generator=generator)
    sources = []
    source_map = {}

    def cite(passage):
        key = passage.url or (passage.chunk_id, passage.title)
        if key not in source_map:
            source_map[key] = len(sources) + 1
            sources.append(passage)
        return source_map[key]

    output = []
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
            output.append(line)
            continue
        name = match.group(2).strip()
        found = evidence.get(name)
        if not found:
            # Never reuse potentially incorrect original generated descriptions.
            output.append('{}{} — {}'.format(match.group(1), name, FALLBACK))
            continue
        passage, english = found
        citation = cite(passage)
        translation = translations.get(name)
        if translation:
            description = translation
        else:
            # Verbatim sourced fact is preferable to an invented Russian claim.
            description = 'По Wikivoyage (оригинал): ' + english[:300]
        output.append('{}{} [Источник {}] — {}'.format(match.group(1), name, citation, description))

    if sources:
        at = next((i for i, line in enumerate(output) if NOTE.match(line)), len(output))
        listing = ['Источники:']
        listing.extend('[Источник {}] {}: {}'.format(i, src.title, src.url)
                       for i, src in enumerate(sources, 1))
        output[at:at] = listing
    return '\n'.join(output).strip(), sources
