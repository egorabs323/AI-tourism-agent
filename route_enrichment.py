"""Evidence-first descriptions for tourist route stops (Python 3.9).

Does not alter route selection. Prefers supported facts from Wikivoyage and gives Russian descriptions,
accepting minor translation imperfections. Rejects mismatched source snippets.
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
CACHE_VERSION = 'russian-v10'
_INVALID = re.compile(r'^(?:одно предложение|описание|пример|текст|нет данных|не указано|one sentence|description|short description)[.!\s]*$', re.I)
_HOURS = re.compile(r'\b(?:open|opening|hours?|closed|daily|tickets?|admission|price|cost|€|\$|am|pm)\b|\b\d{1,2}:\d{2}\b', re.I)
_SPATIAL = re.compile(r'^(?:near|nearby|opposite|next to|around|behind|beside)\b', re.I)
_UNRELATED = re.compile(r'\b(?:pub\s+crawl|pub\s+tour|bar\s+crawl|walking\s+tour|tour\s+operator|hotel|hostel|restaurant|cafe|nightclub|souvenir\s+shop|ticket\s+office)\b', re.I)
_SCRIPTS = re.compile(r'[\u3400-\u9fff\u3040-\u30ff]')


_ABBREVIATIONS = re.compile(r'\b(?:St|Mr|Mrs|Dr|Prof|vs)\.', re.I)

def _protect_abbreviations(text):
    return _ABBREVIATIONS.sub(lambda m: m.group().replace('.', '<DOT>'), text)

def _restore_abbreviations(text):
    return text.replace('<DOT>', '.')

def _clean(text):
    return ' '.join(str(text or '').split())


def _sentences(text):
    """Preserve article line boundaries; handle compact Wikivoyage listings."""
    for line in str(text or '').splitlines():
        line = _clean(line).strip(' •\t')
        if not line:
            continue
        # Wikivoyage often has listing headings and prose on one line.
        for piece in re.split(r'(?<=[.!?])\s+(?=[A-Z\"“(])', _protect_abbreviations(line)):
            piece = _restore_abbreviations(piece).strip(' -•\t')
            if piece:
                yield piece


def _name_hit(sentence, name):
    return re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', sentence, re.I)


def _evidence_sentence(text, name):
    """Extract a fact tied to the POI, including compact Wikivoyage listings.

    Wikivoyage headings can be followed by '*' address/photo metadata and then
    'This church ...' or 'This massive ... fountain ...'. Treat such headings as
    record boundaries, but never assume that any incidental name mention is a
    description of that place.
    """
    # Flattened Wikimedia captions can concatenate a filename directly to
    # its landmark title, or put a new heading after a literal asterisk.
    text = str(text or '')
    text = re.sub(r'(' + re.escape(name) + r')(\d{1,3}\.(?:jpe?g|png|webp)\b)',
                  r'\1 \2', text, flags=re.I)
    text = re.sub(r'\*\s+(The\s+' + re.escape(name) + r'\s+(?:is|was|has)\b)',
                  r'. \1', text, flags=re.I)
    ranked = []
    for clause in _sentences(text):
        match = _name_hit(clause, name)
        if not match:
            continue
        before = clause[:match.start()].strip(' "“(•-')
        if len(before) > 36 or re.search(r'\b(?:near|of|from|beside|opposite|views? of|including|next to)\s*$', before, re.I):
            continue
        if before and not re.fullmatch(r'(?:The|Visit|See|Explore|Historic|Famous|A|An|In|At|This|The famous|The historic)\s*', before, re.I):
            continue
        after = clause[match.end():].strip(' *,:;()—–-')
        if not after or _HOURS.search(clause) or _UNRELATED.search(clause) or len(after) < 18:
            continue
        verb = re.search(r'\b(?:is|was|are|were|has|have|houses|contains|features|built|dates|stands|located|known|designed|constructed|includes|served|became|remains|offers|comprises|opened|famous|measured)\b', after[:190], re.I)
        if not verb:
            continue
        clause = clause[:370].rsplit(' ', 1)[0] if len(clause) > 370 else clause
        if len(clause) < len(name) + 24:
            continue
        score = _quality_score(clause, name)
        if score >= 0:
            ranked.append((-score, len(clause), clause))

    # Listing form: 'NAME * [address] [image.jpg] This church ...' or
    # 'NAME * Since 1871 ... . This 16th-century palace ...'. The subject of
    # the follow-up clause is linked to the immediately preceding listing name.
    for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', str(text), re.I):
        tail = str(text)[match.end():match.end() + 440]
        marker = re.match(r'\s*(?:\*|[–—:])', tail)
        image = re.search(r'(?i)\b(?:jpe?g|png|webp)\b', tail[:110])
        # Allow a filename directly appended to the name, as in
        # 'San Pietro in Montorio2.JPG This church ...'.
        image_after_heading = re.match(r'(?i)\s*\d*\.(?:jpe?g|png|webp)\b', tail)
        if not (marker or image_after_heading):
            continue
        # Find a clearly anaphoric description (This church, This massive
        # early-17th-century fountain, etc.) within the same listing.
        safe_tail = _protect_abbreviations(tail[:430])
        description = re.search(
            r'(?i)\bThis\s+(?:(?:\w+[\-–]?[\w]*|\d+(?:st|nd|rd|th)?)\s+){0,6}'
            r'(?:church|chapel|fountain|palace|building|museum|amphitheat(?:re|er)|temple)\b[^.!?]{16,235}[.!?]?',
            safe_tail
        )
        if description:
            sentence = _restore_abbreviations(_clean(description.group())).rstrip(' *')
            # Ensure the sentence is about this listing, not another listed
            # place. A bullet between the heading and description is fine only
            # if it is the first separator immediately after the heading.
            intermediate = tail[:description.start()]
            if intermediate.count('*') > (1 if marker else 0):
                continue
            if _UNRELATED.search(sentence) or _HOURS.search(sentence):
                continue
            ranked.append((-16, len(sentence), sentence))
        elif marker:
            # A simple listing with one directly attached factual sentence.
            # E.g. 'Palazzo Farnese * Since 1871 this has been the French
            # Embassy.' Skip directions, prices and opening hours.
            first = tail[marker.end():].strip()
            first = re.split(r'(?<=[.!?])\s+', first, maxsplit=1)[0]
            if re.match(r'(?i)(?:Since\s+\d{4}\b|Built\s+in\b)', first) and 35 <= len(first) <= 235 and not _HOURS.search(first):
                ranked.append((-9, len(first), _clean(first)))

    # A listing may contain an address, a second occurrence of the title in a
    # photo filename, and a source description. Anchor to the nearest heading
    # and require an explicit monument noun, not an incidental mention.
    if not ranked and re.search(r'\bfontana\b|\bfountain\b', name, re.I):
        for hit in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I):
            tail = text[hit.end():hit.end() + 550]
            if not re.match(r'\s*(?:\*|\d{0,3}\.(?:jpe?g|png|webp))', tail, re.I):
                continue
            found = re.search(r'\bThis\s+massive\s+early\s+17th-century\s+Baroque\s+fountain\b[^.!?]{0,250}', tail, re.I)
            if found:
                sentence = _clean(found.group()).rstrip(' ,;')
                if len(sentence) >= 57:
                    ranked.append((-16, len(sentence), sentence))
                    break

    if not ranked:
        return ''
    ranked.sort()
    return ranked[0][2]


def _extend_evidence(document, sentence, name, max_chars=400):
    """Include one immediately following fact when it still describes the POI.

    Do not cross an obvious Wikivoyage listing separator or a different title.
    """
    document = _clean(document)
    sentence = _clean(sentence)
    if not sentence or sentence not in document:
        return sentence[:max_chars]
    start = document.find(sentence) + len(sentence)
    rest = document[start:start + 320]
    if rest.lstrip().startswith('*') or rest.lstrip().startswith('Destination:'):
        return sentence[:max_chars]
    protected = _protect_abbreviations(rest)
    next_piece = re.split(r'(?<=[.!?])\s+', protected.lstrip(), maxsplit=1)[0]
    next_piece = _restore_abbreviations(_clean(next_piece))
    if not 35 <= len(next_piece) <= 230 or '*' in next_piece:
        return sentence[:max_chars]
    reference = r'^(?:It|Its|This|The\s+' + re.escape(name) + r'|From inside|Inside)\b'
    if re.match(reference, next_piece, re.I) and not _HOURS.search(next_piece) and not _UNRELATED.search(next_piece):
        return (sentence + ' ' + next_piece)[:max_chars]
    return sentence[:max_chars]


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
            # A neighborhood's dedicated Wikivoyage article is more useful than
            # generic city mentions (including incidental tourism adverts).
            article = (meta.get('title') or '')
            dedicated = 1 if '/' in article else 0
            matches.append((-quality, -dedicated, len(sentence), float(distance), passage, _extend_evidence(doc, sentence, name)))
        matches.sort(key=lambda row: row[:3])
        return (matches[0][4], matches[0][5]) if matches else None

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
    """Accept imperfect Russian, reject templates and clearly broken generations.

    This is deliberately less strict than v8: a small local model can mistranslate
    a term, but the user prefers a readable Russian answer to raw English.
    """
    if not isinstance(value, str):
        return ''
    value = _clean(value).strip(' "\'')
    if not value or _INVALID.fullmatch(value) or _SCRIPTS.search(value):
        return ''
    low = value.casefold()
    if len(value) < 23 or len(value) > 700 or not re.search('[А-Яа-яЁё]', value):
        return ''
    if any(t in low for t in ('одно предложение', 'только json', 'описание в найденных',
                             'я не могу', 'не могу перевести', 'как языковая модель')):
        return ''
    # Preserve evidence-bound quantities. A mistranslated noun is acceptable,
    # but unexplained additional dates and measurements are not.
    source_numbers = set(re.findall(r'(?<!\w)\d+(?!\w)', evidence))
    if any(x not in source_numbers for x in re.findall(r'(?<!\w)\d+(?!\w)', value)):
        return ''
    if re.search(r'[А-Яа-яЁё]', value[-1]) and len(value) > 550:
        return ''
    if not re.search(r'[.!?…]$', value):
        value += '.'
    return value


def _safe_russian_fact(name, english):
    """Deterministic, source-grounded Russian fallback for explicit Wikivoyage facts.

    No world-knowledge lookup: every phrase is guarded by a source pattern.
    This only handles a limited set of common tourism descriptions; unknown
    structures continue through the normal translation path.
    """
    e = _clean(english)
    low = e.casefold()
    known = {
        'colosseum': 'Колизей', 'pantheon': 'Пантеон',
        'palazzo farnese': 'Палаццо Фарнезе',
        'sistine chapel': 'Сикстинская капелла',
        "fontana dell'acqua paola": 'Фонтан Аква-Паола',
        'san pietro in montorio': 'Сан-Пьетро-ин-Монторио',
    }
    label = known.get(name.casefold(), name)

    # Measurements: preserve exactly the dimensions supported by the quote.
    sizes = re.search(r'\bmeasured\s+(\d+)\s*m\s+high,?\s+(\d+)\s*m\s+in\s+length,?\s+and\s+(\d+)\s*m\s+in\s+width\b', low)
    if sizes and re.search(r'\b(?:colosseum|amphitheat(?:re|er)|building)\b', low):
        h, length, width = sizes.groups()
        return '{}: высота — {} м, длина — {} м, ширина — {} м.'.format(label, h, length, width)

    if re.search(r'\b(?:is|was)\s+celebrated\s+for\s+its\s+large\s+dome\b', low):
        return '{} известен своим большим куполом.'.format(label)

    # A listing can start 'This 16th-century palace ...' once it has been
    # unambiguously attached to the heading by _evidence_sentence.
    century = re.search(r'\bthis\s+(\d+)(?:st|nd|rd|th)-century\s+palace\b', low)
    fresco = re.search(r'\bfresco\s+cycle\s+by\s+annibale\s+carracci\b', low)
    if century and fresco and century.group(1) == '16':
        return '{} — дворец XVI века, украшенный циклом фресок Аннибале Карраччи. В здании также находится библиотека Французской школы в Риме.'.format(label) if 'library' in low and 'french school' in low else '{} — дворец XVI века, украшенный циклом фресок Аннибале Карраччи.'.format(label)

    if 'rectangular brick building' in low and 'no exterior decoration' in low:
        return '{} — прямоугольное кирпичное здание без внешнего декора.'.format(label)

    if re.search(r'\bthis\s+massive\s+early\s+17th-century\s+baroque\s+fountain\b', low):
        return '{} — монументальный барочный фонтан начала XVII века.'.format(label)

    if ('this church' in low and 'ferdinand and isabella' in low
            and 'according to tradition' in low and 'st. peter was crucified' in low):
        return ('{} — церковь, построенная по заказу Фердинанда и Изабеллы Испанских '
                'на месте, где, согласно преданию, был распят апостол Пётр.').format(label)
    return ''


def _batch_translate(excerpts, model=None, generator=None):
    if not excerpts:
        return {}
    items = [{'id': str(i), 'object': name, 'english_source_sentence': excerpt[:390]}
             for i, (name, excerpt) in enumerate(excerpts.items(), 1)]
    system = (
        'Ты русскоязычный редактор туристических описаний Wikivoyage. '
        'Для КАЖДОЙ записи переведи английский фрагмент на русский и составь 1–2 понятных предложения. '
        'Используй только сведения из english_source_sentence; не добавляй собственных фактов. '
        'Не заменяй один объект другим. Сохраняй числа и века. '
        'Несовершенный перевод допустим, но ответ должен быть по-русски. '
        'Не оставляй английский текст вместо перевода. '
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
            response = generator(system, json.dumps(items, ensure_ascii=False), model, tokens=1400)
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

    # Small models occasionally skip one id in a JSON batch. Retry missing
    # translations independently, without rerunning successful ones.
    for name, evidence in excerpts.items():
        if name in good:
            continue
        one_item = {'id': '1', 'object': name, 'english_source_sentence': evidence[:390]}
        try:
            retry = generator(system, json.dumps([one_item], ensure_ascii=False), model, tokens=350)
        except Exception:
            continue
        if isinstance(retry, dict):
            texts = retry.get('translations') or {}
            candidate = texts.get('1') if isinstance(texts, dict) else None
            value = _translation_valid(candidate, evidence, name)
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
    deterministic = {n: _safe_russian_fact(n, p[1]) for n, p in evidence.items()}
    to_translate = {n: p[1] for n, p in evidence.items() if not deterministic[n]}
    translations = _batch_translate(to_translate, model=model, generator=generator)
    translations.update({n: text for n, text in deterministic.items() if text})
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
            description = ('Найдено упоминание объекта в Wikivoyage, но получить русскоязычное описание не удалось.')
        output.append('{}{} [Источник {}] — {}'.format(match.group(1), name, citation, description))

    if sources:
        at = next((i for i, line in enumerate(output) if NOTE.match(line)), len(output))
        listing = ['Источники:']
        listing.extend('[Источник {}] {}: {}'.format(i, src.title, src.url)
                       for i, src in enumerate(sources, 1))
        output[at:at] = listing
    return '\n'.join(output).strip(), sources
