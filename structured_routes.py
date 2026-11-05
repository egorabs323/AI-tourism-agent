"""Object records from local Wikivoyage, with source spans bound before generation."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from rag_pipeline import Passage, ollama_configuration


@dataclass(frozen=True)
class Attraction:
    name: str
    passage_index: int
    title: str
    url: str
    evidence: str


def route_days(question: str) -> int:
    q = question.lower()
    number = re.search(r'\b(?:на|за)\s+(\d{1,2})\s*(?:день|дня|дней|days?)\b', q)
    if number:
        return max(1, min(7, int(number.group(1))))
    for word, n in [('один', 1), ('одного', 1), ('два', 2), ('двух', 2), ('три', 3), ('трёх', 3), ('четыре', 4)]:
        if re.search(r'\b(?:на|за)\s+' + word + r'\s+д(?:ень|ня|ней)\b', q):
            return n
    return 1


def is_route_question(question: str) -> bool:
    """Compatibility for the console client; the web UI uses explicit modes."""
    return bool(re.search(r'маршрут|itinerary|route for', question, re.I))


def _parse_json_object(text: str) -> dict | None:
    """Разбирает ответ модели; допускает случайные Markdown-ограждения."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _ollama_json(system: str, user: str, model: str | None, tokens: int = 700) -> dict:
    """Один ограниченный по контексту вызов локальной Ollama.

    Важное ограничение: формат JSON гарантирует синтаксис при успешной
    генерации, но не гарантирует достоверность содержимого.
    """
    url, selected_model = ollama_configuration(model)
    endpoint = url + '/api/chat'
    payload = {
        'model': selected_model,
        'messages': [
            {'role': 'system', 'content': system + '\nOutput only a compact valid JSON object. No explanation.'},
            {'role': 'user', 'content': user},
        ],
        'stream': False,
        'think': False,
        'format': 'json',
        'options': {
            'temperature': 0,
            'num_ctx': 4096,
            'num_predict': tokens,
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')[:300]
        raise RuntimeError(f'Ollama вернула HTTP {exc.code}: {details}') from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError('Ollama недоступна. Проверь: ollama list') from exc
    content = (raw.get('message') or {}).get('content', '')
    value = _parse_json_object(content)
    if value is not None:
        return value
    done_reason = raw.get('done_reason', 'не указан')
    last_reason = f'done_reason={done_reason}, символов в ответе={len(content)}, лимит={tokens}'
    raise RuntimeError(
        'Ollama не смогла сформировать корректный JSON '
        f'({last_reason}). Проверь модель: ollama list.'
    )


def group_attractions(attractions: list[Attraction], days: int, per_day: int = 3) -> list[list[Attraction]]:
    """Prefer highest-ranked attractions and keep same Wikivoyage district together.

    This does NOT calculate distances and is not a shortest-path planner.
    """
    picks = attractions[:days * per_day]
    result: list[list[Attraction]] = [[] for _ in range(days)]
    for attraction in picks:
        available = [i for i in range(days) if len(result[i]) < per_day]
        if not available:
            break
        # Place in a day already containing the same district if capacity allows.
        same_district = [i for i in available
                         if any(a.title == attraction.title for a in result[i])]
        if same_district:
            target = min(same_district, key=lambda i: (len(result[i]), i))
        else:
            target = min(available, key=lambda i: (len(result[i]), i))
        result[target].append(attraction)
    return result


def render_route(groups: list[list[Attraction]], desc: dict[str, str], all_attractions: list[Attraction]) -> str:
    index = {a: str(i) for i, a in enumerate(all_attractions, 1)}
    lines = []
    for day, entries in enumerate(groups, 1):
        lines.append(f'День {day}')
        if not entries:
            lines.append('Подтверждённых объектов для этого дня недостаточно.')
            continue
        for number, a in enumerate(entries, 1):
            summary = desc.get(index[a], '')
            # Имена и ссылки рендерит Python, модель не может добавить новые остановки.
            line = f'{number}. {a.name} [Источник {a.passage_index}]'
            if summary:
                line += f' — {summary}'
            lines.append(line)
        lines.append('')
    sources = sorted({a.passage_index: (a.title, a.url) for group in groups for a in group}.items())
    if sources:
        lines.append('Источники:')
        for i, (title, url) in sources:
            lines.append(f'[Источник {i}] {title}: {url}')
    lines.append('Порядок остановок сгруппирован приблизительно по разделам путеводителя; '
                 'расстояния и время в пути не рассчитаны.')
    return '\n'.join(lines).strip()


# Syntax of proper names, not a city/landmark whitelist. Lower-case particles
# permit names such as Palazzo dei Conservatori and Arc de Triomphe.
_WORD = r"(?:St\.|[A-ZÀ-ÖØ-Þ][\wÀ-ÿ’'’-]*)"
_NAME = re.compile(r"(?<![\w’'])" + _WORD + r"(?:\s+(?:(?:de|di|del|della|dei|degli|des|du|of|the|in|and|la|le|da|dell['’])\s+)?" + _WORD + r"){0,7}")
_VERB = re.compile(r'^\s*(?:\([^)]{1,100}\)\s*)?(?:,\s*[^,]{1,120},\s*)?'
                   r'(?:(?:when completed|originally)\s+)?'
                   r'(?:is|are|was|were|has|have|houses|contains|features|stands|dates|measured|commemorates)\b', re.I)
_FACT = re.compile(r'\b(?:built|construct\w*|design\w*|centur\w*|dome|gothic|baroque|'
                   r'ancient|medieval|renaissance|collection|paintings?|sculptur\w*|'
                   r'fresco\w*|archaeolog\w*|amphitheat\w*|aqueduct\w*|'
                   r'royal|residence|drawbridge|exhibits?|artworks?|'
                   r'commemorat\w*|contains|houses|decorat\w*|measured|'
                   r'founded|dedicated|architecture|largest|tombs?|buried)\b', re.I)
_LOGISTICS = re.compile(r'\b(?:pickpockets?|scams?|thieves|souvenir|hotels?|hostels?|'
                        r'restaurants?|café|cafe|pubs?|booking|bus|metro|tube|station|'
                        r'tickets?|queues?|admission|timetable|tour operator|luggage|airport)\b', re.I)
_RELATION = re.compile(r'\b(?:near|beside|behind|opposite|next to|from|between|towards|'
                       r'in front of|view of|views of)\b', re.I)
_TYPES = {
    'tower': ('tower',), 'bridge': ('bridge', 'drawbridge'),
    'square': ('square', 'piazza'), 'park': ('park', 'garden', 'gardens', 'parco'),
    'church': ('church', 'cathedral', 'cathedrale', 'basilica', 'basilique', 'chapel', 'chiesa', 'cappella'),
    'palace': ('palace', 'palazzo'), 'museum': ('museum', 'museo', 'gallery', 'galleria'),
    'fountain': ('fountain', 'fontana'), 'arch': ('arch', 'arc'),
    'building': ('building', 'house', 'castle', 'fortress', 'auditorium', 'monument', 'memorial', 'observatory'),
    'architecture': ('dome', 'amphitheatre', 'amphitheater', 'temple', 'ruins', 'theatre', 'theater', 'aqueduct'),
}


def name_key(name: str) -> str:
    text = unicodedata.normalize('NFKD', name.casefold())
    return ' '.join(re.findall(r'\w+', ''.join(c for c in text if not unicodedata.combining(c))))


def _clean_name(name: str) -> str:
    name = re.sub(r'^(?:The|A|An)\s+', '', name.strip().rstrip('.'))
    words = name.split()
    for size in range(1, len(words) // 2 + 1):
        if len(words) % size == 0 and words == words[:size] * (len(words) // size):
            return ' '.join(words[:size])
    return name


def _sentences(text: str) -> list[str]:
    # Sentence boundaries cannot cross a record, newline or bullet.
    text = re.sub(r'\b(St|Mr|Dr)\.', r'\1<DOT>', text)
    return [s.replace('<DOT>', '.').strip() for s in
            re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ"“])|[\r\n]+', text) if s.strip()]


def _types(text: str) -> set:
    words = set(name_key(text).split())
    return {kind for kind, labels in _TYPES.items() if words.intersection(labels)}


def _primary_type(text: str) -> set:
    """The first object noun owns a listing; later location nouns do not."""
    for word in name_key(text).split():
        for kind, labels in _TYPES.items():
            if word in labels:
                return {kind}
    return set()


def _informative(text: str) -> bool:
    return 35 <= len(text) <= 650 and bool(_FACT.search(text)) and not _LOGISTICS.search(text)


@lru_cache(maxsize=128)
def _records(text: str) -> tuple:
    result = []
    for block in re.split(r'\*+|\n\s*[-•]\s+', text):
        result.append((block, block.casefold(), _sentences(block)))
    return tuple(result)


def _subject_fact(sentence: str, name: str) -> str:
    """Require the name to be the subject, never a location or image caption.

    Matching a proper noun somewhere in a 220-word chunk is insufficient.
    Only the independent clause naming the object is returned.
    """
    target = name_key(name)
    for match in _NAME.finditer(sentence):
        subject = _clean_name(match.group())
        # Descriptive modifiers (e.g. 'early Gothic Cathédrale de Notre Dame')
        # can precede an exact name; a longer proper name cannot follow it.
        key = name_key(subject)
        if key != target and not key.endswith(' ' + target):
            continue
        if key != target and len(target.split()) == 1:
            continue
        prefix = sentence[:match.start()].strip()
        clause_start = bool(re.search(r',\s*(?:the|a|an)?\s*$', prefix, re.I))
        if prefix and not clause_start and (not re.fullmatch(r'(?:The|A|An|early|late|great|famous|Gothic|Baroque|\s)+', prefix)
                                            or _RELATION.search(prefix)):
            continue
        tail = sentence[match.end():]
        if _VERB.match(tail):
            if re.match(r'\s*(?:is said to have|was born|was voted|is buried|died|lived|founded)\b', tail, re.I):
                continue  # a person's biography is not a building's history
            predicate = re.split(r'\b(?:of|in|near|from|with)\b', tail, maxsplit=1, flags=re.I)[0][:100]
            if re.search(r'\b(?:area|district|city|country|region|neighbou?rhood)\b', predicate, re.I):
                continue
            if re.search(r'\bis not an?\b', predicate, re.I):
                continue
            # A repeated image caption can precede the actual subject.
            at = match.group().casefold().rfind(name.casefold()) if match.group().casefold().count(name.casefold()) > 1 else 0
            fact = sentence[match.start() + max(0, at):].strip()
            # Stop before a new named subject in a compound sentence.
            for other in _NAME.finditer(fact, match.end() - match.start() - max(0, at)):
                if _VERB.match(fact[other.end():]) and name_key(_clean_name(other.group())) != key:
                    boundary = max(fact.rfind(',', 0, other.start()), fact.rfind(';', 0, other.start()))
                    if boundary >= 0:
                        fact = fact[:boundary].rstrip() + '.'
                        break
            if _informative(fact):
                return fact
    return ''


def object_facts(text: str, name: str) -> list[tuple[int, str]]:
    """Separate bullets FIRST. Never attach the text after '*' to a caption.

    Old flattened data has missing listing names. Such records are used only
    when a name occurs inside the record and the primary object type agrees.
    Ambiguous/no-name records are left unavailable rather than guessed.
    """
    result = []
    pattern = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I)
    for block, lowered, sentences in _records(text):
        if name.casefold() not in lowered or not pattern.search(block):
            continue
        prose = [re.split(r'\.(?:jpe?g|png|webp)\b\s*', s, flags=re.I)[-1] for s in sentences]
        for sentence in prose:
            fact = _subject_fact(sentence, name)
            if fact:
                result.append((100, fact))
        if not sentences:
            continue
        # Explicit 'Name. Description.' listings retain unambiguous ownership.
        heading = sentences[0].rstrip('.')
        if name_key(heading) == name_key(name):
            facts = []
            for sentence in sentences[1:4]:
                if _LOGISTICS.search(sentence):
                    break
                if _informative(sentence):
                    facts.append(sentence)
                else:
                    break
            if facts:
                result.append((110, ' '.join(facts[:2])))
        # In a damaged listing no named header can safely be recovered from
        # the PRECEDING bullet. A self-mention inside the SAME bullet plus a
        # matching primary type is required. Reject a different named subject.
        kinds = _types(name)
        if not kinds or len(block) > 5000:
            continue
        # Image filenames and addresses are metadata, not a descriptive subject.
        descriptive = [s for s in prose if _informative(s) and re.match(
            r'^(?:The|A|An|This|These|Built|Completed|Constructed|Designed|Founded|Magnificent|One)\b', s)]
        if not descriptive:
            continue
        first = descriptive[0]
        # In flattened data a generic 'This church ...' without its original
        # named heading has lost its owner. Do not guess it from nearby names.
        # Construction clauses retain a primary object noun and a self-mention
        # inside the same bullet, without borrowing a preceding caption.
        if not re.match(r'^(?:Built|Completed|Constructed|Designed|Founded)\b', first):
            continue
        primary = _primary_type(re.split(r'\b(?:is|was|are|were|has|with|near|from|featur\w*)\b', first, maxsplit=1)[0])
        if re.match(r'^(?:Built|Completed|Constructed|Designed|Founded)\b', first):
            primary = _primary_type(first)
        if not primary.intersection(kinds):
            continue
        other_subject = next((m for m in _NAME.finditer(first)
                              if _VERB.match(first[m.end():]) and
                              name_key(m.group()) != name_key(name) and
                              not name_key(m.group()).endswith(' ' + name_key(name))), None)
        if other_subject:
            continue
        result.append((60, first))
    return sorted(set(result), key=lambda item: (-item[0], len(item[1])))


def _evidence(text: str, name: str) -> str:
    facts = object_facts(text, name)
    return facts[0][1] if facts else ''


def validate_attractions(raw: list, passages: list[Passage], max_items: int = 18) -> list[Attraction]:
    good, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name, source = item.get('name'), item.get('source')
        if not isinstance(name, str) or type(source) is not int or not 1 <= source <= len(passages):
            continue
        name = ' '.join(name.split())
        key = name_key(name)
        passage = passages[source - 1]
        if not 4 <= len(name) <= 90 or len(name.split()) > 10 or key in seen:
            continue
        if not re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', passage.text, re.I):
            continue
        evidence = _evidence(passage.text, name)
        good.append(Attraction(name, source, passage.title, passage.url, evidence))
        seen.add(key)
        if len(good) >= max_items:
            break
    return good


def city_catalogue(passages: list[Passage], destination: str, question: str = '') -> list[Attraction]:
    """Names are corpus substrings; rank only objects with attributable facts.

    Frequency is a weak prominence signal, not a hand-written popularity list.
    Preferences add a score for facts actually present in the selected record.
    """
    counts = Counter()
    main_names = set()
    spellings = {}
    for p in passages:
        for match in _NAME.finditer(p.text):
            full = _clean_name(match.group())
            # Drop descriptive capitalized modifiers without losing the exact
            # corpus substring: 'Gothic Cathédrale de Notre Dame' -> 'Notre Dame'.
            tokens = full.split()
            names = [full]
            if not _types(full):
                names += [' '.join(tokens[i:]) for i in range(1, len(tokens) - 1)
                          if tokens[i][0].isupper()]
            for name in names:
                key = name_key(name)
                if not 4 <= len(name) <= 75 or key == name_key(destination):
                    continue
                if not _types(name) and len(name.split()) > 5:
                    continue
                counts[key] += 1
                spellings.setdefault(key, name)
                if p.title == destination:
                    main_names.add(key)
    entries = []
    for key, count in counts.items():
        name = spellings[key]
        if any(key == name_key(p.title.rsplit('/', 1)[-1]) for p in passages):
            continue  # a city/district article is not an individual stop
        if key in {label for labels in _TYPES.values() for label in labels}:
            continue
        if len(name.split()) == 1 and (key in {
                'this', 'these', 'those', 'there', 'they', 'also', 'here', 'that', 'which', 'what',
                'some', 'most', 'many', 'other', 'one', 'two', 'three', 'four', 'five',
                'see', 'understand', 'landmarks', 'churches', 'museums', 'piazza',
                'tour', 'basilica', 'cathedral', 'palazzo', 'temple', 'garden', 'gardens',
                'tower', 'bridge', 'church', 'museum', 'palace', 'square', 'park', 'arch'}):
            continue
        options = []
        for i, p in enumerate(passages, 1):
            for confidence, fact in object_facts(p.text, name):
                if not _types(name) and not _types(fact) and 'measured' not in fact:
                    continue
                if not _types(name) and not _subject_fact(fact, name):
                    continue
                if not _types(name):
                    # A feature of a neighbouring named place is not a type
                    # statement about the subject (e.g. a district centred on
                    # a park). Require an owned feature or a direct type clause.
                    tail = fact[fact.casefold().find(name.casefold()) + len(name):]
                    own_feature = bool(re.search(r'\b(?:its|measured)\b', tail) or
                                       re.match(r'\s*(?:has|houses|contains|features)\b', tail))
                    type_clause = re.match(r'\s*(?:is|was)\s+(?:a|an|the)\s+', tail)
                    if not own_feature and not (type_clause and _types(tail.split(',')[0])):
                        continue
                options.append((confidence, len(fact), i, p, fact))
        if not options:
            continue
        confidence, _, i, p, fact = max(options, key=lambda x: (x[0], min(x[1], 400)))
        score = min(count, 40) + (8 if key in main_names else 0) + confidence / 20
        if re.search(r'истор|ancient|historic', question, re.I) and re.search(
                r'\b(?:ancient|historic|century|built|medieval|gothic|baroque|roman)\b', fact, re.I):
            score += 12
        if re.search(r'музе|искусств|museum|art', question, re.I) and _types(name) == {'museum'}:
            score += 16
        if re.search(r'парк|сад|природ|park|garden', question, re.I) and _types(name) == {'park'}:
            score += 16
        if re.search(r'главн|известн|перв|famous|main|first', question, re.I):
            score += min(count, 40) / 2 + (8 if key in main_names else 0)
        entries.append((score, Attraction(name, i, p.title, p.url, fact)))
    entries.sort(key=lambda row: (-row[0], name_key(row[1].name)))
    unique, used_facts = [], {}
    for _, attraction in entries:
        identity = (attraction.title, attraction.evidence)
        alias = next((old for old in unique if old.title == attraction.title and
                      (name_key(old.name).endswith(' ' + name_key(attraction.name)) or
                       name_key(attraction.name).endswith(' ' + name_key(old.name))) and
                      old.evidence[old.evidence.casefold().find(old.name.casefold()) + len(old.name):] ==
                      attraction.evidence[attraction.evidence.casefold().find(attraction.name.casefold()) + len(attraction.name):]), None)
        if alias:
            continue
        if identity not in used_facts:
            used_facts[identity] = len(unique)
            unique.append(attraction)
        else:
            index = used_facts[identity]
            if _types(attraction.name) and name_key(attraction.name).endswith(' ' + name_key(unique[index].name)):
                unique[index] = attraction
    return unique


def extract_route_candidates(passages: list[Passage], destination: str, days: int,
                             model: str | None,
                             question: str = '') -> tuple[list[Attraction], dict[str, str]]:
    candidates = _cached_catalogue(tuple((p.text, p.title, p.url, p.chunk_id) for p in passages),
                                   destination, question)
    if not candidates:
        return [], {}
    # Ranking is deterministic. A 1.7B model is unreliable at selecting IDs
    # from a catalogue: actual local validation showed it rejecting confirmed
    # landmarks and keeping generic institutions. It only translates facts.
    return list(candidates[:days * 3]), {}



@lru_cache(maxsize=3)
def _cached_catalogue(rows: tuple, destination: str, question: str) -> tuple:
    return tuple(city_catalogue([Passage(text, title, url, 0.0, cid) for text, title, url, cid in rows],
                                destination, question))


def answer_route(question: str, passages: list[Passage], destination: str | None,
                 model: str | None = None) -> str:
    if not destination:
        return 'Укажи город, чтобы составить маршрут.'
    days = route_days(question)
    candidates, _ = extract_route_candidates(passages, destination, days, model, question=question)
    return render_route(group_attractions(candidates, days), {}, candidates)
