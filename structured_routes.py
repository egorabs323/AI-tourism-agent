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
_CAPITALS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ''.join(chr(i) for i in range(0xC0, 0x250) if chr(i).isupper())
_WORD = r"(?:St\.|[" + re.escape(_CAPITALS) + r"][\w’'’-]*)"
_NAME = re.compile(r"(?<![\w’'])" + _WORD + r"(?:\s+(?:(?:de|di|del|della|delle|dei|degli|des|du|of|the|in|and|la|le|da|dell['’])\s+)?" + _WORD + r"){0,7}")
_VERB = re.compile(r'^\s*(?:\([^)]{1,100}\)\s*)?(?:,\s*[^,]{1,120},\s*)?'
                   r'(?:(?:when completed|originally)\s+)?'
                   r'(?:is|are|was|were|has|have|houses|contains|features|stands|dates|measured|commemorates)\b', re.I)
_FACT = re.compile(r'\b(?:built|construct\w*|design\w*|centur\w*|dome|gothic|baroque|'
                   r'ancient|medieval|renaissance|collection|paintings?|sculptur\w*|'
                   r'fresco\w*|archaeolog\w*|amphitheat\w*|aqueduct\w*|'
                   r'royal|residence|drawbridge|exhibits?|artworks?|'
                   r'commemorat\w*|contains|houses|decorat\w*|measured|'
                   r'founded|dedicated|architecture|largest|oldest|tallest|grandest|'
                   r'imperial|popular|famous|symbol|water|columns?|minarets?|Buddhist|Shinto|'
                   r'blossoms?|tombs?|buried)\b', re.I)
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
    'mosque': ('mosque', 'camii'), 'shrine': ('shrine', 'jingu', 'jinja'),
    'cistern': ('cistern',), 'market': ('bazaar', 'market'),
    'gate': ('gate', 'gates'),
    'lodging': ('hotel', 'hostel', 'ryokan', 'resort'),
    'dining': ('restaurant', 'cafe', 'café', 'bar', 'pub'),
    'sculpture': ('statue', 'sculpture', 'column', 'obelisk'),
    'street': ('street', 'avenue', 'boulevard', 'road', 'promenade'),
    'region': ('neighbourhood', 'neighborhood', 'district', 'ward', 'area'),
}

_SECTIONS = re.compile(r'(?<![\w/#])(?:Get in|Get around|Stay safe|Stay healthy|Go next|Understand|See|Do|Buy|Eat|Drink|Sleep|Connect|Cope|Learn|Work|Respect)(?=\s+(?:[' + re.escape(_CAPITALS) + r']|\*))')
_SECTION_NAMES = {'Get in', 'Get around', 'Stay safe', 'Stay healthy', 'Go next', 'Understand', 'See', 'Do', 'Buy', 'Eat', 'Drink', 'Sleep', 'Connect', 'Cope', 'Learn', 'Work', 'Respect'}
_ROUTE_SECTIONS = {'intro', 'Understand', 'See', 'Do', 'Buy'}
_FACT_SECTIONS = _ROUTE_SECTIONS | {'Get in', 'Get around'}


@dataclass(frozen=True)
class WikiRecord:
    text: str
    section: str
    heading: str = ''
    heading_context: str = ''


def _candidate_names(text):
    for match in _NAME.finditer(text):
        # A suffix is not an alias: Museum of Rome and National Museum of Rome
        # are different institutions. Only document markers are stripped.
        yield _clean_name(match.group())


def _section_events(text):
    for match in _SECTIONS.finditer(text):
        before = text[max(0, match.start() - 12):match.start()]
        after = text[match.end():match.end() + 70].lstrip()
        if re.search(r'(?:see|See)\s+$|["“(]\s*$', before):
            continue
        if match.group() == 'See' and ('#' in after.split(' ', 1)[0] or
                any(after.startswith(section + ' ') for section in _SECTION_NAMES)):
            continue
        yield match


def _detached_heading(text):
    """Recover a trailing listing title, rejecting coordinated POI captions.

    Ownership is provisional here. The following record must agree in type;
    its own explicit heading or named subject always takes precedence.
    """
    tail = _sentences(text[-450:])[-1] if text.strip() else ''
    matches = list(_NAME.finditer(tail))
    if not matches:
        return '', ''
    match = matches[-1]
    name = _clean_name(match.group())
    suffix, prefix = tail[match.end():].strip(), tail[:match.start()].strip()
    if suffix and not suffix.startswith(','):
        return '', ''
    if ' and ' in name and all(_types(part) for part in name.split(' and ')):
        return '', ''
    if re.search(r'\b(?:near|of|at|by|from|to|behind|opposite|beside)(?:\s+the)?$', prefix, re.I):
        return '', ''
    if re.search(r"\b(?:is|was|are|were|has|have|\w+['’](?:s|re))\b", prefix, re.I):
        return '', ''
    if any(_VERB.match(prefix[m.end():]) for m in _NAME.finditer(prefix)):
        return '', ''
    # A bare title, an explicit alias, or a typed photo caption can supply a
    # heading. An arbitrary final capitalized word in prose cannot.
    if prefix and not re.fullmatch(_WORD + r'\s+or', prefix) and not (_types(name) and suffix.startswith(',')):
        return '', ''
    return name, tail


def name_key(name: str) -> str:
    text = unicodedata.normalize('NFKD', name.casefold())
    return ' '.join(re.findall(r'\w+', ''.join(c for c in text if not unicodedata.combining(c))))


def _clean_name(name: str) -> str:
    name = re.sub(r'^(?:The|A|An)\s+', '', name.strip().rstrip('.'))
    name = re.sub(r'^(?:Gothic|Baroque|Medieval|Romanesque|Neoclassical)\s+', '', name)
    words = name.split()
    if len(words) > 2 and words[0].lower().endswith('s') and _types(words[0]).intersection(_types(' '.join(words[1:]))):
        name = ' '.join(words[1:])
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
    return {kind for kind, labels in _TYPES.items() if any(
        word in words or word + 's' in words or word + 'es' in words for word in labels)}


def _primary_type(text: str) -> set:
    """Classify the subject noun phrase, never a later verb or neighbour."""
    if re.match(r'^(?:Built|Completed|Constructed|Designed|Founded)\b', text):
        text = text.split(',', 1)[-1]
    text = re.sub(r'^(?:It|This|These|They)\s+(?:is|was|are|were)\s+', '', text)
    text = re.sub(r'^One of (?:the )?', '', text)
    phrase = re.split(r'\b(?:is|was|are|were|has|have|houses|contains|features|stands|that|which|of|on|in|near|with)\b', text, maxsplit=1, flags=re.I)[0]
    for word in name_key(phrase).split():
        for kind, labels in _TYPES.items():
            if word == 'dome':
                continue
            if any(word in (label, label + 's', label + 'es') for label in labels):
                return {kind}
    return set()


def _informative(text: str) -> bool:
    if not 35 <= len(text) <= 650 or not _FACT.search(text) or _LOGISTICS.search(text):
        return False
    words = {m.group().lower() for m in _FACT.finditer(text)}
    weak = {'popular', 'famous', 'symbol', 'oldest', 'largest', 'tallest', 'grandest'}
    return bool(words - weak or _types(text) or re.search(r'\d', text))


@lru_cache(maxsize=128)
def _records(text: str) -> tuple:
    """Keep section and listing ownership while walking the article once."""
    events = [(m.start(), m.end(), m.group()) for m in _section_events(text)]
    events += [(m.start(), m.end(), '*') for m in re.finditer(r'\*+', text)]
    events.sort()
    records, section, start, heading, context = [], 'intro', 0, '', ''
    for left, right, marker in events + [(len(text), len(text), '')]:
        body = text[start:left].strip()
        if body:
            records.append(WikiRecord(body, section, heading, context))
        if marker == '*':
            heading, context = _detached_heading(body)
        else:
            heading, context = '', ''
            if marker:
                section = marker
        start = right
    return tuple(records)


def _description(text):
    """First complete descriptive sentence after names, addresses and hours."""
    result = []
    sentences = _sentences(text)
    for index, sentence in enumerate(sentences):
        sentence = re.split(r'\.(?:jpe?g|png|webp)\b\s*', sentence, flags=re.I)[-1]
        for match in re.finditer(r'(?<!\w)[' + re.escape(_CAPITALS) + r'][\w’\'-]*\b', sentence):
            quote = sentence[match.start():].strip()
            if len(match.group()) > 1 and match.group().isupper():
                continue
            if re.search(r'\d{1,2}:\d{2}|[€¥£$]|\b(?:Daily|Adult|Entry)\b', quote):
                continue
            if ')' in quote.split('(', 1)[0]:
                continue
            grammatical = bool(re.match(r'^(?:A|An|The|This|These|It|Its|One|Built|Completed|Constructed|Designed|Founded|Opened|Originally|Surrounded|With|Has|Houses|Contains|Features|Displays|Exhibits|Dedicated)\b', quote))
            possessive = bool(re.match(_WORD + r"['’]s\b", quote))
            noun_predicate = bool(_primary_type(quote) and re.search(
                r'\b(?:is|was|were|are|built|housed|designed|houses|contains|features|decorated)\b', quote[:180], re.I))
            if not (grammatical or possessive or noun_predicate):
                continue
            # Once the actual prose starts, an unsupported first sentence is
            # not permission to skip ahead to a neighbour or another subject.
            if _LOGISTICS.search(quote):
                continue
            if not _informative(quote):
                return ''
            if index + 1 < len(sentences):
                following = sentences[index + 1]
                if (re.match(r'^(?:It|This|These|They)\b', following) and _informative(following)
                        and len(quote) + len(following) < 650):
                    quote += ' ' + following
            return quote
        if result:
            break
    return result[0] if result else ''


def _owned_quote(name, text):
    """A listing title cannot own a paragraph about a different primary type."""
    quote = _description(text)
    if not quote:
        return ''
    expected, actual = _types(name), _primary_type(quote)
    if expected and actual and not expected.intersection(actual):
        return ''
    first = next(_NAME.finditer(quote), None)
    pronouns = {'the', 'a', 'an', 'it', 'this', 'these', 'they', 'there', 'one', 'those'}
    if first and _VERB.match(quote[first.end():]) and name_key(_clean_name(first.group())) not in pronouns:
        subject = name_key(_clean_name(first.group()))
        target = name_key(name)
        if subject != target:
            return ''
    return quote


def _listing_names(record):
    """Explicit bullet headings and bilingual heading metadata."""
    names = []
    for match in _NAME.finditer(record.text):
        name = _clean_name(match.group())
        if not _types(name):
            continue
        suffix = record.text[match.end():].lstrip()
        native = bool(re.match(r'\(?[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', suffix))
        photo_match = re.match(r'[^.!?]{0,90}\.(?:jpe?g|png|webp)\b', suffix, re.I)
        photo = bool(photo_match)
        prefix = record.text[:match.start()].rstrip()
        if (native or photo) and re.search(r'\b(?:from|of|near|at|behind|opposite)(?:\s+the)?$', prefix, re.I):
            continue
        if photo and re.match(r'^[a-z]', suffix):
            continue  # prose mentioning the name before an unrelated image filename
        if _VERB.match(suffix) and not native:
            continue  # a named subject is already handled by the direct clause
        if match.start() == 0 or native or photo:
            body = suffix[photo_match.end():] if photo else suffix
            names.append((name, body, native or photo))
    return names


def _subject_fact(sentence, name):
    """Extract an independent named subject, including bilingual definitions."""
    target = name_key(name)
    for match in _NAME.finditer(sentence):
        subject = name_key(_clean_name(match.group()))
        if subject != target:
            continue
        prefix, tail = sentence[:match.start()].strip(), sentence[match.end():]
        clause = bool(re.search(r',\s*(?:the|a|an)?\s*$', prefix, re.I))
        native = bool(re.match(r'\s*\([\u3040-\u30ff\u3400-\u9fff]', tail))
        modifier = bool(re.fullmatch(r'(?:The|A|An|early|late|great|famous|Gothic|Baroque|\s)+', prefix))
        if prefix and not (clause or native or modifier):
            continue
        verb = _VERB.match(tail)
        if not verb:
            continue
        if re.search(r',\s*(?:and|or)\b', tail[:verb.end()], re.I):
            continue  # a coordinated group does not give one member the others' facts
        if re.match(r'\s*(?:is said to have|was born|was voted|is buried|died|lived|founded)\b', tail, re.I):
            continue
        predicate = re.split(r'\b(?:of|in|near|from|with)\b', tail, maxsplit=1, flags=re.I)[0][:100]
        if re.search(r'\b(?:area|district|city|country|region|neighbou?rhood)\b|\bis not an?\b', predicate, re.I):
            continue
        if re.search(r'\b(?:district|ward|neighbou?rhood)\b', tail[:120], re.I):
            continue
        at = match.group().casefold().rfind(name.casefold()) if match.group().casefold().count(name.casefold()) > 1 else 0
        fact = sentence[match.start() + max(0, at):].strip()
        for other in _NAME.finditer(fact, match.end() - match.start() - max(0, at)):
            if _VERB.match(fact[other.end():]) and name_key(_clean_name(other.group())) != subject:
                boundary = max(fact.rfind(',', 0, other.start()), fact.rfind(';', 0, other.start()))
                if boundary >= 0:
                    fact = fact[:boundary].rstrip() + '.'
                    break
        if _informative(fact) and _FACT.search(re.sub(re.escape(name), '', fact, flags=re.I)):
            return fact
    return ''


@lru_cache(maxsize=128)
def _fact_index(text):
    """Index attributable facts once per article, rather than once per name."""
    found = {}
    for record in _records(text):
        if record.section not in _FACT_SECTIONS:
            continue
        names = set(_candidate_names(record.text))
        if record.heading:
            names.add(record.heading)
        listings = _listing_names(record) if record.section in _ROUTE_SECTIONS else []
        names.update(name for name, _, _ in listings)
        for name in names:
            if not 4 <= len(name) <= 90:
                continue
            if record.section == 'Buy' and 'market' not in _types(name):
                continue
            options = []
            sentences = _sentences(record.text)
            for sentence_index, sentence in enumerate(sentences):
                sentence = re.split(r'\.(?:jpe?g|png|webp)\b\s*', sentence, flags=re.I)[-1]
                fact = _subject_fact(sentence, name)
                if fact:
                    if sentence_index + 1 < len(sentences):
                        following = sentences[sentence_index + 1]
                        if re.match(r'^(?:Although|However|Today|Now)\b', following) and len(fact) + len(following) < 650:
                            fact += ' ' + following
                    options.append((100, fact))
            for title, body, native in listings:
                if name_key(title) == name_key(name):
                    fact = _owned_quote(name, body)
                    if fact:
                        options.append((115 if native else 110, fact))
            if record.section in _ROUTE_SECTIONS and record.heading and name_key(name) == name_key(record.heading):
                fact = _owned_quote(name, record.text)
                if fact:
                    options.append((90, fact))
            if options:
                found.setdefault(name_key(name), []).extend(options)
    return {key: tuple(sorted(set(values), key=lambda x: (-x[0], len(x[1]))))
            for key, values in found.items()}


def object_facts(text: str, name: str) -> list[tuple[int, str]]:
    """Look up object-owned records in the article's section-aware fact index."""
    return list(_fact_index(text).get(name_key(name), ()))

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


def _mention_counts(text, keys):
    """Count candidate phrases with a small trie, without rescanning per name."""
    trie = {}
    for key in keys:
        node = trie
        for word in key.split():
            node = node.setdefault(word, {})
        node[''] = key
    words = name_key(text).split()
    counts = Counter()
    for i, word in enumerate(words):
        node = trie.get(word)
        j = i + 1
        while node is not None:
            if '' in node:
                counts[node['']] += 1
            node = node.get(words[j]) if j < len(words) else None
            j += 1
    return counts


def _prominence_key(name):
    """A type designation may help find overview mentions, but never rename a POI.

    This key is used only for ranking: it cannot attach another name's facts.
    """
    words = name.split()
    if len(words) > 2 and _types(words[0]) and words[1].lower() in {'de', 'di', 'of'}:
        return name_key(' '.join(words[2:]))
    return name_key(name)


def evidence_options(passages, name):
    """Shared ownership/availability checks for selection and source rendering."""
    kinds = _types(name)
    options = []
    for i, passage in enumerate(passages, 1):
        for confidence, fact in object_facts(passage.text, name):
            if not kinds:
                if not (_types(fact) - {'region', 'dining', 'lodging'}) and 'measured' not in fact:
                    continue
                if confidence == 100:
                    tail = fact[fact.casefold().find(name.casefold()) + len(name):]
                    verb = _VERB.match(tail)
                    complement = tail[verb.end():] if verb else ''
                    own_feature = re.search(r'\b(?:its|measured|home of)\b', tail)
                    if not own_feature and not (_primary_type(complement) - {'region', 'dining', 'lodging'}):
                        continue
                elif confidence == 90:
                    physical_caption = any(name_key(r.heading) == name_key(name) and _types(r.heading_context)
                                           for r in _records(passage.text))
                    if not physical_caption:
                        continue
            if re.search(r'\bclosed (?:for|until)|permanently closed\b', fact, re.I) and not re.search(r'\bre-?opened\b', fact, re.I):
                continue
            options.append((confidence, min(len(fact), 400), i, passage, fact))
    return options


def city_catalogue(passages: list[Passage], destination: str, question: str = '') -> list[Attraction]:
    """Rank named, owned sightseeing records; transport/hospitality is not searched."""
    spellings, source_text, overview, capitals, tokens = {}, [], [], Counter(), Counter()
    for passage in passages:
        for word in re.findall(r"[^\W\d_][\w’'-]*", passage.text):
            key = name_key(word)
            tokens[key] += 1
            if word[0].isupper():
                capitals[key] += 1
        for record in _records(passage.text):
            if record.section not in _FACT_SECTIONS:
                continue
            source_text.append(record.text)
            if passage.title == destination and record.section == 'See':
                overview.append(record.text)
            for name in list(_candidate_names(record.text)) + ([record.heading] if record.heading else []):
                key = name_key(name)
                if 4 <= len(name) <= 90 and len(name.split()) <= 9:
                    spellings.setdefault(key, name)
    counts = _mention_counts(' '.join(source_text), spellings)
    prominence_keys = {_prominence_key(name) for name in spellings.values()}
    main_counts = _mention_counts(' '.join(overview), set(spellings) | prominence_keys)
    district_keys = {name_key(p.title.rsplit('/', 1)[-1]) for p in passages}
    generic = {form for labels in _TYPES.values() for word in labels for form in (word, word + 's', word + 'es')}
    generic_words = generic | {'of', 'the', 'and', 'national', 'city', 'regional', 'public', 'old', 'new', 'art'}
    entries = []
    for key, name in spellings.items():
        kinds = _types(name)
        if key in district_keys or key in generic or (len(kinds) < 2 and set(key.split()).issubset(generic_words)) or kinds.intersection({'lodging', 'dining'}):
            continue
        if len(name.split()) == 1 and not kinds:
            # Sentence-initial adverbs are not names. Independently repeated,
            # consistently capitalized proper names (e.g. Colosseum) survive.
            if tokens[key] < 2 or capitals[key] / max(tokens[key], 1) < .8:
                continue
        if ' and ' in name and all(_types(part) for part in name.split(' and ')):
            continue
        options = evidence_options(passages, name)
        if not options:
            continue
        confidence, _, i, passage, fact = max(options, key=lambda item: (item[0], item[1]))
        # Prominence in the city's sights overview outweighs repeated mentions
        # in a district guide. Construction dates do not make a church famous.
        overview_mentions = max(main_counts[key], main_counts[_prominence_key(name)])
        score = 2 * min(counts[key], 12) + 10 * min(overview_mentions, 5)
        score += confidence / 40
        if kinds.intersection({'museum', 'palace', 'church', 'mosque', 'shrine', 'cistern'}):
            score += 8
        elif kinds.intersection({'bridge', 'tower', 'square'}):
            score += 4
        if (re.search(r'\b(?:shopping|office building|eating)\b', fact, re.I)
                and not re.search(r'\b(?:museum|temple|shrine|observatory|viewpoint|historic|ancient)\b', fact, re.I)
                and not re.search(r'современн|архитектур|modern|architecture', question, re.I)):
            score -= 10
        rank_context = fact + ' ' + ' '.join(
            record.heading_context for record in _records(passage.text)
            if name_key(record.heading) == key)
        if re.search(r'\btop \d+ (?:most )?visited\b', rank_context, re.I):
            score += 50
        elif re.search(r'\b(?:iconic|symbol|defining image|most visited|major attraction)\b', rank_context, re.I):
            score += 20
        elif re.search(r'\b(?:oldest|largest|tallest|grandest)\b', rank_context, re.I):
            score += 8
        if re.search(r'истор|ancient|historic', question, re.I) and re.search(
                r'\b(?:ancient|historic|century|built|completed|medieval|gothic|baroque|roman)\b', fact, re.I):
            score += 6
        if re.search(r'музе|искусств|museum|art', question, re.I) and 'museum' in kinds:
            score += 20
        if re.search(r'парк|сад|природ|park|garden', question, re.I) and 'park' in kinds:
            score += 20
        entries.append((score, Attraction(name, i, passage.title, passage.url, fact)))
    entries.sort(key=lambda item: (-item[0], name_key(item[1].name)))
    unique = []
    for _, attraction in entries:
        key = name_key(attraction.name)
        duplicate = next((old for old in unique if old.title == attraction.title and (
            old.evidence == attraction.evidence or
            ((name_key(old.name).endswith(' ' + key) or key.endswith(' ' + name_key(old.name))) and
             old.evidence[old.evidence.casefold().find(old.name.casefold()) + len(old.name):] ==
             attraction.evidence[attraction.evidence.casefold().find(attraction.name.casefold()) + len(attraction.name):]))
            ), None)
        if not duplicate:
            unique.append(attraction)
        elif key.endswith(' ' + name_key(duplicate.name)):
            unique[unique.index(duplicate)] = attraction
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
