from __future__ import annotations
import json
import hashlib
from pathlib import Path
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass

from rag_pipeline import Passage, ollama_configuration
from geo_routes import geographic_groups


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
    """Получает JSON из локальной Ollama с повтором для оборванных ответов.

    Важное ограничение: формат JSON гарантирует синтаксис при успешной
    генерации, но не гарантирует достоверность содержимого.
    """
    url, selected_model = ollama_configuration(model)
    endpoint = url + '/api/chat'
    last_reason = 'пустой ответ'
    # Вторая попытка с большим лимитом, если первая завершилась невалидным JSON.
    for attempt in range(2):
        budget = tokens if attempt == 0 else max(tokens * 2, 1600)
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
                'num_predict': budget,
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
        last_reason = f'done_reason={done_reason}, символов в ответе={len(content)}, лимит={budget}'
    raise RuntimeError(
        'Ollama дважды не смогла сформировать корректный JSON '
        f'({last_reason}). Проверь модель: ollama list; '
        'можно попробовать уменьшить --top-k до 5.'
    )


def _evidence(text: str, name: str, width: int = 230) -> str:
    pos = text.casefold().find(name.casefold())
    if pos < 0:
        return ''
    return ' '.join(text[max(0, pos - 55): pos + len(name) + width].split())


def validate_attractions(raw: list, passages: list[Passage], max_items: int = 18) -> list[Attraction]:
    """Главная защита: объект должен ДОСЛОВНО встречаться в указанном чанке."""
    good, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        source = item.get('source')
        if not isinstance(name, str) or not isinstance(source, int) or isinstance(source, bool):
            continue
        name = ' '.join(name.strip().split())
        if not (4 <= len(name) <= 75) or len(name.split()) > 9:
            continue
        if not (1 <= source <= len(passages)):
            continue
        passage = passages[source - 1]
        evidence = _evidence(passage.text, name)
        if not evidence or name.casefold() in seen:
            continue
        # Заголовки/фразы, не являющиеся конкретными объектами, отклоняем.
        if name.casefold() in {'rome', 'italy', 'historic centre', 'city center', 'city centre',
                               'see', 'do', 'museum', 'church', 'palace', 'fountain', 'travel'}:
            continue
        # Эвристика: явные дальние поездки и города в секциях Get out исключаются на этапе поиска.
        good.append(Attraction(name, source, passage.title, passage.url, evidence))
        seen.add(name.casefold())
        if len(good) >= max_items:
            break
    return good


def extract_attractions(passages: list[Passage], destination: str, model: str | None) -> list[Attraction]:
    pieces = []
    for idx, passage in enumerate(passages, 1):
        # Ограничение длины важно для 8 ГБ ОЗУ и небольшой локальной модели.
        pieces.append(f'[Источник {idx}: {passage.title}]\n{passage.text[:700]}')
    system = (
        'Extract names of specific tourist attractions (buildings, monuments, museums, parks, squares) '
        'IN the requested city from provided excerpts. Return ONLY JSON object '
        '{"attractions":[{"name":"exact English substring from excerpt","source":1}]}. '
        'Do not invent names, translate, or include places outside the requested city. '
        'Provide at most 10 objects; keep response very short; if insufficient evidence return fewer.'
    )
    result = _ollama_json(system, f'CITY: {destination}\n\n' + '\n\n'.join(pieces), model, 1400)
    return validate_attractions(result.get('attractions', []), passages)


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


def descriptions(attractions: list[Attraction], model: str | None) -> dict[str, str]:
    if not attractions:
        return {}
    data = [
        {'id': str(i), 'name': a.name, 'evidence': a.evidence[:280]}
        for i, a in enumerate(attractions, 1)
    ]
    system = (
        'You are a careful translator of travel snippets. Return ONLY JSON object '
        '{"descriptions":{"1":"короткое описание по-русски", "2":"..."}}. '
        'For each ID write one short RUSSIAN sentence strictly supported by its evidence. '
        'Do not introduce new place names, dates, addresses, prices or claims. '
        'If the snippet does not support a description, return an empty string for that ID.'
    )
    result = _ollama_json(system, json.dumps(data, ensure_ascii=False), model, 650)
    candidates = result.get('descriptions', {})
    if not isinstance(candidates, dict):
        return {}
    clean = {}
    for key, value in candidates.items():
        if isinstance(value, str) and len(value) <= 240 and '\n' not in value:
            clean[str(key)] = value.strip()
    return clean


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


def is_route_question(question: str) -> bool:
    q = question.casefold()
    return bool(re.search(r'маршрут|на\s+\d+\s+д(?:ень|ня|ней)|за\s+\d+\s+д(?:ень|ня|ней)|itinerary|route for', q))


# Один вызов Ollama — только названия. Описания создаются локально из фактов.

# ROUTE_SELECTION_V2. Универсальное дополнение к Qwen: кандидаты по названиям
# в самом тексте Wikivoyage. Не использует списков городов или внешних API.
_LANDMARK_END = re.compile(
    r"(?<![\w])((?:[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’.-]+\s+){1,4}"
    r"(?:Tower|Cathedral|Museum|Palace|Basilica|Chapel|Bridge|Garden|Gardens|"
    r"Monument|Square|Church|Arch|Colosseum|Pantheon|Louvre))\b"
)
_LANDMARK_SPECIAL = re.compile(r"(?<!\w)(Arc de [A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’-]+|Notre[- ]Dame(?: de [A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’-]+)?)\b")
_GENERIC_NAME = re.compile(r"^(?:The|A|An|This|One|Paris|Rome|London|City|National)$", re.I)

def _landmark_candidates(passages):
    result = []
    seen = set()
    for i, p in enumerate(passages, 1):
        for m in list(_LANDMARK_END.finditer(p.text[:1800])) + list(_LANDMARK_SPECIAL.finditer(p.text[:1800])):
            name = ' '.join(m.group(1).split())
            name = re.sub(r'^(?:The|A|An)\s+', '', name)
            if len(name) > 70 or _GENERIC_NAME.match(name.split()[0]):
                continue
            key = name.casefold()
            if key not in seen:
                result.append({'name': name, 'source': i})
                seen.add(key)
    return result

def _route_focused_excerpt(text, limit=1050):
    text = ' '.join(text.split())
    if len(text) <= limit:
        return text
    # Оставляем начало и область с явными названиями достопримечательностей.
    # Это уменьшает случаи, когда нужное название находится в конце чанка.
    head = text[:min(360, limit // 2)]
    candidates = list(_LANDMARK_END.finditer(text)) + list(_LANDMARK_SPECIAL.finditer(text))
    candidates.sort(key=lambda m: m.start())
    if not candidates:
        return text[:limit]
    # Выбираем первое содержательное упоминание вне начала текста.
    match = next((m for m in candidates if m.start() >= 300), candidates[0])
    window = text[max(0, match.start() - 85):match.start() + (limit - len(head) - 50)]
    return (head + ' ... ' + window).strip()[:limit]

# ROUTE_CITY_CATALOG_V3: названия и значимость по городским статьям Wikivoyage.
# Загружаются только чанки выбранного города, без повторной индексации и сети.
_CITY_NAMED_SIGHT = re.compile(
    r"(?<![\w])((?:(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÿ’'-]+)\s+){1,4}"
    r"(?:Tower|Museum|Cathedral|Basilica|Chapel|Palace|Bridge|Gardens?|"
    r"Square|Church|Monument|Memorial|Gallery|Opera|Fortress))\b"
)
_CITY_SPECIAL_SIGHT = re.compile(
    r"\b(Arc de Triomphe(?: de l[’']Étoile)?|Notre[- ]Dame(?: de Paris)?)\b",
    flags=re.I,
)
_CITY_LOWER_TYPE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][\wÀ-ÿ’'-]{3,})\s+(?:museum|palace|cathedral)\b"
)
_CITY_BAD_NAMES = frozenset({
    'new bridge', 'old bridge', 'the bridge', 'the tower', 'the museum',
    'national museum', 'city museum', 'the church', 'the cathedral',
    'the palace', 'the square', 'the garden', 'the gallery',
})


def _city_names(text: str, city: str = '') -> list[str]:
    """Названия из текста, без названий, придуманных LLM."""
    result = []
    matches = ([(m, False) for m in _CITY_NAMED_SIGHT.finditer(text)]
               + [(m, False) for m in _CITY_SPECIAL_SIGHT.finditer(text)]
               + [(m, True) for m in _CITY_LOWER_TYPE.finditer(text)])
    for m, lower_category in matches:
        name = re.sub(r'^(?:the|a|an)\s+', '', ' '.join(m.group(1).split()), flags=re.I)
        if lower_category:
            # `Louvre museum` -> Louvre; `British Museum` оставляем целиком.
            name = name.strip()
        if (4 <= len(name) <= 70 and name.casefold() not in _CITY_BAD_NAMES
                and name.casefold() != city.casefold()
                and name.split()[0].casefold() not in {'destination', 'source'}):
            result.append(name)
    return list(dict.fromkeys(result))


def augment_route_passages(passages: list[Passage], destination: str,
                           collection, city_titles, max_extra: int = 7):
    """Расширяет маршрут подтверждёнными кандидатами из городских статей.

    Возвращает (passages, prominence). Если Chroma не поддерживает нужный
    метафильтр, сохраняет исходную выдачу. Данные в Chroma не изменяются.
    """
    titles = [t for t in city_titles if isinstance(t, str)
              and (t.casefold() == destination.casefold()
                   or t.casefold().startswith(destination.casefold() + '/'))]
    if not titles or collection is None:
        return passages, {}
    try:
        records = collection.get(where={'title': {'$in': titles[:120]}},
                                 include=['documents', 'metadatas'])
    except Exception:
        return passages, {}
    docs = records.get('documents') or []
    metas = records.get('metadatas') or []
    ids = records.get('ids') or []
    if not docs:
        return passages, {}
    from collections import Counter
    counts = Counter()
    candidates = {}
    for i, doc in enumerate(docs):
        if not isinstance(doc, str):
            continue
        meta = metas[i] or {} if i < len(metas) else {}
        title = str(meta.get('title') or '')
        if title.casefold() != destination.casefold() and not title.casefold().startswith(destination.casefold() + '/'):
            continue
        names = _city_names(doc, destination)
        # Повторения внутри одного чанка не раздувают значимость.
        for name in names:
            key = name.casefold()
            counts[key] += 1
            pos = doc.casefold().find(key)
            window = doc[max(0, pos - 40):pos + len(name) + 280].casefold()
            # Выбираем чанки с описаниями, а не упоминаниями в адресах отелей.
            informative = len(window) > 100 and bool(re.search(
                r'\b(?:built|famous|known|constructed|contains|houses|is|was|iconic|museum|historic|century|architecture)\b', window))
            quality = (2 if title != destination else 0) + (3 if informative else 0)
            if any(x in window for x in ('hotel', 'pub crawl', 'restaurant', 'hostel')):
                quality -= 3
            old = candidates.get(key)
            if old is None or quality > old[0]:
                candidates[key] = (quality, name, doc, meta, ids[i] if i < len(ids) else '')
    if not counts:
        return passages, {}
    # Корректируем частоту по всем городским чанкам (включая упоминания без
    # шаблона имени, но без отдельного загрузочного цикла по каждому объекту).
    city_docs = [str(d).casefold() for d in docs if isinstance(d, str)]
    existing = {p.chunk_id for p in passages}
    prominence = {}
    ranked = []
    for key, (quality, name, doc, meta, chunk_id) in candidates.items():
        mentions = sum(key in text for text in city_docs)
        prominence[key] = mentions
        rank = mentions + min(counts[key], 6) + max(0, quality)
        if len(name.split()) < 2 and mentions < 3:
            continue
        ranked.append((rank, name, doc, meta, chunk_id))
    ranked.sort(key=lambda row: (-row[0], row[1].casefold()))
    extra = []
    for _, name, doc, meta, chunk_id in ranked:
        if len(extra) >= max_extra:
            break
        if chunk_id in existing:
            continue
        title = str(meta.get('title') or destination)
        url = str(meta.get('url') or 'https://en.wikivoyage.org/wiki/' + title.replace(' ', '_'))
        extra.append(Passage(text=doc, title=title, url=url,
                             distance=0.0, chunk_id=str(chunk_id)))
        existing.add(chunk_id)
    return list(passages) + extra, prominence


def extract_route_candidates(passages: list[Passage], destination: str,
                             days: int, model: str | None,
                             cache_dir: Path = Path("data/cache/route_extraction"),
                              prominence: dict | None = None) -> tuple[list[Attraction], dict[str, str]]:
    """Extract source-grounded names in ONE Ollama call; build safe summaries locally.

    Invariant: user-visible stop names must exactly occur in the cited passage.
    Descriptions are conservative Python templates, not claims generated by LLM.
    """
    # ROUTE_SELECTION_V2: не ограничиваем выбор случайным началом чанка.
    excerpts = []
    # Ограничение контекста Ollama: новые источники доступны Python-отбору,
    # но в промпт идут только первые восемь и последние четыре.
    shown = list(range(min(8, len(passages))))
    shown += list(range(max(8, len(passages) - 4), len(passages)))
    for i in dict.fromkeys(shown):
        passage = passages[i]
        excerpt = _route_focused_excerpt(passage.text, 850)
        excerpts.append(f"[Источник {i + 1}, {passage.title}]\n{excerpt}")
    source_data = "\n\n".join(excerpts)
    selected_model = ollama_configuration(model)[1]
    signature = hashlib.sha256(
        json.dumps(["route_selection_v2", destination, days, selected_model, source_data],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_file = cache_dir / (signature + ".json")
    result = None
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("attractions"), list):
                result = cached
        except (OSError, ValueError):
            pass
    if result is None:
        system = (
            'Extract specific sightseeing attractions IN the requested city. '
            'Return ONLY compact JSON: '
            '{"attractions":[{"name":"EXACT English substring","source":1}]}. '
            'Prefer important landmarks, monuments, museums, squares, historic buildings. '
            'Avoid remote day trips, suburbs and incidental minor mentions. '
            'Every name MUST be copied exactly from its source text. '
            'Do not translate names, invent places, or write descriptions. '
            'At most 9 attractions.'
        )
        result = _ollama_json(system, f"CITY: {destination}\nDAYS: {days}\n\n{source_data}",
                              model, tokens=650)
        if isinstance(result.get("attractions"), list):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass  # Caching is optional; generation must still work.
    raw = result.get("attractions", [])
    if not isinstance(raw, list):
        raw = []
    # Добавляем кандидатов, которые явно фигурируют в самом тексте путеводителя.
    # Они должны дословно присутствовать в источнике, список не привязан к городам.
    raw.extend(_landmark_candidates(passages))
    # Городской каталог извлекается из всех найденных статей, включая
    # достопримечательности, которые Qwen 1.7B не перечислила.
    for i, p in enumerate(passages, 1):
        raw.extend({"name": name, "source": i} for name in _city_names(p.text, destination))
    verified = validate_attractions(raw, passages, max_items=120)
    # Score by source relevance (rank), central-city sections, number of explicit mentions.
    # This is a heuristic, NOT a measured popularity or geographic distance.
    def score(a: Attraction) -> float:
        section = a.title.casefold().split("/", 1)[-1]
        key = a.name.casefold()
        mentions = sum(key in passage.text.casefold() for passage in passages)
        # Частые упоминания в разных статьях — приблизительный индикатор
        # заметности объекта, а не географическая близость или официальный рейтинг.
        city_guide = sum(key in p.text.casefold() and '/' not in p.title
                         for p in passages)
        notable = bool(re.search(r'\b(?:iconic|famous|major|main|landmark|must.see|highlight|symbol)\b',
                                 a.evidence, re.I))
        penalty = 0.0
        if any(word in section for word in ("south", "north", "outskirts", "get out")):
            penalty += 1.4
        if any(word in a.evidence.casefold() for word in ("day trip", "outside the city", "airport")):
            penalty += 2.5
        return (4.0 / (1 + a.passage_index * 0.32)
                + min(mentions, 5) * 1.15 + min(city_guide, 3) * 1.8
                + (1.0 if notable else 0.0) - penalty)
    # Важность в полном наборе городских текстов важнее порядка Qwen.
    prominence = prominence or {}
    verified.sort(key=lambda a: score(a) + 3.5 * min(prominence.get(a.name.casefold(), 0), 20),
                  reverse=True)
    # Описания, придуманные LLM, принципиально НЕ показываем пользователю.
    # Для Qwen3 1.7B даже валидный JSON нередко содержит фактические ошибки.
    # Создаём короткие объяснения по проверяемым лексическим признакам
    # исходного фрагмента, не обращаясь к модели второй раз.
    descriptions_by_name = {
        a.name.casefold(): grounded_description(a)
        for a in verified
    }
    return verified, descriptions_by_name


def grounded_description(attraction: Attraction) -> str:
    """Консервативное описание по английскому названию и соседнему тексту.

    Не переводим свободный текст моделью: русская фраза задаётся шаблоном,
    активируемым только проверяемыми словами в названии/цитате.
    При недостатке фактов возвращаем пустую строку.
    Это НЕ полная проверка фактов, а защита от галлюцинаций LLM.
    """
    name = attraction.name.casefold()
    evidence = attraction.evidence.casefold()
    # Название уже дословно найдено в источнике функцией validate_attractions.
    # Порядок правил: более узкие типы — раньше общих.
    if re.search(r"\b(?:aqueduct|acquedott)\w*", name) and re.search(r"\baqueducts?\b", evidence):
        return "Место, связанное с сохранившимися древними акведуками."
    if re.search(r"\b(?:fountain|fontana|fontane)\b", name):
        return "Фонтан или архитектурный комплекс с фонтанами."
    if re.search(r"\b(?:gallery|galleria)\b", name) and re.search(r"\b(?:art|paintings|collection|museum)\b", evidence):
        return "Художественная галерея с произведениями искусства."
    if re.search(r"\b(?:museum|museo|musei)\b", name) and re.search(r"\b(?:museum|collection|artefacts|artifacts|exhibits)\b", evidence):
        return "Музей с экспозицией, упомянутой в путеводителе."
    if re.search(r"\b(?:chapel|cappella)\b", name):
        if re.search(r"\b(?:fresco|frescoes|frescos)\b", evidence):
            return "Капелла, известная своими фресками."
        return "Капелла, упомянутая в путеводителе."
    if re.search(r"\b(?:church|basilica|abbazia)\b", name):
        return "Религиозный архитектурный объект."
    if re.search(r"\b(?:palazzo|palace)\b", name):
        return "Историческое дворцовое здание."
    if re.search(r"\b(?:baths|terme)\b", name) and re.search(r"\b(?:roman|ancient|baths|thermae)\b", evidence):
        return "Комплекс древнеримских терм."
    if re.search(r"\b(?:park|parco|gardens)\b", name):
        return "Парк или сад для прогулок."
    if "colosseum" in name and re.search(r"\b(?:amphitheat\w*|colosseum|roman|ancient)\b", evidence):
        return "Историческая достопримечательность Древнего Рима."
    if "pantheon" in name and re.search(r"\b(?:temple|ancient|roman|church)\b", evidence):
        return "Историческое сооружение Рима."
    if re.search(r"\b(?:forum|foro)\b", name) and re.search(r"\b(?:roman|ancient|ruins|forum)\b", evidence):
        return "Исторический объект с наследием Древнего Рима."
    return ""


def answer_route(question: str, passages: list[Passage], destination: str | None,
                 model: str | None = None, prominence: dict | None = None) -> str:
    if not passages:
        return 'Не удалось найти сведения для маршрута в базе знаний.'
    if not destination:
        return 'Укажи город, чтобы составить маршрут.'
    days = route_days(question)
    candidates, descriptions_by_name = extract_route_candidates(
        passages, destination, days, model, prominence=prominence)
    if not candidates:
        return ('В найденных фрагментах не удалось подтвердить названия достопримечательностей. '
                'Попробуй уточнить запрос или увеличить число фрагментов.')
    geo_result = geographic_groups(candidates, destination, days)
    if geo_result is not None:
        grouped, route_note = geo_result
    else:
        grouped = group_attractions(candidates, days)
        route_note = ('Географическая группировка недоступна: недостаточно координат '
                      'отдельных объектов. Остановки распределены приблизительно '
                      'по разделам путеводителя; расстояния и время в пути не рассчитаны.')
    planned = [attraction for day in grouped for attraction in day]
    summaries = {str(i): descriptions_by_name.get(a.name.casefold(), "")
                 for i, a in enumerate(planned, 1)}
    rendered = render_route(grouped, summaries, planned)
    old_note = ('Порядок остановок сгруппирован приблизительно по разделам путеводителя; '
                'расстояния и время в пути не рассчитаны.')
    return rendered.replace(old_note, route_note)
