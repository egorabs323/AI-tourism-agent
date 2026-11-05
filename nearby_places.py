"""Offline, evidence-only proximity answers from the existing Wikivoyage ChromaDB.

Compatible with Python 3.9 and web_app.py from the previous stages.
No OSM/Wikipedia/network requests and no second SentenceTransformer or LLM.
The underlying Wikivoyage data does NOT contain per-landmark coordinates,
so this module NEVER invents distances or equates a city's coordinates with a POI.
"""
import re
import sys

from context_query import LANDMARK_ALIASES

NEARBY_REQUEST = re.compile(r"\b(?:рядом|поблизости|неподал[её]ку|недалеко|возле|вблизи|ближайш\w*|nearby|near)\b", re.I)
# Deliberately conservative: literal relationship expressions, not a general
# 'city A and city B occur together' heuristic.
PROXIMITY = re.compile(
    r"\b(?:next to|adjacent to|beside|nearby|near the|near |close to|"
    r"opposite|across from|a short walk from|walking distance from|"
    r"just (?:north|south|east|west) of|around the corner from)\b", re.I
)
GENERIC_MENTION = (
    re.compile(r"(?:расскажи(?:те)?(?:\s+мне)?\s+(?:о|об|про)|что\s+знаешь\s+(?:о|об|про))\s+(.+?)(?:[.!?]|$)", re.I),
    re.compile(r"(?:tell me about|what is)\s+(.+?)(?:[.!?]|$)", re.I),
)
_PRONOUN = re.compile(r"^(?:он|она|его|её|нее|ней|ним|них|ними|этим|it|them|there)$", re.I)


def _explicit_place(text):
    for pattern, english in LANDMARK_ALIASES:
        if pattern.search(text):
            return english
    for pattern in GENERIC_MENTION:
        match = pattern.search(text)
        if match:
            subject = re.sub(r"\s+(?:в|во|из)\s+[А-ЯЁA-Z][^,.!?]*$", "", match.group(1)).strip(' «»\"\'.,?!')
            if 2 < len(subject) <= 90 and not _PRONOUN.fullmatch(subject):
                return subject
    return None


def _referent(question, history):
    # Trust explicit *user* mentions only; never parse the LLM's unverified reply.
    messages = [question] + [str(x.get('content') or '') for x in reversed(history[-12:]) if x.get('role') == 'user']
    for text in messages:
        found = _explicit_place(text)
        if found:
            return found
    return None


def _evidence(text, subject):
    """Literal excerpts that mention target and an explicit spatial relation.

    This is not full entity linking; extracts are deliberately presented as
    source excerpts, not as a verified ordered list of nearby attractions.
    """
    cleaned = re.sub(r"\s+", " ", text or '')
    # Split lightly on sentence boundaries and list separators (Wikivoyage is
    # often flattened); keep clauses under control.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\d*])|\s+\*\s+", cleaned)
    needle = subject.casefold()
    matches = []
    for i, sentence in enumerate(sentences):
        if not PROXIMITY.search(sentence):
            continue
        # Require subject and spatial phrase in the SAME sentence. A reference
        # elsewhere in the paragraph is not evidence of geographic proximity.
        if needle not in sentence.casefold():
            continue
        snippet = sentence.strip(' *')
        if 35 <= len(snippet) <= 400 and not snippet.lower().startswith('destination:'):
            matches.append(snippet)
    return matches


def nearby_reply(question, history, destination=None, max_results=4):
    """For proximity questions return (answer, sources); otherwise None.

    Imports get_assistant lazily to reuse web_app.py's single model instance.
    It is executed inside the existing serialization lock in generate_reply.
    """
    if not NEARBY_REQUEST.search(question):
        return None
    subject = _referent(question, history)
    if not subject:
        return ('Не удалось определить, о каком месте идёт речь. Укажи название достопримечательности.', [])

    # Local import avoids import cycle during FastAPI startup and avoids loading
    # the embedding model for ordinary (non-proximity) requests.
    # Reuse the SAME in-memory RagAssistant instance as the running web app.
    # When launched with `python web_app.py`, the module is `__main__`; importing
    # web_app again would load a second encoder and waste RAM.
    app_module = sys.modules.get('web_app') or sys.modules.get('__main__')
    get_assistant = getattr(app_module, 'get_assistant', None)
    if get_assistant is None:
        raise RuntimeError('Не удалось получить текущий RAG-ассистент веб-приложения')
    assistant = get_assistant()
    query = '{} nearby landmarks attractions next to adjacent walking distance'.format(subject)
    try:
        passages, _ = assistant.retrieve(query, destination=destination or None, top_k=18)
    except ValueError:
        # If the user's city field is not a Wikivoyage title, try unfiltered
        # retrieval rather than making claims about unknown geo metadata.
        passages, _ = assistant.retrieve(query, destination=None, top_k=18)

    found = []
    seen = set()
    for passage in passages:
        for snippet in _evidence(passage.text, subject):
            key = re.sub(r'\W+', '', snippet.casefold())[:160]
            if key in seen:
                continue
            seen.add(key)
            found.append((snippet, passage.title, passage.url))
            if len(found) >= max_results:
                break
        if len(found) >= max_results:
            break
    if not found:
        return (
            'В имеющихся фрагментах Wikivoyage не найдено надёжного прямого указания, '
            'какие достопримечательности расположены рядом с {}. '
            'В базе есть описания мест, но координаты туристической статьи нельзя '
            'использовать как координаты каждого объекта; поэтому расстояния не рассчитаны.'
            .format(subject), []
        )
    answer = ('В Wikivoyage найдены следующие фрагменты с упоминанием расположения рядом с {} '
              '(оригинальный текст источников):\n'.format(subject))
    sources = []
    used_urls = set()
    for index, (snippet, title, url) in enumerate(found, 1):
        answer += '\n{}. «{}»'.format(index, snippet)
        if url and url not in used_urls:
            sources.append({'number': len(sources)+1, 'title': title, 'url': url})
            used_urls.add(url)
    answer += ('\n\nЭто выдержки из туристических статей, а не проверенный список ближайших '
               'объектов. Точные расстояния и пешеходный маршрут не рассчитаны.')
    return answer, sources
