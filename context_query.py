"""Lightweight reference resolution for retrieval queries. Python 3.9+.

This only enriches search; it does not claim facts or use a second LLM.
"""
import re

# Explicit surface forms in tourist questions, with English Wikivoyage titles
# for multilingual retrieval. Extend as needed for other destinations.
LANDMARK_ALIASES = (
    (re.compile(r'\bколизе[йяюем]\b', re.I), 'Colosseum'),
    (re.compile(r'\bпантеон[а-я]*\b', re.I), 'Pantheon'),
    (re.compile(r'\bримск(?:ий|ого|ому|им)\s+форум[а-я]*\b', re.I), 'Roman Forum'),
    (re.compile(r'\bсикстинск(?:ая|ой|ую|ую)\s+капелл[а-я]*\b', re.I), 'Sistine Chapel'),
    (re.compile(r'\bэйфелев(?:а|ой|у)\s+башн[а-я]*\b', re.I), 'Eiffel Tower'),
    (re.compile(r'\bлувр[а-я]*\b', re.I), 'Louvre'),
    (re.compile(r'\bколизеум\b', re.I), 'Colosseum'),
    (re.compile(r'\bcolosseum\b', re.I), 'Colosseum'),
    (re.compile(r'\bpantheon\b', re.I), 'Pantheon'),
)
REFERENCE = re.compile(
    r'\b(?:его|ему|ним|нём|нем|он|они|них|ними|него|неё|нее|она|этого|этому|этом|'
    r'этих|этими|эти|них|рядом\s+с\s+ним|возле\s+него)\b', re.I
)


def contextual_search_query(question, history, destination=None):
    """Return a more explicit retrieval query for pronoun follow-ups.

    Selects the last explicitly mentioned known place in *user* turns.
    Does not rewrite the actual question shown to the user.
    """
    if not REFERENCE.search(question):
        return question
    # If the current question already names a place, it is self-contained.
    if any(pattern.search(question) for pattern, _ in LANDMARK_ALIASES):
        return question
    entity = None
    for message in reversed(history[-10:]):
        if message.get('role') != 'user':
            continue
        previous = str(message.get('content') or '')
        for pattern, label in LANDMARK_ALIASES:
            if pattern.search(previous):
                entity = label
                break
        if entity:
            break
    if not entity:
        return question
    base = '{} {} {}'.format(destination or '', entity, question).strip()
    # Make explicit the relational intent, not specific unsupported results.
    if re.search(r'\b(?:рядом|поблизости|неподал[её]ку|около)\b', question, re.I):
        base += ' nearby attractions landmarks walking distance'
    return base
