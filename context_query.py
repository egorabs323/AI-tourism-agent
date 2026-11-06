import re

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
    r'этих|этими|эти|них|там|туда|рядом\s+с\s+ним|возле\s+него)\b', re.I
)


def contextual_search_query(question, history, destination=None):
    if not REFERENCE.search(question):
        return question
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
    if re.search(r'\b(?:рядом|поблизости|неподал[её]ку|около)\b', question, re.I):
        base += ' nearby attractions landmarks walking distance'
    return base
