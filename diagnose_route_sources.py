"""Inspect the real, local ChromaDB route evidence without loading Ollama/embeddings.

Run at the project root: python diagnose_route_sources.py
Creates: route_sources_diagnostic.txt
Does not mutate the database.
"""
from collections import Counter
from pathlib import Path
import re
import traceback

DB = Path('data/chroma')
OUTPUT = Path('route_sources_diagnostic.txt')
COLLECTION = 'wikivoyage_e5_small_v1'
CITY = 'Rome'
NAMES = ['Colosseum', 'Pantheon', 'Palazzo Farnese',
         'Sistine Chapel', "Fontana dell'Acqua Paola", 'San Pietro in Montorio']
MAX_RECORDS = 250
MAX_EXAMPLES = 12


def context(text, name, span=165):
    match = re.search(re.escape(name), text, re.I)
    if not match:
        return ''
    start = max(0, match.start() - span)
    end = min(len(text), match.end() + span)
    return ' '.join(text[start:end].split())


def collect(collection, city, name, limit=MAX_RECORDS):
    # Search both precise city metadata and any sub-article metadata, without
    # depending on destination_titles() or an embedding model.
    results = []
    seen = set()
    for token in dict.fromkeys([name, name.lower(), name.title()]):
        try:
            got = collection.get(where_document={'$contains': token},
                                 include=['documents', 'metadatas'], limit=limit)
        except Exception as exc:
            return [], 'Chroma get failed: %s: %s' % (type(exc).__name__, exc)
        docs = got.get('documents') or []
        metas = got.get('metadatas') or []
        for cid, doc, meta in zip(got.get('ids') or [], docs, metas):
            meta = meta or {}
            title = str(meta.get('title', ''))
            if title != city and not title.startswith(city + '/'):
                continue
            if cid in seen or not re.search(re.escape(name), str(doc), re.I):
                continue
            seen.add(cid)
            results.append((str(cid), str(doc), meta))
    return results, ''


def build_report(collection):
    lines = ['LOCAL ROUTE EVIDENCE DIAGNOSTIC',
             'Database: %s | collection: %s | city: %s' % (DB, COLLECTION, CITY),
             'Matches are literal mentions, not confirmed descriptions.\n']
    for name in NAMES:
        rows, error = collect(collection, CITY, name)
        lines += ['=' * 70, 'OBJECT: ' + name, 'Exact mention chunks: %s' % len(rows)]
        if error:
            lines.append('ERROR: ' + error)
        if rows:
            counts = Counter(str(meta.get('title', '')) for _, _, meta in rows)
            lines.append('Article titles: ' + repr(counts.most_common(12)))
            rows.sort(key=lambda r: (0 if re.search(r'(?i)(?:^|[.!?]\\s+)(?:the\\s+)?'+re.escape(name)+r'\\s+(?:is|was|has|houses|contains|features|built)', r[1]) else 1, len(r[1])))
            for index, (cid, doc, meta) in enumerate(rows[:MAX_EXAMPLES], 1):
                lines += ['  %s) article=%r, chunk=%r' % (index, meta.get('title', ''), cid),
                          '     url=%s' % meta.get('url', ''),
                          '     excerpt=%s' % context(doc, name)]
        else:
            lines.append('No exact mentions detected in this collection for this city.')
    return '\n'.join(lines) + '\n'


def main():
    try:
        import chromadb
        if not DB.exists():
            raise FileNotFoundError('Missing folder: ' + str(DB))
        client = chromadb.PersistentClient(path=str(DB))
        collection = client.get_collection(COLLECTION)
        report = build_report(collection)
    except Exception:
        report = 'DIAGNOSTIC ERROR\n' + traceback.format_exc()
    OUTPUT.write_text(report, encoding='utf-8')
    print('Wrote:', OUTPUT.resolve())
    print('\n'.join(report.splitlines()[:14]))


if __name__ == '__main__':
    main()
