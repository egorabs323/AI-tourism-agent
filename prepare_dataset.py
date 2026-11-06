from __future__ import annotations
import argparse
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASET = "yuiseki/wikivoyage-geotagged"
CONFIG = "20260901.en"
SOURCE_SNAPSHOT = "2026-09-01"
LICENSE = "CC BY-SA 4.0"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", html.unescape(value or ""))
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


def preserve_layout(value: str) -> str:
    text = unicodedata.normalize('NFC', html.unescape(value or '')).replace('\u00a0', ' ')
    return '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip()
                     for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')).strip()


def split_into_chunks(text: str, words_per_chunk: int = 220, overlap_words: int = 40) -> list[str]:
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk должен быть больше нуля")
    if overlap_words < 0 or overlap_words >= words_per_chunk:
        raise ValueError("overlap_words должен быть от 0 до words_per_chunk - 1")
    normalized = normalize_text(text)
    words = normalized.split(" ") if normalized else []
    if not words:
        return []
    step = words_per_chunk - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + words_per_chunk, len(words))
        if start and end == len(words) and len(words) - start <= overlap_words:
            break
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def numeric_coordinate(value: Any, low: float, high: float) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x < low or x > high:
        return None
    return x


def iter_rows(input_jsonl: Path | None, stream: bool) -> Iterator[dict[str, Any]]:
    if input_jsonl is not None:
        with input_jsonl.open("r", encoding="utf-8") as inp:
            for line_num, line in enumerate(inp, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Некорректный JSON в строке {line_num}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Ожидается объект в строке {line_num}")
                yield row
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Установи зависимости: python -m pip install -r requirements.txt") from exc

    ds = load_dataset(DATASET, CONFIG, split="train", streaming=stream)
    for row in ds:
        yield row


def write_jsonl_line(file_obj: Any, data: dict[str, Any]) -> None:
    file_obj.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(
    rows: Iterable[dict[str, Any]],
    output: Path,
    limit: int = 0,
    words_per_chunk: int = 220,
    overlap_words: int = 40,
    min_chars: int = 120,
) -> dict[str, Any]:
    if limit < 0 or min_chars < 0:
        raise ValueError("limit и min_chars не могут быть отрицательными")
    if words_per_chunk <= 0 or not (0 <= overlap_words < words_per_chunk):
        raise ValueError("Неверные настройки чанкинга")

    output.mkdir(parents=True, exist_ok=True)
    counters = {
        "rows_seen": 0,
        "articles_saved": 0,
        "chunks_saved": 0,
        "skipped_missing_fields": 0,
        "skipped_short_text": 0,
        "skipped_invalid_coordinates": 0,
        "skipped_duplicate_id": 0,
        "articles_without_qid": 0,
        "total_article_chars": 0,
    }
    lengths: list[int] = []
    titles_preview: list[str] = []
    chunks_preview: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with (output / "articles.jsonl").open("w", encoding="utf-8") as article_file, \
         (output / "chunks.jsonl").open("w", encoding="utf-8") as chunk_file:
        for row in rows:
            if limit and counters["rows_seen"] >= limit:
                break
            counters["rows_seen"] += 1

            page_id = str(row.get("id") or "").strip()
            title = normalize_text(str(row.get("title") or ""))
            url = str(row.get("url") or "").strip()
            body = normalize_text(str(row.get("text") or ""))
            if not (page_id and title and url.startswith("https://") and body):
                counters["skipped_missing_fields"] += 1
                continue
            if len(body) < min_chars:
                counters["skipped_short_text"] += 1
                continue
            lat = numeric_coordinate(row.get("lat"), -90, 90)
            lon = numeric_coordinate(row.get("lon"), -180, 180)
            if lat is None or lon is None:
                counters["skipped_invalid_coordinates"] += 1
                continue
            if page_id in seen_ids:
                counters["skipped_duplicate_id"] += 1
                continue
            seen_ids.add(page_id)
            qid = str(row.get("qid") or "").strip() or None
            chunks = split_into_chunks(body, words_per_chunk, overlap_words)

            metadata = {
                "article_id": page_id,
                "title": title,
                "url": url,
                "qid": qid,
                "lat": lat,
                "lon": lon,
                "source": "Wikivoyage",
                "snapshot": SOURCE_SNAPSHOT,
                "license": LICENSE,
            }
            article = {**metadata, "text": body, "length_chars": len(body),
                       "structured_text": preserve_layout(str(row.get('text') or ''))}
            write_jsonl_line(article_file, article)
            if len(titles_preview) < 10:
                titles_preview.append(title)

            for index, passage in enumerate(chunks):
                chunk = {
                    **metadata,
                    "chunk_id": f"wv_{page_id}_{index:04d}",
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "content": f"Destination: {title}\n{passage}",
                    "body": passage,
                    "word_count": len(passage.split()),
                }
                write_jsonl_line(chunk_file, chunk)
                if len(chunks_preview) < 3:
                    chunks_preview.append({
                        "chunk_id": chunk["chunk_id"],
                        "title": title,
                        "text_excerpt": passage[:350],
                    })

            counters["articles_saved"] += 1
            counters["chunks_saved"] += len(chunks)
            counters["total_article_chars"] += len(body)
            lengths.append(len(body))
            if counters["articles_saved"] % 1000 == 0:
                print(f"Обработано статей: {counters['articles_saved']:,}; "
                      f"чанков: {counters['chunks_saved']:,}")

    n = counters["articles_saved"]
    sorted_lengths = sorted(lengths)
    stats = {
        "dataset": DATASET,
        "config": CONFIG,
        "snapshot": SOURCE_SNAPSHOT,
        "license": LICENSE,
        "settings": {
            "limit": limit,
            "words_per_chunk": words_per_chunk,
            "overlap_words": overlap_words,
            "min_chars": min_chars,
        },
        **counters,
        "average_article_chars": round(counters["total_article_chars"] / n, 2) if n else 0,
        "min_article_chars": sorted_lengths[0] if n else 0,
        "median_article_chars": sorted_lengths[n // 2] if n else 0,
        "p95_article_chars": sorted_lengths[math.ceil(n * 0.95) - 1] if n else 0,
        "max_article_chars": sorted_lengths[-1] if n else 0,
        "average_chunks_per_article": round(counters["chunks_saved"] / n, 2) if n else 0,
        "sample_titles": titles_preview,
    }
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "preview.json").write_text(json.dumps(chunks_preview, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово. Результаты:")
    print(f"  Просмотрено исходных записей: {counters['rows_seen']:,}")
    print(f"  Сохранено статей: {n:,}")
    print(f"  Сохранено чанков: {counters['chunks_saved']:,}")
    print(f"  Каталог: {output.resolve()}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка Wikivoyage Geotagged для RAG")
    parser.add_argument("--limit", type=int, default=0,
                        help="Сколько исходных строк обработать; 0 = весь датасет")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--input-jsonl", type=Path, default=None,
                        help="Локальный JSONL вместо Hugging Face (для тестирования)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Скачать датасет обычным способом, без потоковой загрузки")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--min-chars", type=int, default=120)
    args = parser.parse_args()
    prepare(
        iter_rows(args.input_jsonl, stream=not args.no_stream),
        args.output,
        limit=args.limit,
        words_per_chunk=args.chunk_words,
        overlap_words=args.overlap_words,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
