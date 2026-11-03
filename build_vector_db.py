from __future__ import annotations
import argparse
import json
from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION = "wikivoyage_e5_small_v1"


def load_chunks(path: Path, limit: int):
    with path.open(encoding="utf-8") as f:
        n = 0
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("chunk_id") or not row.get("content"):
                continue
            yield row
            n += 1
            if limit and n >= limit:
                break


def metadata(row):
    # Chroma принимает в metadata лишь строку, число или bool, но не None.
    result = {}
    for field in ("article_id", "title", "url", "qid", "source", "snapshot", "license",
                  "lat", "lon", "chunk_index", "chunk_count", "word_count"):
        val = row.get(field)
        if isinstance(val, (str, int, float, bool)) and val != "":
            result[field] = val
    return result


def create_collection(db_dir: Path):
    import chromadb
    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    return col


def index_batch(collection, model, rows):
    ids = [r["chunk_id"] for r in rows]
    existing = set(collection.get(ids=ids, include=[])["ids"])
    fresh = [r for r in rows if r["chunk_id"] not in existing]
    if not fresh:
        return 0
    texts = [r["content"] for r in fresh]
    # E5 требует префикс passage: для документов и query: для вопросов.
    vectors = model.encode(["passage: " + text for text in texts],
                           batch_size=16, show_progress_bar=False,
                           normalize_embeddings=True)
    collection.add(ids=[r["chunk_id"] for r in fresh],
                   documents=texts,
                   embeddings=vectors.tolist(),
                   metadatas=[metadata(r) for r in fresh])
    return len(fresh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--db", type=Path, default=Path("data/chroma"))
    p.add_argument("--limit", type=int, default=1000, help="0 = весь корпус")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.batch_size > 1000:
        p.error("limit >= 0, batch-size от 1 до 1000")
    if not args.input.is_file():
        p.error(f"Нет файла {args.input}. Сначала выполни этап 1")

    from sentence_transformers import SentenceTransformer
    print(f"Загрузка модели: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    collection = create_collection(args.db)
    processed = added = 0
    batch = []
    for row in load_chunks(args.input, args.limit):
        batch.append(row)
        if len(batch) < args.batch_size:
            continue
        added += index_batch(collection, model, batch)
        processed += len(batch)
        batch.clear()
        print(f"Пройдено: {processed:,}; добавлено за запуск: {added:,}; всего в БД: {collection.count():,}", flush=True)
    if batch:
        added += index_batch(collection, model, batch)
        processed += len(batch)
    print(f"\nГотово! Пройдено: {processed:,}; добавлено: {added:,}; записей в коллекции: {collection.count():,}")
    print(f"База: {args.db.resolve()}\nКоллекция: {COLLECTION}")


if __name__ == "__main__":
    main()