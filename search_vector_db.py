from __future__ import annotations

import argparse
from pathlib import Path

from build_vector_db import MODEL_NAME, COLLECTION, create_collection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str)
    p.add_argument("--db", type=Path, default=Path("data/chroma"))
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    if args.top_k < 1:
        p.error("top-k должно быть положительным")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    collection = create_collection(args.db)
    if collection.count() == 0:
        p.error("База пустая. Сначала запусти build_vector_db.py")
    vector = model.encode(["query: " + args.question], normalize_embeddings=True)[0].tolist()
    result = collection.query(query_embeddings=[vector], n_results=min(args.top_k, collection.count()),
                              include=["documents", "metadatas", "distances"])
    for n, (doc, meta, distance) in enumerate(zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), 1):
        print(f"\n{n}. {meta.get('title', '')} | cosine distance: {distance:.4f}")
        print(f"Источник: {meta.get('url', '')}")
        print(doc[:700].replace("\n", " ") + ("..." if len(doc) > 700 else ""))


if __name__ == "__main__":
    main()