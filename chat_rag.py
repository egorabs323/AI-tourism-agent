"""Консольный чат с RAG, фильтрацией направления и опциональным LLM.

python chat_rag.py --question "Какие достопримечательности посетить в Риме за два дня?" --no-llm
python chat_rag.py --destination Rome   # интерактивный диалог, требует установленной и запущенной Ollama
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline import RagAssistant, answer_with_llm
from structured_routes import answer_route, is_route_question


def append_log(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-ассистент по туристическому датасету Wikivoyage")
    parser.add_argument("--question", help="Один вопрос. Если не задан — интерактивный режим")
    parser.add_argument("--destination", help="Название статьи/города НА АНГЛИЙСКОМ, например Rome")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true", help="Показать найденные чанки без LLM")
    parser.add_argument("--without-rag", action="store_true", help="Для сравнения: ответ LLM без базы знаний")
    parser.add_argument("--model", default=None, help="По умолчанию qwen3:1.7b")
    parser.add_argument("--db", type=Path, default=Path("data/chroma"))
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.jsonl"))
    parser.add_argument("--log", type=Path, default=Path("data/logs/rag_requests.jsonl"))
    args = parser.parse_args()
    if args.no_llm and args.without_rag:
        parser.error("--no-llm и --without-rag одновременно не поддерживаются")

    assistant = None if args.without_rag else RagAssistant(db=args.db, articles=args.articles)
    history, previous_destination = [], args.destination

    print("Туристический RAG-ассистент. Для выхода: exit")
    while True:
        question = args.question if args.question is not None else input("\nВы: ").strip()
        if question.lower() in ("exit", "quit", "выход"):
            break
        if not question:
            if args.question is not None:
                break
            continue
        try:
            if args.without_rag:
                passages, detected = [], None
            else:
                # Следующий вопрос с новым явно названным городом переключает направление.
                from rag_pipeline import detect_destination
                explicit = detect_destination(question, assistant.titles)
                requested = explicit or args.destination or previous_destination
                passages, detected = assistant.retrieve(question, destination=requested, top_k=args.top_k)
                previous_destination = detected
            if args.no_llm:
                answer = "\n".join(
                    f"{i}. {p.title} | distance={p.distance:.4f}\n{p.url}\n{p.text[:650]}"
                    for i, p in enumerate(passages, 1)) or "Подходящие чанки не найдены"
            elif not args.without_rag and is_route_question(question):
                answer = answer_route(question, passages, detected, args.model)
            else:
                answer = answer_with_llm(question, passages, history, args.model, args.without_rag)
            print("\nАссистент:\n" + answer)
            append_log(args.log, {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "question": question, "destination": detected,
                "mode": "without_rag" if args.without_rag else "retrieval_only" if args.no_llm else "rag",
                "sources": [{"title": p.title, "url": p.url, "distance": p.distance} for p in passages],
                "answer": answer,
            })
            history.extend([{"role": "user", "content": question},
                            {"role": "assistant", "content": answer}])
        except Exception as exc:
            print(f"\nОшибка: {exc}")
        if args.question is not None:
            break


if __name__ == "__main__":
    main()
