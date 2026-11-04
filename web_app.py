from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_pipeline import RagAssistant, answer_with_llm, detect_destination
from structured_routes import answer_route, is_route_question

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATIC = BASE / "static"
HISTORY_DB = DATA / "web_history.sqlite3"
LOG = DATA / "logs" / "web_requests.jsonl"

app = FastAPI(title="Travel RAG — локальный туристический гид", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_busy = threading.Lock()
_assistant = None
_init_lock = threading.Lock()


@contextmanager
def database():
    DATA.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL)")
    connection.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        sources TEXT NOT NULL DEFAULT '[]',
        created TEXT NOT NULL
    )""")
    connection.commit()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def get_assistant():
    global _assistant
    if _assistant is None:
        with _init_lock:
            if _assistant is None:
                # Загрузка SentenceTransformers/ChromaDB лишь при первом запросе.
                _assistant = RagAssistant(db=DATA / "chroma", articles=DATA / "processed" / "articles.jsonl")
    return _assistant


class ConversationInput(BaseModel):
    title: str = Field(default="Новый диалог", max_length=120)


class ChatInput(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    destination: Optional[str] = Field(default=None, max_length=100)
    top_k: int = Field(default=6, ge=1, le=12)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/conversations")
def conversations():
    with database() as db:
        rows = db.execute("SELECT id, title, created FROM conversations ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: ConversationInput):
    conversation_id = str(uuid.uuid4())
    title = payload.title.strip() or "Новый диалог"
    created = timestamp()
    with database() as db:
        db.execute("INSERT INTO conversations(id, title, created) VALUES (?, ?, ?)", (conversation_id, title, created))
    return {"id": conversation_id, "title": title, "created": created}


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: str):
    with database() as db:
        if db.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone() is None:
            raise HTTPException(404, "Диалог не найден")
        rows = db.execute("SELECT id, role, content, sources, created FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    return [dict(id=r['id'], role=r['role'], content=r['content'], sources=json.loads(r['sources']), created=r['created']) for r in rows]


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str):
    with database() as db:
        removed = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        if not removed.rowcount:
            raise HTTPException(404, "Диалог не найден")


def generate_reply(conversation_id: str, question: str, destination: Optional[str], top_k: int):
    """Выполняется вне event loop, последовательно, чтобы избежать пиков памяти."""
    with _busy:
        with database() as db:
            rows = db.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 6", (conversation_id,)).fetchall()
        history = [dict(role=r['role'], content=r['content']) for r in reversed(rows)]
        assistant = get_assistant()
        # Вопрос на русском без латинского названия города может не определиться.
        # Поле destination явно задаётся пользователем в интерфейсе.
        explicit = detect_destination(question, assistant.titles)
        selected = explicit or destination or None
        passages, identified = assistant.retrieve(question, destination=selected, top_k=top_k)
        if is_route_question(question):
            answer = answer_route(question, passages, identified)
        else:
            answer = answer_with_llm(question, passages, history)
        sources = []
        for index, passage in enumerate(passages, 1):
            sources.append({"number": index, "title": passage.title, "url": passage.url})
        return answer, sources, identified


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, payload: ChatInput):
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "Вопрос не должен быть пустым")
    with database() as db:
        current = db.execute("SELECT title FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if current is None:
            raise HTTPException(404, "Диалог не найден")
    try:
        answer, sources, identified = await asyncio.to_thread(
            generate_reply, conversation_id, question, payload.destination.strip() if payload.destination else None, payload.top_k
        )
    except Exception as exc:
        logging.exception("Ошибка RAG")
        # Сообщение не сохраняется, если генерация не удалась, чтобы можно было повторить попытку.
        raise HTTPException(503, f"Не удалось получить ответ: {exc}") from exc
    created = timestamp()
    with database() as db:
        db.execute("INSERT INTO messages(conversation_id, role, content, sources, created) VALUES (?, ?, ?, ?, ?)",
                   (conversation_id, "user", question, "[]", created))
        db.execute("INSERT INTO messages(conversation_id, role, content, sources, created) VALUES (?, ?, ?, ?, ?)",
                   (conversation_id, "assistant", answer, json.dumps(sources, ensure_ascii=False), timestamp()))
        if current['title'] == 'Новый диалог':
            db.execute("UPDATE conversations SET title=? WHERE id=?", (question[:60], conversation_id))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": created, "conversation_id": conversation_id, "question": question,
                                 "destination": identified, "answer": answer, "sources": sources}, ensure_ascii=False) + "\n")
    return {"answer": answer, "sources": sources, "destination": identified}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
