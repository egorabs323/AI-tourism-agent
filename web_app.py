from __future__ import annotations
import asyncio
from contextlib import contextmanager
import json
import logging
import os
import sqlite3
import threading
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from auth_local import COOKIE, SESSION_DAYS, init_auth, create_user, login_user, create_session, read_session, revoke_session, check_origin

from rag_pipeline import RagAssistant, answer_with_llm, detect_destination
from structured_routes import answer_route
from route_enrichment import enrich_route_answer
from qa_answers import answer_factual
from context_query import contextual_search_query
from request_modes import validate_mode, retrieval_query
from user_documents import MAX_BYTES, ALLOWED_SUFFIXES, extract_text, index_document, search_documents, remove_document

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATIC = BASE / "static"
HISTORY_DB = DATA / "web_history.sqlite3"
LOG = DATA / "logs" / "web_requests.jsonl"
UPLOADS = DATA / "uploads"

app = FastAPI(title="Travel RAG — локальный туристический гид", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Один процесс генерации одновременно: важно для ПК с 8 ГБ ОЗУ.
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
    connection.execute("""CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        filename TEXT NOT NULL, suffix TEXT NOT NULL, chunks INTEGER NOT NULL, created TEXT NOT NULL
    )""")
    init_auth(connection)
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
    mode: str = Field(default="reference", max_length=32)


@app.middleware('http')
async def protect_cross_origin(request: Request, call_next):
    if request.url.path.startswith('/api/') and request.method not in ('GET', 'HEAD', 'OPTIONS'):
        try:
            check_origin(request)
        except HTTPException:
            return Response(status_code=403, content='Forbidden')
    return await call_next(request)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


def authenticated_user(request: Request):
    with database() as db:
        user = read_session(db, request.cookies.get(COOKIE))
    if user is None:
        raise HTTPException(401, 'Требуется войти в аккаунт')
    return user


def set_session_cookie(response, token):
    response.set_cookie(key=COOKIE, value=token, httponly=True, secure=False,
                        samesite='strict', max_age=SESSION_DAYS * 86400, path='/')


@app.post('/api/auth/register', status_code=201)
def register(payload: Credentials, response: Response, request: Request):
    check_origin(request)
    with database() as db:
        user_id = create_user(db, payload.username, payload.password)
        token = create_session(db, user_id)
    set_session_cookie(response, token)
    return {'username': payload.username.strip()}


@app.post('/api/auth/login')
def login(payload: Credentials, response: Response, request: Request):
    check_origin(request)
    with database() as db:
        user_id = login_user(db, payload.username, payload.password)
        token = create_session(db, user_id)
    set_session_cookie(response, token)
    return {'username': payload.username.strip()}


@app.post('/api/auth/logout', status_code=204)
def logout(request: Request, response: Response):
    check_origin(request)
    with database() as db:
        revoke_session(db, request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path='/')


@app.get('/api/auth/me')
def current_user(user=Depends(authenticated_user)):
    return {'username': user['username']}


@app.get("/api/conversations")
def conversations(user=Depends(authenticated_user)):
    with database() as db:
        rows = db.execute("SELECT id, title, created FROM conversations WHERE user_id=? ORDER BY created DESC", (user["id"],)).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: ConversationInput, user=Depends(authenticated_user)):
    conversation_id = str(uuid.uuid4())
    title = payload.title.strip() or "Новый диалог"
    created = timestamp()
    with database() as db:
        db.execute("INSERT INTO conversations(id, title, created, user_id) VALUES (?, ?, ?, ?)", (conversation_id, title, created, user["id"]))
    return {"id": conversation_id, "title": title, "created": created}


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: str, user=Depends(authenticated_user)):
    with database() as db:
        if db.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])).fetchone() is None:
            raise HTTPException(404, "Диалог не найден")
        rows = db.execute("SELECT id, role, content, sources, created FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    return [dict(id=r['id'], role=r['role'], content=r['content'], sources=json.loads(r['sources']), created=r['created']) for r in rows]


def ensure_conversation(db, conversation_id, user_id):
    if db.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone() is None:
        raise HTTPException(404, "Диалог не найден")


@app.get("/api/conversations/{conversation_id}/documents")
def list_documents(conversation_id: str, user=Depends(authenticated_user)):
    with database() as db:
        ensure_conversation(db, conversation_id, user["id"])
        records = db.execute("SELECT id, filename, chunks, created FROM documents WHERE conversation_id=? ORDER BY created DESC",
                             (conversation_id,)).fetchall()
    return [dict(r) for r in records]


@app.post("/api/conversations/{conversation_id}/documents", status_code=201)
async def upload_document(conversation_id: str, file: UploadFile = File(...), user=Depends(authenticated_user)):
    with database() as db:
        ensure_conversation(db, conversation_id, user["id"])
    filename = Path(file.filename or '').name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(422, 'Поддерживаются только PDF, DOCX, TXT')
    content = await file.read(MAX_BYTES + 1)
    await file.close()
    try:
        plain_text = extract_text(filename, content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    document_id = str(uuid.uuid4())
    # Общее ограничение конкуренции с Ollama, чтобы не запускать тяжёлые вычисления параллельно.
    try:
        def work():
            with _busy:
                return index_document(get_assistant(), conversation_id, document_id, filename, plain_text)
        chunks = await asyncio.to_thread(work)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception('Ошибка индексирования документа')
        raise HTTPException(503, 'Не удалось проиндексировать документ') from exc
    location = UPLOADS / conversation_id
    location.mkdir(parents=True, exist_ok=True)
    try:
        (location / (document_id + suffix)).write_bytes(content)
        with database() as db:
            ensure_conversation(db, conversation_id, user["id"])
            db.execute('INSERT INTO documents(id, conversation_id, filename, suffix, chunks, created) VALUES(?,?,?,?,?,?)',
                       (document_id, conversation_id, filename, suffix, chunks, timestamp()))
    except Exception:
        with _busy:
            remove_document(get_assistant(), document_id)
        (location / (document_id + suffix)).unlink(missing_ok=True)
        raise
    return {'id': document_id, 'filename': filename, 'chunks': chunks}


@app.get("/api/conversations/{conversation_id}/documents/{document_id}/download")
def download_document(conversation_id: str, document_id: str, user=Depends(authenticated_user)):
    with database() as db:
        ensure_conversation(db, conversation_id, user["id"])
        row = db.execute('SELECT filename, suffix FROM documents WHERE id=? AND conversation_id=?',
                         (document_id, conversation_id)).fetchone()
    if row is None:
        raise HTTPException(404, 'Документ не найден')
    path = UPLOADS / conversation_id / (document_id + row['suffix'])
    if not path.is_file():
        raise HTTPException(404, 'Файл отсутствует')
    return FileResponse(path, filename=row['filename'])


@app.delete("/api/conversations/{conversation_id}/documents/{document_id}", status_code=204)
def delete_document(conversation_id: str, document_id: str, user=Depends(authenticated_user)):
    with database() as db:
        ensure_conversation(db, conversation_id, user["id"])
        row = db.execute('SELECT suffix FROM documents WHERE id=? AND conversation_id=?',
                         (document_id, conversation_id)).fetchone()
        if row is None:
            raise HTTPException(404, 'Документ не найден')
    with _busy:
        remove_document(get_assistant(), document_id)
    with database() as db:
        db.execute('DELETE FROM documents WHERE id=? AND conversation_id=?', (document_id, conversation_id))
    (UPLOADS / conversation_id / (document_id + row['suffix'])).unlink(missing_ok=True)


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user=Depends(authenticated_user)):
    with database() as db:
        ensure_conversation(db, conversation_id, user["id"])
        docs = db.execute('SELECT id FROM documents WHERE conversation_id=?', (conversation_id,)).fetchall()
    if docs:
        with _busy:
            for doc in docs:
                remove_document(get_assistant(), doc['id'])
    with database() as db:
        db.execute('DELETE FROM conversations WHERE id=?', (conversation_id,))
    shutil.rmtree(UPLOADS / conversation_id, ignore_errors=True)


def generate_reply(conversation_id: str, question: str, destination: Optional[str], top_k: int, mode: str = "reference"):
    """Выполняется вне event loop, последовательно, чтобы избежать пиков памяти."""
    with _busy:
        with database() as db:
            rows = db.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 6", (conversation_id,)).fetchall()
        history = [dict(role=r['role'], content=r['content']) for r in reversed(rows)]
        validate_mode(mode)
        assistant = get_assistant()
        # Вопрос на русском без латинского названия города может не определиться.
        # Поле destination явно задаётся пользователем в интерфейсе.
        explicit = detect_destination(question, assistant.titles)
        selected = explicit or destination or None
        own_passages = search_documents(assistant, conversation_id, question, count=3)
        if own_passages:
            # Вопрос о загруженном документе: НЕ смешиваем с общими статьями Wikivoyage.
            passages = own_passages
            identified = selected
        else:
            search_question = contextual_search_query(question, history, destination=selected)
            search_question = retrieval_query(search_question, mode)
            passages, identified = assistant.retrieve(search_question, destination=selected, top_k=top_k)
        if mode == "route" and not own_passages:
            answer = answer_route(question, passages, identified)
            answer, passages = enrich_route_answer(answer, passages, assistant, identified)
        else:
            answer = answer_factual(question, passages, history, mode=mode)
        sources = []
        for index, passage in enumerate(passages, 1):
            sources.append({"number": index, "title": passage.title, "url": passage.url})
        return answer, sources, identified


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, payload: ChatInput, user=Depends(authenticated_user)):
    question = payload.question.strip()
    try:
        validate_mode(payload.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not question:
        raise HTTPException(422, "Вопрос не должен быть пустым")
    with database() as db:
        current = db.execute("SELECT title FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])).fetchone()
        if current is None:
            raise HTTPException(404, "Диалог не найден")
    try:
        answer, sources, identified = await asyncio.to_thread(
            generate_reply, conversation_id, question, payload.destination.strip() if payload.destination else None, payload.top_k, payload.mode
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
        handle.write(json.dumps({"time": created, "conversation_id": conversation_id, "user_id": user["id"], "question": question,
                                 "destination": identified, "mode": payload.mode, "answer": answer, "sources": sources}, ensure_ascii=False) + "\n")
    return {"answer": answer, "sources": sources, "destination": identified}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
