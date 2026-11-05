"""Индексация загруженных PDF/DOCX/TXT в отдельную коллекцию ChromaDB.
Совместимо с Python 3.9. Не изменяет основную коллекцию Wikivoyage.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import List

from rag_pipeline import Passage

USER_COLLECTION = 'travel_user_docs_e5_small_v1'
MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 500_000
MAX_CHUNKS = 180
ALLOWED_SUFFIXES = {'.pdf', '.docx', '.txt'}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError('Поддерживаются только PDF, DOCX и TXT')
    if not content:
        raise ValueError('Файл пуст')
    if len(content) > MAX_BYTES:
        raise ValueError('Размер документа не должен превышать 5 МБ')
    if suffix == '.txt':
        try:
            raw = content.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                raw = content.decode('cp1251')
            except UnicodeDecodeError as exc:
                raise ValueError('Не удалось прочитать TXT. Нужна кодировка UTF-8 или Windows-1251') from exc
    elif suffix == '.pdf':
        if not content.startswith(b'%PDF-'):
            raise ValueError('Неверный формат PDF')
        import fitz
        try:
            with fitz.open(stream=content, filetype='pdf') as pdf:
                if pdf.page_count > 250:
                    raise ValueError('PDF содержит слишком много страниц (максимум 250)')
                raw = '\n'.join(page.get_text('text') for page in pdf)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('Не удалось извлечь текст из PDF') from exc
    else:
        if not content.startswith(b'PK'):
            raise ValueError('Неверный формат DOCX')
        from docx import Document
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 20 * 1024 * 1024:
                    raise ValueError('Распакованный DOCX слишком большой')
            doc = Document(io.BytesIO(content))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(' | '.join(cell.text for cell in row.cells))
            raw = '\n'.join(parts)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('Не удалось извлечь текст из DOCX') from exc
    raw = re.sub(r'[ \t]+', ' ', raw).strip()
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    if len(raw) < 40:
        raise ValueError('В документе слишком мало извлекаемого текста (минимум 40 символов). PDF-сканы без текстового слоя не поддерживаются')
    if len(raw) > MAX_CHARS:
        raise ValueError('В документе слишком много текста (максимум 500 000 символов)')
    return raw


def split_text(text: str, words_per_chunk: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    if not 0 <= overlap < words_per_chunk:
        raise ValueError('Неверное перекрытие чанков')
    step = words_per_chunk - overlap
    chunks = [' '.join(words[start:start + words_per_chunk])
              for start in range(0, len(words), step)]
    if len(chunks) > MAX_CHUNKS:
        raise ValueError('Документ слишком большой для локального индексирования: более 180 чанков')
    return chunks


def get_collection(main_assistant):
    # Используем публичный API, а не внутреннее поле collection._client:
    # оно может вернуть низкоуровневый объект без методов get/count/query.
    import chromadb
    from pathlib import Path
    db_dir = Path(__file__).resolve().parent / 'data' / 'chroma'
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(
        name=USER_COLLECTION, metadata={'hnsw:space': 'cosine'}
    )


def index_document(main_assistant, conversation_id: str, document_id: str, filename: str, text: str) -> int:
    chunks = split_text(text)
    collection = get_collection(main_assistant)
    # Тот же энкодер, который уже используется в основном RAG: новая модель не загружается.
    for start in range(0, len(chunks), 12):
        part = chunks[start:start + 12]
        vectors = main_assistant.encoder.encode(['passage: ' + chunk for chunk in part],
                                                 normalize_embeddings=True,
                                                 batch_size=12, show_progress_bar=False)
        collection.add(ids=['doc_' + document_id + '_' + str(start + i) for i in range(len(part))],
                       documents=part, embeddings=vectors.tolist(),
                       metadatas=[{'conversation_id': conversation_id, 'document_id': document_id,
                                   'title': filename, 'chunk_index': start + i}
                                  for i in range(len(part))])
    return len(chunks)


def search_documents(main_assistant, conversation_id: str, question: str, count: int = 3) -> List[Passage]:
    collection = get_collection(main_assistant)
    # Фильтр по ID диалога обязателен: чужие документы не ищем.
    # Проверяем доступность записей, чтобы не вызывать query() на пустой коллекции.
    existing = collection.get(where={'conversation_id': conversation_id}, limit=1, include=[])
    if not existing['ids']:
        return []
    embedding = main_assistant.encoder.encode(['query: ' + question], normalize_embeddings=True)[0]
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=count,
                               where={'conversation_id': conversation_id},
                               include=['documents', 'metadatas', 'distances'])
    passages = []
    for chunk_id, doc, meta, distance in zip(results['ids'][0], results['documents'][0],
                                             results['metadatas'][0], results['distances'][0]):
        # Локальная ссылка разрешена только в рамках того же диалога.
        document_id = meta['document_id']
        link = 'http://127.0.0.1:8000/api/conversations/{}/documents/{}/download'.format(conversation_id, document_id)
        passages.append(Passage(text=doc, title='Документ: ' + meta['title'], url=link,
                                distance=float(distance), chunk_id=chunk_id))
    return passages


def remove_document(main_assistant, document_id: str):
    collection = get_collection(main_assistant)
    collection.delete(where={'document_id': document_id})
