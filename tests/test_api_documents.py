"""Проверка HTTP API без реальных моделей и ChromaDB."""
import collections
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

if 'rag_pipeline' not in sys.modules:
    rag = types.ModuleType('rag_pipeline')
    rag.Passage = collections.namedtuple('Passage', 'text title url distance chunk_id')
    rag.RagAssistant = type('RagAssistant', (), {})
    rag.answer_with_llm = lambda *args, **kwargs: 'Ответ'
    rag.detect_destination = lambda *args, **kwargs: None
    sys.modules['rag_pipeline'] = rag
if 'structured_routes' not in sys.modules:
    routes = types.ModuleType('structured_routes')
    routes.answer_route = lambda *args, **kwargs: 'Маршрут'
    routes.is_route_question = lambda question: False
    sys.modules['structured_routes'] = routes

import web_app


class ApiDocuments(unittest.TestCase):
    def test_upload_listing_download_and_delete(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / 'data'
            with patch.object(web_app, 'DATA', data), \
                 patch.object(web_app, 'HISTORY_DB', data / 'web_history.sqlite3'), \
                 patch.object(web_app, 'UPLOADS', data / 'uploads'), \
                 patch.object(web_app, 'LOG', data / 'logs' / 'web_requests.jsonl'), \
                 patch.object(web_app, 'get_assistant', return_value=object()), \
                 patch.object(web_app, 'index_document', return_value=1), \
                 patch.object(web_app, 'remove_document') as remove:
                api = TestClient(web_app.app)
                cid = api.post('/api/conversations', json={}).json()['id']
                text = ('Маршрут Лазурная тропа завершается в Саду маяков. ' * 3).encode()
                response = api.post('/api/conversations/{}/documents'.format(cid),
                                    files={'file': ('guide.txt', text, 'text/plain')})
                self.assertEqual(response.status_code, 201, response.text)
                item = response.json()
                self.assertEqual(item['filename'], 'guide.txt')
                records = api.get('/api/conversations/{}/documents'.format(cid)).json()
                self.assertEqual(len(records), 1)
                download = api.get('/api/conversations/{}/documents/{}/download'.format(cid, item['id']))
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.content, text)
                other = api.post('/api/conversations', json={}).json()['id']
                self.assertEqual(api.get('/api/conversations/{}/documents/{}/download'.format(other, item['id'])).status_code, 404)
                self.assertEqual(api.delete('/api/conversations/{}/documents/{}'.format(cid, item['id'])).status_code, 204)
                self.assertEqual(api.get('/api/conversations/{}/documents'.format(cid)).json(), [])
                remove.assert_called_once()


if __name__ == '__main__':
    unittest.main()
