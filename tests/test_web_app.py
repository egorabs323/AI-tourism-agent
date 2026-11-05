"""Lightweight API smoke test, without real ChromaDB/Ollama."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import web_app


class WebAppTests(unittest.TestCase):
    def test_create_chat_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(web_app, 'HISTORY_DB', Path(folder) / 'history.sqlite3'), \
                 patch.object(web_app, 'DATA', Path(folder)), \
                 patch.object(web_app, 'LOG', Path(folder) / 'log.jsonl'):
                with TestClient(web_app.app) as client:
                    self.assertEqual(client.post('/api/auth/register', json={
                        'username': 'webtest', 'password': 'web-test-password'}).status_code, 201)
                    res = client.post('/api/conversations', json={})
                    self.assertEqual(res.status_code, 201)
                    cid = res.json()['id']
                    self.assertEqual(len(client.get('/api/conversations').json()), 1)
                    with patch.object(web_app, 'generate_reply', return_value=(
                        'День 1\n1. Colosseum',
                        [{'number': 1, 'title': 'Rome', 'url': 'https://en.wikivoyage.org/wiki/Rome'}],
                        'Rome')):
                        reply = client.post(f'/api/conversations/{cid}/messages', json={
                            'question': 'Маршрут по Риму', 'destination': 'Rome', 'top_k': 6})
                    self.assertEqual(reply.status_code, 200, reply.text)
                    messages = client.get(f'/api/conversations/{cid}/messages').json()
                    self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
                    self.assertEqual(messages[1]['sources'][0]['title'], 'Rome')
                    self.assertEqual(client.delete(f'/api/conversations/{cid}').status_code, 204)
                    self.assertEqual(client.get('/api/conversations').json(), [])

if __name__ == '__main__':
    unittest.main()
