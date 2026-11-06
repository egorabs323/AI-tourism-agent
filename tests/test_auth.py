import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
import web_app


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = (web_app.DATA, web_app.HISTORY_DB, web_app.LOG, web_app.UPLOADS)
        root = Path(self.tmp.name)
        web_app.DATA = root
        web_app.HISTORY_DB = root / 'web_history.sqlite3'
        web_app.LOG = root / 'logs' / 'web_requests.jsonl'
        web_app.UPLOADS = root / 'uploads'
        self.one = TestClient(web_app.app)
        self.two = TestClient(web_app.app)

    def tearDown(self):
        web_app.DATA, web_app.HISTORY_DB, web_app.LOG, web_app.UPLOADS = self.previous
        self.one.close()
        self.two.close()
        self.tmp.cleanup()

    def test_registration_login_isolation_and_logout(self):
        self.assertEqual(self.one.get('/api/conversations').status_code, 401)
        r = self.one.post('/api/auth/register', json={'username':'alice','password':'password-123456'})
        self.assertEqual(r.status_code,201)
        self.assertIn('httponly',r.headers['set-cookie'].lower())
        conv = self.one.post('/api/conversations',json={}).json()['id']
        self.assertEqual(len(self.one.get('/api/conversations').json()),1)
        self.assertEqual(self.two.post('/api/auth/register',json={'username':'bob','password':'password-123456'}).status_code,201)
        self.assertEqual(self.two.get('/api/conversations').json(),[])
        self.assertEqual(self.two.get('/api/conversations/'+conv+'/messages').status_code,404)
        self.assertEqual(self.two.get('/api/conversations/'+conv+'/documents').status_code,404)
        self.assertEqual(self.two.delete('/api/conversations/'+conv).status_code,404)
        self.assertEqual(self.two.post('/api/conversations/'+conv+'/messages',json={'question':'hello'}).status_code,404)
        self.assertEqual(self.one.post('/api/auth/logout').status_code,204)
        self.assertEqual(self.one.get('/api/conversations').status_code,401)
        self.assertEqual(self.one.post('/api/auth/login',json={'username':'ALICE','password':'password-123456'}).status_code,200)
        self.assertEqual(len(self.one.get('/api/conversations').json()),1)
        self.assertEqual(self.one.post('/api/auth/login',json={'username':'alice','password':'badpass'}).status_code,401)

    def test_legacy_history_assigned_to_first_user(self):
        with sqlite3.connect(web_app.HISTORY_DB) as db:
            db.execute('CREATE TABLE conversations(id TEXT PRIMARY KEY,title TEXT NOT NULL,created TEXT NOT NULL)')
            db.execute("INSERT INTO conversations VALUES('old','Старый диалог','2026-01-01')")
        self.assertEqual(self.one.post('/api/auth/register',json={'username':'first','password':'password-123456'}).status_code,201)
        self.assertEqual(self.one.get('/api/conversations').json()[0]['id'],'old')
        self.two.post('/api/auth/register',json={'username':'other','password':'password-123456'})
        self.assertEqual(self.two.get('/api/conversations').json(),[])


if __name__=='__main__':
    unittest.main()
