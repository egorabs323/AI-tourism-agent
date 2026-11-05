"""Локальная регистрация и cookie-сессии, совместимо с Python 3.9."""
import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Request

COOKIE = 'travel_rag_session'
SESSION_DAYS = 7
USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{3,32}$')


def init_auth(db):
    db.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_salt TEXT NOT NULL, password_hash TEXT NOT NULL, created TEXT NOT NULL
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires TEXT NOT NULL
    )''')
    # Existing dialogues are not silently made public to newly registered users.
    cols = {r['name'] for r in db.execute('PRAGMA table_info(conversations)').fetchall()}
    if 'user_id' not in cols:
        db.execute('ALTER TABLE conversations ADD COLUMN user_id TEXT')
    db.execute('CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations(user_id)')


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), 260000).hex()


def create_user(db, username, password):
    username = username.strip()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(422, 'Логин: 3–32 символа, латинские буквы, цифры и _')
    if len(password) < 10 or len(password) > 128:
        raise HTTPException(422, 'Пароль: от 10 до 128 символов')
    user_id = secrets.token_hex(16)
    salt = secrets.token_hex(16)
    first_user = db.execute('SELECT 1 FROM users LIMIT 1').fetchone() is None
    try:
        db.execute('INSERT INTO users(id,username,password_salt,password_hash,created) VALUES(?,?,?,?,?)',
                   (user_id, username, salt, hash_password(password, salt), datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, 'Этот логин уже используется')
    # Only first registered account inherits legacy local conversations.
    if first_user:
        db.execute('UPDATE conversations SET user_id=? WHERE user_id IS NULL', (user_id,))
    return user_id


def login_user(db, username, password):
    row = db.execute('SELECT id,password_salt,password_hash FROM users WHERE username=? COLLATE NOCASE',
                     (username.strip(),)).fetchone()
    if row is None:
        # Reduce difference between valid and invalid usernames.
        hash_password(password, '0' * 32)
        raise HTTPException(401, 'Неверный логин или пароль')
    if not hmac.compare_digest(hash_password(password, row['password_salt']), row['password_hash']):
        raise HTTPException(401, 'Неверный логин или пароль')
    return row['id']


def create_session(db, user_id):
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    expiry = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
    db.execute('INSERT INTO sessions(token_hash,user_id,expires) VALUES(?,?,?)', (digest, user_id, expiry))
    return token


def read_session(db, token):
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    row = db.execute('''SELECT u.id,u.username FROM sessions s JOIN users u ON s.user_id=u.id
                        WHERE s.token_hash=? AND s.expires>?''',
                     (digest, datetime.now(timezone.utc).isoformat())).fetchone()
    return dict(row) if row else None


def revoke_session(db, token):
    if token:
        db.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))


def check_origin(request):
    # Defense in depth: same-site cookies plus exact origin check for browser writes.
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Недопустимый Origin')
