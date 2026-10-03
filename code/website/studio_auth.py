"""Small, persistent access-code sessions for the password-gated course demo."""
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime


def password_record(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1).hex()
    return {'salt': salt.hex(), 'digest': digest}


def matches(password, record):
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(record['salt']), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(digest, record['digest'])


class Auth:
    def __init__(self, config_path):
        self.path = Path(config_path)
        self.config = json.loads(self.path.read_text())
        self.db = self.path.with_name('sessions.sqlite3')
        self.lock = threading.Lock()
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS sessions
                (digest TEXT PRIMARY KEY, role TEXT, csrf TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS failures (key TEXT, created REAL);''')
            if 'access_id' not in {row[1] for row in db.execute('PRAGMA table_info(sessions)')}:
                db.execute("ALTER TABLE sessions ADD COLUMN access_id TEXT NOT NULL DEFAULT 'shared'")
        self.db.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def access_codes(self):
        return {**self.config.get('guest_codes', {}), 'shared': self.config['shared']}

    def window(self, access_id='shared'):
        share = self.access_codes().get(access_id)
        if share is None:
            return float('inf'), float('-inf')
        return datetime.fromisoformat(share['starts_at']).timestamp(), datetime.fromisoformat(share['ends_at']).timestamp()

    def login(self, password, client, now=None):
        now = time.time() if now is None else now
        if not isinstance(password, str) or not 1 <= len(password) <= 256:
            return None, 401
        with self.lock, self.connect() as db:
            db.execute('DELETE FROM failures WHERE created < ?', (now - 900,))
            db.execute('DELETE FROM sessions WHERE expires <= ?', (now,))
            local = db.execute('SELECT COUNT(*) FROM failures WHERE key=?', (client,)).fetchone()[0]
            total = db.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
            if local >= 8 or total >= 100:
                return None, 429
            # Check every configured code, with a separate validity window for each.
            owner = matches(password, self.config['owner'])
            access_id = None
            end = now
            for key, record in self.access_codes().items():
                matched = matches(password, record['password'])
                start_at, end_at = self.window(key)
                if matched and start_at <= now < end_at:
                    access_id, end = key, end_at
            role = 'owner' if owner else 'guest' if access_id else None
            if role is None:
                db.execute('INSERT INTO failures VALUES (?, ?)', (client, now))
                return None, 401
            db.execute('DELETE FROM failures WHERE key=?', (client,))
            token = secrets.token_urlsafe(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            csrf = secrets.token_urlsafe(24)
            expires = now + 12 * 3600 if role == 'owner' else min(now + 12 * 3600, end)
            db.execute('INSERT INTO sessions (digest, role, csrf, expires, access_id) VALUES (?, ?, ?, ?, ?)',
                       (digest, role, csrf, expires, access_id or 'shared'))
            return {'token': token, 'id': digest, 'role': role, 'csrf': csrf, 'expires': expires}, 200

    def session(self, token, now=None):
        now = time.time() if now is None else now
        if not isinstance(token, str) or not token or len(token) > 128:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as db:
            row = db.execute('SELECT role, csrf, expires, access_id FROM sessions WHERE digest=?', (digest,)).fetchone()
        if not row or row[2] <= now:
            return None
        if row[0] == 'guest':
            start, end = self.window(row[3])
            if not start <= now < end:
                return None
        return {'id': digest, 'role': row[0], 'csrf': row[1], 'expires': row[2]}

    def logout(self, token):
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE digest=?', (hashlib.sha256(token.encode()).hexdigest(),))
