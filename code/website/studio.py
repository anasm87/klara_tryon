"""Klara studio. Bound to loopback behind the domain's HTTPS reverse proxy."""
import argparse
import base64
import binascii
import io
import json
import mimetypes
import os
import re
import secrets
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError
from studio_auth import Auth

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'studio-web'
JOBS = ROOT / 'studio-jobs'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
STATES = {}
BUSY = False
GPU = False
ENGINE = None
APP_VERSION = '1.4.1'
AUTH = None
OWNERS = {}
MAX_BODY = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000


def generation_ready(preview):
    """Live service must restart on CUDA startup failure, never latch preview mode."""
    if preview:
        return False
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not ready. Restart the live service to retry GPU initialization.')
    # Check actual allocation, not only device discovery, before advertising readiness.
    torch.empty(1, device='cuda')
    return True


def restore_jobs():
    """Keep completed downloads usable after a server restart; never restart GPU work."""
    if not JOBS.exists():
        return
    for folder in JOBS.iterdir():
        if not folder.is_dir() or not re.fullmatch(r'[0-9a-f]{32}', folder.name):
            continue
        receipt = folder / 'run.json'
        try:
            job = folder / 'job.json'
            OWNERS[folder.name] = json.loads(job.read_text()).get('owner') if job.is_file() else None
            if receipt.is_file() and (folder / 'result.png').is_file():
                metadata = json.loads(receipt.read_text(encoding='utf-8'))
                STATES[folder.name] = {'status': 'done', 'seconds': round(float(metadata['seconds']), 1),
                    'image': f'/jobs/{folder.name}/result.png', 'receipt': f'/jobs/{folder.name}/run.json'}
            elif (folder / 'job.json').is_file():
                STATES[folder.name] = {'status': 'error', 'message': 'This job was interrupted when the server stopped. You can submit it again.'}
        except (OSError, ValueError, KeyError, TypeError):
            STATES[folder.name] = {'status': 'error', 'message': 'Saved job details could not be read. Check the job folder.'}


def decode_image(data):
    if not isinstance(data, str) or ',' not in data:
        raise ValueError('Choose a JPG, PNG or WebP image.')
    try:
        raw = base64.b64decode(data.split(',', 1)[1], validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Each image must be smaller than 8 MB.')
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Use JPG, PNG or WebP images.')
            if im.width * im.height > 25_000_000 or min(im.size) < 128:
                raise ValueError('Use images at least 128 pixels wide and high, and below 25 megapixels.')
            im = ImageOps.exif_transpose(im)
            rgba = im.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            background.alpha_composite(rgba)
            result = background.convert('RGB')
            result.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            return result
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('That image could not be read. Try another JPG, PNG or WebP.') from exc


def worker(job_id):
    global BUSY
    folder = JOBS / job_id
    try:
        metadata = ENGINE.generate(folder)
        with LOCK:
            STATES[job_id] = {'status': 'done', 'seconds': round(metadata['seconds'], 1),
                             'image': f'/jobs/{job_id}/result.png',
                             'receipt': f'/jobs/{job_id}/run.json'}
    except Exception:
        (folder / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
        with LOCK:
            STATES[job_id] = {'status': 'error', 'message': 'Generation failed. Your inputs are saved. Check the server log before retrying.'}
    finally:
        with LOCK:
            BUSY = False


class Handler(BaseHTTPRequestHandler):
    def log_request(self, code='-', size='-'):
        # Never include URL queries (or credentials mistakenly entered there) in logs.
        self.log_message('%s %s %s', self.command, urlsplit(self.path).path, code)

    def send(self, code, body, content_type='application/json', headers=None):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def cookie_token(self):
        try:
            cookies = SimpleCookie(self.headers.get('Cookie', ''))
            return cookies['klara_session'].value if 'klara_session' in cookies else ''
        except Exception:
            return ''

    def authenticated(self):
        self.session = AUTH.session(self.cookie_token()) if AUTH else {'id': 'local-owner', 'role': 'owner', 'csrf': TOKEN}
        if self.session:
            return True
        if self.path.startswith('/api/') or self.path.startswith('/jobs/'):
            self.send(401, {'error': 'Your session has ended. Please sign in again.'})
        else:
            self.send(302, b'', headers={'Location': '/signin'})
        return False

    def may_read_job(self, job_id):
        return self.session['role'] == 'owner' or OWNERS.get(job_id) == self.session['id']

    def login(self):
        if not AUTH:
            return self.send(200, {'ok': True})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                return self.send(413, {'error': 'Enter your access code.'})
            self.connection.settimeout(15)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError()
            client = self.headers.get('X-Real-IP', self.client_address[0]) if self.client_address[0] in ('127.0.0.1', '::1') else self.client_address[0]
            session, status = AUTH.login(payload.get('password'), client[:100])
        except (ValueError, OSError, TypeError):
            return self.send(400, {'error': 'Enter your access code.'})
        if not session:
            message = 'Too many attempts. Please try again in 15 minutes.' if status == 429 else 'That code is incorrect, not active yet, or has expired.'
            return self.send(status, {'error': message})
        return self.send(200, {'ok': True}, headers={'Set-Cookie': 'klara_session=' + session['token'] + '; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=43200'})

    def do_GET(self):
        path = urlsplit(self.path).path
        public = {'/signin': WEB / 'signin.html', '/login.js': WEB / 'login.js', '/style.css': WEB / 'style.css',
                  '/credits': WEB / 'credits.html', '/signin-person.jpg': WEB / 'samples/person-11246_00.jpg',
                  '/signin-garment.jpg': WEB / 'samples/garment-01095_00.jpg',
                  '/licenses/viton-hd.txt': ROOT / 'LICENSE-VITON-HD.txt',
                  '/licenses/catvton.txt': ROOT / 'vendor/catvton-maskfree/LICENSE'}
        if path in public:
            target = public[path]
            if not target.is_file():
                return self.send(404, {'error': 'Not found'})
            return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'text/html')
        if not self.authenticated():
            return
        if path == '/api/status':
            with LOCK:
                value = {'app': 'klara-studio', 'version': APP_VERSION, 'gpu': GPU, 'busy': BUSY,
                         'token': self.session['csrf'], 'role': self.session['role'],
                         'model': 'Klara attention-adapted CatVTON-MaskFree'}
            return self.send(200, value)
        if path == '/api/examples':
            return self.send(200, (WEB / 'samples/catalog.json').read_bytes())
        if path == '/api/presentation-examples':
            return self.send(200, (WEB / 'presentation-examples.json').read_bytes())
        if path == '/sample-audit':
            return self.send(302, b'', headers={'Location': '/sample-audit/'})
        if path.startswith('/sample-audit/'):
            relative = path[len('/sample-audit/'):] or 'index.html'
            safe = relative in ('index.html', 'findings.csv', 'reviewed-audit.json') or re.fullmatch(r'images/(?:(?:person|garment)-[0-9]{5}_00\.jpg|[0-9]{5}_00--[0-9]{5}_00\.png)', relative)
            target = ROOT / 'sample-audit' / relative
            if not safe or not target.is_file():
                return self.send(404, {'error': 'Not found'})
            return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        if path.startswith('/api/jobs/'):
            with LOCK:
                job_id = path.rsplit('/', 1)[-1]
                value = STATES.get(job_id) if self.may_read_job(job_id) else None
            return self.send(200 if value else 404, value or {'error': 'Job not found. The server may have restarted.'})
        allowed = {'/': WEB / 'index.html', '/app.js': WEB / 'app.js', '/style.css': WEB / 'style.css'}
        for name in ('person.jpg', 'garment.jpg', 'result.png'):
            allowed['/example/' + name] = WEB / 'example' / name
        if re.fullmatch(r'/samples/(person|garment)-[0-9]{5}_00\.jpg', path):
            allowed[path] = WEB / path.lstrip('/')
        parts = path.strip('/').split('/')
        if len(parts) == 3 and parts[0] == 'jobs' and parts[1] in STATES and self.may_read_job(parts[1]) and parts[2] in ('result.png', 'run.json'):
            allowed[path] = JOBS / parts[1] / parts[2]
        target = allowed.get(path)
        if target is None or not target.is_file():
            return self.send(404, {'error': 'Not found'})
        self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')

    def do_POST(self):
        global BUSY
        origin = self.headers.get('Origin')
        if AUTH and origin and origin not in ('https://klara-app.de', 'https://www.klara-app.de'):
            return self.send(403, {'error': 'Open Klara in its own browser tab.'})
        path = urlsplit(self.path).path
        if path == '/api/login':
            return self.login()
        if not self.authenticated():
            return
        if not secrets.compare_digest(self.headers.get('X-Klara-Token', ''), self.session['csrf']):
            return self.send(403, {'error': 'Refresh the page and try again.'})
        if path == '/api/logout':
            if AUTH:
                AUTH.logout(self.cookie_token())
            return self.send(200, {'ok': True}, headers={'Set-Cookie': 'klara_session=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0'})
        if path != '/api/generate':
            return self.send(404, {'error': 'Not found'})
        if not GPU:
            return self.send(503, {'error': 'Generation is available on the GPU VM. This computer is in preview mode.'})
        reserved = False
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                return self.send(413, {'error': 'Upload two images smaller than 8 MB each.'})
            with LOCK:
                if BUSY:
                    return self.send(409, {'error': 'A generation is already running. Please wait.'})
                BUSY = True
                reserved = True
            self.connection.settimeout(30)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Invalid request.')
            if payload.get('permission') is not True:
                raise ValueError('Confirm that you have permission to use both images.')
            person = decode_image(payload.get('person'))
            garment = decode_image(payload.get('garment'))
            job_id = uuid.uuid4().hex
            folder = JOBS / job_id
            folder.mkdir(parents=True)
            person.save(folder / 'person.png')
            garment.save(folder / 'garment.png')
            (folder / 'job.json').write_text(json.dumps({'id': job_id, 'status': 'submitted', 'owner': self.session['id']}), encoding='utf-8')
            with LOCK:
                OWNERS[job_id] = self.session['id']
                STATES[job_id] = {'status': 'running'}
            threading.Thread(target=worker, args=(job_id,), daemon=True).start()
        except (ValueError, TypeError, OSError) as exc:
            if reserved:
                with LOCK:
                    BUSY = False
            return self.send(400, {'error': str(exc)})
        self.send(202, {'job_id': job_id})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--preview', action='store_true', help='Recorded examples only; never load the GPU model')
    args = parser.parse_args()
    auth_path = os.environ.get('KLARA_AUTH_CONFIG')
    if auth_path:
        AUTH = Auth(auth_path)
    elif not args.preview:
        parser.error('KLARA_AUTH_CONFIG is required for the live studio. Use --preview for a local preview without generation.')
    # systemd Restart=on-failure retries in a fresh process if CUDA is not ready at boot.
    GPU = generation_ready(args.preview)
    from studio_engine import Engine
    ENGINE = Engine()
    restore_jobs()
    print(f'Klara: http://127.0.0.1:{args.port} | GPU ready: {GPU}', flush=True)
    print('Uploads are retained in studio-jobs. HTTPS and access-code sessions protect the public demo.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
