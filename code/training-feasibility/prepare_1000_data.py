"""Download only frozen development IDs; fully decode and audit before training."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import zipfile
import zlib

from PIL import Image, ImageDraw
from fetch_viton_sample import RemoteZip

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'edit-data'
PLAN_SHA = 'c84116c645d1077f2a78482e2f679a380006872f85523e9744b09e4f7b0a4066'
tls = threading.local()


class MemberBufferedZip(RemoteZip):
    """Fetch each ZIP member in one range rather than three small HTTP calls."""
    def __init__(self, limit_mb=700):
        self.buffer_start = 0
        self.buffer = b''
        super().__init__(limit_mb)

    def warm(self, entry):
        self.seek(entry.header_offset)
        self.buffer_start = self.position
        self.buffer = super().read(min(entry.compress_size + 4096, self.size - self.position))

    def read(self, count=-1):
        start = self.position - self.buffer_start
        if count >= 0 and start >= 0 and start + count <= len(self.buffer):
            self.position += count
            return self.buffer[start:start+count]
        return super().read(count)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    raw = (ROOT / 'experiment-plan.json').read_bytes()
    if sha(raw) != PLAN_SHA:
        raise ValueError('Frozen 1000-case plan mismatch')
    return json.loads(raw)


def fetch_case(row):
    sid = row['id']
    files = []
    specs = [('person.jpg', f'test/image-edit/{sid}.jpg', 'hf'),
             ('garment.jpg', f'test/cloth/{sid}.jpg', 'zip'),
             ('target.jpg', f'test/image/{sid}.jpg', 'zip'),
             ('edited-mask.png', f'test/image-edit_cloth-mask/{sid}.png', 'hf'),
             ('target-mask.png', f'test/gt_cloth_warped_mask/{sid}.png', 'hf')]
    for name, source, provider in specs:
        dest = DATA / 'samples' / sid / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            prior = Path.home() / 'klara-edit-expanded/edit-data/samples' / sid / name
            if prior.exists():
                shutil.copyfile(prior, dest)
            else:
                for attempt in range(3):
                    try:
                        if provider == 'hf':
                            url = f"https://huggingface.co/datasets/NXN-Labs/VITON-HD-edit/resolve/{plan()['source_revision']}/{source}"
                            with urllib.request.urlopen(url, timeout=45) as response:
                                raw = response.read(8 * 1024**2 + 1)
                            if len(raw) > 8 * 1024**2:
                                raise ValueError('File exceeds bounded size')
                        else:
                            if not hasattr(tls, 'archive'):
                                tls.remote = MemberBufferedZip(limit_mb=700)
                                tls.archive = zipfile.ZipFile(tls.remote)
                            entry = tls.archive.getinfo(source)
                            if entry.file_size > 8 * 1024**2:
                                raise ValueError('Unexpected archive member size')
                            tls.remote.warm(entry)
                            raw = tls.archive.read(source)
                        tmp = dest.with_suffix(dest.suffix + '.part')
                        tmp.write_bytes(raw)
                        tmp.replace(dest)
                        break
                    except (OSError, RuntimeError):
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
        raw = dest.read_bytes()
        files.append({'file': dest.relative_to(DATA).as_posix(), 'source': source,
                      'provider': provider, 'sha256': sha(raw), 'bytes': len(raw)})
    return {**row, 'files': files}


def audit(receipt):
    seen = {'train': set(), 'validation': set()}
    masks = []
    for row in receipt:
        for f in row['files']:
            path = DATA / f['file']
            if sha(path.read_bytes()) != f['sha256']:
                raise ValueError('Hash mismatch ' + str(path))
            with Image.open(path) as im:
                im.load()
                if im.size != (768, 1024):
                    raise ValueError('Dimension mismatch ' + str(path))
                if path.suffix == '.jpg':
                    seen[row['split']].add(('byte', f['sha256']))
                    seen[row['split']].add(('pixel', sha(im.convert('RGB').tobytes())))
                else:
                    hist = im.convert('L').histogram()
                    coverage = sum(hist[128:]) / (768 * 1024)
                    if coverage == 0 or coverage == 1:
                        raise ValueError('Empty/full mask ' + str(path))
                    masks.append({'id': row['id'], 'file': path.name, 'coverage': coverage})
    if seen['train'] & seen['validation']:
        raise ValueError('Cross-split byte or pixel duplicate; amendment required')
    rows = plan()['candidates']
    groups = [{r['group'] for r in rows[s]} for s in ('train', 'validation', 'reserved_test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Cross-split source group overlap')
    # Predetermined, evenly spaced development sample; never inspect reserved test.
    sample = [rows['train'][i] for i in range(0, 1000, 100)] + [rows['validation'][i] for i in range(0, 64, 16)]
    for page in range(0, len(sample), 4):
        canvas = Image.new('RGB', (800, 1160), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(sample[page:page+4]):
            for col, name in enumerate(('person.jpg', 'garment.jpg', 'target.jpg', 'target-mask.png')):
                with Image.open(DATA / 'samples' / row['id'] / name) as im:
                    canvas.paste(im.convert('RGB').resize((192, 256)), (col * 200, j * 290 + 30))
                draw.text((col * 200, j * 290 + 3), row['id'] + ' ' + name, fill='black')
        canvas.save(DATA / f'qa-{page//4+1}.jpg')
    result = {'status': 'technical-checks-passed-visual-review-pending', 'plan_sha256': PLAN_SHA,
              'cases': len(receipt), 'counts': {'train': 1000, 'validation': 64},
              'files_decoded': sum(len(r['files']) for r in receipt),
              'receipt_sha256': sha((DATA/'download-receipt.json').read_bytes()),
              'cross_split_exact_duplicates': 0, 'identity_disjoint': 'unverified',
              'test_images_downloaded': 0, 'visual_sample_ids': [r['id'] for r in sample],
              'mask_diagnostics': masks}
    (DATA/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'mask_diagnostics'}), flush=True)


def main():
    p = plan()
    selected = p['candidates']['train'] + p['candidates']['validation']
    if len(selected) != 1064 or len({r['id'] for r in selected}) != 1064:
        raise ValueError('Unexpected selection')
    DATA.mkdir(exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_case, row): row['id'] for row in selected}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results[row['id']] = row
            if len(results) % 25 == 0 or len(results) == len(selected):
                print(json.dumps({'downloaded_cases': len(results), 'total': len(selected)}), flush=True)
    receipt = [results[r['id']] for r in selected]
    (DATA/'download-receipt.json').write_text(json.dumps(receipt, indent=2))
    audit(receipt)


if __name__ == '__main__':
    main()
