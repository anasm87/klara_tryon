"""Verified, CPU-only input adapter for the frozen VITON-HD-edit audit batch."""
import hashlib
import json
from pathlib import Path

from PIL import Image

PLAN_SHA256 = '01bf63173d45e9d00431de682b7350340c7c741df9d4f03b9031d48bd27f7e6b'
SOURCE_REVISION = '9de94ee10e15f5069fd650ce4c914215f124eb6f'
SIZE = (384, 512)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset path outside bundle')
    return path


def load_edit_batch(folder):
    """Validate mapping, frozen assignments and content before importing any model."""
    root = Path(folder).resolve()
    plan_path = root / 'split-plan.json'
    if sha(plan_path) != PLAN_SHA256:
        raise ValueError('Frozen edit split plan changed')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['source_revision'] != SOURCE_REVISION:
        raise ValueError('Unexpected dataset revision')
    manifest = json.loads((root/'batch-manifest.json').read_text(encoding='utf-8'))
    receipt = json.loads((root/'download-receipt.json').read_text(encoding='utf-8'))
    review = json.loads((root/'visual-audit.json').read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in plan['records']}
    expected = {r['id']: r['split'] for r in plan['inspection_batch']}
    for collection in (manifest, receipt):
        if len(collection) != len(expected) or {r['id'] for r in collection} != set(expected):
            raise ValueError('Batch must exactly match the frozen inspection IDs')
    if (set(review['cases']) != set(expected) or review['rejected_cases']
            or review['reviewer_type'] != 'assistant_visual_inspection'):
        raise ValueError('Agent inspection does not cover this batch')
    cases = validated_cases(root, manifest, receipt, lookup, expected,
                            [r['id'] for r in plan['inspection_batch']], {'train':12,'validation':4})
    return {'cases':cases, 'excluded_cases':[], 'data_provenance':{
        'dataset':'NXN-Labs/VITON-HD-edit', 'revision':SOURCE_REVISION,
        'plan_sha256':PLAN_SHA256, 'manifest_sha256':sha(root/'batch-manifest.json'),
        'receipt_sha256':sha(root/'download-receipt.json'),
        'inspection_sha256':sha(root/'visual-audit.json'),
        'inspection_type':'assistant visual audit, not independent human ratings',
        'official_source_split':'test; custom adaptation split used here',
        'identity_disjoint':'unverified', 'preprocessing':'RGB; full-frame Lanczos resize to 384x512; no crop or compositing',
        'inference_mask_required':False}}


def validated_cases(root, manifest, receipt, lookup, expected, order, expected_counts,
                    quality_status='candidate_after_agent_contact_sheet_and_mask_inspection'):
    """Shared file/mapping checks; callers authenticate their immutable protocol first."""
    for collection in (manifest, receipt):
        if len(collection) != len(expected) or {r['id'] for r in collection} != set(expected):
            raise ValueError('Case coverage differs from the frozen selection')
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError('Frozen case order differs from selection')
    receipts = {r['id']: r for r in receipt}
    hashes = {s:set() for s in expected_counts}
    pixel_hashes = {s:set() for s in expected_counts}
    groups = {s:set() for s in expected_counts}
    cases = []
    for row in manifest:
        sid, split = row['id'], row['split']
        if split not in hashes or split != expected[sid] or split != lookup[sid]['split']:
            raise ValueError('Wrong split or test/excluded data in development batch')
        if receipts[sid]['split'] != split:
            raise ValueError('Receipt split mismatch')
        if row['source_files'] != receipts[sid]['files']:
            raise ValueError('Source receipt changed')
        if row['quality_status'] != quality_status:
            raise ValueError('Missing documented inspection')
        groups[split].add(lookup[sid]['group'])
        specs = {
            'person': ('person.jpg', f'test/image-edit/{sid}.jpg', 'hf'),
            'garment': ('garment.jpg', f'test/cloth/{sid}.jpg', 'zip'),
            'target': ('target.jpg', f'test/image/{sid}.jpg', 'zip'),
            'edited_mask': ('edited-mask.png', f'test/image-edit_cloth-mask/{sid}.png', 'hf'),
            'target_mask': ('target-mask.png', f'test/gt_cloth_warped_mask/{sid}.png', 'hf')}
        files = {f['file']: f for f in row['source_files']}
        if len(files) != 5 or len(row['source_files']) != 5:
            raise ValueError('Expected five source files per case')
        paths = {}
        for role, (name, source, provider) in specs.items():
            relative = f'samples/{sid}/{name}'
            if role in ('person','garment','target') and row[role] != relative:
                raise ValueError('Person/garment/target mapping changed')
            path = safe_path(root, relative)
            entry = files.get(relative)
            if not entry or entry['source'] != source or entry['provider'] != provider:
                raise ValueError('Unexpected source mapping')
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('Dataset file changed: ' + relative)
            with Image.open(path) as im:
                if im.size != (768,1024):
                    raise ValueError('Unexpected source dimensions')
                im.load()
                if role in ('person','garment','target'):
                    hashes[split].add(entry['sha256'])
                    pixel_hashes[split].add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
            paths[role] = str(path)
        cases.append({'id': sid, 'split': split, 'group': lookup[sid]['group'],
                      'edit_paths': paths})
    from itertools import combinations
    for a,b in combinations(expected_counts,2):
        if hashes[a]&hashes[b] or pixel_hashes[a]&pixel_hashes[b] or groups[a]&groups[b]:
            raise ValueError('Cross-split duplicate or group overlap')
    counts = {s: sum(c['split'] == s for c in cases) for s in hashes}
    if counts != expected_counts:
        raise ValueError('Unexpected training/validation counts')
    # Use the frozen batch order, not a potentially reordered manifest.
    case_lookup = {c['id']:c for c in cases}
    return [case_lookup[sid] for sid in order]


def training_images(case):
    """Return source, requested garment, real target in the denoising-loss order."""
    images = []
    for role in ('person','garment','target'):
        with Image.open(case['edit_paths'][role]) as im:
            images.append(im.convert('RGB').resize(SIZE, Image.Resampling.LANCZOS))
    return images
