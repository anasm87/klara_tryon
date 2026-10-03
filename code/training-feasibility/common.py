"""Manifest checks shared by the bounded feasibility commands. No model imports."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIZE = (384, 512)
EXAMPLE_FILES = {'target.png', 'garment-a.png', 'garment-b.png', 'raw-teacher.png', 'source.png', 'mask.png'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Path outside training kit')
    return target


def plan():
    value = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    cases = value['cases']
    if len(cases) != 12 or len({r['id'] for r in cases}) != 12:
        raise ValueError('Expected the frozen twelve-case feasibility plan')
    ids = {s: {r['person_id'] for r in cases if r['split'] == s} for s in ('train', 'validation')}
    if len(ids['train']) != 8 or len(ids['validation']) != 4 or ids['train'] & ids['validation']:
        raise ValueError('Invalid development split')
    image_hashes = {'train': set(), 'validation': set()}
    for row in cases:
        if row['other_garment_id'] == row['person_id'] or row['other_garment_id'] not in ids[row['split']]:
            raise ValueError('Garments must differ and stay inside the same split')
        for field, expected in row['sha256'].items():
            if sha(local(row[field])) != expected:
                raise ValueError('Input changed: ' + row['id'] + '/' + field)
        image_hashes[row['split']].update(row['sha256'][f] for f in ('person', 'garment_a', 'garment_b'))
    if image_hashes['train'] & image_hashes['validation']:
        raise ValueError('Exact image overlap between train and validation')
    supplement = ROOT / 'supplement.json'
    if supplement.exists():
        extra = json.loads(supplement.read_text(encoding='utf-8'))
        if extra['base_plan_sha256'] != sha(ROOT / 'plan.json') or len(extra['cases']) != 1:
            raise ValueError('Invalid supplemental plan')
        case = extra['cases'][0]
        if (case['split'] != 'train' or case['id'] in {r['id'] for r in cases}
                or case['person_id'] in ids['train'] | ids['validation']
                or case['other_garment_id'] not in ids['train']):
            raise ValueError('Supplement must use a new training person and an existing training garment')
        for field, expected in case['sha256'].items():
            if sha(local(case[field])) != expected:
                raise ValueError('Supplement input changed: ' + field)
        if {case['sha256'][f] for f in ('person', 'garment_a', 'garment_b')} & image_hashes['validation']:
            raise ValueError('Supplement overlaps validation')
        case = {**case, 'supplement_sha256': sha(supplement)}
        value = {**value, 'cases': cases + [case], 'supplement_sha256': sha(supplement)}
    return value


def case_status(case):
    """Check saved artifacts before reusing or excluding a previous attempt."""
    folder = ROOT / 'examples' / case['id']
    receipt, rejection = folder / 'receipt.json', folder / 'rejection.json'
    if receipt.exists() and rejection.exists():
        raise ValueError('Conflicting completion and exclusion records: ' + case['id'])
    if rejection.exists():
        record = json.loads(rejection.read_text())
        if record.get('plan_sha256') != sha(ROOT / 'plan.json') or record.get('id') != case['id']:
            raise ValueError('Exclusion does not match plan: ' + case['id'])
        if record.get('reason') not in {'safety_checker_flagged', 'prior_safety_rejection_reported_by_user'}:
            raise ValueError('Unknown exclusion reason: ' + case['id'])
        if record.get('supplement_sha256') != case.get('supplement_sha256'):
            raise ValueError('Exclusion does not match supplement')
        return 'excluded', record
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get('plan_sha256') != sha(ROOT / 'plan.json') or record.get('case', {}).get('id') != case['id']:
            raise ValueError('Examples were generated from another plan')
        if record.get('case', {}).get('supplement_sha256') != case.get('supplement_sha256'):
            raise ValueError('Examples were generated from another supplement')
        if set(record.get('files', {})) != EXAMPLE_FILES:
            raise ValueError('Incomplete example receipt: ' + case['id'])
        for name, digest in record['files'].items():
            if sha(folder / name) != digest:
                raise ValueError('Generated training file changed: ' + case['id'])
        return 'complete', record
    return ('incomplete' if folder.exists() else 'new'), None


def available_cases(value):
    usable, excluded = [], []
    for case in value['cases']:
        status, record = case_status(case)
        if status == 'complete':
            usable.append(case)
        elif status == 'excluded':
            excluded.append(record)
        else:
            raise ValueError('Unfinished example: ' + case['id'])
    return {**value, 'planned_cases': value['cases'], 'cases': usable, 'excluded_cases': excluded}


def reviewed(require_minimum=True):
    value = plan()
    if not (ROOT / 'examples/review.csv').is_file():
        raise ValueError('No reviewed synthetic examples yet. Run make_examples.py, then inspect review.html and complete review.csv.')
    value = available_cases(value)
    with (ROOT / 'examples/review.csv').open(encoding='utf-8-sig', newline='') as stream:
        ratings = list(csv.DictReader(stream))
    if len(ratings) != len(value['cases']) or {r['id'] for r in ratings} != {r['id'] for r in value['cases']}:
        raise ValueError('Review rows do not match the usable cases from the frozen plan')
    lookup = {r['id']: r for r in ratings}
    accepted = []
    for case in value['cases']:
        row = lookup[case['id']]
        decision = row['approved'].strip().upper()
        if decision not in {'YES', 'NO'} or not row['reviewer'].strip():
            raise ValueError('Human review is incomplete: ' + case['id'])
        folder = ROOT / 'examples' / case['id']
        if sha(folder / 'source.png') != row['source_sha256']:
            raise ValueError('Reviewed source changed: ' + case['id'])
        if decision == 'YES':
            accepted.append(case)
        else:
            value['excluded_cases'].append({'id': case['id'], 'reason': 'human_review_rejected',
                'reviewer': row['reviewer'], 'notes': row.get('notes', ''), 'source_sha256': row['source_sha256']})
    value['cases'] = accepted
    counts = {split: sum(r['split'] == split for r in value['cases']) for split in ('train', 'validation')}
    if require_minimum and (counts['train'] < 6 or counts['validation'] < 2):
        raise ValueError(f"Too few approved examples: {counts['train']} train and {counts['validation']} validation; need at least 6 train and 2 validation for this implementation test")
    return value
