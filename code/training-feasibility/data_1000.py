"""Validate the isolated 1000-case run; never opens the reserved test images."""
import json
from pathlib import Path
from edit_data import sha, validated_cases, SOURCE_REVISION

PLAN_SHA = 'c84116c645d1077f2a78482e2f679a380006872f85523e9744b09e4f7b0a4066'


def load_1000_batch(folder):
    root = Path(folder).resolve()
    plan_file = root.parent/'experiment-plan.json'
    if sha(plan_file) != PLAN_SHA:
        raise ValueError('Frozen experiment plan changed')
    plan = json.loads(plan_file.read_text())
    verification = json.loads((root/'verification.json').read_text())
    review = json.loads((root/'visual-review.json').read_text())
    if (verification['plan_sha256'] != PLAN_SHA
            or verification['status'] != 'technical-checks-passed-visual-review-pending'
            or verification['receipt_sha256'] != sha(root/'download-receipt.json')
            or verification['files_decoded'] != 5320
            or verification['cross_split_exact_duplicates'] != 0):
        raise ValueError('Technical verification incomplete')
    if (review['verification_sha256'] != sha(root/'verification.json')
            or review['sample_ids'] != verification['visual_sample_ids']
            or review['status'] != 'sample-reviewed-no-blocking-technical-anomaly'):
        raise ValueError('Documented sample inspection incomplete')
    rows = plan['candidates']['train'] + plan['candidates']['validation']
    lookup = {r['id']: r for r in rows}
    receipt = json.loads((root/'download-receipt.json').read_text())
    manifest = [{**r, 'person': 'samples/'+r['id']+'/person.jpg',
                 'garment': 'samples/'+r['id']+'/garment.jpg',
                 'target': 'samples/'+r['id']+'/target.jpg',
                 'source_files': r['files'], 'quality_status': 'technical-audit-plus-sampled-visual-review'}
                for r in receipt]
    cases = validated_cases(root, manifest, receipt, lookup,
                            {r['id']:r['split'] for r in rows}, [r['id'] for r in rows],
                            {'train':1000, 'validation':64},
                            quality_status='technical-audit-plus-sampled-visual-review')
    pins = json.loads((root.parent/'vendor/catvton-maskfree/revisions.json').read_text())
    expected = {'code_revision': '7818397f25613beedb3d861a34769f607cfcf3b1', 'models': {
        'timbrooks/instruct-pix2pix': '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d',
        'zhengchong/CatVTON-MaskFree': 'b034d7700d6bf0e8ae1c01f9398f2eb54df5f52a',
        'stabilityai/sd-vae-ft-mse': '31f26fdeee1355a5c34592e401dd41e45d25a493'}}
    if pins != expected or plan['source_revision'] != SOURCE_REVISION:
        raise ValueError('Pinned revisions changed')
    return {'cases':cases, 'excluded_cases':[], 'experiment':plan, 'expected_revisions':expected,
            'data_provenance': {'dataset':'NXN-Labs/VITON-HD-edit', 'revision':SOURCE_REVISION,
                'plan_sha256':PLAN_SHA, 'receipt_sha256':sha(root/'download-receipt.json'),
                'verification_sha256':sha(root/'verification.json'),
                'visual_review_sha256':sha(root/'visual-review.json'),
                'visual_review_coverage':len(review['sample_ids']),
                'identity_disjoint':'unverified', 'official_source_split':'test; custom split',
                'training_count':1000, 'validation_count':64}}


def epoch_indices(count, epochs=3, seed=9026):
    import random
    rng = random.Random(seed)
    order = []
    for _ in range(epochs):
        epoch = list(range(count))
        rng.shuffle(epoch)
        order.extend(epoch)
    return order
