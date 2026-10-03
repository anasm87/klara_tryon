"""Verified paired analysis of all three checkpoints; CPU-only, no generation."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import skimage
from PIL import Image
from analyze_edit_evaluation import metrics, load_image

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'reports/viton-edit-1000'
OUT = BASE/'validation-1000-01'
NAMES = ['garment_mae','garment_ssim','outside_mae_input','outside_mae_target']
VARIANTS = ['pretrained','adapted-63','adapted-1000']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(case_means, ids, variants=VARIANTS):
    summary = {}
    for metric in NAMES:
        eligible = [c for c in ids if c in case_means and all(
            v in case_means[c] and case_means[c][v][metric] is not None for v in variants)]
        values = {v:np.array([case_means[c][v][metric] for c in eligible]) for v in variants}
        result = {'n':len(eligible),'cases':eligible,'models':{},'comparisons':{}}
        for v,a in values.items():
            result['models'][v] = {'mean':float(a.mean()) if len(a) else None,
                                   'median':float(np.median(a)) if len(a) else None}
        for newer,older in [('adapted-1000','pretrained'),('adapted-1000','adapted-63'),('adapted-63','pretrained')]:
            if newer not in variants or older not in variants or not eligible:
                continue
            delta = values[newer]-values[older]
            improvement = delta if metric=='garment_ssim' else -delta
            ci = None
            if len(eligible)>=8:
                rng = np.random.default_rng(41026)
                samples = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
                ci = [float(x) for x in np.quantile(samples,[.025,.975])]
            result['comparisons'][newer+'_vs_'+older] = {
                'mean_delta_new_minus_old':float(delta.mean()),
                'median_paired_delta':float(np.median(delta)),
                'relative_mean_change_percent':float(100*delta.mean()/values[older].mean()) if values[older].mean()!=0 else None,
                'wins':int((improvement>0).sum()),'ties':int((improvement==0).sum()),
                'losses':int((improvement<0).sum()),'descriptive_paired_bootstrap_95_interval':ci}
        summary[metric] = result
    return summary


def main():
    plan_file = BASE/'evaluation-plan.json'
    if sha(plan_file)!=(BASE/'evaluation-plan.sha256').read_text().strip():
        raise ValueError('Evaluation plan changed')
    plan = json.loads(plan_file.read_text())
    result = json.loads((OUT/'results.json').read_text())
    if (result['status']!='completed' or result['plan']!=plan or result['plan_sha256']!=sha(plan_file)
            or result['evaluator_sha256']!=sha(ROOT/'training-feasibility/evaluate_1000.py')
            or sha(ROOT/'analyze_edit_evaluation.py')!=plan['metric_source_sha256']
            or skimage.__version__!=plan['ssim']['version']):
        raise ValueError('Result/protocol/code/metric provenance mismatch')
    expected = {(c,v,s) for c in plan['case_ids'] for v in plan['variants'] for s in plan['seeds']}
    actual = [(r['case_id'],r['variant'],r['seed']) for r in result['records']]
    if len(actual)!=384 or len(set(actual))!=384 or set(actual)!=expected:
        raise ValueError('Incomplete/duplicate evaluation attempts')
    references = {}
    for sid in plan['case_ids']:
        folder = OUT/'references'/sid
        references[sid] = {}
        for role in ('person','garment','target','edited_mask','target_mask'):
            path = folder/(role+'.png')
            if sha(path)!=result['reference_hashes'][sid][role]:
                raise ValueError('Reference hash mismatch')
            references[sid][role] = load_image(path,mask=role.endswith('mask'))
    rows = []
    for row in result['records']:
        item = dict(row)
        if row['status'] not in ('ok','safety-excluded'):
            raise ValueError('Unexpected attempt status')
        if row['status']=='ok':
            path = (OUT/row['file']).resolve()
            if not path.is_relative_to(OUT.resolve()) or sha(path)!=row['sha256']:
                raise ValueError('Output path/hash mismatch')
            with Image.open(path) as im:
                im.load()
                if im.size!=(384,512):
                    raise ValueError('Output dimensions mismatch')
            ref = references[row['case_id']]
            item['metrics'] = metrics(ref['person'],ref['target'],load_image(path),ref['edited_mask'],ref['target_mask'])
        rows.append(item)
    case_means = {}
    for sid in plan['case_ids']:
        case_means[sid] = {}
        for variant in plan['variants']:
            selected = [r for r in rows if r['case_id']==sid and r['variant']==variant]
            if len(selected)==2 and all(r['status']=='ok' for r in selected):
                case_means[sid][variant] = {m:float(np.mean([r['metrics'][m] for r in selected]))
                    if all(r['metrics'][m] is not None for r in selected) else None for m in NAMES}
    complete = [sid for sid in plan['case_ids'] if all(v in case_means[sid] for v in VARIANTS)]
    strata = {'all_64':plan['case_ids'], **plan['validation_strata']}
    joint = {}
    for older in ('pretrained','adapted-63'):
        counts = collections.Counter()
        for sid in complete:
            new = case_means[sid]['adapted-1000']
            old = case_means[sid][older]
            if any(x is None for x in (new['garment_mae'],old['garment_mae'],new['outside_mae_input'],old['outside_mae_input'])):
                counts['metric-unavailable']+=1
                continue
            garment = new['garment_mae']<old['garment_mae']
            preserve = new['outside_mae_input']<old['outside_mae_input']
            counts[('garment-better' if garment else 'garment-not-better')+' / '+('preservation-better' if preserve else 'preservation-not-better')]+=1
        joint[older] = dict(counts)
    analysis = {'scope':plan['scope'],'plan_sha256':sha(plan_file),'results_sha256':sha(OUT/'results.json'),
      'analyzer_sha256':sha(Path(__file__)),'metric_source_sha256':plan['metric_source_sha256'],
      'attempts':384,'images_verified':sum(r['status']=='ok' for r in rows),
      'status_by_variant':{v:dict(collections.Counter(r['status'] for r in rows if r['variant']==v)) for v in VARIANTS},
      'complete_cases':complete,'excluded_cases':[c for c in plan['case_ids'] if c not in complete],
      'case_means':case_means,
      'summaries':{name:summarize(case_means,ids) for name,ids in strata.items()},
      'pairwise_complete_sensitivity':{a+'_vs_'+b:summarize(case_means,plan['case_ids'],[a,b]) for a,b in plan['comparisons']},
      'joint_outcomes_new_vs':joint,'rows':rows,'scikit_image_version':skimage.__version__,
      'numpy_version':np.__version__,'limitations':plan['limits'],'generation_seconds':result['generation_seconds']}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2))
    print(json.dumps({k:analysis[k] for k in ('attempts','images_verified','status_by_variant','excluded_cases','joint_outcomes_new_vs')}))
    print(json.dumps(analysis['summaries']['all_64'],indent=2))


if __name__=='__main__':
    main()
