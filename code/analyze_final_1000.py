"""Recompute the frozen final-test metrics from all recorded attempts; CPU only."""
import argparse, collections, csv, hashlib, json
from pathlib import Path
import numpy as np
import skimage
from PIL import Image
from analyze_edit_evaluation import metrics, load_image
from analyze_1000_evaluation import summarize, NAMES

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def analyze(out, plan_file, evaluator):
    out=Path(out).resolve(); plan_file=Path(plan_file)
    plan=json.loads(plan_file.read_text())
    r=json.loads((out/'results.json').read_text())
    if (r['status']!='completed' or r['plan']!=plan or r['plan_sha256']!=sha(plan_file)
        or r['evaluator_sha256']!=sha(evaluator) or not r['safety_checker']):
        raise ValueError('Final protocol/result/evaluator mismatch')
    if sha(Path(__file__).with_name('analyze_edit_evaluation.py'))!=plan['metric_source_sha256']:
        raise ValueError('Metric implementation changed')
    if skimage.__version__!=plan['ssim']['version']:raise ValueError('SSIM version mismatch')
    variants=plan['variants']; ids=plan['case_ids']; seeds=plan['seeds']
    expected={(c,v,s) for c in ids for v in variants for s in seeds}
    actual=[(x['case_id'],x['variant'],x['seed']) for x in r['records']]
    if len(actual)!=plan['expected_attempts'] or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Missing, duplicate or unexpected final attempts')
    refs={}
    for sid in ids:
        refs[sid]={}
        for role in ('person','garment','target','edited_mask','target_mask'):
            p=out/'references'/sid/(role+'.png')
            if sha(p)!=r['reference_hashes'][sid][role]:raise ValueError('Reference changed')
            refs[sid][role]=load_image(p,role.endswith('mask'))
    rows=[]
    for row in r['records']:
        row=dict(row)
        if row['status']=='ok':
            p=(out/row['file']).resolve()
            if not p.is_relative_to(out) or sha(p)!=row['sha256']:raise ValueError('Output changed')
            with Image.open(p) as im:
                im.load()
                if im.size!=tuple(plan['size']):raise ValueError('Output resolution mismatch')
            ref=refs[row['case_id']]
            row['metrics']=metrics(ref['person'],ref['target'],load_image(p),ref['edited_mask'],ref['target_mask'])
        elif row['status']!='safety-excluded':raise ValueError('Unresolved generation error')
        rows.append(row)
    case_means={sid:{} for sid in ids}
    for sid in ids:
        for v in variants:
            selected=[row for row in rows if row['case_id']==sid and row['variant']==v]
            if all(row['status']=='ok' for row in selected):
                case_means[sid][v]={m:float(np.mean([x['metrics'][m] for x in selected]))
                    if all(x['metrics'][m] is not None for x in selected) else None for m in NAMES}
    complete=[sid for sid in ids if all(v in case_means[sid] for v in variants)]
    joint=collections.Counter()
    for sid in complete:
        a=case_means[sid]['adapted-1000'];b=case_means[sid]['pretrained']
        if any(x is None for x in (a['garment_mae'],b['garment_mae'],a['outside_mae_input'],b['outside_mae_input'])):
            joint['metric-unavailable']+=1
        else:
            joint[('garment-better' if a['garment_mae']<b['garment_mae'] else 'garment-not-better')+' / '+
                  ('preservation-better' if a['outside_mae_input']<b['outside_mae_input'] else 'preservation-not-better')]+=1
    analysis={'scope':plan['scope'],'plan_sha256':sha(plan_file),'results_sha256':sha(out/'results.json'),
      'analyzer_sha256':sha(__file__),'metric_source_sha256':plan['metric_source_sha256'],
      'attempts':len(rows),'images_verified':sum(x['status']=='ok' for x in rows),
      'status_by_variant':{v:dict(collections.Counter(x['status'] for x in rows if x['variant']==v)) for v in variants},
      'complete_cases':complete,'excluded_cases':[sid for sid in ids if sid not in complete],
      'case_means':case_means,'summary':summarize(case_means,ids,variants),
      'joint_outcomes':dict(joint),'rows':rows,'scikit_image_version':skimage.__version__,
      'numpy_version':np.__version__,'limitations':plan['limits'],'generation_seconds':r['generation_seconds']}
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2))
    with (out/'per-case-metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','variant','both_seeds_valid',*NAMES]);writer.writeheader()
        for sid in ids:
            for v in variants:writer.writerow({'case_id':sid,'variant':v,'both_seeds_valid':v in case_means[sid],**case_means[sid].get(v,{})})
    print(json.dumps({k:analysis[k] for k in ('attempts','images_verified','excluded_cases','status_by_variant','joint_outcomes','summary')},indent=2))
    return analysis

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent/'reports/viton-edit-1000'
    p.add_argument('--output',type=Path,default=base/'final-test-1000-01')
    p.add_argument('--plan',type=Path,default=base/'final-evaluation-plan.json')
    p.add_argument('--evaluator',type=Path,default=Path(__file__).resolve().parent/'training-feasibility/evaluate_final_1000.py')
    args=p.parse_args();analyze(args.output,args.plan,args.evaluator)
