"""Acquire all preselected final cases; check integrity and development overlap."""
import concurrent.futures, hashlib, json
from pathlib import Path
from PIL import Image
import prepare_1000_data as fetch
from data_final_1000 import PLAN_SHA

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan_file=ROOT/'final-evaluation-plan.json'
    if sha(plan_file)!=PLAN_SHA:raise ValueError('Final plan changed')
    final=json.loads(plan_file.read_text())
    if sha(ROOT/'experiment-plan.json')!=final['parent_plan_sha256']:raise ValueError('Parent plan changed')
    parent=json.loads((ROOT/'experiment-plan.json').read_text())
    rows=parent['candidates']['reserved_test']
    if [r['id'] for r in rows]!=final['case_ids'] or len(rows)!=32:raise ValueError('Final IDs mismatch')
    dev=parent['candidates']['train']+parent['candidates']['validation']
    if {r['group'] for r in rows}&{r['group'] for r in dev}:raise ValueError('Source-group overlap')
    fetch.DATA=ROOT/'final-test-data'
    fetch.DATA.mkdir(exist_ok=True)
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for row in pool.map(fetch.fetch_case,rows):
            results[row['id']]=row
            print(json.dumps({'stage':'final-input-download','cases':len(results),'total':32}),flush=True)
    receipt=[results[r['id']] for r in rows]
    (fetch.DATA/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
    seen=set(); files=0
    for row in receipt:
        for f in row['files']:
            p=fetch.DATA/f['file']
            if sha(p)!=f['sha256']:raise ValueError('Final hash mismatch')
            with Image.open(p) as im:
                im.load()
                if im.size!=(768,1024):raise ValueError('Final dimensions mismatch')
                if p.suffix=='.jpg':
                    seen.add(('byte',f['sha256']))
                    seen.add(('pixel',hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()))
                else:
                    hist=im.convert('L').histogram()
                    area=sum(hist[128:])
                    if area==0 or area==768*1024:raise ValueError('Empty/full mask')
            files+=1
    development=json.loads((ROOT/'edit-data/download-receipt.json').read_text())
    if {r['id'] for r in development}!={r['id'] for r in dev}:raise ValueError('Development coverage changed')
    checked=0
    for row in development:
        for f in row['files']:
            if not f['file'].endswith('.jpg'):continue
            p=ROOT/'edit-data'/f['file']
            if sha(p)!=f['sha256']:raise ValueError('Development hash mismatch')
            with Image.open(p) as im:
                pixel=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if ('byte',f['sha256']) in seen or ('pixel',pixel) in seen:
                raise ValueError('Final/development exact overlap: '+row['id'])
            checked+=1
    result={'status':'passed','plan_sha256':PLAN_SHA,'cases':32,'files_decoded':files,
        'development_images_checked':checked,'cross_split_exact_duplicates':0,
        'receipt_sha256':sha(fetch.DATA/'download-receipt.json'),
        'identity_disjoint':'unverified','generation_started':False}
    (fetch.DATA/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
