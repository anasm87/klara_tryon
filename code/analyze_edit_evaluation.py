"""Local matched validation analysis; run with workspace scikit-image on PYTHONPATH."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import skimage
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'reports/viton-edit-audit'
OUT=ROOT/'reports/remote-training/edit-validation-01'


def metrics(person,target,output,edited_mask,target_mask):
    if person.shape!=target.shape or target.shape!=output.shape or target.ndim!=3 or target.shape[2]!=3:
        raise ValueError('Expected equal RGB image shapes')
    if target_mask.shape!=target.shape[:2] or edited_mask.shape!=target_mask.shape:
        raise ValueError('Mask dimensions do not match images')
    garment=np.asarray(Image.fromarray((target_mask*255).astype('uint8')).filter(ImageFilter.MinFilter(7)))>0
    garment[:3,:]=False;garment[-3:,:]=False;garment[:,:3]=False;garment[:,-3:]=False
    union=Image.fromarray(((edited_mask|target_mask)*255).astype('uint8'))
    outside=np.asarray(union.filter(ImageFilter.MaxFilter(17)))==0
    _,ssim_map=structural_similarity(target,output,channel_axis=2,data_range=255,
        win_size=7,gaussian_weights=False,use_sample_covariance=True,full=True)
    def mean(x,m):return float(x[m].mean()) if m.sum()>=64 else None
    target_error=np.abs(target.astype(np.float64)-output.astype(np.float64)).mean(axis=2)/255
    person_error=np.abs(person.astype(np.float64)-output.astype(np.float64)).mean(axis=2)/255
    return {'garment_ssim':mean(ssim_map.mean(axis=2),garment),
            'garment_mae':mean(target_error,garment),
            'outside_mae_input':mean(person_error,outside),
            'outside_mae_target':mean(target_error,outside)}


def load_image(path,mask=False):
    with Image.open(path) as im:
        im=im.convert('L' if mask else 'RGB').resize((384,512),Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
        return np.asarray(im)>=128 if mask else np.asarray(im)


def main():
    result=json.loads((OUT/'results.json').read_text());plan=result['plan']
    assert result['status']=='completed' and skimage.__version__==plan['ssim']['version']
    local_plan=ROOT/'training-feasibility/edit-evaluation-plan.json'
    assert hashlib.sha256(local_plan.read_bytes()).hexdigest()==result['plan_sha256']
    expected={(c,v,s) for c in plan['case_ids'] for v in plan['variants'] for s in plan['seeds']}
    actual=[(r['case_id'],r['variant'],r['seed']) for r in result['records']]
    assert len(actual)==len(set(actual))==16 and set(actual)==expected
    rows=[];images={};references={}
    for case in plan['case_ids']:
        folder=DATA/'samples'/case
        references[case]={k:load_image(folder/f) for k,f in [('person','person.jpg'),('garment','garment.jpg'),('target','target.jpg')]}
        references[case]['edited_mask']=load_image(folder/'edited-mask.png',True)
        references[case]['target_mask']=load_image(folder/'target-mask.png',True)
    for r in result['records']:
        row=dict(r)
        if r['status']=='ok':
            p=OUT/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            image=load_image(p);ref=references[r['case_id']]
            images[(r['case_id'],r['variant'],r['seed'])]=image
            row['metrics']=metrics(ref['person'],ref['target'],image,ref['edited_mask'],ref['target_mask'])
        rows.append(row)
    complete=[c for c in plan['case_ids'] if all((c,v,s) in images for v in plan['variants'] for s in plan['seeds'])]
    case_means={};means={};wins={}
    names=['garment_ssim','garment_mae','outside_mae_input','outside_mae_target']
    for c in complete:
        case_means[c]={v:{m:float(np.mean([r['metrics'][m] for r in rows if r['case_id']==c and r['variant']==v]))
                          if all(r['metrics'][m] is not None for r in rows if r['case_id']==c and r['variant']==v) else None
                          for m in names} for v in plan['variants']}
    eligible={m:[c for c in complete if all(case_means[c][v][m] is not None for v in plan['variants'])] for m in names}
    for v in plan['variants']:
        means[v]={m:float(np.mean([case_means[c][v][m] for c in eligible[m]])) if eligible[m] else None for m in names}
    for m in names:
        wins[m]=sum(case_means[c]['adapted'][m]>case_means[c]['pretrained'][m] if m=='garment_ssim'
                    else case_means[c]['adapted'][m]<case_means[c]['pretrained'][m] for c in eligible[m])
    analysis={'complete_cases':complete,'excluded_cases':[c for c in plan['case_ids'] if c not in complete],
              'images_verified':len(images),'means':means,'case_means':case_means,'adapted_case_wins':wins,
              'eligible_cases_by_metric':eligible,'rows':rows,'scikit_image_version':skimage.__version__,
              'numpy_version':np.__version__,'scope':'four-case exploratory validation; not an untouched test'}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2))
    for seed in plan['seeds']:
        canvas=Image.new('RGB',(1200,4*350),'white');draw=ImageDraw.Draw(canvas)
        for i,c in enumerate(plan['case_ids']):
            cells=[('Input',references[c]['person']),('Garment',references[c]['garment']),
                   ('Pretrained',images.get((c,'pretrained',seed))),('Adapted',images.get((c,'adapted',seed))),
                   ('Target',references[c]['target'])]
            for col,(label,array) in enumerate(cells):
                draw.text((col*240+4,i*350+8),f'{c} {label}',fill='black')
                if array is not None:
                    im=Image.fromarray(array);im.thumbnail((234,312));canvas.paste(im,(col*240,i*350+30))
                else:draw.text((col*240+10,i*350+150),'Excluded; not scored',fill='black')
        canvas.save(OUT/f'comparison-{seed}.jpg',quality=94)
    print(json.dumps({k:analysis[k] for k in ['complete_cases','excluded_cases','images_verified','means','adapted_case_wins']}))


if __name__=='__main__':main()
