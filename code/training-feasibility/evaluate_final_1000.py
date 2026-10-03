"""Frozen two-model final test; no training or automatic output retries."""
import argparse, json, random, time, traceback
from pathlib import Path
from common import ROOT, sha
from edit_data import training_images
from data_final_1000 import checked_setup, PLAN_SHA

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan,value,cases,folders = checked_setup()
    print(json.dumps({'stage':'preflight-passed','cases':len(cases),'attempts':128,'plan_sha256':PLAN_SHA}),flush=True)
    if args.preflight:
        return
    output = ROOT/'final-test-1000-01'
    if output.exists():
        raise ValueError('Output already exists; preserve it, no automatic retries')
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from models import load_pipeline
    from training_core import apply_adapter
    load_started = time.monotonic()
    pipe,revisions = load_pipeline()
    if revisions != value['expected_revisions'] or pipe.skip_safety_check or pipe.safety_checker is None:
        raise ValueError('Model revisions/safety configuration mismatch')
    pipe.unet.eval()
    output.mkdir()
    (output/'evaluation-plan.json').write_bytes((ROOT/'final-evaluation-plan.json').read_bytes())
    prepared = {}
    reference_hashes = {}
    for case in cases:
        person,garment,target = training_images(case)
        prepared[case['id']] = (person,garment)
        folder = output/'references'/case['id']
        folder.mkdir(parents=True)
        images = {'person':person,'garment':garment,'target':target}
        for name in ('edited_mask','target_mask'):
            with Image.open(case['edit_paths'][name]) as im:
                images[name] = im.convert('L').resize((384,512),Image.Resampling.NEAREST)
        reference_hashes[case['id']] = {}
        for name,image in images.items():
            path = folder/(name+'.png')
            image.save(path)
            reference_hashes[case['id']][name] = sha(path)
    with Image.open(ROOT/'vendor/catvton-maskfree/resource/img/NSFW.jpg') as im:
        warning = np.asarray(im.resize((384,512)).convert('RGB'))
    record = {'status':'started','plan':plan,'plan_sha256':PLAN_SHA,'revisions':revisions,
              'evaluator_sha256':sha(__file__),'data_provenance':value['data_provenance'],
              'reference_hashes':reference_hashes,'torch':torch.__version__,
              'safety_checker':True,'model_load_seconds':time.monotonic()-load_started,'records':[]}
    started = time.monotonic()
    try:
        for variant in plan['variants']:
            if variant != 'pretrained':
                apply_adapter(pipe.unet, load_file(str(folders[variant]/'attention-adapter.safetensors')))
            print(json.dumps({'stage':'variant-started','variant':variant}),flush=True)
            for case in cases:
                person,garment = prepared[case['id']]
                for seed in plan['seeds']:
                    if time.monotonic()-started > 25*60:
                        raise TimeoutError('25-minute generation ceiling reached')
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    before = time.monotonic()
                    with torch.inference_mode():
                        image = pipe(image=person,condition_image=garment,width=384,height=512,
                                     num_inference_steps=20,guidance_scale=2.5,eta=1.0,
                                     generator=torch.Generator('cuda').manual_seed(seed))[0]
                    flagged = np.array_equal(np.asarray(image.convert('RGB')),warning)
                    row = {'case_id':case['id'],'variant':variant,'seed':seed,
                           'status':'safety-excluded' if flagged else 'ok',
                           'seconds':time.monotonic()-before}
                    if not flagged:
                        name = f"{case['id']}-{variant}-{seed}.png"
                        image.save(output/name)
                        row.update(file=name,sha256=sha(output/name))
                    record['records'].append(row)
                    tmp = output/'results.tmp'
                    tmp.write_text(json.dumps(record,indent=2))
                    tmp.replace(output/'results.json')
                    print(json.dumps({**row,'attempt':len(record['records']),'total':128}),flush=True)
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (output/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        record['generation_seconds'] = time.monotonic()-started
        (output/'results.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'status':record['status'],'attempts':len(record['records']),
                      'generation_seconds':record['generation_seconds']}),flush=True)


if __name__ == '__main__':
    main()
