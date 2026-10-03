"""Isolated 1000-case adaptation: three shuffled epochs, unchanged denoising loss."""
import argparse
import json
import time
import traceback
from common import ROOT, SIZE, reviewed, sha


def main(expanded=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--steps', type=int, default=3000 if expanded else 20)
    parser.add_argument('--output', default='training-1000-01' if expanded else 'training-run-01')
    parser.add_argument('--preservation-weight', type=float, default=0.0)
    parser.add_argument('--edit-data', default=str(ROOT/'edit-data') if expanded else None,
                        help='Verified VITON-HD-edit folder; replaces the original synthetic examples')
    args = parser.parse_args()
    if not 0 <= args.preservation_weight <= 1:
        parser.error('This pilot allows preservation weight between zero and one')
    if expanded and (args.steps != 3000 or args.preservation_weight):
        parser.error('Frozen 1000-case experiment requires exactly 3000 updates and standard denoising')
    if not expanded and not 1 <= args.steps <= 50:
        parser.error('This feasibility script is limited to 1-50 optimizer updates')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == ROOT.resolve():
        parser.error('Output must be a new folder inside this kit')
    if args.edit_data:
        if args.preservation_weight:
            parser.error('Edit-data compatibility run uses standard denoising only; original semantic-region masks do not apply')
        from edit_data import load_edit_batch, training_images
        if expanded:
            from data_1000 import load_1000_batch, epoch_indices
            value = load_1000_batch(args.edit_data)
        else:
            value = load_edit_batch(args.edit_data)
    else:
        value = reviewed()  # Original review checks remain unchanged.
    mask_plan = None
    if args.preservation_weight > 0:
        from common import local
        mask_plan = json.loads((ROOT / 'preservation-plan.json').read_text())
        mask_lookup = {c['id']: c for c in mask_plan['cases']}
        if set(mask_lookup) != {c['id'] for c in value['cases']}:
            raise ValueError('Preservation masks do not match approved cases')
        for case in mask_lookup.values():
            for region in case['regions'].values():
                if sha(local(region['path'])) != region['sha256']:
                    raise ValueError('Preservation mask changed')
    if args.preflight:
        print(json.dumps({'reviewed_cases': len(value['cases']),
                          'train': sum(r['split'] == 'train' for r in value['cases']),
                          'validation': sum(r['split'] == 'validation' for r in value['cases']),
                          'excluded_cases': value['excluded_cases'],
                          'steps': args.steps, 'preservation_weight': args.preservation_weight,
                          'scope': '1000-case attention adaptation' if expanded else 'feasibility',
                          'data_provenance': value.get('data_provenance', {'dataset':'original reviewed synthetic feasibility examples'}),
                          'downloads': False, 'gpu_started': False}))
        return
    if output.exists():
        raise ValueError('Output exists. Preserve previous results and choose a new name.')
    import torch
    import diffusers, transformers, accelerate, platform
    from PIL import Image
    from diffusers import DDPMScheduler
    from safetensors.torch import save_file, load_file
    from models import load_pipeline
    from training_core import configure_attention, frozen_digest, denoising_loss, adapter_state, apply_adapter
    torch.manual_seed(9026)
    load_started = time.monotonic()
    pipe, revisions = load_pipeline()
    if expanded and revisions != value['expected_revisions']:
        raise ValueError('Loaded model revisions differ from expanded protocol')
    load_seconds = time.monotonic() - load_started
    from utils import prepare_image
    output.mkdir()
    record = {'status': 'started', 'scope': '1000-case exploratory attention adaptation' if expanded else 'training implementation feasibility only',
               'plan_sha256': value['data_provenance']['plan_sha256'] if args.edit_data else sha(ROOT / 'plan.json'),
               'review_sha256': None if args.edit_data else sha(ROOT / 'examples/review.csv'),
               'data_provenance': value.get('data_provenance'),
               'supplement_sha256': value.get('supplement_sha256'),
              'case_ids': [r['id'] for r in value['cases']], 'excluded_cases': value['excluded_cases'],
              'revisions': revisions, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'model_load_seconds': load_seconds,
              'python': platform.python_version(),
              'packages': {module.__name__: module.__version__ for module in (diffusers, transformers, accelerate)},
              'resolution': SIZE, 'steps_requested': args.steps, 'seed': 9026,
              'optimizer': 'AdamW', 'learning_rate': 1e-5, 'weight_decay': .01,
              'condition_dropout': .1, 'preservation_loss': args.preservation_weight > 0,
              'preservation_weight': args.preservation_weight,
              'preservation_objective': 'Equal-region weighted noise MSE on protected person latents; a proxy, not decoded image error.',
              'preservation_plan_sha256': sha(ROOT / 'preservation-plan.json') if mask_plan else None,
              'method': 'Proposed standard denoising objective; not a reproduction of DREAM training.',
              'frozen_weights_unchanged': False, 'adapter_reload_verified': False,
              'quality_improvement_established': False}
    (output / 'run.json').write_text(json.dumps(record, indent=2))
    started = time.monotonic()
    try:
        pipe.vae.requires_grad_(False).eval()
        pipe.safety_checker.requires_grad_(False).eval()
        pipe.unet.float().train()
        selected = configure_attention(pipe.unet)
        pipe.unet.enable_gradient_checkpointing()
        scheduler = DDPMScheduler.from_config(pipe.noise_scheduler.config)
        if pipe.unet.config.in_channels != 8:
            raise ValueError('Expected the eight-channel mask-free U-Net')
        record['prediction_type'] = scheduler.config.prediction_type
        record['epochs'] = 3
        record['initialization'] = 'pinned pretrained; not the 63-case checkpoint'
        record['sampling'] = 'seeded shuffled full passes'
        record['runner_sha256'] = sha(__file__)
        record['training_core_sha256'] = sha(ROOT/'training_core.py')
        record['trainable_parameters'] = sum(p.numel() for p in selected.values())
        if expanded and record['trainable_parameters'] != 49574080:
            raise ValueError('Expanded protocol trainable parameter count differs')
        record['total_unet_parameters'] = sum(p.numel() for p in pipe.unet.parameters())
        record['frozen_before_sha256'] = frozen_digest(pipe.unet)
        initial = adapter_state(selected)
        cached = {'train': [], 'validation': []}
        cached_regions = {'train': [], 'validation': []}
        for case_index, case in enumerate(value['cases']):
            if time.monotonic() - started > 40 * 60:
                raise TimeoutError('40-minute processing limit during caching')
            encoded = []
            if args.edit_data:
                images = training_images(case)
            else:
                images = []
                for filename in ('source.png', 'garment-a.png', 'target.png'):
                    with Image.open(ROOT / 'examples' / case['id'] / filename) as image:
                        images.append(image.convert('RGB'))
            for image in images:
                pixels = prepare_image(image).to('cuda', dtype=pipe.vae.dtype)
                with torch.no_grad():
                    latent = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
                encoded.append(latent.detach())
            cached[case['split']].append([tensor.cpu() for tensor in encoded])
            if (case_index + 1) % 50 == 0:
                print(json.dumps({'stage':'latent-cache', 'cases':case_index+1, 'total':len(value['cases'])}), flush=True)
            if mask_plan:
                import numpy as np
                arrays = []
                for region in mask_lookup[case['id']]['regions'].values():
                    with Image.open(local(region['path'])) as mask_image:
                        arrays.append((np.asarray(mask_image.convert('L')) > 0).astype('float32'))
                masks = torch.from_numpy(np.stack(arrays)).unsqueeze(0).to('cuda')
                # Only fully protected latent cells contribute; mixed boundary cells are omitted.
                masks = (torch.nn.functional.adaptive_avg_pool2d(masks, encoded[0].shape[-2:]) > .999).float()
                if not masks.any():
                    raise ValueError('No protected latent cells: ' + case['id'])
                cached_regions[case['split']].append(masks)
        optimizer = torch.optim.AdamW(list(selected.values()), lr=1e-5, weight_decay=.01, foreach=False)
        def validation_loss():
            values = []
            pipe.unet.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for index, item in enumerate(cached['validation']):
                    generator = torch.Generator('cuda').manual_seed(8100 + index)
                    values.append(float(denoising_loss(pipe.unet, scheduler, *[x.to('cuda') for x in item], generator, dropout=0)))
            pipe.unet.train()
            return sum(values) / len(values)
        record['validation_denoising_mse_before'] = validation_loss()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator('cuda').manual_seed(9026)
        order = epoch_indices(len(cached['train']))
        if len(order) != args.steps:
            raise ValueError('Training order length differs from protocol')
        training_ids = [c['id'] for c in value['cases'] if c['split']=='train']
        for step in range(args.steps):
            if time.monotonic() - started > 40 * 60:
                raise TimeoutError('40-minute processing limit reached before another training update')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            before = time.monotonic()
            item = [x.to('cuda') for x in cached['train'][order[step]]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                masks_for_step = cached_regions['train'][step % len(cached['train'])] if mask_plan else None
                loss = denoising_loss(pipe.unet, scheduler, *item, generator,
                    preservation_masks=masks_for_step, preservation_weight=args.preservation_weight)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            grads = [p.grad for p in selected.values() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise ValueError('Missing or non-finite attention gradients')
            norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0)
            if not torch.isfinite(norm) or norm.item() <= 0:
                raise ValueError('Zero or non-finite gradient norm')
            optimizer.step()
            torch.cuda.synchronize()
            event = {'step': step + 1, 'loss': loss.item(), 'grad_norm_before_clip': norm.item(),
                     'training_case_id': training_ids[order[step]], 'epoch': step // len(cached['train']) + 1,
                     'seconds': time.monotonic() - before,
                     'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            with (output / 'steps.jsonl').open('a') as log:
                log.write(json.dumps(event) + '\n')
            if (step + 1) % 25 == 0:
                print(json.dumps(event), flush=True)
            if (step + 1) % 1000 == 0:
                epoch_file = output / ('epoch-' + str((step+1)//1000) + '.safetensors')
                save_file(adapter_state(selected), str(epoch_file), metadata={'scope':'intermediate recovery only; final checkpoint prespecified'})
        record['validation_denoising_mse_after'] = validation_loss()
        record['frozen_after_sha256'] = frozen_digest(pipe.unet)
        if record['frozen_before_sha256'] != record['frozen_after_sha256']:
            raise ValueError('Frozen U-Net weights changed')
        record['frozen_weights_unchanged'] = True
        state = adapter_state(selected)
        change = sum((state[k] - initial[k]).abs().sum().item() for k in state)
        record['mean_absolute_attention_update'] = change / record['trainable_parameters']
        if change <= 0:
            raise ValueError('No attention weight update occurred')
        adapter = output / 'attention-adapter.safetensors'
        save_file(state, str(adapter), metadata={'scope': record['scope'], 'plan_sha256': record['plan_sha256']})
        apply_adapter(pipe.unet, initial)
        reloaded = load_file(str(adapter))
        apply_adapter(pipe.unet, reloaded)
        if not all(torch.equal(p.detach().cpu(), reloaded[name]) for name, p in selected.items()):
            raise ValueError('Adapter reload mismatch')
        record['adapter_reload_verified'] = True
        record['adapter_sha256'] = sha(adapter)
        record['steps_completed'] = args.steps
        if expanded:
            # Matched image generation is a separate evaluation phase. Avoid an
            # extra unplanned validation image or treating denoising loss as quality.
            record['status'] = 'adaptation-complete-evaluation-pending'
            print(json.dumps({'status': record['status'], 'steps_completed': args.steps,
                              'adapter_sha256': record['adapter_sha256']}), flush=True)
            return
        record['status'] = 'training-checks-passed-inference-pending'
        (output / 'run.json').write_text(json.dumps(record, indent=2))
        # Release optimizer and cached tensors before the safety-enabled inference.
        optimizer.zero_grad(set_to_none=True)
        del optimizer, cached, initial, state, reloaded, selected, grads, loss, item, encoded, pixels, latent
        pipe.unet.eval().to(dtype=torch.bfloat16)
        pipe.unet.disable_gradient_checkpointing()
        torch.cuda.empty_cache()
        case = next(r for r in value['cases'] if r['split'] == 'validation')
        from common import local
        if args.edit_data:
            person, garment, _ = training_images(case)
        else:
            person = Image.open(local(case['person'])).convert('RGB')
            garment = Image.open(local(case['garment_b'])).convert('RGB')
        with torch.inference_mode():
            result = pipe(image=person, condition_image=garment, width=SIZE[0], height=SIZE[1],
                          num_inference_steps=20, guidance_scale=2.5,
                          generator=torch.Generator('cuda').manual_seed(9026))[0]
        result.save(output / 'reload-inference.png')
        record.update(status='feasibility-passed', inference_case=case['id'], inference_seed=9026,
                      inference_steps=20, inference_output_sha256=sha(output / 'reload-inference.png'),
                      inference_note=('Edited person + original catalog garment; target is original photograph. ' if args.edit_data else 'Real person A + different garment B. ') +
                      'This one output is not a quality assessment. The safety checker remains enabled.')
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (output / 'error.log').write_text(traceback.format_exc())
        raise
    finally:
        record['processing_seconds'] = time.monotonic() - started
        (output / 'run.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
