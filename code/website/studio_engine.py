"""GPU inference using Klara's verified attention-adapted CatVTON checkpoint."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER_SHA256 = 'e4f332704879fd3929120c4c038cbb65837c8a9c694bdb9fc75bd5a18924de38'
MODEL_NAME = 'Klara attention-adapted CatVTON-MaskFree'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Engine:
    def __init__(self):
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        checkpoint = Path(os.environ.get('KLARA_ADAPTER_PATH', str(ROOT / 'models/attention-adapter.safetensors')))
        if not checkpoint.is_file() or sha(checkpoint) != ADAPTER_SHA256:
            raise RuntimeError('The verified Klara trained checkpoint is missing or has changed.')
        import torch
        from huggingface_hub import snapshot_download
        if not torch.cuda.is_available():
            raise RuntimeError('Generation requires the GPU VM.')
        vendor = ROOT / 'vendor' / 'catvton-maskfree'
        self.versions = json.loads((vendor / 'revisions.json').read_text())
        specs = {
            'timbrooks/instruct-pix2pix': ['scheduler/*', 'feature_extractor/*', 'safety_checker/config.json', 'safety_checker/model.safetensors', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'],
            'zhengchong/CatVTON-MaskFree': ['mix-48k-1024/attention/model.safetensors'],
            'stabilityai/sd-vae-ft-mse': ['config.json', 'diffusion_pytorch_model.safetensors'],
        }
        paths = {repo: snapshot_download(repo, revision=self.versions['models'][repo], allow_patterns=patterns)
                 for repo, patterns in specs.items()}
        os.environ['KLARA_CATVTON_VAE'] = paths['stabilityai/sd-vae-ft-mse']
        sys.path.insert(0, str(vendor))
        from model.pipeline import CatVTONPix2PixPipeline
        pipe = CatVTONPix2PixPipeline(
            base_ckpt=paths['timbrooks/instruct-pix2pix'],
            attn_ckpt=paths['zhengchong/CatVTON-MaskFree'],
            attn_ckpt_version='mix-48k-1024', weight_dtype=torch.bfloat16, device='cuda',
            skip_safety_check=False)
        self.apply_trained(pipe, checkpoint)
        pipe.unet.eval()
        self.pipe = pipe  # Publish only after all trained weights have been verified.

    def apply_trained(self, pipe, checkpoint):
        import torch
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device='cpu')
        parameters = dict(pipe.unet.named_parameters())
        expected = {name for name in parameters if '.attn1.' in name}
        if set(state) != expected:
            raise RuntimeError('Trained checkpoint does not match the model attention layers.')
        for name, value in state.items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise RuntimeError('Invalid trained tensor: ' + name)
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
            for name, value in state.items():
                if not torch.equal(parameters[name], value.to(parameters[name].device, dtype=parameters[name].dtype)):
                    raise RuntimeError('Trained weight verification failed: ' + name)
        self.adapter_parameters = sum(value.numel() for value in state.values())

    def generate(self, job):
        import torch
        from PIL import Image
        started = time.monotonic()
        self.load()
        load_seconds = time.monotonic() - started
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with Image.open(job / 'person.png') as p, Image.open(job / 'garment.png') as g:
            with torch.inference_mode():
                output = self.pipe(image=p.convert('RGB'), condition_image=g.convert('RGB'),
                    width=768, height=1024, num_inference_steps=50, guidance_scale=2.5,
                    generator=torch.Generator('cuda').manual_seed(42))[0]
        output.save(job / 'result.png')
        torch.cuda.synchronize()
        metadata = {
            'model': MODEL_NAME, 'base_model': 'zhengchong/CatVTON-MaskFree', **self.versions,
            'adapter_sha256': ADAPTER_SHA256, 'adapter_parameters': self.adapter_parameters,
            'adapter_weights_verified': True, 'training_run': 'training-1000-01',
            'seed': 42, 'steps': 50, 'guidance': 2.5, 'width': 768, 'height': 1024,
            'mask': None, 'safety_checker': True, 'gpu': torch.cuda.get_device_name(),
            'seconds': time.monotonic() - started, 'load_seconds': load_seconds,
            'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
            'person_sha256': sha(job / 'person.png'), 'garment_sha256': sha(job / 'garment.png'),
            'output_sha256': sha(job / 'result.png'),
            'limitations': 'Research prototype. Output may alter identity, clothing details and unrelated regions. No physical fit prediction.',
        }
        (job / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return metadata
