"""Pinned model loading; only called by explicit GPU commands."""
import json
import os
import sys
from common import ROOT


def load_pipeline(teacher=False):
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. No model downloads started.')
    vendor = ROOT / 'vendor/catvton-maskfree'
    revisions = json.loads((vendor / 'revisions.json').read_text())
    if teacher:
        teacher_revs = json.loads((ROOT / 'teacher-revisions.json').read_text())
        base_id = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
        attn_id = 'zhengchong/CatVTON'
        pins = {k: v['revision'] for k, v in teacher_revs.items()}
        base_files = ['scheduler/*', 'feature_extractor/*', 'safety_checker/config.json',
                      'safety_checker/pytorch_model.bin', 'unet/config.json', 'unet/diffusion_pytorch_model.bin']
    else:
        base_id = 'timbrooks/instruct-pix2pix'
        attn_id = 'zhengchong/CatVTON-MaskFree'
        pins = revisions['models']
        base_files = ['scheduler/*', 'feature_extractor/*', 'safety_checker/config.json',
                      'safety_checker/model.safetensors', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors']
    vae_id = 'stabilityai/sd-vae-ft-mse'
    base = snapshot_download(base_id, revision=pins[base_id], allow_patterns=base_files)
    attn = snapshot_download(attn_id, revision=pins[attn_id], allow_patterns=['mix-48k-1024/attention/model.safetensors'])
    vae = snapshot_download(vae_id, revision=pins[vae_id], allow_patterns=['config.json', 'diffusion_pytorch_model.safetensors'])
    os.environ['KLARA_CATVTON_VAE'] = vae
    sys.path.insert(0, str(vendor))
    from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline
    cls = CatVTONPipeline if teacher else CatVTONPix2PixPipeline
    result = cls(base_ckpt=base, attn_ckpt=attn,
                 attn_ckpt_version='mix' if teacher else 'mix-48k-1024',
                 weight_dtype=torch.bfloat16, device='cuda', skip_safety_check=False)
    return result, {'code_revision': revisions['code_revision'], 'models': pins}
