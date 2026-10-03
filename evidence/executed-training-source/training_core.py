"""Small, testable denoising/adaptation primitives. No weights are downloaded here."""
import hashlib
import torch


def configure_attention(unet):
    unet.requires_grad_(False)
    selected = {}
    for name, parameter in unet.named_parameters():
        if '.attn1.' in name:
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError('No self-attention parameters found')
    return selected


def frozen_digest(unet):
    digest = hashlib.sha256()
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            tensor = parameter.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def regional_noise_loss(prediction, expected, preservation_masks=None, preservation_weight=0.0):
    """Region-weighted noise prediction proxy, not a decoded-image preservation loss.

    Masks have shape B,K,H,W for the person half only. Each nonempty region is
    normalized independently so large backgrounds do not dominate small faces.
    """
    if preservation_weight < 0:
        raise ValueError('Preservation weight must be nonnegative')
    error = (prediction.float() - expected.float()).square()
    base = error.mean()
    if preservation_weight == 0:
        return base
    if preservation_masks is None:
        raise ValueError('Preservation masks required')
    b, _, h, ww = error.shape
    masks = preservation_masks
    if masks.ndim != 4 or masks.shape[0] != b or masks.shape[2:] != (h, ww // 2) or ww % 2:
        raise ValueError('Preservation mask shape mismatch')
    if not torch.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
        raise ValueError('Invalid preservation mask values')
    areas = masks.sum(dim=(-2,-1))
    present = areas > 0
    if not present.any():
        raise ValueError('No protected region at latent resolution')
    per_pixel = error[..., :ww//2].mean(dim=1, keepdim=True)
    regional = (per_pixel * masks).sum(dim=(-2,-1)) / areas.clamp_min(1)
    return base + preservation_weight * regional[present].mean()


def denoising_loss(unet, scheduler, source, garment, target, generator, dropout=.1,
                   preservation_masks=None, preservation_weight=0.0):
    if source.shape != garment.shape or target.shape != source.shape:
        raise ValueError('Latents must share shape')
    clean = torch.cat([target, garment], dim=-1)
    drop = torch.rand((), device=source.device, generator=generator).item() < dropout
    condition = torch.cat([source, torch.zeros_like(garment) if drop else garment], dim=-1)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],),
                             device=clean.device, generator=generator)
    noisy = scheduler.add_noise(clean, noise, timesteps)
    model_input = torch.cat([noisy, condition], dim=1)
    # Reentrant gradient checkpointing needs a differentiable input even though
    # the VAE and early convolution weights are frozen.
    if torch.is_grad_enabled():
        model_input.requires_grad_(True)
    prediction = unet(model_input, timesteps, encoder_hidden_states=None, return_dict=False)[0]
    mode = scheduler.config.prediction_type
    if mode == 'epsilon':
        expected = noise
    elif mode == 'v_prediction':
        expected = scheduler.get_velocity(clean, noise, timesteps)
    else:
        raise ValueError('Unsupported scheduler prediction_type: ' + mode)
    return regional_noise_loss(prediction, expected, preservation_masks, preservation_weight)


def adapter_state(selected):
    return {name: value.detach().cpu().contiguous().clone() for name, value in selected.items()}


def apply_adapter(unet, state):
    parameters = dict(unet.named_parameters())
    expected = {name for name in parameters if '.attn1.' in name}
    if set(state) != expected:
        raise ValueError('Adapter keys do not exactly match the self-attention layers')
    for name, value in state.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError('Invalid adapter tensor: ' + name)
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
