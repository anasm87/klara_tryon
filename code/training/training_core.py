"""Self-attention adaptation and the final experiment's noise-prediction loss."""

import hashlib
import torch


def configure_attention(unet):
    """Freeze the U-Net, then enable gradients only for existing attn1 weights."""
    unet.requires_grad_(False)
    selected = {}
    for name, parameter in unet.named_parameters():
        if ".attn1." in name:
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError("No self-attention parameters found")
    return selected


def frozen_digest(unet):
    """Fingerprint frozen parameter names, dtypes and bytes for a before/after check."""
    digest = hashlib.sha256()
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            tensor = parameter.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def denoising_loss(unet, scheduler, source, garment, target, generator, dropout=0.1):
    if source.shape != garment.shape or target.shape != source.shape:
        raise ValueError("Latents must share shape")
    # Width: desired person on the left, garment on the right.
    clean = torch.cat([target, garment], dim=-1)
    drop = torch.rand((), device=source.device, generator=generator).item() < dropout
    condition = torch.cat(
        [source, torch.zeros_like(garment) if drop else garment], dim=-1
    )
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (clean.shape[0],),
        device=clean.device,
        generator=generator,
    )
    noisy = scheduler.add_noise(clean, noise, timesteps)
    model_input = torch.cat([noisy, condition], dim=1)
    # Reentrant checkpointing needs a differentiable input to frozen early layers.
    if torch.is_grad_enabled():
        model_input.requires_grad_(True)
    prediction = unet(
        model_input, timesteps, encoder_hidden_states=None, return_dict=False
    )[0]
    mode = scheduler.config.prediction_type
    if mode == "epsilon":
        expected = noise
    else:
        raise ValueError("Unsupported scheduler prediction_type: " + mode)
    return (prediction.float() - expected.float()).square().mean()


def adapter_state(selected):
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in selected.items()
    }


def apply_adapter(unet, state):
    parameters = dict(unet.named_parameters())
    expected = {name for name in parameters if ".attn1." in name}
    if set(state) != expected:
        raise ValueError("Adapter keys do not exactly match the self-attention layers")
    for name, value in state.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid adapter tensor: " + name)
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(
                value.to(parameters[name].device, dtype=parameters[name].dtype)
            )
