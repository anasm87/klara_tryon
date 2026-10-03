"""Small, testable denoising/adaptation primitives. No weights are downloaded here."""
import hashlib
import torch


def configure_attention(unet):
    # Freeze all U-Net weights first. Frozen layers still calculate outputs.
    unet.requires_grad_(False)
    # Store references to trainable tensors, not copies of their values.
    selected = {}
    # Each parameter is one tensor (e.g. a layer's weight matrix or bias vector),
    # not one scalar weight and not necessarily an entire layer.
    for name, parameter in unet.named_parameters():
        if '.attn1.' in name:
            # Train existing self-attention Q/K/V and output projections.
            # Convolutions, normalization, feed-forward and timestep weights stay frozen.
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError('No self-attention parameters found')
    return selected


def frozen_digest(unet):
    # Create a fingerprint of frozen weights to compare before/after training.
    # SHA-256 is hashing, not encryption; it does not store recoverable weights.
    digest = hashlib.sha256()
    for name, parameter in unet.named_parameters():
        if not parameter.requires_grad:
            # encode() converts the name string to UTF-8 bytes.
            # Each update() feeds more bytes into the same hash calculation.
            digest.update(name.encode())
            # detach() disconnects this view from autograd; it does NOT freeze
            # the original parameter. The condition above already selected frozen ones.
            # Move to CPU and ensure contiguous storage for byte inspection.
            tensor = parameter.detach().cpu().contiguous()
            # Include the dtype name as bytes, e.g. b'torch.float32'.
            digest.update(str(tensor.dtype).encode())
            # view(uint8) reinterprets storage as individual bytes, preserving bits.
            # It is NOT a numerical conversion to integers: float32 occupies 4 bytes.
            # This differs from view(2, 4), which changes the tensor's shape.
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    # One hexadecimal fingerprint for all the names, dtypes and values above.
    return digest.hexdigest()


def regional_noise_loss(prediction, expected, preservation_masks=None, preservation_weight=0.0):
    """Region-weighted noise prediction proxy, not a decoded-image preservation loss.

    Masks have shape B,K,H,W for the person half only. Each nonempty region is
    normalized independently so large backgrounds do not dominate small faces.
    """
    if preservation_weight < 0:
        raise ValueError('Preservation weight must be nonnegative')
    # This function scores predictions; denoising_loss() below prepares inputs
    # and runs the U-Net before calling it.
    # For epsilon prediction, both tensors are [B, 4, H, 2*W] noise canvases.
    # float() numerically converts to float32 for the loss calculation.
    error = (prediction.float() - expected.float()).square()
    # Average across batch, channels and both spatial axes: one scalar MSE.
    base = error.mean()
    if preservation_weight == 0:
        # The latest 1,000-case run uses this path: ordinary noise MSE only.
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
    # Optional regional term: masks cover protected areas of the PERSON half.
    # Ignore empty regions and normalize each region by its own area.
    present = areas > 0
    if not present.any():
        raise ValueError('No protected region at latent resolution')
    per_pixel = error[..., :ww//2].mean(dim=1, keepdim=True)
    regional = (per_pixel * masks).sum(dim=(-2,-1)) / areas.clamp_min(1)
    # Global MSE plus extra weight on protected-region noise errors.
    # This is not a loss between decoded RGB images.
    return base + preservation_weight * regional[present].mean()


def denoising_loss(unet, scheduler, source, garment, target, generator, dropout=.1,
                   preservation_masks=None, preservation_weight=0.0):
    # Prepare ONE training prediction, then delegate its error calculation.
    # Inputs are VAE latents, already encoded/scaled by the training runner.
    # Research resolution: each is [B, 4, 64, 48].
    # source = original person; garment = requested clothing; target = desired person.
    if source.shape != garment.shape or target.shape != source.shape:
        raise ValueError('Latents must share shape')
    # dim=-1 joins WIDTH: [target | garment] -> [B, 4, 64, 96].
    # This clean canvas is the desired latent content we will corrupt with noise.
    clean = torch.cat([target, garment], dim=-1)
    # Occasionally hide the garment CONDITION to train the no-garment CFG branch.
    # The clean canvas above still contains the garment. Batch size is 1 in our run.
    drop = torch.rand((), device=source.device, generator=generator).item() < dropout
    condition = torch.cat([source, torch.zeros_like(garment) if drop else garment], dim=-1)
    # Fresh Gaussian noise at every coordinate, covering BOTH clean-canvas halves.
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    # One noise-level index per example, shared by all its spatial positions/channels.
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],),
                             device=clean.device, generator=generator)
    # z_t = sqrt(alpha_bar_t)*clean + sqrt(1-alpha_bar_t)*noise.
    # The conditioning canvas stays clean; there is no iterative denoising loop here.
    noisy = scheduler.add_noise(clean, noise, timesteps)
    # dim=1 joins CHANNELS: noisy(4) + condition(4) -> [B, 8, 64, 96].
    model_input = torch.cat([noisy, condition], dim=1)
    # Reentrant gradient checkpointing needs a differentiable input even though
    # the VAE and early convolution weights are frozen.
    if torch.is_grad_enabled():
        model_input.requires_grad_(True)
    # One U-Net forward pass -> [B, 4, 64, 96]. No text conditioning is supplied.
    prediction = unet(model_input, timesteps, encoder_hidden_states=None, return_dict=False)[0]
    mode = scheduler.config.prediction_type
    if mode == 'epsilon':
        # Our noise-prediction objective: the answer is the exact noise we added.
        expected = noise
    elif mode == 'v_prediction':
        # Compatibility branch for a different diffusion prediction parameterization.
        expected = scheduler.get_velocity(clean, noise, timesteps)
    else:
        raise ValueError('Unsupported scheduler prediction_type: ' + mode)
    # Return one scalar loss. The caller later runs backward() and optimizer.step().
    return regional_noise_loss(prediction, expected, preservation_masks, preservation_weight)


def adapter_state(selected):
    # Snapshot only the selected attention tensors for saving. clone() makes
    # independent storage, so later training cannot change this saved snapshot.
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
