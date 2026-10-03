"""Regional pixel error and structural similarity for the saved test images."""

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity


def metrics(person, target, output, edited_mask, target_mask):
    if (
        person.shape != target.shape
        or target.shape != output.shape
        or target.ndim != 3
        or (target.shape[2] != 3)
    ):
        raise ValueError("Expected equal RGB image shapes")
    if target_mask.shape != target.shape[:2] or edited_mask.shape != target_mask.shape:
        raise ValueError("Mask dimensions do not match images")
    # Erode the target clothing mask by 3 pixels; exclude the image border.
    garment = (
        np.asarray(
            Image.fromarray((target_mask * 255).astype("uint8")).filter(
                ImageFilter.MinFilter(7)
            )
        )
        > 0
    )
    garment[:3, :] = False
    garment[-3:, :] = False
    garment[:, :3] = False
    garment[:, -3:] = False
    union = Image.fromarray(((edited_mask | target_mask) * 255).astype("uint8"))
    # Protect pixels outside both clothing masks plus an 8-pixel boundary.
    outside = np.asarray(union.filter(ImageFilter.MaxFilter(17))) == 0
    _, ssim_map = structural_similarity(
        target,
        output,
        channel_axis=2,
        data_range=255,
        win_size=7,
        gaussian_weights=False,
        use_sample_covariance=True,
        full=True,
    )

    def mean(x, m):
        return float(x[m].mean()) if m.sum() >= 64 else None

    target_error = (
        np.abs(target.astype(np.float64) - output.astype(np.float64)).mean(axis=2) / 255
    )
    person_error = (
        np.abs(person.astype(np.float64) - output.astype(np.float64)).mean(axis=2) / 255
    )
    return {
        "garment_ssim": mean(ssim_map.mean(axis=2), garment),
        "garment_mae": mean(target_error, garment),
        "outside_mae_input": mean(person_error, outside),
        "outside_mae_target": mean(target_error, outside),
    }


def load_image(path, mask=False):
    with Image.open(path) as im:
        im = im.convert("L" if mask else "RGB").resize(
            (384, 512), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
        )
        return np.asarray(im) >= 128 if mask else np.asarray(im)
