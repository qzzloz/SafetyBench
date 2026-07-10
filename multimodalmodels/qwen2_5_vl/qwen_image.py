"""
Differentiable Qwen2.5-VL image preprocessing for JailBound Stage 2.

To produce a SAVEABLE adversarial image (the guard model does its own
preprocessing, so we cannot hand it Qwen's patch tensor), the attack must
operate in raw pixel space. That requires reproducing Qwen2.5-VL's image
preprocessing - smart_resize -> rescale -> normalize -> patchify - as
differentiable torch ops, so gradients flow from the fused representation back
to the raw [0,1] image.

The patchify order is intricate and version-sensitive, so `selfcheck` compares
this implementation's `pixel_values` against the real HF processor on a clean
image. RUN THE SELF-CHECK FIRST (see crossing.py / the CLI) before trusting the
attack: if it fails, the preprocessing must be fixed before any crossing result
is meaningful.

Constants match Qwen2.5-VL: patch_size=14, merge_size=2, temporal_patch_size=2,
factor=patch_size*merge_size=28, OpenAI-CLIP mean/std.
"""

from __future__ import annotations
import math
from typing import Tuple

PATCH = 14
MERGE = 2
TEMPORAL = 2
FACTOR = PATCH * MERGE          # 28
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(h: int, w: int, factor: int = FACTOR,
                 min_pixels: int = 4 * 28 * 28,
                 max_pixels: int = 1280 * 28 * 28) -> Tuple[int, int]:
    """Qwen's smart_resize: nearest multiple of `factor`, within pixel bounds,
    aspect ratio preserved."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio too extreme for Qwen smart_resize")
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def _pil_to_chw(img):
    """PIL RGB -> (C,H,W) float tensor in [0,1], no torchvision dependency."""
    import numpy as np, torch
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0   # (H,W,C)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()          # (C,H,W)
    return t.clamp(0, 1)


def resized_chw_from_pil(pil_img, min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28):
    """Return a raw [0,1] tensor (C,H,W) resized to Qwen's smart_resize target."""
    from PIL import Image
    img = pil_img.convert("RGB")
    W0, H0 = img.size
    H, W = smart_resize(H0, W0, min_pixels=min_pixels, max_pixels=max_pixels)
    img = img.resize((W, H), Image.BICUBIC)
    return _pil_to_chw(img), (H, W)


def resized_chw_to_grid(pil_img, grid_h: int, grid_w: int):
    """Resize to EXACTLY the processor's chosen patch grid: (grid_h, grid_w) are
    in PATCH units (14px), so target size is (grid_h*14, grid_w*14). This makes
    our pixel_values shape match the HF processor regardless of how min/max
    pixel bounds are parsed - the robust way to stay in lockstep."""
    from PIL import Image
    img = pil_img.convert("RGB")
    H, W = grid_h * PATCH, grid_w * PATCH
    img = img.resize((W, H), Image.BICUBIC)
    return _pil_to_chw(img), (H, W)


def chw_to_pil(t):
    """(C,H,W) float [0,1] -> PIL image, no torchvision dependency."""
    import numpy as np
    from PIL import Image
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
    return Image.fromarray(arr.round().astype("uint8"))


def patchify(img_chw, device=None, dtype=None):
    """Differentiable normalize + patchify. img_chw: (C,H,W) in [0,1], H,W % 28 == 0.
    Returns (flatten_patches (grid_t*grid_h*grid_w, 1176), grid_thw tuple)."""
    import torch
    x = img_chw
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype)
    C, H, W = x.shape
    mean = torch.tensor(IMAGE_MEAN, device=x.device, dtype=x.dtype).view(C, 1, 1)
    std = torch.tensor(IMAGE_STD, device=x.device, dtype=x.dtype).view(C, 1, 1)
    x = (x - mean) / std                                    # normalize

    # tile temporal (image -> TEMPORAL identical frames)
    frames = x.unsqueeze(0).repeat(TEMPORAL, 1, 1, 1)       # (T, C, H, W)
    grid_t = TEMPORAL // TEMPORAL                            # = 1
    grid_h, grid_w = H // PATCH, W // PATCH

    p = frames.reshape(
        grid_t, TEMPORAL, C,
        grid_h // MERGE, MERGE, PATCH,
        grid_w // MERGE, MERGE, PATCH,
    )
    p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    flatten = p.reshape(grid_t * grid_h * grid_w,
                        C * TEMPORAL * PATCH * PATCH)       # (N, 1176)
    return flatten, (grid_t, grid_h, grid_w)


def selfcheck(model, image_path: str, atol: float = 1e-2) -> bool:
    """Confirm our differentiable patchify+normalize matches the HF processor,
    ISOLATED from resize interpolation.

    Resize once (ours) to the grid the processor picked, round-trip through uint8,
    then feed the SAME pixels to both our patchify and the processor (which won't
    resize again - the image is already a multiple of 28). Any remaining diff is
    pure patchify/normalize, the thing that must be correct. Resize-interpolation
    differences on the original image are harmless because the attack saves the
    adversarial image at this resized resolution as lossless PNG, so no second
    resize happens at inference."""
    import torch
    from PIL import Image

    pil = Image.open(image_path).convert("RGB")
    grid0 = tuple(model.processor(text=["<x>"], images=[pil],
                                  return_tensors="pt")["image_grid_thw"][0].tolist())

    chw, _ = resized_chw_to_grid(pil, grid0[1], grid0[2])
    resized_pil = chw_to_pil(chw)                 # uint8 at the target grid size
    chw2 = _pil_to_chw(resized_pil)               # exact pixels the processor sees

    ours_pv, ours_grid = patchify(chw2)
    ours_pv = ours_pv.float()

    ref = model.processor(text=["<x>"], images=[resized_pil], return_tensors="pt")
    ref_pv = ref["pixel_values"].float()
    ref_grid = tuple(ref["image_grid_thw"][0].tolist())

    if (ours_pv.shape != ref_pv.shape) or (tuple(ours_grid) != ref_grid):
        print(f"[selfcheck] SHAPE MISMATCH ours={tuple(ours_pv.shape)}/{ours_grid} "
              f"ref={tuple(ref_pv.shape)}/{ref_grid}")
        return False
    diff = (ours_pv - ref_pv).abs().max().item()
    ok = diff <= atol
    print(f"[selfcheck] (resize-isolated) grid={ours_grid} shape={tuple(ours_pv.shape)} "
          f"max|diff|={diff:.4g} -> {'OK' if ok else 'MISMATCH'}")
    return ok