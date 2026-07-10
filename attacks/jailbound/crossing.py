"""
JailBound Stage 2 (part 1): Safety Boundary Crossing in image space.

Given the probed boundary (Stage 1), perturb the raw image (L_inf <= 8/255) so
the fused representation of a currently-refused harmful input crosses into the
comply region. Text-suffix search is added next; the image perturbation is the
core and is validated first.

Sign convention (from Stage 1 labeling): y=1 comply/unsafe, y=0 refuse/safe;
v points refuse->comply. signed distance d_l(h)=v_l.h+offset_l (offset=b/||w||),
positive = comply side. We PUSH d up toward the margin eps_l.

Losses (validated on mock tensors):
  L_align = mean_l w_l * relu(eps_l - d_l(h))        cross toward comply
  L_geo   = mean_l w_l * ||delta_orth||^2/||delta||^2 keep the shift along v_l
  L_total = L_align + lambda_geo * L_geo             (L_sem enters with suffix)

Layer selection uses Stage-1 CV accuracy: only layers whose boundary generalizes
(cv_acc >= layer_min_cv), optionally weighted by (cv_acc-0.5).

RUN selfcheck_preprocessing FIRST: the raw-image attack is only valid if our
differentiable pixel_values match the HF processor.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossingConfig:
    boundary_path: str = "attacks/jailbound/cache/boundary.pt"
    layer_min_cv: float = 0.80      # use layers whose Stage-1 CV acc >= this
    last_n: Optional[int] = None    # or: use the last N layers (overrides min_cv)
    weight_by_cv: bool = True       # weight layer losses by (cv_acc - 0.5)
    eps: float = 8.0 / 255.0        # L_inf image budget in raw [0,1] space
    n_iters: int = 150
    eta: float = 1.0 / 255.0        # PGD step in raw space
    lambda_geo: float = 1.0
    eps_scale: float = 1.0          # scale the target margin eps_l
    pool: str = "last"


# ---- boundary loading + layer selection ---------------------------------- #
def load_boundary(cfg: CrossingConfig, device="cuda"):
    import torch, math
    b = torch.load(cfg.boundary_path, map_location="cpu")
    layers = b["layers"]
    per_cv = b.get("per_layer_acc",
                   {l: layers[l].get("acc", float("nan")) for l in layers})

    if cfg.last_n:
        chosen = sorted(layers)[-cfg.last_n:]
    else:
        chosen = [l for l in sorted(layers)
                  if not math.isnan(per_cv.get(l, float("nan")))
                  and per_cv[l] >= cfg.layer_min_cv]
        if not chosen:
            chosen = sorted(layers, key=lambda l: (per_cv.get(l, 0) or 0))[-10:]
            logger.warning("[cross] no layer >= layer_min_cv=%.2f; using best 10",
                           cfg.layer_min_cv)

    B = {}
    for l in chosen:
        L = layers[l]
        B[l] = {
            "v": L["v"].to(device).float(),
            "offset": float(L["bias"]) / float(L["w_norm"]),
            "eps": float(L["eps"]) * cfg.eps_scale,
            "w": (max(0.0, (per_cv.get(l, 0.5) or 0.5) - 0.5) if cfg.weight_by_cv else 1.0),
        }
    tot = sum(d["w"] for d in B.values()) or 1.0
    for d in B.values():
        d["w"] = d["w"] / tot if cfg.weight_by_cv else 1.0 / len(B)
    logger.info(f"[cross] using {len(B)} layers: {sorted(B)}")
    return B


# ---- losses (validated) --------------------------------------------------- #
def signed_distance(h, v, offset):
    return (h * v).sum(-1) + offset


def L_align(h, B):
    import torch
    loss = h[next(iter(B))].new_zeros(())
    for l, d in B.items():
        dist = signed_distance(h[l], d["v"], d["offset"])
        loss = loss + d["w"] * torch.relu(d["eps"] - dist).mean()
    return loss


def L_geo(h, h0, B):
    loss = h[next(iter(B))].new_zeros(())
    for l, d in B.items():
        delta = h[l] - h0[l]
        proj = (delta * d["v"]).sum(-1, keepdim=True) * d["v"]
        orth = delta - proj
        loss = loss + d["w"] * (orth.pow(2).sum(-1) / (delta.pow(2).sum(-1) + 1e-6)).mean()
    return loss


def _mean_dist(h, B):
    return float(sum(signed_distance(h[l], d["v"], d["offset"]).mean().item()
                     for l, d in B.items()) / len(B))


# ---- image-space PGD crossing -------------------------------------------- #
def crossing_image(model, text: str, image_path: str, B, cfg: CrossingConfig):
    """Return (adv_pil_image, info). Perturbs the raw image so the fused rep
    crosses the boundary; text unchanged in this part."""
    import torch
    from PIL import Image
    from multimodalmodels.qwen2_5_vl.qwen_image import resized_chw_to_grid, patchify, chw_to_pil

    device = model.model.device
    pil = Image.open(image_path).convert("RGB")

    inp = model.clean_text_inputs(text, pil)
    input_ids, attn, grid = inp["input_ids"], inp["attention_mask"], inp["image_grid_thw"]

    gh, gw = int(grid[0][1].item()), int(grid[0][2].item())
    base, (H, W) = resized_chw_to_grid(pil, gh, gw)      # match processor grid
    base = base.to(device)

    with torch.no_grad():
        pv0, _ = patchify(base, device=device, dtype=model.dtype)
        h0 = model.hidden_states_from_pixels(input_ids, attn, pv0, grid, cfg.pool)
        h0 = {l: v.detach() for l, v in h0.items()}
        d0 = _mean_dist(h0, B)

    delta = torch.zeros_like(base, requires_grad=True)
    for it in range(cfg.n_iters):
        adv = (base + delta).clamp(0, 1)
        pv, _ = patchify(adv, device=device, dtype=model.dtype)
        h = model.hidden_states_from_pixels(input_ids, attn, pv, grid, cfg.pool)
        loss = L_align(h, B) + cfg.lambda_geo * L_geo(h, h0, B)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta - cfg.eta * grad.sign()).clamp(-cfg.eps, cfg.eps)
            delta = (base + delta).clamp(0, 1) - base
        delta.requires_grad_(True)
        if it % 25 == 0 or it == cfg.n_iters - 1:
            with torch.no_grad():
                dm = _mean_dist(h, B)
            logger.info(f"[cross] it={it:>3} loss={loss.item():.4f} "
                        f"signed_dist {d0:+.3f}->{dm:+.3f}")

    adv = (base + delta.detach()).clamp(0, 1)
    adv_pil = chw_to_pil(adv)
    with torch.no_grad():
        pv, _ = patchify(adv, device=device, dtype=model.dtype)
        hf = model.hidden_states_from_pixels(input_ids, attn, pv, grid, cfg.pool)
        d_final = _mean_dist(hf, B)
    info = {"d0": d0, "d_final": d_final,
            "linf": float((adv - base).abs().max().item()),
            "crossed": d_final > 0}
    return adv_pil, info


def selfcheck_preprocessing(model, image_path: str) -> bool:
    from multimodalmodels.qwen2_5_vl.qwen_image import selfcheck
    return selfcheck(model, image_path)


# --------------------------------------------------------------------------- #
# CLI: verify preprocessing, then run a 1-sample image crossing
#   python -m attacks.jailbound.crossing --model_name ... --image X.jpg \
#          --text "..." --boundary_path attacks/jailbound/cache/boundary.pt
# --------------------------------------------------------------------------- #
def _main():
    import argparse, logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import importlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", required=True, help="a real MM-SafetyBench SD_TYPO image")
    ap.add_argument("--text", required=True, help="the paired harmful question")
    ap.add_argument("--boundary_path", default="attacks/jailbound/cache/boundary.pt")
    ap.add_argument("--layer_min_cv", type=float, default=0.80)
    ap.add_argument("--n_iters", type=int, default=150)
    ap.add_argument("--out", default="attacks/jailbound/cache/adv_probe.png")
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28,
                    help="cap image tokens to save memory (lower = less VRAM)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable gradient checkpointing (needs more VRAM)")
    args = ap.parse_args()

    mod = importlib.import_module("multimodalmodels.qwen2_5_vl.qwen_whitebox")
    model = mod.QwenVLWhiteBox(model_name=args.model_name, device=args.device,
                               dtype="float16", max_pixels=args.max_pixels,
                               gradient_checkpointing=not args.no_grad_ckpt)

    # STEP 1 - preprocessing must match the HF processor or the attack is invalid
    logger.info("[cross] STEP 1: preprocessing self-check")
    ok = selfcheck_preprocessing(model, args.image)
    if not ok:
        logger.error("[cross] preprocessing self-check FAILED - fix qwen_image.patchify "
                     "before trusting any crossing result. Aborting.")
        return

    # STEP 2 - one-sample crossing
    logger.info("[cross] STEP 2: single-sample image crossing")
    cfg = CrossingConfig(boundary_path=args.boundary_path,
                         layer_min_cv=args.layer_min_cv, n_iters=args.n_iters)
    B = load_boundary(cfg, device=model.model.device)
    adv, info = crossing_image(model, args.text, args.image, B, cfg)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    adv.save(args.out)
    logger.info(f"[cross] d0={info['d0']:+.3f} -> d_final={info['d_final']:+.3f} "
                f"crossed={info['crossed']} Linf={info['linf']:.4f}")
    logger.info(f"[cross] saved adversarial image -> {args.out}")
    # quick behavioral read: did the model's response change?
    clean_resp = model.generate(args.text, args.image, max_new_tokens=64)
    adv_resp = model.generate(args.text, args.out, max_new_tokens=64)
    from attacks.jailbound.data import is_refusal
    logger.info(f"[cross] clean refuses={is_refusal(clean_resp)} | adv refuses={is_refusal(adv_resp)}")


if __name__ == "__main__":
    _main()