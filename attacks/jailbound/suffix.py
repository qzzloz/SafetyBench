"""
JailBound Stage 2 (part 2): text suffix search.

Optimizes a short token suffix appended to the user request so the fused
representation crosses the safety boundary, while staying fluent. Paper's
embedding-space update (Algorithm 2): gradient of the total loss w.r.t. the
suffix embeddings, step in embedding space, hard-project each position to its
nearest vocabulary token (cheaper than full GCG).

Robust design: the image is FIXED here, so we take pixel_values straight from
the HF processor and let the MODEL do its own image merge; a forward hook
splices in the differentiable suffix embedding. No version-specific vision-tower
plumbing, no manual merge, no parity check needed.

Losses:
  L_align, L_geo : boundary-crossing (same as image stage)
  L_sem          : fluency = NLL of the suffix under the model's LM head
  L_total = L_align + lambda_geo*L_geo + lambda_sem*L_sem
"""

from __future__ import annotations
from dataclasses import dataclass
import logging

from attacks.jailbound.crossing import L_align, L_geo, _mean_dist

logger = logging.getLogger(__name__)


@dataclass
class SuffixConfig:
    suffix_len: int = 20
    n_iters: int = 100
    eta_t: float = 0.01         # Adam lr on the continuous suffix embedding
    project_every: int = 10     # snap to nearest tokens every k steps (0=only at end)
    lambda_geo: float = 1.0
    lambda_sem: float = 2.0
    init_token: str = " x"
    pool: str = "last"


def build_ids_with_suffix(model, text, pil_image, suffix_len, init_token):
    """Insert `suffix_len` init tokens right before the user turn's closing
    <|im_end|>. Returns (input_ids, attention_mask, image_grid_thw,
    pixel_values, suffix_slice). pixel_values come from the processor (image is
    fixed during the suffix search)."""
    import torch
    inp = model.build_inputs(text, pil_image, add_generation_prompt=True)
    ids = inp["input_ids"][0].tolist()
    grid = inp["image_grid_thw"]
    pv = inp["pixel_values"]

    tok = model.processor.tokenizer
    ie = tok.convert_tokens_to_ids("<|im_end|>")
    pos = max(i for i, t in enumerate(ids) if t == ie)          # user-turn closer

    init_id = tok(init_token, add_special_tokens=False)["input_ids"][0]
    new_ids = ids[:pos] + [init_id] * suffix_len + ids[pos:]
    suffix_slice = slice(pos, pos + suffix_len)

    device = model.model.device
    input_ids = torch.tensor([new_ids], device=device)
    attn = torch.ones_like(input_ids)
    return input_ids, attn, grid, pv, suffix_slice


def _suffix_nll(logits, input_ids, suffix_slice):
    """Fluency NLL of the suffix tokens. logit at position p-1 predicts token p."""
    import torch.nn.functional as F
    lp = F.log_softmax(logits[0].float(), dim=-1)               # (T, V)
    s, e = suffix_slice.start, suffix_slice.stop
    idx = input_ids[0, s:e]
    pred = lp[s - 1:e - 1]                                      # (S, V)
    return -pred.gather(1, idx.unsqueeze(1)).mean()


def _project_to_tokens(z, E):
    """Nearest vocab token per position: argmin ||E - z||^2. z may be fp32
    (Adam param) while E is fp16 - cast to E's dtype for the matmul."""
    zc = z.to(E.dtype)
    return (zc @ E.t() - 0.5 * (E * E).sum(-1)).argmax(dim=-1)


def suffix_search(model, text, pil_image, B, cfg: SuffixConfig):
    """Optimize the suffix on a FIXED (adversarial) image. Returns
    (suffix_string, info).

    GCG-style gradient token selection: gradient of the loss w.r.t. the current
    suffix embeddings gives, to first order, the loss change for swapping each
    position to any vocab token (~ grad . E[v]). Pick the argmin per position -
    no step size, tokens actually move. We keep the best (lowest-loss) suffix
    seen and return that."""
    import torch

    input_ids, attn, grid, pv, sl = build_ids_with_suffix(
        model, text, pil_image, cfg.suffix_len, cfg.init_token)
    E = model.token_embedding_matrix()                          # (V,H)
    cur_ids = input_ids[0, sl].clone()

    with torch.no_grad():
        h0, _ = model.forward_with_suffix(input_ids, attn, pv, grid, sl,
                                          E[cur_ids].detach(), pool=cfg.pool)
        h0 = {l: v.detach() for l, v in h0.items()}
        d0 = _mean_dist(h0, B)

    best_ids = cur_ids.clone()
    best_loss = float("inf")
    # GC re-runs forward during backward; since we remove the hook after forward,
    # the suffix drops out of the recomputed graph -> zero grad. The suffix stage
    # backprops only through text, so disable GC for the whole stage (restore
    # after, for the next sample's image crossing).
    gc_on = getattr(model.model, "is_gradient_checkpointing", False)
    if gc_on:
        model.model.gradient_checkpointing_disable()
    try:
        for it in range(cfg.n_iters):
            input_ids[0, sl] = cur_ids
            se = E[cur_ids].detach().clone().requires_grad_(True)   # (S,H)
            reps, logits = model.forward_with_suffix(
                input_ids, attn, pv, grid, sl, se, need_logits=True, pool=cfg.pool)
            loss = (L_align(reps, B) + cfg.lambda_geo * L_geo(reps, h0, B)
                    + cfg.lambda_sem * _suffix_nll(logits, input_ids, sl))
            grad = torch.autograd.grad(loss, se)[0]                 # (S,H)

            lval = loss.item()
            if lval < best_loss:
                best_loss = lval; best_ids = cur_ids.clone()

            with torch.no_grad():
                scores = grad.to(E.dtype) @ E.t()                   # (S,V)
                new_ids = scores.argmin(dim=-1)

            if it == 0:
                logger.info(f"[suffix] it=0 grad_norm={float(grad.norm()):.4g} "
                            f"tokens_changing={int((new_ids != cur_ids).sum())}/{len(cur_ids)}")
            cur_ids = new_ids

            if it % 10 == 0 or it == cfg.n_iters - 1:
                with torch.no_grad():
                    dm = _mean_dist(reps, B)
                logger.info(f"[suffix] it={it:>3} loss={lval:.4f} "
                            f"signed_dist {d0:+.3f}->{dm:+.3f}")
    finally:
        if gc_on:
            model.model.gradient_checkpointing_enable()

    with torch.no_grad():
        reps, _ = model.forward_with_suffix(input_ids, attn, pv, grid, sl,
                                            E[best_ids], pool=cfg.pool)
        d_final = _mean_dist(reps, B)
    suffix_str = model.processor.tokenizer.decode(best_ids.tolist())
    return suffix_str, {"d0": d0, "d_final": d_final, "crossed": d_final > 0,
                        "best_loss": best_loss}