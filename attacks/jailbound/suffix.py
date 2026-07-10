"""
JailBound Stage 2 (part 2): text suffix search.

Optimizes a short token suffix appended to the user request so the fused
representation crosses the safety boundary, while staying fluent. Uses the
paper's embedding-space update (Algorithm 2): take the gradient of the total
loss w.r.t. the suffix embeddings, step in embedding space, then hard-project
each position to its nearest vocabulary token. This is cheaper than full GCG
(no per-candidate re-scoring).

Losses:
  L_align, L_geo  : same boundary-crossing objective as the image stage
  L_sem           : fluency = NLL of the suffix under the model's own LM head
  L_total = L_align + lambda_geo*L_geo + lambda_sem*L_sem

Requires the inputs_embeds path in the wrapper (build_merged_embeds /
forward_from_embeds). RUN model.embed_parity_check(...) FIRST: if the manual
image merge does not match the model's internal forward, the suffix stage is
invalid (attack.py gates on this automatically).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging

from attacks.jailbound.crossing import L_align, L_geo, _mean_dist

logger = logging.getLogger(__name__)


@dataclass
class SuffixConfig:
    suffix_len: int = 20
    n_iters: int = 100
    eta_t: float = 0.0005       # embedding-space step (paper)
    lambda_geo: float = 1.0
    lambda_sem: float = 2.0     # fluency weight (paper lambda_1)
    init_token: str = " x"      # repeated to initialize the suffix
    pool: str = "last"


def _im_end_id(model):
    return model.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")


def build_ids_with_suffix(model, text, pil_image, suffix_len, init_token):
    """Insert `suffix_len` init tokens right before the user turn's closing
    <|im_end|>. Returns (input_ids, attention_mask, image_grid_thw,
    suffix_slice)."""
    import torch
    inp = model.build_inputs(text, pil_image, add_generation_prompt=True)
    ids = inp["input_ids"][0].tolist()
    grid = inp["image_grid_thw"]

    tok = model.processor.tokenizer
    ie = _im_end_id(model)
    pos = max(i for i, t in enumerate(ids) if t == ie)     # user-turn closer

    init_id = tok(init_token, add_special_tokens=False)["input_ids"][0]
    suffix_ids = [init_id] * suffix_len

    new_ids = ids[:pos] + suffix_ids + ids[pos:]
    suffix_slice = slice(pos, pos + suffix_len)
    device = model.model.device
    input_ids = torch.tensor([new_ids], device=device)
    attn = torch.ones_like(input_ids)
    return input_ids, attn, grid, suffix_slice


def _suffix_nll(logits, input_ids, suffix_slice):
    """Fluency NLL of the suffix tokens under the LM head. logit at position p-1
    predicts token p."""
    import torch
    import torch.nn.functional as F
    lp = F.log_softmax(logits[0].float(), dim=-1)          # (T, V)
    s, e = suffix_slice.start, suffix_slice.stop
    idx = input_ids[0, s:e]                                # (S,)
    pred = lp[s - 1:e - 1]                                 # (S, V)
    return -pred.gather(1, idx.unsqueeze(1)).mean()


def suffix_search(model, text, pil_image, B, cfg: SuffixConfig):
    """Optimize the suffix. Returns (suffix_string, info). Image is FIXED here
    (pass the adversarial image from the image stage)."""
    import torch
    from multimodalmodels.qwen2_5_vl.qwen_image import resized_chw_to_grid, patchify

    device = model.model.device
    input_ids, attn, grid, sl = build_ids_with_suffix(
        model, text, pil_image, cfg.suffix_len, cfg.init_token)

    gh, gw = int(grid[0][1].item()), int(grid[0][2].item())
    chw, _ = resized_chw_to_grid(pil_image, gh, gw)
    pv, _ = patchify(chw, device=device, dtype=model.dtype)

    E = model.token_embedding_matrix()                     # (V,H)
    base_embeds = model.build_merged_embeds(input_ids, pv, grid).detach()

    # clean reps for L_geo (with the init suffix in place)
    with torch.no_grad():
        h0, _ = model.forward_from_embeds(base_embeds, attn, pool=cfg.pool)
        h0 = {l: v.detach() for l, v in h0.items()}
        d0 = _mean_dist(h0, B)

    suffix_ids = input_ids[0, sl].clone()
    for it in range(cfg.n_iters):
        suffix_embeds = E[suffix_ids].detach().clone().unsqueeze(0).requires_grad_(True)
        embeds = base_embeds.clone()
        embeds[0, sl] = suffix_embeds[0]
        reps, logits = model.forward_from_embeds(embeds, attn, need_logits=True, pool=cfg.pool)
        loss = (L_align(reps, B) + cfg.lambda_geo * L_geo(reps, h0, B)
                + cfg.lambda_sem * _suffix_nll(logits, input_ids, sl))
        grad = torch.autograd.grad(loss, suffix_embeds)[0][0]      # (S,H)

        with torch.no_grad():
            target = E[suffix_ids] - cfg.eta_t * grad              # (S,H)
            # nearest vocab embedding per position: argmin ||E - target||^2
            #   = argmax (E·target - 0.5||E||^2)
            scores = target @ E.t() - 0.5 * (E * E).sum(-1)        # (S,V)
            suffix_ids = scores.argmax(dim=-1)
            input_ids[0, sl] = suffix_ids

        if it % 20 == 0 or it == cfg.n_iters - 1:
            with torch.no_grad():
                dm = _mean_dist(reps, B)
            logger.info(f"[suffix] it={it:>3} loss={loss.item():.4f} "
                        f"signed_dist {d0:+.3f}->{dm:+.3f}")

    suffix_str = model.processor.tokenizer.decode(suffix_ids.tolist())
    with torch.no_grad():
        se = E[suffix_ids].unsqueeze(0)
        embeds = base_embeds.clone(); embeds[0, sl] = se[0]
        reps, _ = model.forward_from_embeds(embeds, attn, pool=cfg.pool)
        d_final = _mean_dist(reps, B)
    return suffix_str, {"d0": d0, "d_final": d_final, "crossed": d_final > 0}