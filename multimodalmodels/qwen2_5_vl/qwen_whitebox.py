"""
White-box handle on Qwen2.5-VL for JailBound (arXiv:2505.19610).

Unlike `models/qwen_local_model.py` (inference-only), this wrapper exposes the
model internals JailBound needs:

  * Stage 1 (Safety Boundary Probing):
      `fused_representations(...)` -> the fusion-layer hidden states h^(l)
      (last-token vector at every LLM decoder layer). 28 layers for the 7B,
      which matches the paper's "28 classifiers in Qwen2.5-VL".

  * Stage 2 (Safety Boundary Crossing) [built later on top of this]:
      differentiable access to `pixel_values` and to the input token
      embeddings, so we can backprop the boundary-crossing loss to an image
      perturbation delta_v and to a text suffix.

V100 note
---------
V100 (Volta) has NO hardware bf16. Qwen2.5-VL ships bf16 by default; loading
that here would error / run in slow emulation and NaN under backprop. So we
load in **fp16** and keep the probing math (logistic regression, distances) in
fp32 on CPU. For Stage 2, gradient-sensitive reductions should be cast to fp32.

The `target_model` string used by the attack config points here, mirroring how
UMK loads its MiniGPT-4 target from `multimodalmodels`:
    target_model: "qwen2_5_vl.qwen_whitebox.QwenVLWhiteBox"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Sequence
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class QwenVLWhiteBox:
    """Minimal white-box wrapper around Qwen2.5-VL for latent-space attacks."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",          # V100 -> fp16, NOT bfloat16
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        gradient_checkpointing: bool = False,   # set True for Stage 2 backprop
    ):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.device = device
        self.model_name = model_name

        torch_dtype = getattr(torch, dtype, torch.float16)
        if torch_dtype == torch.bfloat16 and not (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ):
            logger.warning(
                "[QwenWhiteBox] bfloat16 requested but this GPU lacks bf16 HW "
                "(e.g. V100); forcing float16."
            )
            torch_dtype = torch.float16
        if torch_dtype == torch.bfloat16:
            logger.info("[QwenWhiteBox] using bfloat16 (bf16-capable GPU detected).")
        self.dtype = torch_dtype

        logger.info(f"[QwenWhiteBox] loading processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, max_pixels=max_pixels, min_pixels=min_pixels
        )

        logger.info(f"[QwenWhiteBox] loading weights ({dtype}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()
        # Probing/attack need gradients on inputs, never on weights.
        self.model.requires_grad_(False)

        # Stage 2 backprop through a 7B on a 32GB V100 needs gradient checkpointing.
        # enable_input_require_grads() makes the (image/text) input path carry grad
        # so checkpointing works even though the weights are frozen.
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            logger.info("[QwenWhiteBox] gradient checkpointing ON")

        # Qwen2.5-VL-7B language model has 28 decoder layers -> 28 fusion layers.
        # hidden_states from a forward() has length num_layers + 1 (idx 0 = embeddings).
        # NOTE: depending on the transformers version, num_hidden_layers lives either
        # at config top level or nested under config.text_config (the LLM sub-config).
        self.num_layers = self._resolve_num_layers(self.model.config)
        logger.info(f"[QwenWhiteBox] ready. fusion layers = {self.num_layers}")

    @staticmethod
    def _resolve_num_layers(cfg) -> Optional[int]:
        if hasattr(cfg, "num_hidden_layers"):
            return int(cfg.num_hidden_layers)
        for sub in ("text_config", "llm_config", "language_config"):
            sc = getattr(cfg, sub, None)
            if sc is not None and hasattr(sc, "num_hidden_layers"):
                return int(sc.num_hidden_layers)
        # Fall back: leave unknown; fused_representations derives it from the
        # actual hidden_states length at the first forward pass.
        logger.warning("[QwenWhiteBox] could not read num_hidden_layers from config; "
                       "will infer from hidden_states at first forward.")
        return None

    # ------------------------------------------------------------------ #
    # input building
    # ------------------------------------------------------------------ #
    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def build_inputs(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        add_generation_prompt: bool = True,
    ):
        """Return processor inputs on the model device.

        `add_generation_prompt=True` appends the assistant turn header, so the
        last token position corresponds to "the model is about to answer" -
        the natural place to read the fused safety representation from.
        """
        content = []
        images = []
        if image is not None:
            pil = self._load_image(image)
            images.append(pil)
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        inputs = self.processor(
            text=[chat_text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        return inputs

    # ------------------------------------------------------------------ #
    # generation (used to label inputs by the model's own comply/refuse behavior)
    # ------------------------------------------------------------------ #
    def generate(self, text: str, image=None, max_new_tokens: int = 96) -> str:
        """Greedy-decode a response. Used by Stage-1 behavior labeling:
        whether the model complies with or refuses a harmful input is exactly
        the 'safety label determined by the VLM'."""
        torch = self.torch
        inputs = self.build_inputs(text, image, add_generation_prompt=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        resp = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        del out, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return resp

    # ------------------------------------------------------------------ #
    # Stage 1: fused representations h^(l)
    # ------------------------------------------------------------------ #
    def _pool_last_token(self, hidden, attention_mask):
        """h^(l) = hidden state of the last real (non-pad) token at layer l.

        hidden: (B, T, H); attention_mask: (B, T). Returns (B, H).
        """
        torch = self.torch
        # index of last non-pad token per row
        lengths = attention_mask.sum(dim=1) - 1            # (B,)
        idx = lengths.clamp(min=0).long()
        b = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[b, idx]                               # (B, H)

    @property
    def pool_modes(self):
        return ("last", "mean")

    def fused_representations(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        pool: str = "last",
    ) -> Dict[int, "self.torch.Tensor"]:
        """Return {layer_index -> h vector (fp32, cpu)} for all 28 fusion layers.

        layer_index runs 1..num_layers (i.e. AFTER each decoder layer);
        index 0 (raw embeddings) is intentionally dropped.
        The paper reports early layers (0-4) probe poorly and deep layers ~100%.
        """
        torch = self.torch
        inputs = self.build_inputs(text, image, add_generation_prompt=True)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                              # tuple len = L+1
        attn = inputs["attention_mask"]

        reps: Dict[int, "torch.Tensor"] = {}
        for l in range(1, len(hs)):
            layer = hs[l]                                   # (B, T, H)
            if pool == "mean":
                m = attn.unsqueeze(-1).to(layer.dtype)
                vec = (layer * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                vec = self._pool_last_token(layer, attn)
            reps[l] = vec.squeeze(0).float().cpu()          # (H,) fp32 cpu

        # free GPU tensors between samples (Qwen-VL memory accumulates otherwise)
        del out, hs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return reps

    def batched_fused_representations(
        self,
        samples: Sequence[dict],
        pool: str = "last",
        log_every: int = 200,
    ):
        """samples: list of {"text": str, "image": path-or-PIL-or-None}.

        Yields (i, {layer -> vec}) one sample at a time. Batch size is 1 on
        purpose: it is memory-trivial on a 32GB V100 for a 7B model, avoids
        left/right padding pitfalls in last-token pooling, and the whole thing
        is a one-time cached pass anyway.
        """
        n = len(samples)
        for i, s in enumerate(samples):
            reps = self.fused_representations(
                text=s["text"], image=s.get("image"), pool=pool
            )
            if log_every and (i % log_every == 0):
                logger.info(f"[QwenWhiteBox] features {i}/{n}")
            yield i, reps

    # ------------------------------------------------------------------ #
    # Stage 2 helpers (used by crossing.py, added in the next step)
    # ------------------------------------------------------------------ #
    def input_embeddings(self):
        """The token embedding matrix (V, H) for embedding-space suffix search."""
        return self.model.get_input_embeddings().weight

    def get_dtype(self):
        return self.dtype

    # ------------------------------------------------------------------ #
    # Stage 2 (text suffix): differentiable inputs_embeds path
    # ------------------------------------------------------------------ #
    def token_embedding_matrix(self):
        """(V, H) input embedding matrix for embedding-space suffix search."""
        return self.model.get_input_embeddings().weight

    def _image_embeds(self, pixel_values, image_grid_thw):
        """Version-robust image features. Newer transformers move the vision
        tower under model.model.visual and expose get_image_features(); older
        ones have model.visual(pixel_values, grid_thw=...)."""
        torch = self.torch
        pv = pixel_values.to(self.model.device, self.dtype)
        # 1) preferred: high-level API. NOTE: some transformers versions return
        #    a ModelOutput (e.g. BaseModelOutputWithPooling) here instead of a
        #    tensor, which is why the JOINT path no longer relies on this method
        #    (forward_joint delegates to forward_with_suffix and lets the model do
        #    its own merge). Kept robust for any other callers.
        if hasattr(self.model, "get_image_features"):
            img = self.model.get_image_features(pixel_values=pv, image_grid_thw=image_grid_thw)
            if hasattr(img, "last_hidden_state"):          # ModelOutput wrapper
                img = img.last_hidden_state
            if isinstance(img, (list, tuple)):
                img = torch.cat([x for x in img], dim=0)
            return img.reshape(-1, img.shape[-1])
        # 2) fallback: locate the visual tower and call it directly
        for getter in (lambda: self.model.visual,
                       lambda: self.model.model.visual,
                       lambda: self.model.model.model.visual):
            try:
                vt = getter()
            except AttributeError:
                continue
            if vt is not None:
                out = vt(pv, grid_thw=image_grid_thw)
                return out.reshape(-1, out.shape[-1])
        raise AttributeError("could not locate the Qwen2.5-VL vision tower "
                             "(tried get_image_features, .visual, .model.visual)")

    def _image_token_id(self):
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index"):
            if getattr(cfg, attr, None) is not None:
                return getattr(cfg, attr)
        raise AttributeError("no image_token_id/image_token_index on config")

    def build_merged_embeds(self, input_ids, pixel_values, image_grid_thw):
        """Manually build inputs_embeds with image features scattered in, so we
        can then swap in a differentiable suffix embedding. Mirrors what the
        model does internally (verified by embed_parity_check)."""
        emb = self.model.get_input_embeddings()(input_ids)          # (1,T,H)
        image_embeds = self._image_embeds(pixel_values, image_grid_thw)
        mask = (input_ids == self._image_token_id())
        emb = emb.clone()
        emb[mask] = image_embeds.to(emb.dtype)
        return emb

    def forward_from_embeds(self, inputs_embeds, attention_mask,
                            need_logits=False, pool: str = "last"):
        """Forward with pre-merged inputs_embeds (no pixel_values). Returns
        ({layer -> last-token hidden state fp32}, logits-or-None)."""
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        reps = {}
        for l in range(1, len(hs)):
            layer = hs[l]
            if pool == "mean":
                m = attention_mask.unsqueeze(-1).to(layer.dtype)
                vec = (layer * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                vec = self._pool_last_token(layer, attention_mask)
            reps[l] = vec.squeeze(0).float()
        return reps, (out.logits if need_logits else None)

    def forward_with_suffix(self, input_ids, attention_mask, pixel_values,
                            image_grid_thw, suffix_slice, suffix_embeds,
                            need_logits=False, pool: str = "last"):
        """Forward the model normally (input_ids + pixel_values, so the model does
        its OWN image merge - no version-specific vision-tower plumbing), but
        splice a differentiable suffix embedding into the token-embedding output
        via a forward hook. Gradients flow to suffix_embeds. Returns
        ({layer -> last-token hidden state fp32}, logits-or-None)."""
        torch = self.torch
        s, e = suffix_slice.start, suffix_slice.stop
        emb_module = self.model.get_input_embeddings()

        def _hook(module, inp, out):
            out = out.clone()
            out[0, s:e] = suffix_embeds.to(out.dtype)
            return out

        handle = emb_module.register_forward_hook(_hook)
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(self.model.device, self.dtype),
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            handle.remove()

        hs = out.hidden_states
        reps = {}
        for l in range(1, len(hs)):
            layer = hs[l]
            if pool == "mean":
                m = attention_mask.unsqueeze(-1).to(layer.dtype)
                vec = (layer * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                vec = self._pool_last_token(layer, attention_mask)
            reps[l] = vec.squeeze(0).float()
        return reps, (out.logits if need_logits else None)

    # ------------------------------------------------------------------ #
    # Stage 2 (JOINT): image delta_v AND suffix embeds differentiable together
    # ------------------------------------------------------------------ #
    def forward_joint(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                      suffix_slice, suffix_embeds, need_logits=False, pool: str = "last"):
        """One forward whose fused hidden states are differentiable w.r.t. BOTH
        the (perturbed) pixel_values and the suffix embeddings simultaneously -
        the requirement for the paper's joint update (Algorithm 2, one L_total
        per iteration k).

        Implementation: delegate to `forward_with_suffix`, i.e. pass input_ids +
        pixel_values so the model performs its OWN image merge (no version-
        specific vision-tower plumbing, which broke on some transformers builds
        where get_image_features returns a ModelOutput), and splice the
        differentiable suffix via a forward hook on the token-embedding module.

        Gradient checkpointing is safe here: the token-embedding + image merge run
        in the model's OUTER forward, before the checkpointed decoder blocks, so
        the (suffix-spliced) inputs_embeds is what autograd saves as each block's
        checkpoint input. Removing the hook after forward() returns therefore does
        NOT drop the suffix from the graph. Both pixel_values and suffix_embeds
        keep gradients. Run embed_parity_check() once on your transformers version
        to confirm both grads are non-zero before trusting a full run.

        Returns ({layer -> last-token hidden state fp32}, logits-or-None).
        """
        return self.forward_with_suffix(
            input_ids, attention_mask, pixel_values, image_grid_thw,
            suffix_slice, suffix_embeds, need_logits=need_logits, pool=pool)

    def embed_parity_check(self, text: str, image, atol: float = 1e-2,
                           suffix_len: int = 4, init_token: str = "!") -> bool:
        """Validate the JOINT forward on THIS transformers version: run
        forward_joint exactly as the attack does (differentiable pixel_values +
        differentiable suffix embeddings, one forward under the current gradient-
        checkpointing setting) and confirm a scalar loss backprops NON-ZERO
        gradients to BOTH inputs. This is the real precondition for Algorithm 2 -
        if either grad is missing/zero, the joint update silently degrades to a
        single-modality attack.

        Returns True iff both grads are present and non-trivial.
        """
        torch = self.torch
        from attacks.jailbound.suffix import build_suffix_ids

        # mirror the attack's input construction
        input_ids, attn, grid, sl = build_suffix_ids(
            self, text, image, suffix_len=suffix_len, init_token=init_token)
        inp = self.build_inputs(text, image, add_generation_prompt=True)

        # differentiable pixel_values (leaf)
        pv = inp["pixel_values"].to(self.model.device, self.dtype).clone().detach()
        pv.requires_grad_(True)
        # differentiable suffix embeds (leaf), seeded from the init tokens
        emb_w = self.token_embedding_matrix()
        se = emb_w[input_ids[0, sl]].clone().detach().to(self.dtype)
        se.requires_grad_(True)

        reps, _ = self.forward_joint(input_ids, attn, pv, grid, sl, se,
                                     need_logits=False, pool="last")
        l = max(reps.keys())
        loss = reps[l].float().pow(2).sum()          # arbitrary scalar
        loss.backward()

        g_img = None if pv.grad is None else float(pv.grad.abs().sum().item())
        g_sfx = None if se.grad is None else float(se.grad.abs().sum().item())
        ok_img = g_img is not None and g_img > 0.0
        ok_sfx = g_sfx is not None and g_sfx > 0.0
        ok = ok_img and ok_sfx
        logger.info(f"[parity] joint grad-flow: |grad pixel|={g_img} "
                    f"|grad suffix|={g_sfx} -> {'OK' if ok else 'MISMATCH'} "
                    f"(pixel {'ok' if ok_img else 'ZERO/None'}, "
                    f"suffix {'ok' if ok_sfx else 'ZERO/None'})")
        return ok

    # ------------------------------------------------------------------ #
    # Stage 2: differentiable pixel_values -> fused hidden states
    # ------------------------------------------------------------------ #
    def clean_text_inputs(self, text: str, image):
        """Run the processor once on the clean image to get input_ids /
        attention_mask / image_grid_thw. These stay fixed during the image
        attack; only pixel_values are perturbed."""
        inputs = self.build_inputs(text, image, add_generation_prompt=True)
        return inputs

    def hidden_states_from_pixels(self, input_ids, attention_mask,
                                  pixel_values, image_grid_thw, pool: str = "last"):
        """Differentiable forward: given (possibly perturbed) pixel_values,
        return {layer -> last-token hidden state} WITH gradients enabled so the
        crossing loss can backprop to the image. Only pixel_values carry grad;
        weights are frozen."""
        torch = self.torch
        pv = pixel_values.to(self.model.device, self.dtype)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pv,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        reps = {}
        for l in range(1, len(hs)):
            layer = hs[l]
            if pool == "mean":
                m = attention_mask.unsqueeze(-1).to(layer.dtype)
                vec = (layer * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                vec = self._pool_last_token(layer, attention_mask)
            reps[l] = vec.squeeze(0).float()          # fp32 for stable loss
        return reps