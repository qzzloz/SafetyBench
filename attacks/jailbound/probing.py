"""
JailBound Stage 1: Safety Boundary Probing.

Pipeline:
  1. build contrastive (safe/unsafe) examples from the 70% train split
  2. run the white-box Qwen2.5-VL forward once per example, cache the fused
     representation h^(l) at every fusion layer (last-token pooled)
  3. fit one logistic-regression classifier per layer  ->  w^(l), b^(l)
  4. store, per layer:  v^(l)=w/||w||,  ||w^(l)||,  b^(l),  eps^(l),  accuracy
     (eps^(l) = mean signed distance of the UNSAFE class to the boundary; the
      crossing stage uses it to set the target margin. P0 threshold is applied
      in Stage 2, not here.)

The artifact `boundary.pt` is everything Stage 2 (crossing.py) needs.

Validation: the paper reports probing accuracy ~100% on deep layers and poor
accuracy on layers 0-4. Print `per_layer_acc` after running; if deep layers
are not near 1.0 something is wrong (wrong pooling position, bf16 NaNs, or the
label construction collapsed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProbingConfig:
    target_model: str = "qwen2_5_vl.qwen_whitebox.QwenVLWhiteBox"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "float16"                 # V100 -> fp16
    pool: str = "last"                     # "last" | "mean"
    cache_dir: str = "attacks/jailbound/cache"
    boundary_path: str = "attacks/jailbound/cache/boundary.pt"
    C: float = 1.0                         # logistic-regression inverse reg
    max_iter: int = 2000


def _load_whitebox(cfg: ProbingConfig):
    """Instantiate the white-box target from cfg.target_model.

    We deliberately avoid `import multimodalmodels.<...>` because the repo's
    `multimodalmodels/__init__.py` eagerly imports MiniGPT4 (pulling omegaconf
    and the whole MiniGPT-4 dependency stack). JailBound doesn't use MiniGPT4,
    so we first try a normal import and, if the package __init__ blows up on a
    missing optional dependency, fall back to loading the module file directly
    (which bypasses the package __init__ entirely).
    """
    import importlib

    module_path, cls = cfg.target_model.rsplit(".", 1)   # e.g. qwen2_5_vl.qwen_whitebox . QwenVLWhiteBox
    try:
        mod = importlib.import_module(f"multimodalmodels.{module_path}")
    except Exception as e:
        logger.warning(
            f"[probe] normal import of multimodalmodels.{module_path} failed "
            f"({type(e).__name__}: {e}); loading the module file directly."
        )
        import importlib.util
        from pathlib import Path

        # <repo>/attacks/jailbound/probing.py  ->  <repo>/multimodalmodels/<...>.py
        repo_root = Path(__file__).resolve().parents[2]
        rel = module_path.replace(".", "/") + ".py"
        file_path = repo_root / "multimodalmodels" / rel
        if not file_path.exists():
            raise FileNotFoundError(
                f"white-box model file not found: {file_path}"
            ) from e
        spec = importlib.util.spec_from_file_location(f"_jb_{module_path.replace('.', '_')}", file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    Model = getattr(mod, cls)
    return Model(model_name=cfg.model_name, device=cfg.device, dtype=cfg.dtype)


def extract_features(samples, model, cfg: ProbingConfig):
    """Return (X_by_layer: {l -> (N,H) float tensor}, y: (N,) long tensor).

    Features are cached to disk keyed by (case_id, pool) so re-runs are cheap.
    """
    import torch

    cache = Path(cfg.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    feats_by_layer: Dict[int, List[torch.Tensor]] = {}
    labels: List[int] = []

    req = [{"text": s.text, "image": s.image} for s in samples]
    ids = [s.case_id for s in samples]

    for i, reps in model.batched_fused_representations(req, pool=cfg.pool):
        # per-sample cache
        cpath = cache / f"feat_{ids[i]}_{cfg.pool}.pt"
        torch.save(reps, cpath)
        for l, vec in reps.items():
            feats_by_layer.setdefault(l, []).append(vec)
        labels.append(samples[i].label)

    X_by_layer = {l: torch.stack(v) for l, v in feats_by_layer.items()}
    y = torch.tensor(labels, dtype=torch.long)
    logger.info(f"[probe] features: {len(labels)} samples, "
                f"{len(X_by_layer)} layers, H={next(iter(X_by_layer.values())).shape[1]}")
    return X_by_layer, y


def fit_boundary_one_layer(X, y, C: float, max_iter: int) -> dict:
    """Fit LR on one layer's features. X:(N,H) tensor, y:(N,) tensor.

    Reports CROSS-VALIDATED balanced accuracy, not training accuracy: with
    N < H (e.g. 52 samples in 3584 dims) the data is ALWAYS linearly separable,
    so training accuracy is 1.0 at every layer regardless of meaning. CV
    balanced accuracy is the honest probe metric and reveals the real
    early-poor / deep-sharp profile.

    The final boundary is fit with feature standardization + L2, then folded
    back to RAW hidden-state space so Stage 2 can operate on raw activations:
      x' = (x - mu)/sd,  w'.x' + b'  ==>  w = w'/sd,  b = b' - sum(w'*mu/sd)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    Xn = X.numpy().astype("float64")
    yn = y.numpy().astype(int)
    minority = int(min((yn == 0).sum(), (yn == 1).sum()))

    # ---- cross-validated balanced accuracy (the honest metric) ----
    cv_acc = float("nan")
    if minority >= 2:
        n_splits = max(2, min(5, minority))
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        cv_acc = float(cross_val_score(pipe, Xn, yn, cv=skf,
                                       scoring="balanced_accuracy").mean())

    # ---- final boundary on all data, standardized then folded to raw space ----
    scaler = StandardScaler().fit(Xn)
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
    clf.fit(scaler.transform(Xn), yn)
    w_s = clf.coef_[0]; b_s = float(clf.intercept_[0])
    mu, sd = scaler.mean_, scaler.scale_
    w = w_s / sd
    b = b_s - float(np.sum(w_s * mu / sd))
    w_norm = float(np.linalg.norm(w))
    v = (w / w_norm).astype("float32")                 # unit normal, safe->unsafe
    dist = (Xn @ w + b) / w_norm
    eps = float(dist[yn == 1].mean())                  # mean margin of unsafe class
    train_acc = float(clf.score(scaler.transform(Xn), yn))
    return {"v": v, "w_norm": w_norm, "bias": b, "eps": eps,
            "acc": cv_acc, "train_acc": train_acc, "minority": minority}


def label_by_model_behavior(harmful_items, model, max_new_tokens: int = 96,
                            cache_dir: Optional[str] = None):
    """Label each harmful (image, question) by the model's OWN behavior.

    This is the "safety label determined by the VLM": run generation, then
      complied (not a refusal) -> y=1 (unsafe: harmful output produced)
      refused                  -> y=0 (safe: safety mechanism fired)
    Both classes come from the SAME harmful input distribution, so the probe
    cannot cheat on surface features - early layers stay near chance and the
    boundary sharpens with depth, which is the genuine safety boundary the
    crossing stage attacks.

    Labels are cached to <cache_dir>/behavior_labels.json and reused on reruns,
    so a killed job resumes instead of regenerating everything.
    """
    import json
    from attacks.jailbound.data import is_refusal, Sample

    cache_path = None
    labels_cache = {}
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / "behavior_labels.json"
        if cache_path.exists():
            labels_cache = json.loads(cache_path.read_text())
            logger.info(f"[probe] resuming: {len(labels_cache)} cached behavior labels")

    def _flush():
        if cache_path:
            cache_path.write_text(json.dumps(labels_cache))

    samples, n_comply, n_refuse = [], 0, 0
    n = len(harmful_items)
    for i, it in enumerate(harmful_items):
        cid = it["case_id"]
        if cid in labels_cache:
            label = int(labels_cache[cid])
        else:
            resp = model.generate(it["text"], it.get("image_path"),
                                  max_new_tokens=max_new_tokens)
            label = 0 if is_refusal(resp) else 1
            labels_cache[cid] = label
            if i % 50 == 0:
                _flush()
        n_refuse += int(label == 0); n_comply += int(label == 1)
        samples.append(Sample(it["text"], it["image_path"], label,
                              cid.split("_")[0], cid))
        if i % 100 == 0:
            logger.info(f"[probe] behavior-labeling {i}/{n}")
    _flush()
    logger.info(f"[probe] behavior labels -> comply/unsafe(1)={n_comply}, "
                f"refuse/safe(0)={n_refuse}")
    if n_comply == 0 or n_refuse == 0:
        logger.warning(
            "[probe] only ONE class present after behavior labeling. The probe "
            "needs both. If all refused, the attack images aren't jailbreaking "
            "this model (try image_type=SD_TYPO, more samples, or add benign "
            "inputs for the comply class); if all complied, add harder inputs.")
    return samples


def run_probing(samples, cfg: Optional[ProbingConfig] = None, model=None) -> dict:
    """Full Stage 1. Returns the boundary dict and saves it to cfg.boundary_path.
    Pass a preloaded `model` to reuse it (e.g. from behavior labeling)."""
    import torch

    cfg = cfg or ProbingConfig()
    if model is None:
        model = _load_whitebox(cfg)

    # guard: probing needs both classes
    labels_present = set(int(s.label) for s in samples)
    if len(labels_present) < 2:
        raise ValueError(f"probing needs 2 classes, got labels={labels_present}. "
                         f"See the behavior-labeling warning above.")

    X_by_layer, y = extract_features(samples, model, cfg)

    boundary = {"layers": {}, "meta": {"pool": cfg.pool,
                                        "model_name": cfg.model_name,
                                        "num_layers": model.num_layers}}
    per_layer_acc = {}
    for l in sorted(X_by_layer):
        res = fit_boundary_one_layer(X_by_layer[l], y, cfg.C, cfg.max_iter)
        boundary["layers"][l] = {
            "v": torch.tensor(res["v"]),
            "w_norm": res["w_norm"],
            "bias": res["bias"],
            "eps": res["eps"],
            "acc": res["acc"],
            "train_acc": res["train_acc"],
            "minority": res["minority"],
        }
        per_layer_acc[l] = res["acc"]

    boundary["per_layer_acc"] = per_layer_acc
    outp = Path(cfg.boundary_path); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(boundary, outp)

    logger.info("[probe] per-layer CV balanced-accuracy (train_acc in parens is "
                "meaningless when N<H; watch the CV number):")
    for l in sorted(per_layer_acc):
        L = boundary["layers"][l]
        logger.info(f"    layer {l:>2}: cv_acc={per_layer_acc[l]:.3f} "
                    f"(train={L['train_acc']:.3f}) eps={L['eps']:+.3f}")

    # --- leakage / validity diagnostic (uses CV, not training, accuracy) ------
    import math
    early = [per_layer_acc[l] for l in sorted(per_layer_acc)
             if l <= 4 and not math.isnan(per_layer_acc[l])]
    deep = [per_layer_acc[l] for l in sorted(per_layer_acc)
            if l >= (model.num_layers - 8) and not math.isnan(per_layer_acc[l])]
    minority = boundary["layers"][sorted(boundary["layers"])[0]].get("minority", 0)
    if minority < 10:
        logger.warning(
            "[probe] minority class has only %d samples - CV accuracy is noisy. "
            "Run on more data (drop --max_per_category, or raise it to >=50) for a "
            "trustworthy profile.", minority)
    if early and (sum(early) / len(early)) > 0.9:
        logger.warning(
            "[probe] even CROSS-VALIDATED early-layer accuracy is high (mean=%.3f). "
            "If this persists on the full dataset, the two behavior classes are "
            "separable by a shallow cue - inspect a few generations to confirm the "
            "refusal/compliance labels are correct.", sum(early) / len(early))
    elif early and deep:
        logger.info("[probe] profile: early cv_acc=%.3f -> deep cv_acc=%.3f "
                    "(rising with depth = genuine safety boundary).",
                    sum(early)/len(early), sum(deep)/len(deep))

    logger.info(f"[probe] saved boundary -> {outp}")
    return boundary


# --------------------------------------------------------------------------- #
# CLI entry: python -m attacks.jailbound.probing --mm_root /path/to/MM-SafetyBench
# --------------------------------------------------------------------------- #
def _main():
    import argparse, logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from attacks.jailbound.data import (
        ProbingDataConfig, load_mm_safetybench, stratified_split, build_contrastive,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_root", required=True)
    ap.add_argument("--image_type", default="SD_TYPO")
    ap.add_argument("--text_field", default="auto")
    ap.add_argument("--label_mode", default="model_behavior",
                    choices=["model_behavior", "benign_contrast", "continuation", "contrast_text"])
    ap.add_argument("--benign_img_dir", default="")
    ap.add_argument("--benign_questions_file", default="")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", default="last")
    ap.add_argument("--cache_dir", default="attacks/jailbound/cache")
    ap.add_argument("--max_new_tokens", type=int, default=96,
                    help="generation length for behavior labeling")
    ap.add_argument("--max_per_category", type=int, default=None,
                    help="cap items per category for a quick smoke test")
    args = ap.parse_args()

    dcfg = ProbingDataConfig(
        mm_root=args.mm_root, image_type=args.image_type,
        text_field=args.text_field, label_mode=args.label_mode,
        benign_img_dir=args.benign_img_dir,
        benign_questions_file=args.benign_questions_file,
        cache_dir=args.cache_dir, max_per_category=args.max_per_category,
    )
    per_cat = load_mm_safetybench(dcfg)
    train, _test = stratified_split(per_cat, dcfg.train_ratio, dcfg.seed)

    pcfg = ProbingConfig(model_name=args.model_name, device=args.device,
                         pool=args.pool, cache_dir=args.cache_dir)
    model = _load_whitebox(pcfg)   # load once, reuse for labeling + features

    if dcfg.label_mode == "model_behavior":
        samples = label_by_model_behavior(train, model,
                                          max_new_tokens=args.max_new_tokens,
                                          cache_dir=args.cache_dir)
    else:
        samples = build_contrastive(train, dcfg)

    run_probing(samples, pcfg, model=model)


if __name__ == "__main__":
    _main()