"""
Loss ablation for JailBound (reproduces paper Figure 4).

Paper reports (image crossing ASR):
    full (L_align + L_geo) : 91.40%
    remove L_align         : 82.67%   (biggest drop)
    remove L_geo           : 85.79%

We can't match absolute numbers (different eval basis / model), but the ORDERING
    full  >  no-geo  >  no-align
is the reproduction signal: it shows each loss term is implemented and
contributes as the paper claims. L_align (crossing pressure) should matter most.

For speed we run on a stratified subsample (default 5 per category = 65) and
measure refusal-string ASR by generating on the adversarial image + original
prompt (suffix OFF, to isolate the image-loss ablation).

Usage (GPU):
  python -m attacks.jailbound.ablation \
    --behaviors dataset/jailbound_mmsb_test.json \
    --model_name /scratch/x3411a15/models/Qwen2.5-VL-7B-Instruct \
    --per_category 5 --out output_qwen/loss_ablation.json
"""
import argparse, json, importlib, logging
from collections import defaultdict
from pathlib import Path


def stratified_subsample(behaviors, per_category, seed=42):
    import random
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for b in behaviors:
        by_cat[b["id"].rsplit("_", 1)[0]].append(b)
    out = []
    for c, items in by_cat.items():
        rng.shuffle(items)
        out.extend(items[:per_category])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from attacks.jailbound.data import is_refusal
    from attacks.jailbound.crossing import CrossingConfig, load_boundary, crossing_image

    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--boundary_path", default="attacks/jailbound/cache/boundary.pt")
    ap.add_argument("--per_category", type=int, default=5)
    ap.add_argument("--n_iters", type=int, default=150)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="output_qwen/loss_ablation.json")
    args = ap.parse_args()

    behaviors = json.loads(Path(args.behaviors).read_text())
    sub = stratified_subsample(behaviors, args.per_category)
    logging.info(f"[ablation] {len(sub)} samples "
                 f"({args.per_category}/category)")

    mod = importlib.import_module("multimodalmodels.qwen2_5_vl.qwen_whitebox")
    model = mod.QwenVLWhiteBox(model_name=args.model_name, device="cuda",
                               dtype="float16", max_pixels=args.max_pixels,
                               gradient_checkpointing=True)

    # three settings; lambdas control which loss term is active
    settings = {
        "full":     dict(lambda_align=1.0, lambda_geo=1.0),
        "no_align": dict(lambda_align=0.0, lambda_geo=1.0),
        "no_geo":   dict(lambda_align=1.0, lambda_geo=0.0),
    }

    results = {}
    per_sample = defaultdict(dict)

    # resumable cache: {setting: {test_case_id: {"complied":bool,"d_final":float}}}
    cache_path = Path(args.out).with_suffix(".progress.json")
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            done_n = {k: len(v) for k, v in cache.items()}
            logging.info(f"[ablation] resuming from cache: {done_n}")
        except Exception:
            cache = {}

    for name, lam in settings.items():
        cfg = CrossingConfig(boundary_path=args.boundary_path, n_iters=args.n_iters,
                             **lam)
        B = load_boundary(cfg, device=model.model.device)
        cache.setdefault(name, {})
        for i, b in enumerate(sub):
            cid = b["id"]
            if cid in cache[name]:                      # already done -> skip
                rec = cache[name][cid]
            else:
                adv_pil, info = crossing_image(model, b["original_prompt"],
                                               b["image_path"], B, cfg)
                resp = model.generate(b["original_prompt"], adv_pil,
                                      max_new_tokens=args.max_new_tokens)
                rec = {"complied": (not is_refusal(resp)), "d_final": info.get("d_final")}
                cache[name][cid] = rec
                cache_path.write_text(json.dumps(cache))     # checkpoint each sample
            per_sample[cid][name] = rec
            if i % 20 == 0:
                logging.info(f"[ablation:{name}] {i}/{len(sub)}")
        n_comply = sum(int(cache[name][b["id"]]["complied"]) for b in sub)
        asr = 100.0 * n_comply / len(sub)
        results[name] = asr
        logging.info(f"[ablation] {name:>8} ASR = {asr:.1f}%  ({n_comply}/{len(sub)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"asr": results, "n": len(sub), "per_sample": per_sample}, indent=2))

    # summary + ordering check
    print("\n" + "=" * 60)
    print(" Loss ablation (refusal-string ASR, image crossing only)")
    print("=" * 60)
    print(f"   full  (align+geo) : {results['full']:5.1f}%   [paper 91.40%]")
    print(f"   no_geo            : {results['no_geo']:5.1f}%   [paper 85.79%]")
    print(f"   no_align          : {results['no_align']:5.1f}%   [paper 82.67%]")
    order_ok = results["full"] >= results["no_geo"] >= results["no_align"]
    align_worst = results["no_align"] <= results["no_geo"]
    print("-" * 60)
    print(f"   ordering full >= no_geo >= no_align : {'YES' if order_ok else 'NO'}")
    print(f"   removing L_align hurts most         : {'YES' if align_worst else 'NO'}")
    print("   (both YES = paper Fig 4 trend reproduced -> each loss term works)")
    print("=" * 60)


if __name__ == "__main__":
    main()