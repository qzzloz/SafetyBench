"""
Build the JailBound evaluation behaviors_file from MM-SafetyBench.

CRITICAL: the attack must run on the HELD-OUT 30% test split - the same split
Stage-1 probing used (seed 42, train_ratio 0.70). The 70% train split was used
to learn the safety boundary, so attacking it would be train/test leakage. This
matches the paper's protocol ("ASR measured only on the held-out 30%").

Output rows match the pipeline's behaviors_file schema:
    {image_path, original_prompt, style, main_category, subcategory, id}

Usage:
    python -m attacks.jailbound.build_behaviors \
        --mm_root /scratch/x3411a15/MM-SafetyBench/data \
        --out dataset/jailbound_mmsb_test.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from attacks.jailbound.data import (
        ProbingDataConfig, load_mm_safetybench, stratified_split,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_root", required=True)
    ap.add_argument("--image_type", default="SD_TYPO")
    ap.add_argument("--text_field", default="auto")
    ap.add_argument("--train_ratio", type=float, default=0.70)  # must match probing
    ap.add_argument("--seed", type=int, default=42)             # must match probing
    ap.add_argument("--out", default="dataset/jailbound_mmsb_test.json")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    args = ap.parse_args()

    dcfg = ProbingDataConfig(mm_root=args.mm_root, image_type=args.image_type,
                             text_field=args.text_field, train_ratio=args.train_ratio,
                             seed=args.seed)
    per_cat = load_mm_safetybench(dcfg)
    train, test = stratified_split(per_cat, dcfg.train_ratio, dcfg.seed)
    items = {"test": test, "train": train, "all": train + test}[args.split]

    rows, skipped = [], 0
    for it in items:
        if not it["image_path"]:            # need an image on disk to attack
            skipped += 1
            continue
        cat = it["case_id"].split("_")[0]
        rows.append({
            "image_path": str(Path(it["image_path"]).resolve()),
            "original_prompt": it["text"],
            "style": "declarative",
            "main_category": cat,
            "subcategory": cat,
            "id": it["case_id"],
        })

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    logging.info(f"[build] split={args.split}  wrote {len(rows)} behaviors "
                 f"(skipped {skipped} without images) -> {outp}")
    if skipped:
        logging.warning("[build] some items had no image on disk - download the "
                        "MM-SafetyBench image zip if you expected all of them.")


if __name__ == "__main__":
    main()