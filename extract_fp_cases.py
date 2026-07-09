#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP 케이스 추출: judge 가 safe(success=False)라 했는데
가드가 unsafe 로 막은 케이스만 모은다.

= 가드가 '과하게 막은' 케이스 (오탐/over-blocking).
  무해한 응답을 가드가 차단한 것 → 가드의 과민성/false positive 를 본다.

responses(가드 판정) + evaluations(judge 판정)을 test_case_id 로 조인해서,
  judge success=True AND 가드 verdict=safe 인 것만 추출.

각 FN 케이스에 대해 프롬프트/응답/가드판정/judge판정/이미지경로를 저장.

사용:
  python extract_fp_cases.py \
      --responses output_qwen/responses/llama_guard_3/attack_elite_model_qwen2.5-vl-7b-local.jsonl \
      --evaluations output_qwen/evaluations/attack_elite_model_qwen2.5-vl-7b-local_defense_llama_guard_3_evaluator_default_judge.jsonl \
      --out fn_cases.json \
      --csv fn_cases.csv
"""

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def md(row):
    """metadata 딕셔너리를 안전하게 꺼낸다."""
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def get_guard_verdict(row):
    """가드 판정(safe/unsafe)을 여러 가능한 필드에서 추출."""
    m = md(row)
    v = m.get("guard_verdict") or row.get("guard_verdict")
    if v in ("safe", "unsafe"):
        return v
    # 차단 흔적으로 추정
    if m.get("guard_blocked") is True or m.get("defense_blocked") is True:
        return "unsafe"
    # 가드가 붙었는데 차단 안 함 → safe
    if m.get("guard_name") or m.get("defense_method") not in (None, "None"):
        return "safe"
    return "unknown"


def get_field(row, *keys, default=""):
    """row 본문과 metadata 양쪽에서 키를 찾는다."""
    m = md(row)
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
        v = m.get(k)
        if v not in (None, ""):
            return v
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="가드 판정 jsonl")
    ap.add_argument("--evaluations", required=True, help="judge 판정 jsonl")
    ap.add_argument("--out", default="fp_cases.json", help="FP 케이스 JSON 저장 경로")
    ap.add_argument("--csv", default="", help="CSV 도 저장하려면 경로 지정")
    ap.add_argument("--full", action="store_true",
                    help="응답/프롬프트 전체 저장 (기본은 미리보기로 자름)")
    args = ap.parse_args()

    responses = load_jsonl(Path(args.responses))
    evaluations = load_jsonl(Path(args.evaluations))

    # judge 판정 인덱스: test_case_id -> (success, judge_score, judge_reason)
    judge_idx = {}
    for e in evaluations:
        tid = e.get("test_case_id")
        if tid is None:
            continue
        judge_idx[tid] = {
            "success": bool(e.get("success")),
            "judge_score": e.get("judge_score", e.get("score", "")),
            "judge_reason": e.get("judge_reason", e.get("combined_reason", "")),
        }

    fn_cases = []
    n_total = 0
    n_judge_safe = 0
    for row in responses:
        tid = row.get("test_case_id")
        if tid is None or tid not in judge_idx:
            continue  # judge 채점 안 된 케이스는 제외
        n_total += 1

        j = judge_idx[tid]
        if j["success"]:
            continue  # judge 가 unsafe 라 한 건 FP 대상 아님
        n_judge_safe += 1

        verdict = get_guard_verdict(row)
        if verdict != "unsafe":
            continue  # 가드가 safe 또는 unknown → FP 아님

        # === 여기 도달 = FP (judge safe, 가드 unsafe) ===
        prompt = get_field(row, "original_prompt", "jailbreak_prompt", "prompt")
        response = get_field(row, "model_response", "response", "output")
        image_path = get_field(row, "image_path", "original_image_path",
                               "jailbreak_image_path")

        def cut(s, n=300):
            s = str(s)
            return s if args.full or len(s) <= n else s[:n] + " …(잘림)"

        fn_cases.append({
            "test_case_id": tid,
            "image_path": image_path,
            "prompt": cut(prompt),
            "response": cut(response),
            "guard_verdict": verdict,
            "guard_unsafe_prob": md(row).get("guard_unsafe_prob_norm",
                                              md(row).get("guard_unsafe_prob")),
            "judge_success": j["success"],
            "judge_score": j["judge_score"],
            "judge_reason": cut(j["judge_reason"], 400),
            "category": get_field(row, "main_category", "subcategory",
                                  "category", default=""),
        })

    # 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(fn_cases, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  judge 채점된 케이스        : {n_total}")
    print(f"  judge=safe (not success)  : {n_judge_safe}")
    print(f"  FP (judge safe+가드unsafe) : {len(fn_cases)}")
    if n_judge_safe:
        print(f"  → 가드 과차단 비율         : {len(fn_cases)/n_judge_safe*100:.1f}%")
    print("=" * 60)
    print(f"[+] 저장: {out_path}")

    if args.csv and fn_cases:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fn_cases[0].keys()))
            w.writeheader()
            w.writerows(fn_cases)
        print(f"[+] CSV 저장: {args.csv}")

    # 카테고리별 분포 (어떤 종류를 많이 놓치는지)
    if fn_cases:
        from collections import Counter
        cats = Counter(c["category"] for c in fn_cases)
        print("\n[+] FP 카테고리별 분포 (가드가 과하게 막는 유형):")
        for c, n in cats.most_common():
            print(f"      {c or '(없음)':25s} {n}")


if __name__ == "__main__":
    main()