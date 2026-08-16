#!/bin/sh
#SBATCH -J omni_jb
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment=pytorch
#SBATCH -o /scratch/x3411a15/OmniSafeBench-MM/logs/%x/%x_%j.out
#SBATCH -e /scratch/x3411a15/OmniSafeBench-MM/logs/%x/%x_%j.err

# =====================================================================
#  통합 가드 평가 파이프라인 (타깃+가드 한 흐름 → judge → 분석)
#
#  통합 방식: config 의 defenses 에 가드(llama_guard_3)를 켜면,
#    run_pipeline.py 가 타깃 응답 생성 + 가드 판정을 한 흐름에 처리한다.
#    (가드가 reply_directly/block_input 로 응답을 metadata 에 넣으면
#     파이프라인이 그걸 최종 출력으로 사용 → 타깃 중복 호출 없음)
#
#  한 번의 실행에서 두 지표가 모두 나온다:
#    - ASR(실전 효과): 가드 끼운 시스템을 공격이 뚫는 비율
#    - recall/FPR(분류 성능): 가드 verdict 를 라벨과 비교
#
#  실행:  sbatch run_full_pipeline.sh
#  모니터: squeue -u $USER  /  tail -f omni_guard_<jobid>.out
#   squeue -p cas_v100_4 -t PENDING --sort=-p,i | head -20
#   scontrol show job 852533
# nns, nqs
# =====================================================================

set -e

# ---- 환경 ----
module purge
source /apps/applications/Miniconda/25.11.1/bin/activate /scratch/$USER/omni
cd /scratch/$USER/OmniSafeBench-MM

export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1          # 로컬 weight 만 사용

CONFIG=config/guard_eval_config.yaml
ATTACK=jailbound
TARGET=qwen2.5-vl-7b-local
RESP=output_qwen/responses/None/attack_${ATTACK}_model_${TARGET}.jsonl

echo "=============================================="
echo " START  $(date)   node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

# ---- [1/4] 테스트케이스 생성 ----
# echo "[1/4] test_case_generation  $(date)"
python run_pipeline.py --config $CONFIG --stage test_case_generation

# ---- [2/4] 통합 응답 생성 (타깃+가드 한 흐름) ----
#   config 의 defenses 에 llama_guard_3 가 켜져 있어야 통합으로 동작.
#   가드가 켜지면 가드가 응답 생성+판정, 파이프라인은 그 결과를 사용.
echo "[2/4] response_generation (target + guard integrated)  $(date)"
python run_pipeline.py --config $CONFIG --stage response_generation

# ---- [3/4] judge 채점 (70B judge 가 최종 출력의 유해성 판정) ----
#   가드를 거친 최종 응답(responses/llama_guard_3/)을 채점.
# echo "[3/4] evaluation (70B judge)  $(date)"
# python run_pipeline.py --config $CONFIG --stage evaluation \
#     --input-file $RESP

# ---- [4/4] 분석: ASR + recall/FPR 동시 산출 ----
# echo "[4/4] analyze  $(date)"
# python analyze_guard_results.py \
#     --responses-dir output_qwen/responses/llama_guard_3 \
#     --evaluations-dir output_qwen/evaluations \
#     --judge qwen2.5-vl-32b-judge \
#     --out output_qwen/guard_report

echo "=============================================="
echo " DONE   $(date)"
echo "=============================================="