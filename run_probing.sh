#!/bin/sh
#SBATCH -J jb_SD_TYPO_probe                # job name
#SBATCH -p amd_a100nv_8              # V100 partition (check `sinfo` for availability)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # probing uses ONE GPU (single model, batch 1)
#SBATCH --time=04:00:00           
#SBATCH --comment=pytorch          # REQUIRED on Neuron or the job is rejected
#SBATCH -o /scratch/x3411a15/OmniSafeBench-MM/logs/%x/%x_%j.out
#SBATCH -e /scratch/x3411a15/OmniSafeBench-MM/logs/%x/%x_%j.err

set -e

# --- environment -----------------------------------------------------------
module purge
source /apps/applications/Miniconda/25.11.1/bin/activate /scratch/$USER/omni
cd /scratch/$USER/OmniSafeBench-MM

# avoid tokenizer thread oversubscription on the compute node
export TOKENIZERS_PARALLELISM=false

# --- run -------------------------------------------------------------------

python -m attacks.jailbound.probing \
    --mm_root /scratch/x3411a15/MM-SafetyBench/data \
    --image_type SD_TYPO \
    --label_mode model_behavior \
    --behavior_criterion safe \
    --max_new_tokens 256 \
    --cache_dir attacks/jailbound/SD_TYPO/cache \
    --model_name /scratch/x3411a15/models/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16

# Notes:
#  * Behavior labels are checkpointed to attacks/jailbound/cache/behavior_labels.json.
#    If this job is ever killed/timed out, just resubmit - it resumes labeling.
#  * For a quick functional test on a compute node use gdebug (interactive):
#       ssh gdebug01
#       python -m attacks.jailbound.probing --mm_root ... --model_name ... --max_per_category 20
#    but run the FULL job through sbatch (login/debug nodes have a 2h CPU cap).