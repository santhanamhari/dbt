#!/bin/bash
#SBATCH --output=/groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/jobs/slurm-%j.out
# Convenience wrapper to submit distributed finetuning to SLURM.
#
# Usage:
#   sh demo/finetune_distributed.sh                      # 4 nodes (8 GPUs)
#   NODES=8 sh demo/finetune_distributed.sh              # 8 nodes (16 GPUs)
#   sh demo/finetune_distributed.sh --init_lr 1e-4       # Pass extra training flags

set -euo pipefail

NODES=${NODES:-4}
echo "Submitting distributed finetuning on ${NODES} nodes ($(( NODES * 2 )) GPUs)..."

sbatch --nodes="${NODES}" /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/scripts/slurm_finetune.sh "$@"
