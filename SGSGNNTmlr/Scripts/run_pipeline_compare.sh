#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMMON_ARGS=(
  --dataset Reddit
  --mode learned
  --runs 1
  --epochs 2
  --save_csv True
  --edge_mlp_type GCN
  --GNN GCN
  --log False
  --sparse_edge_mlp True
  --conditional True
  --reg1 True
  --reg2 True
  --stats True
  --hybrid_checkpoint True
)

run_pipeline () {
  local pipeline="$1"
  local log_file="${ROOT_DIR}/pipeline_${pipeline}.log"
  echo "=== Running pipeline: ${pipeline} ==="
  (cd "${ROOT_DIR}" && python main.py "${COMMON_ARGS[@]}" --pipeline "${pipeline}") | tee "${log_file}"
  echo "--- Stats (${pipeline}) ---"
  grep -n "\\[stats\\]" "${log_file}" || true
  echo ""  
}

run_pipeline "two_pass"
run_pipeline "straight_through"
run_pipeline "hybrid"
