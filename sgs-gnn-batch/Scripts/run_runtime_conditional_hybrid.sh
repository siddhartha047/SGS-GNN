#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

PIPELINE="${PIPELINE:-hybrid}"
DATASETS_CSV="${DATASETS_CSV:-}"
RUNS="${RUNS:-1}"
EPOCHS="${EPOCHS:-20}"
EDGE_MLP_TYPE="${EDGE_MLP_TYPE:-GCN}"
GNN="${GNN:-GCN}"
NHID="${NHID:-64}"
SPARSE_EDGE_MLP="${SPARSE_EDGE_MLP:-True}"
REG1="${REG1:-True}"
REG2="${REG2:-True}"
EVAL="${EVAL:-False}"
HYBRID_CHECKPOINT="${HYBRID_CHECKPOINT:-True}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
RUN_TAG="${RUN_TAG:-}"

if [[ -n "${DATASETS_CSV}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
else
  DATASETS=(
    Tolokers
    genius
    # pokec
    arxiv-year
    # snap-patents
    Reddit
    # SmallCora
  )
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_ARGS_STR})
fi

LOG_SUFFIX=""
if [[ -n "${RUN_TAG}" ]]; then
  LOG_SUFFIX="_${RUN_TAG}"
fi

SUMMARY_FILE="${LOG_DIR}/runtime_conditional_${PIPELINE}${LOG_SUFFIX}_summary.txt"
echo "runtime_conditional_summary pipeline=${PIPELINE} runs=${RUNS} epochs=${EPOCHS} $(date -Is)" > "${SUMMARY_FILE}"

COMMON_ARGS=(
  --mode learned
  --runs "${RUNS}"
  --epochs "${EPOCHS}"
  --save_csv True
  --edge_mlp_type "${EDGE_MLP_TYPE}"
  --GNN "${GNN}"
  --log False
  --sparse_edge_mlp "${SPARSE_EDGE_MLP}"
  --reg1 "${REG1}"
  --reg2 "${REG2}"
  --nhid "${NHID}"
  --eval "${EVAL}"
  --stats True
  --hybrid_checkpoint "${HYBRID_CHECKPOINT}"
)

run_runtime () {
  local dataset="$1"
  local conditional="$2"
  local cond_tag="$3"
  local log_file="${LOG_DIR}/runtime_conditional_${dataset}_${PIPELINE}_${cond_tag}${LOG_SUFFIX}.log"
  echo "=== Running runtime: dataset=${dataset} pipeline=${PIPELINE} conditional=${conditional} ==="
  (cd "${ROOT_DIR}" && python main.py "${COMMON_ARGS[@]}" --dataset "${dataset}" --pipeline "${PIPELINE}" --conditional "${conditional}" "${EXTRA_ARGS[@]}") | tee "${log_file}"
  echo "--- Stats (${dataset} | conditional=${conditional} | ${PIPELINE}) ---"
  grep -n "\\[stats\\]" "${log_file}" || true
  if grep -q "\\[stats\\]" "${log_file}"; then
    grep "\\[stats\\]" "${log_file}" | awk -v d="${dataset}" -v c="${conditional}" -v p="${PIPELINE}" '{print "[" d " cond=" c " pipeline=" p "] " $0}' >> "${SUMMARY_FILE}"
  else
    echo "[${dataset} cond=${conditional} pipeline=${PIPELINE}] [stats] NOT FOUND" >> "${SUMMARY_FILE}"
  fi
  echo ""
}

for dataset in "${DATASETS[@]}"; do
  run_runtime "${dataset}" True "condTrue"
  run_runtime "${dataset}" False "condFalse"
done
