#!/bin/bash
#SBATCH --job-name=uq_report
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail

# Usage:
#   sbatch scripts/report_v2.slurm
#
# Environment overrides:
#   OUTPUTS_ROOT=outputs           Root(s) to scan for uq_eval_metrics.json
#   LATENCY_DIR=outputs/latency    Directory with per-config latency profiles
#   OUTPUT_DIR=outputs/report      Where to write XLSX and plots
#   METRIC=prr                     Performance metric to report
#   PLOT_FORMAT=png                Plot format (pdf/png/svg)

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

REPORT_ARGS=()

[ -n "${OUTPUTS_ROOT:-}" ] && REPORT_ARGS+=(--outputs_root "$OUTPUTS_ROOT")
[ -n "${LATENCY_DIR:-}" ] && REPORT_ARGS+=(--latency_dir "$LATENCY_DIR")
[ -n "${OUTPUT_DIR:-}" ] && REPORT_ARGS+=(--output_dir "$OUTPUT_DIR")
[ -n "${METRIC:-}" ] && REPORT_ARGS+=(--metric "$METRIC")
[ -n "${PLOT_FORMAT:-}" ] && REPORT_ARGS+=(--format "$PLOT_FORMAT")

print_uq_header "Report: XLSX + Pareto plots"

run_in_uq_container cpu python -m src.cli.report "${REPORT_ARGS[@]}"

echo
echo "Report finished at: $(date)"
