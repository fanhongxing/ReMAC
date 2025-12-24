#!/usr/bin/env bash
set -euo pipefail

# Tunable knobs via env vars (override as needed)
# SRC_DIR="${SRC_DIR:-data/cocoa}"
# DST_DIR="${DST_DIR:-results/cocoa}"

# LLM model choices: gpt4o|qwen for general, gemini|qwen for check, gemini_flash|qwen for ranking
LLM_GENERAL="${LLM_GENERAL:-gpt4o}"
LLM_CHECK="${LLM_CHECK:-gemini}"
LLM_RANK="${LLM_RANK:-gemini_flash}"

cmd=(
  python batch_cocoa_test.py
  # --src-dir "$SRC_DIR"
  # --dst-dir "$DST_DIR"
  --seg-backend "${SEG_BACKEND:-xsam}"
  --boundary-mode boundary_bbox
  --dilate-k 7
  --dilate-iters 2
  --edge-grow-px 10
  --num-images 3
  --generation-mode simple
  --llm-general "$LLM_GENERAL"
  --llm-check "$LLM_CHECK"
  --llm-rank "$LLM_RANK"
  --disable-boundary-analysis
  --restore-square-crop
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
