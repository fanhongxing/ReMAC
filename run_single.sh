#!/usr/bin/env bash
set -euo pipefail

# Tunable via env
IMAGE="${IMAGE:-asset/spraypaintcan.png}"
SEG_TEXT="${SEG_TEXT:-spraypaintcan}"
PROMPT="${PROMPT:-}"
SEG_URL="${SEG_URL:-}"
SEG_BACKEND="${SEG_BACKEND:-xsam}"

# LLM model choices: gpt4o|qwen for general, gemini|qwen for check, gemini_flash|qwen for ranking
LLM_GENERAL="${LLM_GENERAL:-gpt4o}"
LLM_CHECK="${LLM_CHECK:-gemini}"
# LLM_RANK="${LLM_RANK:-gemini_flash}"

# mkdir -p "$(dirname "$OUT")"

cmd=(
  python main.py
  --image "$IMAGE"
  --seg-text "$SEG_TEXT"
  --boundary-mode boundary_bbox
  --dilate-k 7
  --dilate-iters 2
  --edge-grow-px 10
  --seg-backend "$SEG_BACKEND"
  --num-images 1
  --generation-mode simple
  --llm-general "$LLM_GENERAL"
  --llm-check "$LLM_CHECK"
  # --llm-rank "$LLM_RANK"
  --restore-square-crop
  --save-intermediate
)

if [[ -n "$PROMPT" ]]; then
  cmd+=(--prompt "$PROMPT")
fi
if [[ -n "$SEG_URL" ]]; then
  cmd+=(--seg-url "$SEG_URL")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
