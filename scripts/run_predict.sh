#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-single}"
TARGET="${2:-}"
CHECKPOINT="${3:-outputs/models/resnet18/best.pt}"
CONFIG="${4:-src/config/resnet18.yaml}"
TOP_K="${5:-5}"

if [[ -z "$TARGET" ]]; then
  echo "Usage:"
  echo "  $0 single path/to/image.jpg [checkpoint] [config] [top_k]"
  echo "  $0 folder path/to/images_dir [checkpoint] [config] [top_k]"
  exit 1
fi

if [[ "$MODE" == "single" ]]; then
  python -m src.infer.predict \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --image_path "$TARGET" \
    --top_k "$TOP_K"
elif [[ "$MODE" == "folder" ]]; then
  python -m src.infer.predict \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --image_dir "$TARGET" \
    --top_k "$TOP_K"
else
  echo "MODE must be 'single' or 'folder'"
  exit 1
fi
