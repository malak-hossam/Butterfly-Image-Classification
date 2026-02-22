#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/models/resnet18/best.pt}"
CONFIG="${2:-src/config/resnet18.yaml}"
SPLIT="${3:-test}"

python -m src.eval.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT" --split "$SPLIT"
