#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_cnn --config src/config/cnn.yaml
