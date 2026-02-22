#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_resnet18 --config src/config/resnet18.yaml
