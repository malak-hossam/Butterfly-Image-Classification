#!/usr/bin/env bash
set -euo pipefail

python -m src.data.sanity_check --config src/config/default.yaml --save_grid
