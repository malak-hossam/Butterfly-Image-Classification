from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_config = cfg.pop("base_config", None)
    if base_config is None:
        return cfg

    base_path = (config_path.parent / base_config).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    base_cfg = load_config(base_path)
    return _deep_merge(base_cfg, cfg)
