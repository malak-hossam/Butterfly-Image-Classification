from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(pipeline_name: str, outputs_root: str = "outputs") -> dict[str, Path]:
    root = resolve_project_path(outputs_root)
    model_dir = ensure_dir(root / "models" / pipeline_name)
    metrics_dir = ensure_dir(root / "metrics" / pipeline_name)
    plots_dir = ensure_dir(root / "plots")
    predictions_dir = ensure_dir(root / "predictions")
    return {
        "root": root,
        "model_dir": model_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "predictions_dir": predictions_dir,
    }
