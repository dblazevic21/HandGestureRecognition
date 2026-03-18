from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    background_labels: frozenset[str] = field(default_factory=lambda: frozenset({"background"}))
    ignored_project_dirs: frozenset[str] = field(
        default_factory=lambda: frozenset({"artifacts", "mediapipe_app", "__pycache__", ".venv", "venv"})
    )
    random_state: int = 42
    min_samples_per_class: int = 5

    @property
    def workspace_dir(self) -> Path:
        return self.project_dir.parent

    @property
    def dataset_roots(self) -> tuple[Path, ...]:
        workspace = self.workspace_dir
        return (
            workspace / "24000-900-300t",
            workspace / "L-thumbs",
            workspace / "L-thumbs" / "dataset",
            workspace / "leapGestRecog",
            workspace / "ASL" / "captured",
            self.project_dir,
            workspace / "rock-paper-scissors",
            workspace / "rock-paper-scissors" / "HandGesture",
            workspace / "rock-paper-scissors" / "HandGesture" / "images",
        )

    @property
    def artifacts_dir(self) -> Path:
        return self.project_dir / "artifacts"

    @property
    def asl_dir(self) -> Path:
        return self.workspace_dir / "ASL"

    @property
    def asl_capture_dir(self) -> Path:
        return self.asl_dir / "captured"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "gesture_model.joblib"

    @property
    def summary_path(self) -> Path:
        return self.artifacts_dir / "training_summary.json"

    @property
    def dataset_summary_path(self) -> Path:
        return self.artifacts_dir / "dataset_summary.json"

    def ensure_artifacts_dir(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
