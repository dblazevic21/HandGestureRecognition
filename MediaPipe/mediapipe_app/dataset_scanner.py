from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

from .config import AppConfig


@dataclass(slots=True)
class ImageRecord:
    path: Path
    source_dataset: str
    raw_label: str
    display_label: str
    label_key: str
    split: str


class DatasetScanner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def collect_images(self, max_images_per_class: int | None = None) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        seen_absolute_paths: set[str] = set()
        seen_normalized_variants: set[str] = set()
        per_label_counter: Counter[str] = Counter()

        for root in self.config.dataset_roots:
            if not root.exists():
                continue

            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self.config.image_extensions:
                    continue
                if self._should_ignore_project_file(file_path):
                    continue

                absolute_key = file_path.resolve().as_posix().lower()
                dedupe_key = self._build_dedupe_key(root, file_path)
                if absolute_key in seen_absolute_paths or dedupe_key in seen_normalized_variants:
                    continue

                record = self._build_record(root, file_path)
                if max_images_per_class is not None and per_label_counter[record.label_key] >= max_images_per_class:
                    continue

                seen_absolute_paths.add(absolute_key)
                seen_normalized_variants.add(dedupe_key)
                records.append(record)
                per_label_counter[record.label_key] += 1

        records.sort(key=lambda item: (item.source_dataset.lower(), item.display_label.lower(), str(item.path).lower()))
        return records

    def build_summary(self, records: list[ImageRecord]) -> dict:
        per_dataset: defaultdict[str, int] = defaultdict(int)
        per_label: defaultdict[str, int] = defaultdict(int)
        per_split: defaultdict[str, int] = defaultdict(int)

        for record in records:
            per_dataset[record.source_dataset] += 1
            per_label[f"{record.source_dataset} / {record.display_label}"] += 1
            per_split[record.split] += 1

        top_labels = dict(sorted(per_label.items(), key=lambda item: (-item[1], item[0]))[:25])
        dataset_breakdown = dict(sorted(per_dataset.items(), key=lambda item: item[0].lower()))

        return {
            "ukupno_slika": len(records),
            "dataseti": dataset_breakdown,
            "splitovi": dict(sorted(per_split.items(), key=lambda item: item[0])),
            "top_klase": top_labels,
        }

    def export_summary_file(self, records: list[ImageRecord]) -> dict:
        self.config.ensure_artifacts_dir()
        summary = self.build_summary(records)
        summary["dataset_roots"] = [str(path) for path in self.config.dataset_roots if path.exists()]
        self.config.dataset_summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def _build_record(self, root: Path, file_path: Path) -> ImageRecord:
        raw_label = file_path.parent.name.strip()
        display_label = self._format_label(raw_label)
        source_dataset = self._resolve_source_dataset(root)
        label_key = f"{source_dataset}::{self._canonicalize_label(display_label)}"
        split = self._infer_split(file_path.relative_to(root).parts)
        return ImageRecord(
            path=file_path,
            source_dataset=source_dataset,
            raw_label=raw_label,
            display_label=display_label,
            label_key=label_key,
            split=split,
        )

    def _build_dedupe_key(self, root: Path, file_path: Path) -> str:
        relative_parts = list(file_path.relative_to(root).parts)
        while relative_parts and relative_parts[0].lower() == root.name.lower():
            relative_parts = relative_parts[1:]
        normalized_path = Path(*relative_parts) if relative_parts else Path(file_path.name)
        return f"{root.resolve().as_posix().lower()}::{normalized_path.as_posix().lower()}"

    def _format_label(self, raw_label: str) -> str:
        trimmed = raw_label.strip()
        if "_" in trimmed:
            prefix, suffix = trimmed.split("_", 1)
            if prefix.isdigit():
                trimmed = suffix
        return trimmed.replace("_", " ")

    def _canonicalize_label(self, label: str) -> str:
        return "_".join(label.lower().split())

    def _infer_split(self, parts: tuple[str, ...]) -> str:
        lowered = [part.lower() for part in parts]
        if "train" in lowered or "training_set" in lowered:
            return "train"
        if "test" in lowered or "test_set" in lowered:
            return "test"
        return "unspecified"

    def _should_ignore_project_file(self, file_path: Path) -> bool:
        if self.config.project_dir not in file_path.parents:
            return False

        relative_parts = file_path.relative_to(self.config.project_dir).parts
        if not relative_parts:
            return False
        return relative_parts[0] in self.config.ignored_project_dirs

    def _resolve_source_dataset(self, root: Path) -> str:
        try:
            return root.relative_to(self.config.workspace_dir).parts[0]
        except ValueError:
            return root.name
