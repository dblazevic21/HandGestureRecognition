from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .config import AppConfig
from .dataset_scanner import ImageRecord
from .feature_extractor import MediaPipeFeatureExtractor


class GestureModelService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def train(self, records: list[ImageRecord], min_samples_per_class: int, progress_every: int = 500) -> dict:
        self.config.ensure_artifacts_dir()
        total_records = len(records)
        start_time = perf_counter()
        input_per_label: Counter[str] = Counter(record.label_key for record in records)

        features: list[np.ndarray] = []
        labels: list[str] = []
        label_metadata: dict[str, dict] = {}
        detected_per_label: Counter[str] = Counter()
        skipped_per_label: Counter[str] = Counter()
        source_detected: Counter[str] = Counter()
        feature_source_counter: Counter[str] = Counter()
        processed_source_counter: Counter[str] = Counter()
        detection_reason_counter: Counter[str] = Counter()
        skipped_examples: defaultdict[str, list[str]] = defaultdict(list)
        skipped_examples_by_reason: defaultdict[str, list[str]] = defaultdict(list)
        fallback_metric_samples: defaultdict[str, list[float]] = defaultdict(list)
        diagnostics_rows: list[dict[str, str | int | float]] = []

        with MediaPipeFeatureExtractor(static_image_mode=True) as extractor:
            print(f"\nKrecem MediaPipe obradu slika: {total_records} ukupno.", flush=True)
            for index, record in enumerate(records, start=1):
                allow_background = record.display_label.lower() in self.config.background_labels
                detection = extractor.extract_from_image_path(record.path, allow_background=allow_background)

                if detection is not None:
                    processed_source_counter[detection.source] += 1
                    if detection.debug_reason:
                        detection_reason_counter[detection.debug_reason] += 1
                    if "fallback_score" in detection.debug_metrics:
                        bucket = f"{detection.source}_fallback_score"
                        fallback_metric_samples[bucket].append(float(detection.debug_metrics["fallback_score"]))
                    if "fallback_ratio" in detection.debug_metrics:
                        bucket = f"{detection.source}_fallback_ratio"
                        fallback_metric_samples[bucket].append(float(detection.debug_metrics["fallback_ratio"]))

                diagnostics_rows.append(
                    self._build_diagnostics_row(
                        record=record,
                        detection=detection,
                    )
                )

                if detection is None or detection.feature_vector is None:
                    skipped_per_label[record.label_key] += 1
                    if len(skipped_examples[record.label_key]) < 5:
                        reason = detection.debug_reason if detection is not None else "image_read_failed"
                        skipped_examples[record.label_key].append(f"{reason} | {record.path}")
                    reason = detection.debug_reason if detection is not None else "image_read_failed"
                    if len(skipped_examples_by_reason[reason]) < 5:
                        skipped_examples_by_reason[reason].append(str(record.path))
                else:
                    features.append(detection.feature_vector)
                    labels.append(record.label_key)
                    detected_per_label[record.label_key] += 1
                    source_detected[record.source_dataset] += 1
                    feature_source_counter[detection.source] += 1
                    label_metadata.setdefault(
                        record.label_key,
                        {
                            "source_dataset": record.source_dataset,
                            "display_label": record.display_label,
                            "raw_label": record.raw_label,
                        },
                    )

                if progress_every > 0 and (index % progress_every == 0 or index == total_records):
                    elapsed = perf_counter() - start_time
                    speed = index / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[obrada] {index}/{total_records} | "
                        f"uspjesno={len(features)} | preskoceno={index - len(features)} | "
                        f"mp={feature_source_counter['mediapipe']} | "
                        f"fallback={feature_source_counter['fallback']} | "
                        f"bg={feature_source_counter['background']} | "
                        f"odbijeno={processed_source_counter['none']} | "
                        f"brzina={speed:.2f} slika/s",
                        flush=True,
                    )

        if not features:
            raise RuntimeError("Nije pronadjen nijedan valjan uzorak za treniranje.")

        print("\nMediaPipe obrada zavrsena. Krecem pripremu skupa za treniranje.", flush=True)
        label_counts = Counter(labels)
        kept_labels = {label for label, count in label_counts.items() if count >= min_samples_per_class}
        removed_labels = {label: count for label, count in label_counts.items() if count < min_samples_per_class}

        filtered_features = [feature for feature, label in zip(features, labels) if label in kept_labels]
        filtered_labels = [label for label in labels if label in kept_labels]

        if len(set(filtered_labels)) < 2:
            raise RuntimeError(
                "Nakon filtriranja nema dovoljno razlicitih klasa za treniranje. "
                "Smanji --min-samples-per-class ili provjeri detekciju."
            )

        x_data = np.vstack(filtered_features).astype(np.float32)
        y_data = np.asarray(filtered_labels)

        print(
            f"Ulaz u model: {x_data.shape[0]} uzoraka, {x_data.shape[1]} znacajki, "
            f"{len(set(filtered_labels))} klasa.",
            flush=True,
        )
        print("Krecem evaluaciju na holdout skupu...", flush=True)
        evaluation = self._evaluate(x_data, y_data)
        print("Evaluacija gotova. Krecem finalno treniranje modela...", flush=True)
        model = self._build_model()
        model.fit(x_data, y_data)

        bundle = {
            "model": model,
            "label_metadata": label_metadata,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "feature_count": int(x_data.shape[1]),
            "training_sample_count": int(x_data.shape[0]),
            "class_count": int(len(set(filtered_labels))),
            "evaluation": evaluation,
        }
        joblib.dump(bundle, self.config.model_path)
        total_elapsed = perf_counter() - start_time
        print(f"Model spremljen: {self.config.model_path}", flush=True)
        print(f"Ukupno vrijeme: {total_elapsed / 60:.2f} min", flush=True)

        summary = {
            "model_path": str(self.config.model_path),
            "summary_path": str(self.config.summary_path),
            "class_csv_path": str(self.config.training_class_csv_path),
            "reason_csv_path": str(self.config.training_reason_csv_path),
            "fallback_csv_path": str(self.config.training_fallback_csv_path),
            "diagnostics_csv_path": str(self.config.training_diagnostics_csv_path),
            "ukupno_slika_u_ulazu": len(records),
            "uspjesno_obradeni_uzorci": int(len(features)),
            "uzorci_za_treniranje": int(x_data.shape[0]),
            "broj_klasa": int(len(set(filtered_labels))),
            "uspjesno_po_datasetu": dict(sorted(source_detected.items(), key=lambda item: item[0].lower())),
            "izvor_znacajki": dict(sorted(feature_source_counter.items(), key=lambda item: item[0])),
            "obrada_po_izvoru": dict(sorted(processed_source_counter.items(), key=lambda item: item[0])),
            "razlozi_obrade": dict(sorted(detection_reason_counter.items(), key=lambda item: (-item[1], item[0]))),
            "fallback_dijagnostika": {
                metric_name: self._summarize_metric(metric_values)
                for metric_name, metric_values in sorted(fallback_metric_samples.items(), key=lambda item: item[0])
            },
            "uspjesno_po_klasi": self._friendly_label_counts(detected_per_label, label_metadata),
            "preskoceno_po_klasi": self._friendly_label_counts(skipped_per_label, label_metadata),
            "uklonjene_klase_zbog_malog_broja_uzoraka": self._friendly_label_counts(removed_labels, label_metadata),
            "primjeri_preskocenih_datoteka": {
                self._friendly_label(label_key, label_metadata): paths
                for label_key, paths in sorted(skipped_examples.items(), key=lambda item: item[0])
            },
            "primjeri_preskocenih_po_razlogu": {
                reason: paths
                for reason, paths in sorted(skipped_examples_by_reason.items(), key=lambda item: item[0])
            },
            "evaluacija": evaluation,
            "ukupno_vrijeme_min": round(total_elapsed / 60, 2),
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self._export_csv_reports(
            input_per_label=input_per_label,
            label_metadata=label_metadata,
            detected_per_label=detected_per_label,
            skipped_per_label=skipped_per_label,
            label_counts=label_counts,
            kept_labels=kept_labels,
            removed_labels=removed_labels,
            detection_reason_counter=detection_reason_counter,
            fallback_metric_samples=fallback_metric_samples,
            diagnostics_rows=diagnostics_rows,
        )
        return summary

    def load_bundle(self) -> dict:
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model nije pronadjen na putanji: {self.config.model_path}. "
                "Prvo pokreni 'python mainMedia.py train'."
            )
        bundle = joblib.load(self.config.model_path)
        model = bundle.get("model")
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        return bundle

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        bundle = self.load_bundle()
        return self.predict_with_bundle(bundle, feature_vector)

    def predict_with_bundle(self, bundle: dict, feature_vector: np.ndarray) -> tuple[str, float]:
        model = bundle["model"]
        probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]
        best_index = int(np.argmax(probabilities))
        label_key = model.classes_[best_index]
        confidence = float(probabilities[best_index])
        return label_key, confidence

    def label_to_text(self, label_key: str, label_metadata: dict[str, dict]) -> str:
        metadata = label_metadata.get(label_key)
        if not metadata:
            return label_key
        return f"{metadata['source_dataset']} / {metadata['display_label']}"

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            random_state=self.config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )

    def _evaluate(self, x_data: np.ndarray, y_data: np.ndarray) -> dict:
        label_counts = Counter(y_data.tolist())
        if min(label_counts.values()) < 2 or len(label_counts) < 2:
            return {
                "status": "preskoceno",
                "razlog": "Nema dovoljno uzoraka za stratificirani holdout test.",
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=self.config.random_state,
            stratify=y_data,
        )
        model = self._build_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        return {
            "status": "ok",
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "test_uzorci": int(len(y_test)),
            "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_avg_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        }

    def _friendly_label_counts(self, counts: Counter | dict[str, int], label_metadata: dict[str, dict]) -> dict[str, int]:
        return {
            self._friendly_label(label_key, label_metadata): int(count)
            for label_key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        }

    def _friendly_label(self, label_key: str, label_metadata: dict[str, dict]) -> str:
        metadata = label_metadata.get(label_key)
        if not metadata:
            return label_key
        return f"{metadata['source_dataset']} / {metadata['display_label']}"

    def _summarize_metric(self, values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float32)
        return {
            "count": int(array.size),
            "min": round(float(array.min()), 4),
            "mean": round(float(array.mean()), 4),
            "max": round(float(array.max()), 4),
        }

    def _build_diagnostics_row(self, record: ImageRecord, detection) -> dict[str, str | int | float]:
        metrics = detection.debug_metrics if detection is not None else {}
        return {
            "path": str(record.path),
            "source_dataset": record.source_dataset,
            "display_label": record.display_label,
            "label_key": record.label_key,
            "raw_label": record.raw_label,
            "split": record.split,
            "allow_background": int(record.display_label.lower() in self.config.background_labels),
            "feature_source": detection.source if detection is not None else "read_failed",
            "feature_vector_ok": int(detection is not None and detection.feature_vector is not None),
            "detected": int(detection.detected) if detection is not None else 0,
            "debug_reason": detection.debug_reason if detection is not None else "image_read_failed",
            "fallback_variant": metrics.get("fallback_variant", ""),
            "fallback_score": metrics.get("fallback_score", ""),
            "fallback_ratio": metrics.get("fallback_ratio", ""),
            "fallback_bbox_area_ratio": metrics.get("fallback_bbox_area_ratio", ""),
            "fallback_extent": metrics.get("fallback_extent", ""),
            "fallback_aspect_ratio": metrics.get("fallback_aspect_ratio", ""),
            "fallback_solidity": metrics.get("fallback_solidity", ""),
            "fallback_touches_border": metrics.get("fallback_touches_border", ""),
            "fallback_accepted": metrics.get("fallback_accepted", ""),
        }

    def _export_csv_reports(
        self,
        input_per_label: Counter[str],
        label_metadata: dict[str, dict],
        detected_per_label: Counter[str],
        skipped_per_label: Counter[str],
        label_counts: Counter[str],
        kept_labels: set[str],
        removed_labels: dict[str, int],
        detection_reason_counter: Counter[str],
        fallback_metric_samples: dict[str, list[float]],
        diagnostics_rows: list[dict[str, str | int | float]],
    ) -> None:
        self._write_class_summary_csv(
            input_per_label=input_per_label,
            label_metadata=label_metadata,
            detected_per_label=detected_per_label,
            skipped_per_label=skipped_per_label,
            label_counts=label_counts,
            kept_labels=kept_labels,
            removed_labels=removed_labels,
        )
        self._write_reason_summary_csv(detection_reason_counter)
        self._write_fallback_summary_csv(fallback_metric_samples)
        self._write_diagnostics_csv(diagnostics_rows)

    def _write_class_summary_csv(
        self,
        input_per_label: Counter[str],
        label_metadata: dict[str, dict],
        detected_per_label: Counter[str],
        skipped_per_label: Counter[str],
        label_counts: Counter[str],
        kept_labels: set[str],
        removed_labels: dict[str, int],
    ) -> None:
        fieldnames = [
            "label_key",
            "source_dataset",
            "display_label",
            "input_images",
            "successful_features",
            "skipped_images",
            "kept_for_training",
            "removed_for_low_samples",
            "samples_after_filter",
        ]
        all_labels = sorted(set(input_per_label) | set(detected_per_label) | set(skipped_per_label) | set(label_metadata))
        with self.config.training_class_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label_key in all_labels:
                metadata = label_metadata.get(label_key, {})
                writer.writerow(
                    {
                        "label_key": label_key,
                        "source_dataset": metadata.get("source_dataset", label_key.split("::", 1)[0] if "::" in label_key else ""),
                        "display_label": metadata.get("display_label", label_key),
                        "input_images": int(input_per_label.get(label_key, 0)),
                        "successful_features": int(detected_per_label.get(label_key, 0)),
                        "skipped_images": int(skipped_per_label.get(label_key, 0)),
                        "kept_for_training": int(label_key in kept_labels),
                        "removed_for_low_samples": int(label_key in removed_labels),
                        "samples_after_filter": int(label_counts.get(label_key, 0) if label_key in kept_labels else 0),
                    }
                )

    def _write_reason_summary_csv(self, detection_reason_counter: Counter[str]) -> None:
        fieldnames = ["reason", "count"]
        with self.config.training_reason_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for reason, count in sorted(detection_reason_counter.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow({"reason": reason, "count": int(count)})

    def _write_fallback_summary_csv(self, fallback_metric_samples: dict[str, list[float]]) -> None:
        fieldnames = ["metric_name", "count", "min", "mean", "max"]
        with self.config.training_fallback_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metric_name, values in sorted(fallback_metric_samples.items(), key=lambda item: item[0]):
                summary = self._summarize_metric(values)
                writer.writerow(
                    {
                        "metric_name": metric_name,
                        "count": int(summary.get("count", 0)),
                        "min": summary.get("min", ""),
                        "mean": summary.get("mean", ""),
                        "max": summary.get("max", ""),
                    }
                )

    def _write_diagnostics_csv(self, diagnostics_rows: list[dict[str, str | int | float]]) -> None:
        fieldnames = [
            "path",
            "source_dataset",
            "display_label",
            "label_key",
            "raw_label",
            "split",
            "allow_background",
            "feature_source",
            "feature_vector_ok",
            "detected",
            "debug_reason",
            "fallback_variant",
            "fallback_score",
            "fallback_ratio",
            "fallback_bbox_area_ratio",
            "fallback_extent",
            "fallback_aspect_ratio",
            "fallback_solidity",
            "fallback_touches_border",
            "fallback_accepted",
        ]
        with self.config.training_diagnostics_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diagnostics_rows)
