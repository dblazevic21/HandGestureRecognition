from __future__ import annotations

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

        features: list[np.ndarray] = []
        labels: list[str] = []
        label_metadata: dict[str, dict] = {}
        detected_per_label: Counter[str] = Counter()
        skipped_per_label: Counter[str] = Counter()
        source_detected: Counter[str] = Counter()
        feature_source_counter: Counter[str] = Counter()
        skipped_examples: defaultdict[str, list[str]] = defaultdict(list)

        with MediaPipeFeatureExtractor(static_image_mode=True) as extractor:
            print(f"\nKrecem MediaPipe obradu slika: {total_records} ukupno.", flush=True)
            for index, record in enumerate(records, start=1):
                allow_background = record.display_label.lower() in self.config.background_labels
                detection = extractor.extract_from_image_path(record.path, allow_background=allow_background)

                if detection is None or detection.feature_vector is None:
                    skipped_per_label[record.label_key] += 1
                    if len(skipped_examples[record.label_key]) < 5:
                        skipped_examples[record.label_key].append(str(record.path))
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
            "ukupno_slika_u_ulazu": len(records),
            "uspjesno_obradeni_uzorci": int(len(features)),
            "uzorci_za_treniranje": int(x_data.shape[0]),
            "broj_klasa": int(len(set(filtered_labels))),
            "uspjesno_po_datasetu": dict(sorted(source_detected.items(), key=lambda item: item[0].lower())),
            "izvor_znacajki": dict(sorted(feature_source_counter.items(), key=lambda item: item[0])),
            "uspjesno_po_klasi": self._friendly_label_counts(detected_per_label, label_metadata),
            "preskoceno_po_klasi": self._friendly_label_counts(skipped_per_label, label_metadata),
            "uklonjene_klase_zbog_malog_broja_uzoraka": self._friendly_label_counts(removed_labels, label_metadata),
            "primjeri_preskocenih_datoteka": {
                self._friendly_label(label_key, label_metadata): paths
                for label_key, paths in sorted(skipped_examples.items(), key=lambda item: item[0])
            },
            "evaluacija": evaluation,
            "ukupno_vrijeme_min": round(total_elapsed / 60, 2),
        }

        self.config.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def load_bundle(self) -> dict:
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model nije pronadjen na putanji: {self.config.model_path}. "
                "Prvo pokreni 'python mainMedia.py train'."
            )
        return joblib.load(self.config.model_path)

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
