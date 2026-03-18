from __future__ import annotations

import argparse
import json

from mediapipe_app.config import AppConfig
from mediapipe_app.asl_capture_service import ASLCaptureService
from mediapipe_app.dataset_scanner import DatasetScanner
from mediapipe_app.model_service import GestureModelService
from mediapipe_app.webcam_service import GestureWebcamService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MediaPipe pipeline za analizu slika i prepoznavanje gesti ruke preko kamere."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Skenira dataset foldere i ispisuje sazetak.")
    scan_parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Ogranici broj slika po klasi pri skeniranju.",
    )

    train_parser = subparsers.add_parser("train", help="Prolazi slike, izvlaci MediaPipe znacajke i trenira model.")
    train_parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Ogranici broj slika po klasi. Korisno za brzi test.",
    )
    train_parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=None,
        help="Minimalan broj uspjesno obradjenih uzoraka po klasi da bi klasa ostala u modelu.",
    )
    train_parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Koliko slika obraditi prije ispisa napretka.",
    )

    webcam_parser = subparsers.add_parser("webcam", help="Pokrece kameru i prepoznavanje gesti u realnom vremenu.")
    webcam_parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks kamere. Najcesce je 0.",
    )

    capture_parser = subparsers.add_parser("capture-asl", help="Otvori kameru i spremaj oznacene ASL slike po slovima.")
    capture_parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks kamere. Najcesce je 0.",
    )

    full_parser = subparsers.add_parser("full", help="Prvo trenira model pa odmah pokrece kameru.")
    full_parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Ogranici broj slika po klasi tijekom treniranja.",
    )
    full_parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=None,
        help="Minimalan broj uzoraka po klasi da bi klasa ostala u modelu.",
    )
    full_parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks kamere. Najcesce je 0.",
    )
    full_parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Koliko slika obraditi prije ispisa napretka tijekom treniranja.",
    )

    return parser


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def run_training(
    config: AppConfig,
    scanner: DatasetScanner,
    model_service: GestureModelService,
    max_images_per_class: int | None,
    min_samples_per_class: int | None,
    progress_every: int,
) -> None:
    records = scanner.collect_images(max_images_per_class=max_images_per_class)
    summary = scanner.export_summary_file(records)
    print("Sazetak pronadjenih slika:")
    print_json(summary)

    training_summary = model_service.train(
        records=records,
        min_samples_per_class=min_samples_per_class or config.min_samples_per_class,
        progress_every=progress_every,
    )
    print("\nSazetak treniranja:")
    print_json(training_summary)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AppConfig()
    scanner = DatasetScanner(config)
    model_service = GestureModelService(config)
    webcam_service = GestureWebcamService(config, model_service)
    asl_capture_service = ASLCaptureService(config)

    if args.command == "scan":
        records = scanner.collect_images(max_images_per_class=args.max_images_per_class)
        print_json(scanner.export_summary_file(records))
        return

    if args.command == "train":
        run_training(
            config=config,
            scanner=scanner,
            model_service=model_service,
            max_images_per_class=args.max_images_per_class,
            min_samples_per_class=args.min_samples_per_class,
            progress_every=args.progress_every,
        )
        return

    if args.command == "webcam":
        webcam_service.run(camera_index=args.camera_index)
        return

    if args.command == "capture-asl":
        asl_capture_service.run(camera_index=args.camera_index)
        return

    if args.command == "full":
        run_training(
            config=config,
            scanner=scanner,
            model_service=model_service,
            max_images_per_class=args.max_images_per_class,
            min_samples_per_class=args.min_samples_per_class,
            progress_every=args.progress_every,
        )
        webcam_service.run(camera_index=args.camera_index)
        return


if __name__ == "__main__":
    main()
