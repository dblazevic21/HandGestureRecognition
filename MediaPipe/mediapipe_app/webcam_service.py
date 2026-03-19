from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppConfig
from .feature_extractor import MediaPipeFeatureExtractor
from .model_service import GestureModelService


WINDOW_NAME = "MediaPipe Hand Gesture"


@dataclass(slots=True)
class PredictionMode:
    mode_id: str
    title: str
    source_dataset: str | None
    shortcut: str
    label_keys: tuple[str, ...]


class GestureWebcamService:
    def __init__(self, config: AppConfig, model_service: GestureModelService) -> None:
        self.config = config
        self.model_service = model_service
        self._overlay_font_path = self._find_overlay_font()
        self._button_hitboxes: list[tuple[tuple[int, int, int, int], str]] = []
        self._mode_by_id: dict[str, PredictionMode] = {}
        self._current_mode_id = "all"

    def run(self, camera_index: int = 0) -> None:
        bundle = self.model_service.load_bundle()
        model = bundle["model"]
        label_metadata = bundle["label_metadata"]
        class_count = int(bundle.get("class_count", len(getattr(model, "classes_", []))))
        feature_count = int(bundle.get("feature_count", 0))
        created_at = str(bundle.get("created_at", "-"))
        modes = self._build_modes(model.classes_, label_metadata)
        self._mode_by_id = {mode.mode_id: mode for mode in modes}
        self._current_mode_id = "all" if "all" in self._mode_by_id else modes[0].mode_id

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Kamera s indeksom {camera_index} nije dostupna.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)

        with MediaPipeFeatureExtractor(static_image_mode=False) as extractor:
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Ne mogu procitati frame s kamere.")

                    frame = cv2.flip(frame, 1)
                    detection = extractor.analyze_frame(frame)

                    preview = self._build_preview(detection.annotated_frame, detection.color_mask)
                    mode = self._mode_by_id[self._current_mode_id]
                    prediction_text = "Nema prepoznate ruke"
                    confidence_text = "-"
                    status_text = f"Cekam prepoznatu ruku | Mode: {mode.title}"
                    feature_source_text = detection.source
                    finger_text = self._format_finger_states(detection.finger_states)

                    if detection.feature_vector is not None:
                        prediction_text, confidence_text = self._predict_in_mode(
                            model=model,
                            label_metadata=label_metadata,
                            feature_vector=detection.feature_vector,
                            mode=mode,
                        )
                        status_text = f"Mode aktivan: {mode.title}"

                    preview = self._draw_overlay(
                        image=preview,
                        prediction_text=prediction_text,
                        confidence_text=confidence_text,
                        hand_detected=detection.detected,
                        feature_source_text=feature_source_text,
                        finger_text=finger_text,
                        class_count=class_count,
                        feature_count=feature_count,
                        created_at=created_at,
                        status_text=status_text,
                        mode=mode,
                        modes=modes,
                    )

                    cv2.imshow(WINDOW_NAME, preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    self._handle_key(key, modes)
            finally:
                capture.release()
                cv2.destroyAllWindows()

    def _build_modes(self, model_classes: np.ndarray, label_metadata: dict[str, dict]) -> list[PredictionMode]:
        class_list = [str(label_key) for label_key in model_classes.tolist()]

        def keys_for_dataset(dataset_name: str) -> tuple[str, ...]:
            return tuple(
                label_key
                for label_key in class_list
                if label_metadata.get(label_key, {}).get("source_dataset") == dataset_name
            )

        modes: list[PredictionMode] = [
            PredictionMode(
                mode_id="all",
                title="ALL",
                source_dataset=None,
                shortcut="1",
                label_keys=tuple(class_list),
            ),
        ]

        dataset_modes = [
            ("asl", "ASL", "ASL", "2"),
            ("rps", "RPS", "rock-paper-scissors", "3"),
            ("numbers", "0-19", "24000-900-300t", "4"),
            ("leap", "LEAP", "leapGestRecog", "5"),
        ]
        for mode_id, title, dataset_name, shortcut in dataset_modes:
            label_keys = keys_for_dataset(dataset_name)
            if label_keys:
                modes.append(
                    PredictionMode(
                        mode_id=mode_id,
                        title=title,
                        source_dataset=dataset_name,
                        shortcut=shortcut,
                        label_keys=label_keys,
                    )
                )
        return modes

    def _predict_in_mode(
        self,
        model,
        label_metadata: dict[str, dict],
        feature_vector: np.ndarray,
        mode: PredictionMode,
    ) -> tuple[str, str]:
        probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]
        class_to_index = {str(label_key): index for index, label_key in enumerate(model.classes_)}
        allowed_indices = [class_to_index[label_key] for label_key in mode.label_keys if label_key in class_to_index]

        if not allowed_indices:
            return "Odabrani mode nema klasa u modelu", "-"

        allowed_probabilities = probabilities[allowed_indices]
        local_best_pos = int(np.argmax(allowed_probabilities))
        best_index = allowed_indices[local_best_pos]
        label_key = str(model.classes_[best_index])

        subset_sum = float(np.sum(allowed_probabilities))
        raw_confidence = float(probabilities[best_index])
        normalized_confidence = raw_confidence / subset_sum if subset_sum > 1e-9 else raw_confidence

        friendly_label = self.model_service.label_to_text(label_key, label_metadata)
        if mode.mode_id != "all":
            friendly_label = friendly_label.split(" / ", 1)[-1]

        confidence_text = f"{normalized_confidence:.2%} (mode) | {raw_confidence:.2%} (raw)"
        return friendly_label, confidence_text

    def _handle_key(self, key: int, modes: list[PredictionMode]) -> None:
        if key == 0xFF:
            return
        for mode in modes:
            if key == ord(mode.shortcut):
                self._current_mode_id = mode.mode_id
                return

    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), mode_id in self._button_hitboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._current_mode_id = mode_id
                return

    def _build_preview(self, annotated_frame: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
        left = annotated_frame.copy()
        right = color_mask.copy()
        return np.hstack([left, right])

    def _draw_overlay(
        self,
        image: np.ndarray,
        prediction_text: str,
        confidence_text: str,
        hand_detected: bool,
        feature_source_text: str,
        finger_text: str,
        class_count: int,
        feature_count: int,
        created_at: str,
        status_text: str,
        mode: PredictionMode,
        modes: list[PredictionMode],
    ) -> np.ndarray:
        top_bar_height = 154
        bottom_bar_height = 116
        left_padding = 20
        top_line_start_y = 28
        top_line_step = 24
        summary_line_start_y = 30
        summary_line_step = 24
        bottom_line_step = 28
        value_gap = 8
        height, width = image.shape[:2]

        canvas = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        canvas[top_bar_height : top_bar_height + height, :] = image
        pil_image, pil_draw = self._build_pil_draw_context(canvas)

        top_lines = [
            ("Predikcija", f": {prediction_text}", (170, 255, 170)),
            ("Pouzdanost", f": {confidence_text}", (190, 235, 255)),
            ("Mode", f": {mode.title} | Aktivne klase: {len(mode.label_keys)}", (255, 245, 180)),
            ("Detekcija ruke", f": {'DA' if hand_detected else 'NE'} | Izvor znacajki: {feature_source_text}", (190, 235, 255)),
        ]
        summary_lines = [
            f"Ukupno klasa: {class_count}",
            f"Znacajki: {feature_count}",
            f"Model: {created_at}",
        ]
        bottom_lines = [
            ("Prsti", finger_text.replace("Prsti", ""), (255, 255, 255)),
            ("Status", f": {status_text}", (255, 245, 180)),
            ("KEYBINDS", ": Klik na gumb ili tipke 1-5 | Q/ESC = izlaz", (255, 120, 255)),
        ]

        for idx, (label, value, color) in enumerate(top_lines):
            y = top_line_start_y + (idx * top_line_step)
            canvas = self._draw_text(canvas, label, (left_padding, y), 20, color, pil_draw)
            x_offset = left_padding + self._text_width(label, 20) + value_gap
            canvas = self._draw_text(canvas, value, (x_offset, y), 20, color, pil_draw)

        summary_x = max(width // 2 + 10, 760)
        for idx, text in enumerate(summary_lines):
            y = summary_line_start_y + (idx * summary_line_step)
            canvas = self._draw_text(canvas, text, (summary_x, y), 18, (190, 235, 255), pil_draw)

        self._button_hitboxes = self._draw_mode_buttons(
            canvas=canvas,
            pil_draw=pil_draw,
            width=width,
            top_bar_height=top_bar_height,
            modes=modes,
            active_mode_id=mode.mode_id,
        )

        bottom_start_y = top_bar_height + height + 42
        for idx, (label, value, color) in enumerate(bottom_lines):
            y = bottom_start_y + (idx * bottom_line_step)
            canvas = self._draw_text(canvas, label, (left_padding, y), 20, color, pil_draw)
            x_offset = left_padding + self._text_width(label, 20) + value_gap
            canvas = self._draw_text(canvas, value, (x_offset, y), 20, color, pil_draw)

        if pil_image is not None:
            canvas = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return canvas

    def _draw_mode_buttons(
        self,
        canvas: np.ndarray,
        pil_draw: ImageDraw.ImageDraw | None,
        width: int,
        top_bar_height: int,
        modes: list[PredictionMode],
        active_mode_id: str,
    ) -> list[tuple[tuple[int, int, int, int], str]]:
        if not modes:
            return []

        button_y1 = top_bar_height - 42
        button_y2 = top_bar_height - 10
        start_x = 20
        gap = 10
        buttons: list[tuple[tuple[int, int, int, int], str]] = []

        for mode in modes:
            label = f"{mode.shortcut}. {mode.title}"
            button_width = max(110, self._text_width(label, 18) + 24)
            x1 = start_x
            x2 = min(x1 + button_width, width - 20)
            is_active = mode.mode_id == active_mode_id
            fill_color = (70, 70, 70) if not is_active else (70, 120, 220)
            text_color = (210, 210, 210) if not is_active else (255, 255, 255)
            if pil_draw is not None:
                pil_draw.rounded_rectangle(
                    (x1, button_y1, x2, button_y2),
                    radius=8,
                    fill=(fill_color[2], fill_color[1], fill_color[0]),
                    outline=(180, 180, 180),
                    width=1,
                )
            else:
                cv2.rectangle(canvas, (x1, button_y1), (x2, button_y2), fill_color, -1)
                cv2.rectangle(canvas, (x1, button_y1), (x2, button_y2), (180, 180, 180), 1)
            self._draw_text(canvas, label, (x1 + 12, button_y1 + 22), 18, text_color, pil_draw)
            buttons.append(((x1, button_y1, x2, button_y2), mode.mode_id))
            start_x = x2 + gap
        return buttons

    def _format_finger_states(self, finger_states: dict[str, str]) -> str:
        if not finger_states:
            return "Prsti: nema podataka"

        ordered_names = ["palac", "kaziprst", "srednji", "prstenjak", "mali"]
        parts = [f"{name}={finger_states[name]}" for name in ordered_names if name in finger_states]
        return "Prsti: " + " | ".join(parts)

    def _text_width(self, text: str, pixel_size: int) -> int:
        font = self._get_overlay_font(pixel_size)
        if font is not None:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0])

        cv_scale = max(pixel_size / 50.0, 0.3)
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cv_scale, 1)
        return width

    def _find_overlay_font(self) -> Path | None:
        font_candidates = [
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ]
        for path in font_candidates:
            if path.exists():
                return path
        return None

    def _get_overlay_font(self, pixel_size: int) -> ImageFont.FreeTypeFont | None:
        if self._overlay_font_path is None:
            return None
        try:
            return ImageFont.truetype(str(self._overlay_font_path), max(pixel_size, 12))
        except OSError:
            return None

    def _build_pil_draw_context(self, image: np.ndarray) -> tuple[Image.Image | None, ImageDraw.ImageDraw | None]:
        if self._overlay_font_path is None:
            return None, None
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return pil_image, ImageDraw.Draw(pil_image)

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        pixel_size: int,
        color: tuple[int, int, int],
        draw: ImageDraw.ImageDraw | None = None,
    ) -> np.ndarray:
        font = self._get_overlay_font(pixel_size)
        if font is None or draw is None:
            cv_scale = max(pixel_size / 50.0, 0.3)
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, cv_scale, color, 1, cv2.LINE_AA)
            return image

        rgb_color = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((origin[0], origin[1] - pixel_size), text, font=font, fill=rgb_color)
        return image
