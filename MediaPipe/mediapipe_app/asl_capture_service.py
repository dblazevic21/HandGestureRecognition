from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import AppConfig
from .feature_extractor import MediaPipeFeatureExtractor


UP_KEYS = {2490368, 82, ord("w"), ord("W")}
DOWN_KEYS = {2621440, 84, ord("s"), ord("S")}
LEFT_KEYS = {2424832, 81, ord("a"), ord("A")}
RIGHT_KEYS = {2555904, 83, ord("d"), ord("D")}


@dataclass(slots=True)
class SaveResult:
    saved: bool
    message: str
    path: Path | None = None


class ASLCaptureService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.letters = list(ascii_uppercase)
        self.blur_threshold = 85.0
        self.min_capture_fps = 5.0
        self.max_capture_fps = 30.0
        self.capture_fps = 10.0
        self.capture_interval_seconds = 1.0 / self.capture_fps
        self.key_repeat_cooldown_seconds = 0.14
        self._last_action_key = -1
        self._last_action_time = 0.0
        self._overlay_font_path = self._find_overlay_font()

    def run(self, camera_index: int = 0) -> None:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Kamera s indeksom {camera_index} nije dostupna.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        current_index = 0
        status_text = "Drži Space za kontinuirano spremanje. Gore/dolje mijenjaju slovo."
        burst_mode = False
        last_capture_save = 0.0
        letter_counts = self._load_letter_counts()

        with MediaPipeFeatureExtractor(static_image_mode=False) as extractor:
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Ne mogu procitati frame s kamere.")

                    frame = cv2.flip(frame, 1)
                    detection = extractor.analyze_frame(frame)

                    preview = self._build_preview(detection.annotated_frame, detection.color_mask)
                    current_letter = self.letters[current_index]
                    saved_count = letter_counts[current_letter]

                    now = perf_counter()
                    if self._is_space_pressed() and (now - last_capture_save) >= self.capture_interval_seconds:
                        result = self._save_current_sample(
                            letter=current_letter,
                            frame=frame,
                            bbox=detection.bbox,
                            counts=letter_counts,
                        )
                        if result.saved:
                            last_capture_save = now
                        status_text = f"[SPACE] {result.message}"

                    if burst_mode and (now - last_capture_save) >= self.capture_interval_seconds:
                        result = self._save_current_sample(
                            letter=current_letter,
                            frame=frame,
                            bbox=detection.bbox,
                            counts=letter_counts,
                        )
                        if result.saved:
                            last_capture_save = now
                        status_text = f"[BURST] {result.message}"

                    preview = self._draw_overlay(
                        preview,
                        current_letter,
                        saved_count,
                        letter_counts,
                        status_text,
                        detection.detected,
                        detection.finger_states,
                        burst_mode,
                    )

                    cv2.imshow("ASL Capture", preview)
                    key = cv2.waitKeyEx(1)

                    if key in (27, ord("q"), ord("Q")):
                        break
                    if key != -1 and self._should_debounce_key(key, now):
                        continue
                    if key in UP_KEYS:
                        current_index = min(current_index + 1, len(self.letters) - 1)
                        status_text = f"Trenutno slovo: {self.letters[current_index]}"
                        continue
                    if key in DOWN_KEYS:
                        current_index = max(current_index - 1, 0)
                        status_text = f"Trenutno slovo: {self.letters[current_index]}"
                        continue
                    if key in LEFT_KEYS:
                        previous_rate = self.capture_fps
                        self._set_capture_fps(self.capture_fps - 1.0)
                        if self.capture_fps != previous_rate:
                            status_text = f"Capture rate smanjen na {self.capture_fps:.0f} fps."
                        else:
                            status_text = f"Capture rate je već na minimumu ({self.min_capture_fps:.0f} fps)."
                        continue
                    if key in RIGHT_KEYS:
                        previous_rate = self.capture_fps
                        self._set_capture_fps(self.capture_fps + 1.0)
                        if self.capture_fps != previous_rate:
                            status_text = f"Capture rate povećan na {self.capture_fps:.0f} fps."
                        else:
                            status_text = f"Capture rate je već na maksimumu ({self.max_capture_fps:.0f} fps)."
                        continue
                    if key == 32 and not self._supports_key_state():
                        if (now - last_capture_save) < self.capture_interval_seconds:
                            status_text = f"[SPACE] Limit {self.capture_fps:.0f} fps aktivan."
                            continue

                        result = self._save_current_sample(
                            letter=current_letter,
                            frame=frame,
                            bbox=detection.bbox,
                            counts=letter_counts,
                        )
                        if result.saved:
                            last_capture_save = now
                        status_text = f"[SPACE] {result.message}"
                        continue
                    if key in (ord("b"), ord("B")):
                        burst_mode = not burst_mode
                        last_capture_save = 0.0
                        status_text = f"Burst mode: {'UKLJUČEN' if burst_mode else 'ISKLJUČEN'} | Rate: {self.capture_fps:.0f} fps"
                        continue
            finally:
                capture.release()
                cv2.destroyAllWindows()

    def _save_current_sample(
        self,
        letter: str,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        counts: dict[str, int],
    ) -> SaveResult:
        if bbox is None:
            return SaveResult(
                saved=False,
                message=f"Nije spremljeno za {letter}: ruka nije dovoljno jasno detektirana.",
            )

        target_dir = self.config.asl_capture_dir / letter
        crops_dir = target_dir / "crops"
        frames_dir = target_dir / "frames"
        crops_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        image_index = counts[letter] + 1
        crop_path = crops_dir / f"{letter}_{image_index:05d}_crop.png"
        frame_path = frames_dir / f"{letter}_{image_index:05d}_frame.png"

        x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape)
        crop = frame[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            return SaveResult(saved=False, message=f"Nije spremljeno za {letter}: prazan crop.")

        blur_score = self._blur_score(crop)
        if blur_score < self.blur_threshold:
            return SaveResult(
                saved=False,
                message=f"Nije spremljeno za {letter}: slika je mutna ({blur_score:.1f} < {self.blur_threshold:.1f}).",
            )

        crop_ok, crop_encoded = cv2.imencode(".png", crop)
        frame_ok, frame_encoded = cv2.imencode(".png", frame)
        if not crop_ok or not frame_ok:
            return SaveResult(saved=False, message=f"Nije spremljeno za {letter}: encode nije uspio.")

        crop_path.write_bytes(crop_encoded.tobytes())
        frame_path.write_bytes(frame_encoded.tobytes())
        counts[letter] = image_index
        return SaveResult(
            saved=True,
            message=f"Spremljeno {letter}: crop + frame #{image_index:05d} | blur={blur_score:.1f}",
            path=crop_path,
        )

    def _build_preview(
        self,
        annotated_frame: np.ndarray,
        color_mask: np.ndarray,
    ) -> np.ndarray:
        left = annotated_frame.copy()
        right = color_mask.copy()
        return np.hstack([left, right])

    def _draw_overlay(
        self,
        image: np.ndarray,
        current_letter: str,
        saved_count: int,
        letter_counts: dict[str, int],
        status_text: str,
        hand_detected: bool,
        finger_states: dict[str, str],
        burst_mode: bool,
    ) -> np.ndarray:
        top_bar_height = 116
        bottom_bar_height = 116
        left_padding = 20
        top_line_start_y = 28
        top_line_step = 26
        summary_line_start_y = 30
        summary_line_step = 26
        bottom_line_step = 28
        value_gap = 8
        height, width = image.shape[:2]
        canvas = np.zeros((height + top_bar_height + bottom_bar_height, width, 3), dtype=np.uint8)
        canvas[top_bar_height : top_bar_height + height, :] = image
        pil_image, pil_draw = self._build_pil_draw_context(canvas)

        total_saved = sum(letter_counts.values())
        top_lines = [
            ("Trenutno slovo", f": {current_letter}", (170, 255, 170)),
            ("Broj spremljenih slika", f" za {current_letter}: {saved_count}", (190, 235, 255)),
            ("Ukupno spremljeno", f": {total_saved} | Rate: {self.capture_fps:.0f} fps | Burst: {'DA' if burst_mode else 'NE'}", (190, 235, 255)),
            ("Detekcija ruke", f": {'DA' if hand_detected else 'NE'} | Blur threshold: {self.blur_threshold:.1f}", (190, 235, 255)),
        ]
        bottom_lines = [
            ("Prsti", self._format_finger_states(finger_states).replace("Prsti", ""), (255, 255, 255)),
            ("Status", f": {status_text}", (255, 245, 180)),
            ("KEYBINDS", f": Drži Space = capture {self.capture_fps:.0f} fps | B = burst | Lijevo/Desno = rate | Gore/Dolje = slovo | Q/ESC = izlaz", (255, 120, 255)),
        ]

        for idx, (label, value, color) in enumerate(top_lines):
            y = top_line_start_y + (idx * top_line_step)
            canvas = self._draw_text(canvas, label, (left_padding, y), 20, color, pil_draw)
            x_offset = left_padding + self._text_width(label, 20) + value_gap
            canvas = self._draw_text(canvas, value, (x_offset, y), 20, color, pil_draw)

        summary_lines = self._build_letter_summary_lines(letter_counts)
        base_y = summary_line_start_y
        summary_x = max(width // 2 + 10, 760)
        for idx, text in enumerate(summary_lines):
            y = base_y + (idx * summary_line_step)
            canvas = self._draw_text(canvas, text, (summary_x, y), 18, (190, 235, 255), pil_draw)

        bottom_start_y = top_bar_height + height + 42
        for idx, (label, value, color) in enumerate(bottom_lines):
            y = bottom_start_y + (idx * bottom_line_step)
            canvas = self._draw_text(canvas, label, (left_padding, y), 20, color, pil_draw)
            x_offset = left_padding + self._text_width(label, 20) + value_gap
            canvas = self._draw_text(canvas, value, (x_offset, y), 20, color, pil_draw)

        if pil_image is not None:
            canvas = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return canvas

    def _format_finger_states(self, finger_states: dict[str, str]) -> str:
        if not finger_states:
            return "Prsti: nema podataka"

        ordered_names = ["palac", "kaziprst", "srednji", "prstenjak", "mali"]
        parts = [f"{name}={finger_states[name]}" for name in ordered_names if name in finger_states]
        return "Prsti: " + " | ".join(parts)

    def _load_letter_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for letter in self.letters:
            crops_dir = self.config.asl_capture_dir / letter / "crops"
            if not crops_dir.exists():
                counts[letter] = 0
                continue
            counts[letter] = sum(
                1 for path in crops_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        return counts

    def _build_letter_summary_lines(self, letter_counts: dict[str, int]) -> list[str]:
        groups = [
            self.letters[0:9],
            self.letters[9:18],
            self.letters[18:26],
        ]
        lines: list[str] = []
        for group in groups:
            parts = [f"{letter}:{letter_counts.get(letter, 0)}" for letter in group]
            lines.append(" | ".join(parts))
        return lines

    def _blur_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _text_width(self, text: str, font_scale: float) -> int:
        font = self._get_overlay_font(int(round(font_scale)))
        if font is not None:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0])

        cv_scale = max(font_scale / 50.0, 0.3)
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cv_scale, 1)
        return width

    def _set_capture_fps(self, fps: float) -> None:
        self.capture_fps = min(max(fps, self.min_capture_fps), self.max_capture_fps)
        self.capture_interval_seconds = 1.0 / self.capture_fps

    def _should_debounce_key(self, key: int, now: float) -> bool:
        if key == 32:
            return False
        if key != self._last_action_key:
            self._last_action_key = key
            self._last_action_time = now
            return False

        if (now - self._last_action_time) < self.key_repeat_cooldown_seconds:
            return True

        self._last_action_time = now
        return False

    def _supports_key_state(self) -> bool:
        return hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32")

    def _is_space_pressed(self) -> bool:
        if not self._supports_key_state():
            return False
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000)

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

    def _expand_bbox(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = max(int((x2 - x1) * 0.18), 18)
        pad_y = max(int((y2 - y1) * 0.18), 18)
        return (
            max(x1 - pad_x, 0),
            max(y1 - pad_y, 0),
            min(x2 + pad_x, width - 1),
            min(y2 + pad_y, height - 1),
        )
