from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .feature_extractor import MediaPipeFeatureExtractor
from .model_service import GestureModelService


class GestureWebcamService:
    def __init__(self, config: AppConfig, model_service: GestureModelService) -> None:
        self.config = config
        self.model_service = model_service

    def run(self, camera_index: int = 0) -> None:
        bundle = self.model_service.load_bundle()
        model = bundle["model"]
        label_metadata = bundle["label_metadata"]

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Kamera s indeksom {camera_index} nije dostupna.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        with MediaPipeFeatureExtractor(static_image_mode=False) as extractor:
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Ne mogu procitati frame s kamere.")

                    frame = cv2.flip(frame, 1)
                    detection = extractor.analyze_frame(frame)

                    left_panel = detection.annotated_frame
                    right_panel = detection.color_mask.copy()

                    prediction_text = "Nema prepoznate ruke"
                    confidence_text = "Pouzdanost: -"
                    finger_text = "Prsti: nema podataka"

                    if detection.feature_vector is not None:
                        probabilities = model.predict_proba(detection.feature_vector.reshape(1, -1))[0]
                        best_index = int(np.argmax(probabilities))
                        label_key = model.classes_[best_index]
                        confidence = float(probabilities[best_index])
                        prediction_text = self.model_service.label_to_text(label_key, label_metadata)
                        confidence_text = f"Pouzdanost: {confidence:.2%}"
                        finger_text = self._format_finger_states(detection.finger_states)

                    self._draw_panel_header(left_panel, "Kamera", prediction_text, confidence_text, finger_text)
                    self._draw_panel_header(
                        right_panel,
                        "Obojana ruka",
                        "Palac/Prsti/Dlan po bojama",
                        "Crno = pozadina",
                        "Zglobovi i savijanje prstiju prate se preko MediaPipe landmarka",
                    )

                    combined = np.hstack([left_panel, right_panel])
                    cv2.imshow("MediaPipe Hand Gesture", combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
            finally:
                capture.release()
                cv2.destroyAllWindows()

    def _draw_panel_header(self, image: np.ndarray, title: str, line_one: str, line_two: str, line_three: str) -> None:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 118), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.38, image, 0.62, 0, image)
        cv2.putText(image, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, line_one, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, line_two, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, line_three, (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    def _format_finger_states(self, finger_states: dict[str, str]) -> str:
        if not finger_states:
            return "Prsti: nema podataka"

        ordered_names = ["palac", "kaziprst", "srednji", "prstenjak", "mali"]
        parts = [f"{name}={finger_states[name]}" for name in ordered_names if name in finger_states]
        return "Prsti: " + " | ".join(parts)
