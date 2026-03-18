from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MASK_GRID_SIZE = 20
MASK_FEATURE_VECTOR_SIZE = (MASK_GRID_SIZE * MASK_GRID_SIZE) + 7 + 11


@dataclass(slots=True)
class HandDetection:
    feature_vector: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray
    color_mask: np.ndarray
    annotated_frame: np.ndarray
    detected: bool
    source: str
    finger_states: dict[str, str] = field(default_factory=dict)


class MediaPipeFeatureExtractor:
    def __init__(
        self,
        static_image_mode: bool,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "MediaPipeFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def extract_from_image_path(self, image_path: Path, allow_background: bool = False) -> HandDetection | None:
        image = self._read_bgr_image(image_path)
        if image is None:
            return None

        detection = self.analyze_frame(image)
        if detection.feature_vector is not None:
            return detection

        fallback_mask = self._extract_mask_from_image(image)
        fallback_vector = self._build_mask_feature_vector(fallback_mask)
        fallback_bbox = self._mask_to_bbox(fallback_mask)

        if fallback_vector is None:
            if allow_background:
                return HandDetection(
                    feature_vector=np.zeros(MASK_FEATURE_VECTOR_SIZE, dtype=np.float32),
                    bbox=None,
                    mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    color_mask=np.zeros((*image.shape[:2], 3), dtype=np.uint8),
                    annotated_frame=image.copy(),
                    detected=False,
                    source="background",
                    finger_states={},
                )
            return HandDetection(
                feature_vector=None,
                bbox=None,
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                color_mask=np.zeros((*image.shape[:2], 3), dtype=np.uint8),
                annotated_frame=image.copy(),
                detected=False,
                source="none",
                finger_states={},
            )

        annotated = image.copy()
        if fallback_bbox is not None:
            x1, y1, x2, y2 = fallback_bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 170, 255), 2)

        return HandDetection(
            feature_vector=fallback_vector,
            bbox=fallback_bbox,
            mask=fallback_mask,
            color_mask=cv2.cvtColor(fallback_mask, cv2.COLOR_GRAY2BGR),
            annotated_frame=annotated,
            detected=True,
            source="fallback",
            finger_states={},
        )

    def analyze_frame(self, frame_bgr: np.ndarray) -> HandDetection:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        empty_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        empty_color_mask = np.zeros((*frame_bgr.shape[:2], 3), dtype=np.uint8)
        annotated = frame_bgr.copy()

        if not results.multi_hand_landmarks:
            return HandDetection(
                feature_vector=None,
                bbox=None,
                mask=empty_mask,
                color_mask=empty_color_mask,
                annotated_frame=annotated,
                detected=False,
                source="none",
                finger_states={},
            )

        hand_landmarks = results.multi_hand_landmarks[0]
        pixel_points = self._landmarks_to_pixel_points(hand_landmarks, frame_bgr.shape)
        bbox = self._build_bbox(pixel_points, frame_bgr.shape)
        mask = self._build_binary_mask(pixel_points, frame_bgr.shape)
        color_mask = self._build_color_mask(pixel_points, frame_bgr.shape)
        feature_vector = self._build_mask_feature_vector(mask)
        finger_states = self._estimate_finger_states(hand_landmarks)

        self._drawer.draw_landmarks(
            annotated,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._drawing_styles.get_default_hand_landmarks_style(),
            self._drawing_styles.get_default_hand_connections_style(),
        )
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 80), 2)

        return HandDetection(
            feature_vector=feature_vector,
            bbox=bbox,
            mask=mask,
            color_mask=color_mask,
            annotated_frame=annotated,
            detected=feature_vector is not None,
            source="mediapipe",
            finger_states=finger_states,
        )

    def _read_bgr_image(self, image_path: Path) -> np.ndarray | None:
        try:
            file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        except OSError:
            return None
        if file_bytes.size == 0:
            return None
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def _extract_mask_from_image(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mask_normal = self._largest_component_mask(thresh)
        mask_inverse = self._largest_component_mask(thresh_inv)

        scored_masks = [
            (self._score_mask(mask_normal, image_bgr.shape[:2]), mask_normal),
            (self._score_mask(mask_inverse, image_bgr.shape[:2]), mask_inverse),
        ]
        scored_masks.sort(key=lambda item: item[0], reverse=True)
        best_mask = scored_masks[0][1]

        if best_mask is None:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        return best_mask

    def _largest_component_mask(self, threshold_mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 25:
            return None

        mask = np.zeros_like(threshold_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _score_mask(self, mask: np.ndarray | None, image_shape: tuple[int, int]) -> float:
        if mask is None:
            return -1.0

        white_pixels = int(np.count_nonzero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        ratio = white_pixels / max(total_pixels, 1)
        if ratio < 0.01 or ratio > 0.9:
            return -1.0

        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            return -1.0

        x1, y1, x2, y2 = bbox
        box_area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)
        extent = white_pixels / box_area

        touches_border = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= image_shape[1] - 2) + int(y2 >= image_shape[0] - 2)
        border_penalty = touches_border * 0.08

        return ratio + extent - border_penalty

    def _build_mask_feature_vector(self, mask: np.ndarray) -> np.ndarray | None:
        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        crop = mask[y1 : y2 + 1, x1 : x2 + 1]
        resized = cv2.resize(crop, (MASK_GRID_SIZE, MASK_GRID_SIZE), interpolation=cv2.INTER_AREA)
        resized = (resized > 32).astype(np.float32).flatten()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu = np.sign(hu) * np.log1p(np.abs(hu))

        mask_area = float(np.count_nonzero(mask))
        total_area = float(mask.shape[0] * mask.shape[1])
        bbox_w = float(x2 - x1 + 1)
        bbox_h = float(y2 - y1 + 1)
        aspect_ratio = bbox_w / max(bbox_h, 1.0)
        extent = mask_area / max(bbox_w * bbox_h, 1.0)
        solidity = area / max(hull_area, 1.0)
        area_ratio = mask_area / max(total_area, 1.0)
        perimeter_ratio = perimeter / max((bbox_w + bbox_h), 1.0)

        stats = np.asarray(
            [
                area_ratio,
                aspect_ratio,
                extent,
                solidity,
                perimeter_ratio,
                bbox_w / mask.shape[1],
                bbox_h / mask.shape[0],
                x1 / mask.shape[1],
                y1 / mask.shape[0],
                x2 / mask.shape[1],
                y2 / mask.shape[0],
            ],
            dtype=np.float32,
        )
        return np.concatenate([resized, hu.astype(np.float32), stats]).astype(np.float32)

    def _mask_to_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _landmarks_to_pixel_points(self, hand_landmarks, frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        points: list[list[int]] = []
        for landmark in hand_landmarks.landmark:
            x = int(np.clip(landmark.x * width, 0, width - 1))
            y = int(np.clip(landmark.y * height, 0, height - 1))
            points.append([x, y])
        return np.asarray(points, dtype=np.int32)

    def _build_bbox(self, pixel_points: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        if pixel_points.size == 0:
            return None

        height, width = frame_shape[:2]
        min_x = max(int(pixel_points[:, 0].min()) - 18, 0)
        min_y = max(int(pixel_points[:, 1].min()) - 18, 0)
        max_x = min(int(pixel_points[:, 0].max()) + 18, width - 1)
        max_y = min(int(pixel_points[:, 1].max()) + 18, height - 1)
        return min_x, min_y, max_x, max_y

    def _build_binary_mask(self, pixel_points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if pixel_points.size == 0:
            return mask

        palm_indices = np.array([0, 1, 2, 5, 9, 13, 17], dtype=np.int32)
        palm_polygon = pixel_points[palm_indices]
        palm_hull = cv2.convexHull(palm_polygon)
        cv2.fillConvexPoly(mask, palm_hull, 255)

        bbox = self._build_bbox(pixel_points, frame_shape)
        if bbox is None:
            return mask

        x1, y1, x2, y2 = bbox
        hand_span = max(x2 - x1, y2 - y1, 1)
        palm_thickness = max(7, hand_span // 13)
        base_thickness = max(6, hand_span // 18)
        joint_radius = max(4, base_thickness // 2)
        fingertip_radius = joint_radius + max(1, hand_span // 80)

        palm_chain = (0, 1, 2, 5, 9, 13, 17, 0)
        for start_idx, end_idx in zip(palm_chain[:-1], palm_chain[1:]):
            cv2.line(mask, tuple(pixel_points[start_idx]), tuple(pixel_points[end_idx]), 255, palm_thickness)

        for palm_idx in palm_chain[:-1]:
            cv2.circle(mask, tuple(pixel_points[palm_idx]), palm_thickness // 2, 255, -1)

        finger_chains = (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        )
        chain_thickness = (
            max(3, base_thickness - 2),
            base_thickness,
            base_thickness,
            max(4, base_thickness - 1),
            max(3, base_thickness - 2),
        )

        for chain, thickness in zip(finger_chains, chain_thickness):
            for start_idx, end_idx in zip(chain[:-1], chain[1:]):
                cv2.line(mask, tuple(pixel_points[start_idx]), tuple(pixel_points[end_idx]), 255, thickness)

        for joint_idx, point in enumerate(pixel_points):
            if joint_idx in {4, 8, 12, 16, 20}:
                radius = fingertip_radius
            elif joint_idx in {0, 1, 2, 5, 9, 13, 17}:
                radius = joint_radius + 1
            else:
                radius = joint_radius
            cv2.circle(mask, tuple(point), radius, 255, -1)

        blur_kernel = 7 if hand_span > 180 else 5
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        _, mask = cv2.threshold(mask, 108, 255, cv2.THRESH_BINARY)

        kernel_size = max(3, (hand_span // 65) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return mask

    def _build_color_mask(self, pixel_points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
        color_mask = np.zeros((*frame_shape[:2], 3), dtype=np.uint8)
        if pixel_points.size == 0:
            return color_mask

        bbox = self._build_bbox(pixel_points, frame_shape)
        if bbox is None:
            return color_mask

        x1, y1, x2, y2 = bbox
        hand_span = max(x2 - x1, y2 - y1, 1)
        palm_thickness = max(7, hand_span // 13)
        base_thickness = max(6, hand_span // 18)
        joint_radius = max(4, base_thickness // 2)
        fingertip_radius = joint_radius + max(1, hand_span // 80)

        colors = {
            "palm": (80, 80, 255),
            "thumb": (200, 225, 255),
            "index": (170, 100, 210),
            "middle": (0, 220, 255),
            "ring": (80, 230, 80),
            "pinky": (255, 150, 60),
        }

        palm_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        palm_indices = np.array([0, 1, 2, 5, 9, 13, 17], dtype=np.int32)
        palm_polygon = pixel_points[palm_indices]
        palm_hull = cv2.convexHull(palm_polygon)
        cv2.fillConvexPoly(palm_mask, palm_hull, 255)

        palm_chain = (0, 1, 2, 5, 9, 13, 17, 0)
        for start_idx, end_idx in zip(palm_chain[:-1], palm_chain[1:]):
            cv2.line(palm_mask, tuple(pixel_points[start_idx]), tuple(pixel_points[end_idx]), 255, palm_thickness)
        for palm_idx in palm_chain[:-1]:
            cv2.circle(palm_mask, tuple(pixel_points[palm_idx]), palm_thickness // 2, 255, -1)

        palm_mask = self._soften_part_mask(palm_mask, hand_span, erode=False)
        color_mask[palm_mask > 0] = colors["palm"]

        finger_specs = [
            ("thumb", (1, 2, 3, 4), max(3, base_thickness - 2)),
            ("index", (5, 6, 7, 8), base_thickness),
            ("middle", (9, 10, 11, 12), base_thickness),
            ("ring", (13, 14, 15, 16), max(4, base_thickness - 1)),
            ("pinky", (17, 18, 19, 20), max(3, base_thickness - 2)),
        ]

        for finger_name, chain, thickness in finger_specs:
            finger_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
            for start_idx, end_idx in zip(chain[:-1], chain[1:]):
                cv2.line(finger_mask, tuple(pixel_points[start_idx]), tuple(pixel_points[end_idx]), 255, thickness)
            for joint_idx in chain[:-1]:
                cv2.circle(finger_mask, tuple(pixel_points[joint_idx]), joint_radius, 255, -1)
            cv2.circle(finger_mask, tuple(pixel_points[chain[-1]]), fingertip_radius, 255, -1)

            finger_mask = self._soften_part_mask(finger_mask, hand_span, erode=True)
            color_mask[finger_mask > 0] = colors[finger_name]

        return color_mask

    def _soften_part_mask(self, mask: np.ndarray, hand_span: int, erode: bool) -> np.ndarray:
        blur_kernel = 5 if hand_span > 180 else 3
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        _, mask = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)

        kernel_size = 3
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)
        if erode:
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return mask

    def _estimate_finger_states(self, hand_landmarks) -> dict[str, str]:
        coords = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
            dtype=np.float32,
        )

        finger_triplets = {
            "palac": (2, 3, 4),
            "kaziprst": (5, 6, 8),
            "srednji": (9, 10, 12),
            "prstenjak": (13, 14, 16),
            "mali": (17, 18, 20),
        }

        states: dict[str, str] = {}
        for finger_name, (mcp_idx, pip_idx, tip_idx) in finger_triplets.items():
            angle = self._joint_angle(coords[mcp_idx], coords[pip_idx], coords[tip_idx])
            if angle >= 155:
                state = "ispruzen"
            elif angle >= 110:
                state = "polu-savijen"
            else:
                state = "savijen"
            states[finger_name] = state

        return states

    def _joint_angle(self, first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float:
        vector_one = first - middle
        vector_two = last - middle

        norm_one = np.linalg.norm(vector_one)
        norm_two = np.linalg.norm(vector_two)
        if norm_one < 1e-6 or norm_two < 1e-6:
            return 0.0

        cosine = float(np.dot(vector_one, vector_two) / (norm_one * norm_two))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))
