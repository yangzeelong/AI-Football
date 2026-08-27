from __future__ import annotations

import json
from time import perf_counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence
from enum import Enum

import cv2
import numpy as np
from loguru import logger

Point = tuple[int, int]
BBox = tuple[float, float, float, float]


# BGRColor
class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    PURPLE = (255, 0, 255)
    ORANGE = (0, 165, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_sec: float
    image: np.ndarray
    read_ms: float = 0.0


@dataclass(frozen=True)
class Detection:
    frame_index: int
    label: str
    confidence: float
    bbox: BBox
    class_id: int | None = None
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class RoiConfig:
    video_key: str
    width: int
    height: int
    points: list[Point]


class VideoReader:

    def __init__(
        self,
        video_path: str | Path,
        stride: int = 1,
        target_fps: float | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.stride = _resolve_stride(stride, target_fps, self.fps)
        self.target_fps = target_fps
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        logger.info(
            "video info: total_frames={} fps={} target_fps={} stride={} "
            "process_frames={}({} // {})",
            self.frame_count,
            self.fps,
            self.target_fps,
            self.stride,
            self.frame_count // self.stride,
            self.frame_count,
            self.stride,
        )

    def frames(self, max_frames: int | None = None) -> Iterator[VideoFrame]:
        yielded = 0
        frame_index = 0
        while True:
            read_started = perf_counter()
            ok, image = self._cap.read()
            read_ms = (perf_counter() - read_started) * 1000
            if not ok:
                break

            if frame_index % self.stride == 0:
                timestamp = frame_index / self.fps if self.fps else 0.0
                yield VideoFrame(index=frame_index,
                                 timestamp_sec=timestamp,
                                 image=image,
                                 read_ms=read_ms)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break

            frame_index += 1

    def read_frame(self, frame_index: int = 0) -> VideoFrame:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = self._cap.read()
        if not ok:
            raise ValueError(
                f"Cannot read frame {frame_index} from {self.video_path}")
        timestamp = frame_index / self.fps if self.fps else 0.0
        return VideoFrame(index=frame_index,
                          timestamp_sec=timestamp,
                          image=image)

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.release()


def _resolve_stride(
    stride: int,
    target_fps: float | None,
    source_fps: float,
) -> int:
    if target_fps is None:
        return max(1, stride)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive: {target_fps}")
    if source_fps <= 0:
        logger.warning(
            "source fps is unavailable; fallback to explicit stride={}",
            stride)
        return max(1, stride)
    return max(1, int(round(source_fps / target_fps)))


class RoiManager:

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._items = self._load()

    @staticmethod
    def video_key(video_path: str | Path) -> str:
        return Path(video_path).name

    def get(self, video_path: str | Path) -> RoiConfig | None:
        key = self.video_key(video_path)
        raw = self._items.get(key)
        if not raw:
            return None
        return RoiConfig(
            video_key=key,
            width=int(raw["width"]),
            height=int(raw["height"]),
            points=[tuple(point) for point in raw["points"]],
        )

    def save(self, config: RoiConfig) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._items[config.video_key] = asdict(config)
        self.config_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def filter_detections(self, detections: Sequence[Detection],
                          roi: RoiConfig | None) -> list[Detection]:
        if not roi or len(roi.points) < 3:
            return list(detections)

        polygon = np.array(roi.points, dtype=np.int32)
        kept: list[Detection] = []
        for detection in detections:
            cx, cy = detection.center
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)),
                                    False) >= 0:
                kept.append(detection)
        return kept

    def _load(self) -> dict:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))


class RoiAnnotator:

    def __init__(self,
                 roi_manager: RoiManager,
                 window_name: str = "ROI Annotator") -> None:
        self.roi_manager = roi_manager
        self.window_name = window_name
        self.points: list[Point] = []
        self._base_image: np.ndarray | None = None

    def annotate(self,
                 video_path: str | Path,
                 frame_index: int = 0) -> RoiConfig:
        with VideoReader(video_path) as reader:
            frame = reader.read_frame(frame_index)
            self.points = []
            self._base_image = frame.image.copy()
            existing = self.roi_manager.get(video_path)
            if existing:
                self.points = list(existing.points)

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._on_mouse)

            while True:
                canvas = self._render_annotation_canvas()
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (13, ord("s")):
                    if len(self.points) < 3:
                        print("ROI needs at least 3 points.")
                        continue
                    config = RoiConfig(
                        video_key=RoiManager.video_key(video_path),
                        width=reader.width,
                        height=reader.height,
                        points=self.points,
                    )
                    self.roi_manager.save(config)
                    cv2.destroyWindow(self.window_name)
                    return config
                if key == ord("r"):
                    self.points = []
                if key in (ord("u"), 8):
                    if self.points:
                        self.points.pop()
                if key in (ord("q"), 27):
                    cv2.destroyWindow(self.window_name)
                    raise KeyboardInterrupt("ROI annotation cancelled.")

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def _render_annotation_canvas(self) -> np.ndarray:
        if self._base_image is None:
            raise RuntimeError("No image loaded for ROI annotation.")
        canvas = self._base_image.copy()
        draw_roi(canvas,
                 self.points,
                 color=Color.CYAN.value,
                 closed=len(self.points) >= 3)
        help_text = "left click: add | right/u: undo | r: reset | s/enter: save | q/esc: cancel"
        cv2.putText(canvas, help_text, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    Color.GREEN.value, 2)
        return canvas


class RfdetrDetector:
    WEIGHTS = {
        "nano": (
            "rf-detr-nano.pth",
            "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        ),
        "small": (
            "rf-detr-small.pth",
            "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        ),
        "base": (
            "rf-detr-base.pth",
            "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        ),
        "medium": (
            "rf-detr-medium.pth",
            "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
        ),
        "large": (
            "rf-detr-large.pth",
            "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        ),
    }

    def __init__(
        self,
        confidence: float = 0.25,
        class_names: Sequence[str] | None = None,
        size: str = "medium",
        device: str | None = None,
        model_dir: str | Path = "models/rfdetr",
    ) -> None:
        try:
            from rfdetr import (
                RFDETRBase,
                RFDETRLarge,
                RFDETRMedium,
                RFDETRNano,
                RFDETRSmall,
            )
            from rfdetr.util.coco_classes import COCO_CLASSES
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is required for RF-DETR detection. Install it with: pip install rfdetr"
            ) from exc

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "base": RFDETRBase,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        if size not in model_classes:
            raise ValueError(
                f"Unsupported RF-DETR size: {size}. Expected one of {sorted(model_classes)}"
            )

        rfdetr_device = self._normalize_device(device)
        weights_path = self._resolve_weights(size, model_dir)
        model_kwargs = {"pretrain_weights": str(weights_path)}
        if rfdetr_device:
            model_kwargs["device"] = rfdetr_device
        self.model = model_classes[size](**model_kwargs)
        self.confidence = confidence
        self.class_names = set(class_names or ["person", "sports ball"])
        self.device = device
        self.coco_classes = getattr(self.model, "class_names", COCO_CLASSES)
        self.person_tracker = self._build_person_tracker()

    def detect(self, frame: VideoFrame) -> list[Detection]:
        rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(rgb_image, threshold=self.confidence)
        return self._parse_predictions(frame, predictions)

    def track(self,
              frame: VideoFrame,
              tracker: str = "botsort.yaml",
              persist: bool = True) -> list[Detection]:
        detections = self.detect(frame)
        return self._track_persons(detections)

    def _parse_predictions(self, frame: VideoFrame,
                           predictions) -> list[Detection]:
        detections: list[Detection] = []
        xyxy = getattr(predictions, "xyxy", [])
        confidences = getattr(predictions, "confidence", [])
        class_ids = getattr(predictions, "class_id", [])
        data = getattr(predictions, "data", {}) or {}
        class_names = data.get("class_name") if isinstance(data,
                                                           dict) else None

        for index, box in enumerate(xyxy):
            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            label = self._label_for(class_id, class_names, index)
            if self.class_names and label not in self.class_names:
                continue
            confidence = float(
                confidences[index]) if index < len(confidences) else 0.0
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append(
                Detection(
                    frame_index=frame.index,
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    track_id=None,
                ))
        return detections

    def _label_for(self, class_id: int, class_names, index: int) -> str:
        if class_names is not None and index < len(class_names):
            return str(class_names[index])
        if isinstance(self.coco_classes, dict):
            return str(self.coco_classes.get(class_id, class_id))
        if 0 <= class_id < len(self.coco_classes):
            return str(self.coco_classes[class_id])
        return str(class_id)

    def _normalize_device(self, device: str | None) -> str | None:
        if not device:
            return None
        if device == "0" or device.startswith("cuda"):
            return "cuda"
        return device

    def _resolve_weights(self, size: str, model_dir: str | Path) -> Path:
        filename, url = self.WEIGHTS[size]
        weights_path = Path(model_dir) / filename
        if weights_path.exists():
            return weights_path
        raise FileNotFoundError(
            f"RF-DETR weights not found: {weights_path}. Download from {url} "
            f"and save as {weights_path}.")

    def _build_person_tracker(self):
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "supervision is required for RF-DETR person tracking."
            ) from exc
        return sv.ByteTrack(
            track_activation_threshold=self.confidence,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=60,
            minimum_consecutive_frames=1,
        )

    def _track_persons(self,
                       detections: Sequence[Detection]) -> list[Detection]:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "supervision is required for RF-DETR person tracking."
            ) from exc

        persons = [
            detection for detection in detections
            if detection.label == "person"
        ]
        others = [
            detection for detection in detections
            if detection.label != "person"
        ]
        if not persons:
            self.person_tracker.update_with_detections(sv.Detections.empty())
            return list(detections)

        # RF-DETR 输出检测框，ByteTrack 在这里补齐任务1必需的单镜头 track_id。
        tracked = self.person_tracker.update_with_detections(
            sv.Detections(
                xyxy=np.array([person.bbox for person in persons],
                              dtype=np.float32),
                confidence=np.array([person.confidence for person in persons],
                                    dtype=np.float32),
                class_id=np.array(
                    [
                        person.class_id if person.class_id is not None else -1
                        for person in persons
                    ],
                    dtype=np.int32,
                ),
            ))
        if tracked.tracker_id is None:
            return list(detections)

        tracked_persons: list[Detection] = []
        for index, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = [float(value) for value in tracked.xyxy[index]]
            class_id = int(tracked.class_id[index]
                           ) if tracked.class_id is not None else None
            confidence = (float(tracked.confidence[index])
                          if tracked.confidence is not None else 0.0)
            tracked_persons.append(
                Detection(
                    frame_index=persons[0].frame_index,
                    label="person",
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    track_id=int(track_id),
                ))
        return tracked_persons + others


class ResultVideoWriter:

    def __init__(self,
                 output_path: str | Path,
                 fps: float,
                 frame_size: tuple[int, int],
                 fourcc: str = "mp4v") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        logger.info("video writer opened: path={} fps={:.2f} size={}x{}",
                    self.output_path, fps, frame_size[0], frame_size[1])

    def write(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        expected_width, expected_height = self.frame_size
        if (width, height) != self.frame_size:
            image = cv2.resize(image, (expected_width, expected_height),
                               interpolation=cv2.INTER_AREA)
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()
        logger.info("video writer closed: path={}", self.output_path)


class VideoShow:

    def __init__(
        self,
        window_name: str = "AI Football",
        fps: float = 60.0,
        display_width: int = 1920,
        display_height: int = 1080,
    ) -> None:
        self.window_name = window_name
        self.fps = fps
        self.wait_ms = max(1, int(round(1000 / fps))) if fps > 0 else 1
        self.paused = False
        self.display_width = display_width
        self.display_height = display_height
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width,
                         self.display_height)
        self._center_window()

    def show(self, image: np.ndarray) -> bool:
        cv2.imshow(self.window_name, self._fit_to_display(image))
        return self._handle_key()

    def _fit_to_display(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if width == self.display_width and height == self.display_height:
            return image

        scale = min(self.display_width / width, self.display_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height),
                             interpolation=cv2.INTER_AREA)

        canvas = np.zeros((self.display_height, self.display_width, 3),
                          dtype=np.uint8)
        x = (self.display_width - resized_width) // 2
        y = (self.display_height - resized_height) // 2
        canvas[y:y + resized_height, x:x + resized_width] = resized
        return canvas

    def _center_window(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            x = max(0, (screen_width - self.display_width) // 2)
            y = max(0, (screen_height - self.display_height) // 2)
            cv2.moveWindow(self.window_name, x, y)
        except Exception:
            pass

    def _handle_key(self) -> bool:
        wait = 0 if self.paused else self.wait_ms
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.paused = not self.paused
        return True

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)


def draw_roi(image: np.ndarray,
             points: Sequence[Point],
             color=Color.CYAN.value,
             closed: bool = True) -> None:
    if not points:
        return
    for point in points:
        cv2.circle(image, point, 5, color, -1)
    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], closed, color,
                      3)


def draw_detection(image: np.ndarray, detection: Detection) -> None:
    x1, y1, x2, y2 = [int(value) for value in detection.bbox]
    color = Color.GREEN.value if detection.label == "person" else Color.ORANGE.value
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{detection.label} {detection.confidence:.2f}"
    if detection.track_id is not None:
        label = f"{label} id={detection.track_id}"
    cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


def render_detection_frame(frame: VideoFrame,
                           detections: Sequence[Detection],
                           roi: RoiConfig | None = None,
                           fps: float | None = None) -> np.ndarray:
    canvas = frame.image.copy()
    if roi:
        draw_roi(canvas, roi.points, color=Color.CYAN.value, closed=True)
    for detection in detections:
        draw_detection(canvas, detection)

    cv2.putText(
        canvas,
        f"frame={frame.index} time={frame.timestamp_sec:.2f}s det={len(detections)}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        Color.GREEN.value,
        2,
    )
    if fps is not None:
        cv2.putText(
            canvas,
            f"fps={fps:.2f}",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            Color.GREEN.value,
            2,
        )
    return canvas


def draw_debug_panel(
    image: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (24, 110),
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_gap = 8
    padding = 10
    text_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
    ]
    width = max(size[0] for size in text_sizes)
    height = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1)

    x, y = origin
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - padding),
        (x + width + padding, y + height + padding),
        Color.BLACK.value,
        -1,
    )
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    text_y = y
    for line, (text_width, text_height) in zip(lines, text_sizes):
        cv2.putText(
            image,
            line,
            (x, text_y + text_height),
            font,
            font_scale,
            Color.WHITE.value,
            thickness,
        )
        text_y += text_height + line_gap

