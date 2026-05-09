# ============================================================
# VoxSense - Object Detection Service (YOLOv8)
# ============================================================

import time
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

import sys

from ultralytics import YOLO
sys.path.insert(0, '/home/claude/voxsense')
from config.settings import settings, config
from backend.utils.logger import logger


@dataclass
class Detection:
    """Single object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    position: str  # "left", "center", "right"
    distance: str  # "very close", "close", "medium", "far"
    guidance: str  # voice guidance text


@dataclass
class DetectionResult:
    """Complete frame detection result."""
    detections: List[Detection] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    annotated_frame: Optional[np.ndarray] = None
    voice_alerts: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# Position and distance thresholds
DISTANCE_ZONES = {
    "very close": 0.35,
    "close": 0.20,
    "medium": 0.10,
    "far": 0.0,
}

# Priority objects that always get announced
PRIORITY_OBJECTS = {"person", "car", "bus", "truck", "stairs", "door", "bicycle", "motorcycle"}

# Human-friendly position labels
POSITION_LABELS = {
    "left": "on your left",
    "center": "ahead",
    "right": "on your right",
}

# Voice guidance templates
GUIDANCE_TEMPLATES = {
    "person": "Person {position}",
    "chair": "Chair {position}",
    "table": "Table {position}",
    "door": "Door {position}",
    "stairs": "Stairs {position} — caution",
    "car": "Car {position} — caution",
    "bicycle": "Bicycle {position}",
    "motorcycle": "Motorcycle {position}",
    "bus": "Bus {position} — caution",
    "truck": "Truck {position} — caution",
    "traffic light": "Traffic light {position}",
    "stop sign": "Stop sign {position}",
    "bench": "Bench {position}",
    "bottle": "Bottle {position}",
    "cup": "Cup {position}",
    "cell phone": "Phone {position}",
    "laptop": "Laptop {position}",
    "book": "Book {position}",
    "backpack": "Backpack {position}",
    "umbrella": "Umbrella {position}",
    "handbag": "Bag {position}",
    "suitcase": "Luggage {position}",
    "bed": "Bed {position}",
    "toilet": "Toilet {position}",
    "sink": "Sink {position}",
}


class ObjectDetectionService:
    """
    Real-time object detection service for blind user navigation.
    Uses YOLOv8 for detection and generates voice guidance.
    """

    def __init__(self):
        self.model = None
        self.confidence = config.get("vision", {}).get("yolo", {}).get("confidence", 0.5)
        self.iou = config.get("vision", {}).get("yolo", {}).get("iou", 0.45)
        self.target_classes = set(
            config.get("vision", {}).get("yolo", {}).get("target_classes", list(GUIDANCE_TEMPLATES.keys()))
        )
        self.alert_cooldown = config.get("vision", {}).get("guidance", {}).get("alert_cooldown_seconds", 3)
        self._last_alerts: Dict[str, float] = {}
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model (lazy loading)."""
        try:
            from ultralytics import YOLO
            model_name = settings.yolo_model
            logger.info(f"📦 Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)
            logger.info("✅ YOLO model loaded successfully")
        except ImportError:
            logger.warning("⚠️ ultralytics not installed. Object detection unavailable.")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")

    def _get_position(self, x1: int, x2: int, frame_width: int) -> str:
        """Determine horizontal position (left/center/right)."""
        center_x = (x1 + x2) / 2
        ratio = center_x / frame_width
        if ratio < 0.33:
            return "left"
        elif ratio > 0.66:
            return "right"
        else:
            return "center"

    def _get_distance(self, y1: int, y2: int, frame_height: int) -> str:
        """Estimate distance from bounding box height ratio."""
        bbox_height_ratio = (y2 - y1) / frame_height
        if bbox_height_ratio > DISTANCE_ZONES["very close"]:
            return "very close"
        elif bbox_height_ratio > DISTANCE_ZONES["close"]:
            return "close"
        elif bbox_height_ratio > DISTANCE_ZONES["medium"]:
            return "medium"
        else:
            return "far"

    def _should_alert(self, class_name: str) -> bool:
        """Check if we should alert for this class (cooldown)."""
        now = time.time()
        last = self._last_alerts.get(class_name, 0)
        if now - last >= self.alert_cooldown:
            self._last_alerts[class_name] = now
            return True
        return False

    def _build_guidance(self, class_name: str, position: str, distance: str) -> str:
        """Build voice guidance text for a detection."""
        template = GUIDANCE_TEMPLATES.get(class_name, f"{class_name.title()} {{position}}")
        pos_label = POSITION_LABELS.get(position, position)
        guidance = template.format(position=pos_label)

        # Add distance qualifier for very close items
        if distance == "very close":
            guidance = f"Warning! {guidance} — very close"
        elif distance == "close" and class_name in PRIORITY_OBJECTS:
            guidance = f"{guidance} — nearby"

        return guidance

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run object detection on a single video frame.
        Returns DetectionResult with all detections and voice guidance.
        """
        if self.model is None:
            return DetectionResult()

        h, w = frame.shape[:2]
        result = DetectionResult(frame_width=w, frame_height=h)

        try:
            # Run YOLO inference
            predictions = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                verbose=False,
            )

            # Annotated frame for display
            annotated = frame.copy()

            for pred in predictions:
                boxes = pred.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id].lower()

                    # Filter to target classes
                    if class_name not in self.target_classes:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    position = self._get_position(x1, x2, w)
                    distance = self._get_distance(y1, y2, h)
                    guidance = self._build_guidance(class_name, position, distance)

                    detection = Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        position=position,
                        distance=distance,
                        guidance=guidance,
                    )
                    result.detections.append(detection)

                    # Draw bounding box
                    color = (0, 255, 0) if class_name not in PRIORITY_OBJECTS else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(annotated, distance, (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            result.annotated_frame = annotated

            # Build voice alert list (with cooldown)
            for det in sorted(result.detections, key=lambda d: d.distance == "very close", reverse=True):
                if self._should_alert(det.class_name):
                    result.voice_alerts.append(det.guidance)

            if result.voice_alerts:
                logger.debug(f"🔊 Alerts: {result.voice_alerts}")

        except Exception as e:
            logger.error(f"❌ Detection error: {e}")

        return result

    def detect_from_base64(self, image_b64: str) -> DetectionResult:
        """Decode base64 image and run detection."""
        try:
            img_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")
            return self.detect_frame(frame)
        except Exception as e:
            logger.error(f"❌ base64 detection error: {e}")
            return DetectionResult()

    def frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """Encode annotated frame to base64 JPEG."""
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            logger.error(f"❌ Frame encoding error: {e}")
            return ""


# Singleton
_detection_service: Optional[ObjectDetectionService] = None


def get_detection_service() -> ObjectDetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = ObjectDetectionService()
    return _detection_service
