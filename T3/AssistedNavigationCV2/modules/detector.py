"""
Object Detection Module (Generic, supports YOLOv8, RT-DETR, or others)
"""

from ultralytics import YOLO, RTDETR
import numpy as np


class ObjectDetector:
    """Generic object detector supporting YOLOv8, RT-DETR, etc."""

    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25, device='cpu'):
        """
        Initializes the object detector.

        Args:
            model_name: Model file or name (e.g., yolov8n.pt, rtdetr-r18.pt)
            conf_threshold: Confidence threshold for detections
        """

        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.device = device

        # Detect model type by name
        if "rtdetr" in model_name.lower():
            print(f"  Carreagando RT-DETR: {model_name}")
            self.model = RTDETR(model_name)
            self.is_rtdetr = True
        else:
            print(f"  Carregando YOLO: {model_name}")
            self.model = YOLO(model_name.split('/')[-1])
            self.is_rtdetr = False

    def track(self, frame, tracker_type="bytetrack.yaml"):
        """
        Performs object detection and tracking using the selected model.

        Args:
            frame: Video frame (numpy array)
            tracker_type: Tracker type ('bytetrack.yaml', 'botsort.yaml', etc.)

        Returns:
            List of tracked objects in the format:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class': int,
                'class_name': str,
                'track_id': int  # Unique tracker ID
            }]
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            tracker=tracker_type,
            persist=True,  # Keep consistent IDs across frames
            verbose=False,
            device=self.device
        )

        tracked_objects = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]

                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                tracked_objects.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class": cls,
                        "class_name": class_name,
                        "track_id": track_id,
                    }
                )

        return tracked_objects


def initialize_detector(model_name="yolov8n.pt", conf_threshold=0.25, device='cpu'):
    """
    Helper function to initialize the detector.

    Args:
        model_name: Model file or name
        conf_threshold: Confidence threshold

    Returns:
        Instance of ObjectDetector
    """
    return ObjectDetector(model_name, conf_threshold, device)
