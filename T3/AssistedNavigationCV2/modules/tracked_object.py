class TrackedObject:
    """Represents an object tracked by the system"""

    __slots__ = [
        "id",
        "bbox",
        "class_id",
        "class_name",
        "confidence",
        "raw_dist",
        "smooth_dist",
        "velocity",
        "roi_data",
    ]

    def __init__(
        self,
        detection_data=None,
    ):

        self.id = detection_data.get("track_id")
        self.bbox = detection_data["bbox"]
        self.class_id = detection_data["class"]
        self.class_name = detection_data["class_name"]
        self.confidence = detection_data["confidence"]

        # Additional data for navigation
        self.raw_dist = None
        self.smooth_dist = None
        self.velocity = 0.0
        self.roi_data = None
