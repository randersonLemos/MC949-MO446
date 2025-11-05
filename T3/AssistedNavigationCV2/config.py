"""
Assisted Navigation System Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Union


def _get_video_source() -> Union[int, str]:
    """Get video source from environment variable or default"""
    env_source = os.environ.get("VIDEO_SOURCE")
    if env_source is not None:
        # Convert to int if it's a digit (webcam index)
        if env_source.isdigit():
            return int(env_source)
        return env_source
    return "videos/corredor.mp4"  # Default value

@dataclass
class VideoConfig:
    source: Union[int, str] = field(
        default_factory=lambda: _get_video_source()
    )  # 0 = webcam, or file path
    max_fps: int = 0  # 0 = no limit_
    display_width: int = 0  # 0 = original size
    # Video output settings
    save_output: bool = True  # Enable/disable video saving
    output_path: str = "output/navigation_output.mp4"  # Output file path
    output_codec: str = "mp4v"  # Video codec (mp4v, XVID, H264)
    output_fps: int = 30  # Output video FPS (0 = use measured FPS)
    auto_fps: bool = True  # Use measured FPS instead of fixed FPS


@dataclass
class DetectionConfig:
    model: str = "models/yolo11n.pt"
    confidence_threshold: float = 0.4
    max_priority_objects: int = 100
    device: str | int = 0
    tracker_type: str = "bytetrack.yaml"


@dataclass
class DepthConfig:
    model: str = "MiDaS"  # Opções: 'MiDaS', 'MiDaS_small', 'DPT_Hybrid'
    device: str | int = 0


@dataclass
class RiskConfig:
    distance_threshold: float = 0.7
    velocity_weight: float = 0.3


@dataclass
class TTSConfig:
    rate: int = 150
    volume: float = 1.0
    cooldown: float = 2.0


@dataclass
class AlertConfig:
    cooldown: float = 3.0


@dataclass
class KalmanConfig:
    process_variance: float = 1e-5
    measurement_variance: float = 1e-2


@dataclass
class UIConfig:
    show_visualization: bool = True
    show_depth_map: bool = False
    show_roi_map: bool = True


@dataclass
class SystemConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Global configuration instance
CONFIG = SystemConfig()
