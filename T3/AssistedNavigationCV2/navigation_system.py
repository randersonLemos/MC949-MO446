import threading
import cv2
import numpy as np
import time
from typing import Dict, Any
import os

from config import CONFIG
from modules.detector import initialize_detector
from modules.tracked_object import TrackedObject
from modules.depth_estimation import initialize_depth_model, calculate_distance
from modules.kalman_filter import initialize_kalman_filter

# from modules.risk_calculator import calculate_risk
from modules.tts_engine import TTSEngine
from modules.narration_logic import NarrationLogic
from modules.visualization import display_results

CLASS_TRANSLATIONS = {
    "person": "pessoa",
    "car": "carro",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "bus": "ônibus",
    "truck": "caminhão",
    "traffic light": "semáforo",
    "stop sign": "placa de pare",
    "chair": "banco",
    # Adicione outros conforme necessário
}


def __init__(self, config=None):
    self.last_tts_time = 0


class NavigationSystem:
    """Main assisted navigation system"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.detector = None
        self.depth_model = None
        self.tts_engine = None
        self.narration_logic = None
        self.object_filters: Dict[int, Any] = {}

        # Video output
        self.video_writer = None
        self.output_frame_size = None
        self.measured_fps = 0
        self.fps_samples = []
        self.fps_sample_count = 0

        # Metrics
        self.frame_count = 0
        self.fps_counter = 0
        self.current_fps = 0
        self.fps_time = time.time()

        # Cooldown TTS
        self.last_tts_time = 0

        # UI
        self.window_name = "Sistema de Detecção de Colisão e Alerta"
        self.show_depth = self.config.ui.show_depth_map
        self.show_roi = self.config.ui.show_roi_map

    def get_best_direction(self, tracked_objects, frame_width):
        """
        Analisa os objetos e sugere a melhor direção para virar (nunca retorna 'em frente').
        """
        left_dist = 0
        right_dist = 0
        for obj in tracked_objects:
            if obj.smooth_dist is None:
                continue
            x1, y1, x2, y2 = obj.bbox
            bbox_center_x = (x1 + x2) / 2
            if bbox_center_x < frame_width / 2:
                left_dist += obj.smooth_dist
            else:
                right_dist += obj.smooth_dist
        # Retorna a direção com maior espaço livre
        if left_dist >= right_dist:
            return "esquerda"
        else:
            return "direita"

    def initialize(self):
        print("Inicializando módulos...")

        print(f"  Detector: {self.config.detection.model}")
        self.detector = initialize_detector(
            model_name=self.config.detection.model,
            conf_threshold=self.config.detection.confidence_threshold,
            device=self.config.detection.device,
        )

        print(f"  Profundidade: {self.config.depth.model}")
        self.depth_model = initialize_depth_model(
            model_type=self.config.depth.model, device=self.config.depth.device
        )

        # Inicializa TTS Engine (Coqui TTS)
        from modules.tts_engine import TTSEngine

        self.tts_engine = TTSEngine()

    def setup_video_capture(self):
        print(f"Abrindo fonte de vídeo: {self.config.video.source}")
        cap = cv2.VideoCapture(self.config.video.source)

        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir a fonte de vídeo")

        return cap

    def setup_ui(self):
        """Set up user interface"""
        if self.config.ui.show_visualization:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 0)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        import time
        start = time.time()
        tracker_type = self.config.detection.tracker_type
        tracked_detections = self.detector.track(frame, tracker_type=tracker_type)

        tracked_objects = []
        for detection in tracked_detections:
            if detection.get("track_id") is not None:
                tracked_obj = TrackedObject(detection_data=detection)
                tracked_objects.append(tracked_obj)

        depth_map = self.depth_model.predict(frame)

        from modules.depth_estimation import auto_calibrate_from_object_type, calculate_distance
        for obj in tracked_objects:
            if hasattr(obj, 'class_name') and obj.class_name in auto_calibrate_from_object_type.__globals__['KNOWN_OBJECTS']:
                scale_factor = auto_calibrate_from_object_type(depth_map, obj.bbox, obj.class_name, frame.shape[:2])
                raw_dist, roi_data = calculate_distance(obj.bbox, depth_map, scale_factor=scale_factor)
            else:
                raw_dist, roi_data = calculate_distance(obj.bbox, depth_map)
            obj.raw_dist = raw_dist
            obj.roi_data = roi_data

            kalman_filter = self.object_filters.get(obj.id)
            if kalman_filter is None:
                kalman_filter = initialize_kalman_filter()
                self.object_filters[obj.id] = kalman_filter

            obj.smooth_dist = kalman_filter.update(raw_dist)

            # Verifica se o objeto está na região central da imagem
            x1, y1, x2, y2 = obj.bbox
            bbox_center_x = (x1 + x2) / 2
            frame_center_x = frame.shape[1] / 2
            central_margin = frame.shape[1] * 0.2  # 20% de margem central

            current_time = time.time()
            if abs(bbox_center_x - frame_center_x) < central_margin:
                # Só alerta se passou o cooldown
                if current_time - self.last_tts_time > 8:
                    tipo_pt = CLASS_TRANSLATIONS.get(obj.class_name, obj.class_name)
                    dist_metros = obj.raw_dist
                    import math
                    dist_metros_int = math.ceil(dist_metros)
                    NUM_WORDS = {
                        1: "um",
                        2: "dois",
                        3: "três",
                        4: "quatro",
                        5: "cinco",
                        6: "seis",
                        7: "sete",
                        8: "oito",
                        9: "nove",
                        10: "dez",
                    }
                    dist_str = NUM_WORDS.get(dist_metros_int, str(dist_metros_int))
                    best_dir = self.get_best_direction(tracked_objects, frame.shape[1])
                    dir_msg = ""
                    if best_dir == "esquerda":
                        dir_msg = "Vire para a esquerda"
                    elif best_dir == "direita":
                        dir_msg = "Vire para a direita"

                    # Lógica de comandos
                    if dist_metros <= 3:
                        print(f"[TTS] PARE! {tipo_pt} à frente a {dist_str} metros. {dir_msg}")
                        if self.tts_engine:
                            threading.Thread(
                                target=self.tts_engine.speak,
                                args=(f"PARE! {tipo_pt} à frente, a {dist_str} metros. {dir_msg}",),
                                daemon=True,
                            ).start()
                    elif dist_metros <= 8:
                        print(f"[TTS] Atenção: {tipo_pt} à frente a {dist_str} metros.")
                        if self.tts_engine:
                            threading.Thread(
                                target=self.tts_engine.speak,
                                args=(f"Atenção! {tipo_pt} à frente, a {dist_str} metros.",),
                                daemon=True,
                            ).start()
                    self.last_tts_time = current_time

        return tracked_objects, depth_map

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter

            # Coletar amostras de FPS para calcular média
            if self.fps_counter > 0:  # Apenas se houver FPS válido
                self.fps_samples.append(self.fps_counter)
                self.fps_sample_count += 1

                # Manter apenas as últimas 10 amostras
                if len(self.fps_samples) > 10:
                    self.fps_samples.pop(0)

                # Calcular FPS médio
                self.measured_fps = sum(self.fps_samples) / len(self.fps_samples)

            self.fps_counter = 0
            self.fps_time = time.time()

    def handle_keyboard(self, key: int) -> bool:
        """Process keyboard input. Returns False to exit"""
        if key == ord("q") or key == 27:  # 'q' or ESC
            print("Saindo...")
            return False
        elif key == ord("d"):
            self.show_depth = not self.show_depth
            print(
                f"Mapa de profundidade: {'ATIVADO' if self.show_depth else 'DESATIVADO'}"
            )
        elif key == ord("r"):
            self.show_roi = not self.show_roi
            print(f"ROI Map: {'ATIVADO' if self.show_roi else 'DESATIVADO'}")
        # elif key == ord("r"):  # Record video
        #     self.config.video.save_output = not self.config.video.save_output
        #     print(
        #         f"Gravação de vídeo: {'ATIVADA' if self.config.video.save_output else 'DESATIVADA'}"
        #     )
        # elif key == ord("f"):  # FPS info
        #     print(f"\n=== INFORMAÇÕES DE FPS ===")
        #     print(f"FPS atual: {self.current_fps}")
        #     print(f"FPS médio medido: {self.measured_fps:.1f}")
        #     print(f"Amostras coletadas: {len(self.fps_samples)}")
        #     print(f"Auto FPS ativado: {self.config.video.auto_fps}")
        #     print(f"FPS configurado: {self.config.video.output_fps}")
        #     print("========================\n")
        # elif key == ord("h"):
        #     self.print_help()
        return True

    def run(self):
        """Run the main system"""
        try:
            # Initialization
            self.initialize()
            cap = self.setup_video_capture()
            self.setup_ui()

            print("Sistema iniciado! Pressione 'h' para ajuda")

            # Main loop
            video_writer_initialized = False
            while True:

                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Fim do vídeo ou erro na captura")
                    break

                self.frame_count += 1

                # Process
                tracked_objects, depth_map = self.process_frame(frame)

                # Update FPS
                self.update_fps()

                # Visualization
                output_frame = None
                if self.config.ui.show_visualization or self.config.video.save_output:
                    output_frame = display_results(
                        frame,
                        tracked_objects,
                        depth_map,
                        self.show_depth,
                        show_roi=self.show_roi,
                    )

                    # Add FPS
                    cv2.putText(
                        output_frame,
                        f"FPS: {self.current_fps}",
                        (output_frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Initialize video writer após coletar algumas amostras de FPS
                    if not video_writer_initialized and self.config.video.save_output:
                        # Se auto_fps está ativado, aguardar pelo menos 3 amostras de FPS
                        should_init = True
                        if self.config.video.auto_fps:
                            should_init = len(self.fps_samples) >= 3
                            if not should_init:
                                # Mostrar progresso de inicialização
                                progress_text = (
                                    f"Medindo FPS... {len(self.fps_samples)}/3"
                                )
                                cv2.putText(
                                    output_frame,
                                    progress_text,
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2,
                                )

                # Display frame (only if visualization is enabled)
                if self.config.ui.show_visualization and output_frame is not None:
                    cv2.imshow(self.window_name, output_frame)

                    # Check if window was closed
                    if (
                        cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                        < 1
                    ):
                        print("Janela fechada pelo usuário")
                        break

                key = cv2.waitKey(10) & 0xFF
                if not self.handle_keyboard(key):
                    break

                # # Periodic log
                # if self.frame_count % 30 == 0:
                #     print(
                #         f"Frame {self.frame_count}: {len(tracked_objects)} objetos | FPS: {self.current_fps}"
                #     )

                # # Limit FPS
                # if self.config.video.max_fps > 0:
                #     time.sleep(1.0 / self.config.video.max_fps)

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        except Exception as e:
            print(f"Erro: {e}")
            import traceback

            traceback.print_exc()
        finally:
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Vídeo salvo em: {self.config.video.output_path}")
            cv2.destroyAllWindows()
            print("Sistema encerrado")
