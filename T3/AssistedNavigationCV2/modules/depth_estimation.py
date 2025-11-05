# Dicionário de tamanhos típicos
KNOWN_OBJECTS = {
    'person': {'height': 1.7, 'width': 0.5},      # pessoa adulta
    'car': {'height': 1.5, 'width': 1.8},         # carro
    'door': {'height': 2.1, 'width': 0.6},        # porta padrão
    'monitor': {'height': 0.4, 'width': 0.6},     # monitor 24"
    'bottle': {'height': 0.25, 'width': 0.08},    # garrafa 1L
    'chair': {'height': 0.9, 'width': 0.5},       # cadeira
    'bicycle': {'height': 1.0, 'width': 0.5},     # bicicleta
}

def auto_calibrate_from_object_type(depth_map, bbox, object_type, frame_shape):
    """
    Calibra baseado em tipo de objeto detectado.
    Args:
        depth_map: Mapa de profundidade
        bbox: [x1, y1, x2, y2]
        object_type: 'person', 'car', 'door', etc
        frame_shape: (height, width) do frame
    Returns:
        scale_factor
    """
    if object_type not in KNOWN_OBJECTS:
        print(f"Objeto '{object_type}' não tem dimensões conhecidas")
        return 0.40
    obj_dims = KNOWN_OBJECTS[object_type]
    x1, y1, x2, y2 = map(int, bbox)
    # Dimensões em pixels
    obj_height_pixels = y2 - y1
    obj_width_pixels = x2 - x1
    # Extrai profundidade
    h, w = depth_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.40
    lower = np.percentile(roi, 10)
    upper = np.percentile(roi, 90)
    roi_filtered = roi[(roi > lower) & (roi < upper)]
    if roi_filtered.size == 0:
        roi_filtered = roi
    depth_value = np.median(roi_filtered)
    # Estima distância baseado em altura (mais confiável)
    frame_height = frame_shape[0]
    real_height = obj_dims['height']
    # Heurística simplificada
    estimated_distance = (real_height * frame_height) / (obj_height_pixels )

    # Suavização: mantém histórico das últimas N estimativas
    if not hasattr(auto_calibrate_from_object_type, "dist_history"):
        auto_calibrate_from_object_type.dist_history = []
    N = 5  # número de estimativas para suavizar
    auto_calibrate_from_object_type.dist_history.append(estimated_distance)
    if len(auto_calibrate_from_object_type.dist_history) > N:
        auto_calibrate_from_object_type.dist_history.pop(0)
    smoothed_distance = np.median(auto_calibrate_from_object_type.dist_history)

    scale_factor = depth_value / smoothed_distance
    return float(scale_factor)
import torch
import cv2
import numpy as np
from typing import Tuple, List, Optional


def depth_to_meters(depth_value, min_depth=1500, max_depth=6000, min_m=2, max_m=0.5):
    """
    Converte valor de profundidade MiDaS para metros reais (escala invertida).
    
    Args:
        depth_value: Valor de profundidade do MiDaS
        min_depth: valor de profundidade para distância mais longe (ex: 1500 → 2m)
        max_depth: valor de profundidade para distância mais perto (ex: 6000 → 0.5m)
        min_m: distância correspondente ao min_depth
        max_m: distância correspondente ao max_depth
    
    Returns:
        Distância em metros
    """
    # Protege contra divisão por zero
    if max_depth == min_depth:
        return min_m
    
    metros = min_m + (max_m - min_m) * (depth_value - min_depth) / (max_depth - min_depth)
    return max(metros, 0)


class DepthEstimator:
    """Depth estimator using MiDAS and DPT models with auto-calibration support"""

    def __init__(self, model_type="MiDaS_small", device='cpu'):
        # Validation
        supported_models = ["MiDaS", "MiDaS_small", "DPT_Hybrid", "DPT_Large"]
        if model_type not in supported_models:
            raise ValueError(
                f"Modelo '{model_type}' não suportado. Use: {supported_models}"
            )

        self.model_type = model_type
        self.device = torch.device(device)

        # Load model
        self.model = torch.hub.load("intel-isl/MiDAS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDAS", "transforms")

        if model_type in ["DPT_Hybrid", "DPT_Large"]:
            self.transform = midas_transforms.dpt_transform
        elif model_type == "MiDaS":
            self.transform = midas_transforms.default_transform
        else:  # MiDaS_small
            self.transform = midas_transforms.small_transform
        
        # Auto-calibration data
        self.scale_factor = 0.40  # Default
        self.is_calibrated = False
        self.calibration_samples = []  # [(depth_value, real_distance), ...]

    def predict(self, frame):
        """Estimate depth map for a frame"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_batch = self.transform(img_rgb).to(self.device)

        # Prediction
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map
    
    def add_calibration_sample(self, bbox, depth_map, real_distance_meters):
        """
        Adiciona uma amostra de calibração.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] do objeto
            depth_map: Mapa de profundidade
            real_distance_meters: Distância real do objeto em metros
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        
        # Garante limites válidos
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        roi = depth_map[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Filtra outliers
            lower_bound = np.percentile(roi, 10)
            upper_bound = np.percentile(roi, 90)
            roi_filtered = roi[(roi > lower_bound) & (roi < upper_bound)]
            
            if roi_filtered.size == 0:
                roi_filtered = roi
            
            depth_value = np.median(roi_filtered)
            self.calibration_samples.append((float(depth_value), float(real_distance_meters)))
            
            # Recalcula scale_factor automaticamente
            self._update_calibration()
    
    def _update_calibration(self):
        """Atualiza o scale_factor baseado nas amostras coletadas"""
        if len(self.calibration_samples) == 0:
            return
        
        depths = np.array([s[0] for s in self.calibration_samples])
        distances = np.array([s[1] for s in self.calibration_samples])
        
        # Calcula scale_factor como média de depth/distance
        # depth_value / scale_factor = distance_meters
        # scale_factor = depth_value / distance_meters
        scale_factors = depths / distances
        
        # Usa mediana para ser robusto a outliers
        self.scale_factor = float(np.median(scale_factors))
        self.is_calibrated = True
        
        print(f"[Calibração] Atualizado scale_factor = {self.scale_factor:.4f} ({len(self.calibration_samples)} amostras)")
    
    def calibrate_from_objects(self, depth_map, known_objects):
        """
        Calibra de uma vez com múltiplos objetos conhecidos.
        
        Args:
            depth_map: Mapa de profundidade da cena
            known_objects: Lista de tuplas (bbox, distancia_real_metros)
                          bbox = [x1, y1, x2, y2]
        
        Example:
            known_objects = [
                ([100, 100, 200, 200], 1.5),  # objeto a 1.5m
                ([300, 150, 400, 250], 3.0),  # objeto a 3.0m
            ]
        """
        self.calibration_samples.clear()
        
        for bbox, real_distance in known_objects:
            self.add_calibration_sample(bbox, depth_map, real_distance)
    
    def reset_calibration(self):
        """Reseta a calibração para o padrão"""
        self.calibration_samples.clear()
        self.scale_factor = 0.40
        self.is_calibrated = False
        print("[Calibração] Resetada para padrão (scale_factor = 0.40)")
    
    def get_calibration_info(self):
        """Retorna informações sobre a calibração atual"""
        return {
            'is_calibrated': self.is_calibrated,
            'scale_factor': self.scale_factor,
            'num_samples': len(self.calibration_samples),
            'samples': self.calibration_samples.copy()
        }


def initialize_depth_model(model_type="MiDaS_small", device='cpu'):
    """
    Inicializa o modelo de estimativa de profundidade com suporte a auto-calibração.
    
    Args:
        model_type: Tipo do modelo ("MiDaS", "MiDaS_small", "DPT_Hybrid", "DPT_Large")
        device: Dispositivo para inferência ('cpu' ou 'cuda')
    
    Returns:
        DepthEstimator: Instância do estimador de profundidade
    """
    return DepthEstimator(model_type, device)


def calculate_distance(bbox, depth_map, scale_factor=0.40):
    """
    Calculate estimated distance of an object based on depth map.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        depth_map: Depth map (raw MiDAS values)
        scale_factor: Scale factor to convert to meters (default: 0.40)

    Returns:
        Tuple (distance_in_meters, roi_array)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Ensure coordinates are within bounds
    h, w = depth_map.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    # Extract region of interest
    roi = depth_map[y1:y2, x1:x2]

    if roi.size == 0:
        return float("inf"), np.array([])

    # Filter outliers
    lower_bound = np.percentile(roi, 10)
    upper_bound = np.percentile(roi, 90)
    roi_filtered = roi[(roi > lower_bound) & (roi < upper_bound)]

    if roi_filtered.size == 0:
        roi_filtered = roi

    # Use median of region as depth
    depth_value = np.median(roi_filtered)

    # Convert to meters using scale factor
    depth_meters = depth_value / scale_factor if scale_factor != 0 else float('inf')
    
    return float(depth_meters), roi


def auto_calibrate_from_scene(depth_map, known_objects):
    """
    Auto calibra baseado em objetos conhecidos na cena.
    Função standalone para compatibilidade.
    
    Args:
        depth_map: Mapa de profundidade
        known_objects: Lista de tuplas (bbox, distancia_real_metros)
                      onde bbox = [x1, y1, x2, y2]
    
    Returns:
        scale_factor: Fator de escala calibrado
    
    Example:
        known_objects = [
            ([100, 100, 200, 200], 1.5),  # objeto a 1.5m
            ([300, 150, 400, 250], 3.0),  # objeto a 3.0m
        ]
        scale_factor = auto_calibrate_from_scene(depth_map, known_objects)
    """
    depths = []
    distances = []
    
    for bbox, real_distance in known_objects:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        
        # Garante limites válidos
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        roi = depth_map[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Filtra outliers
            lower_bound = np.percentile(roi, 10)
            upper_bound = np.percentile(roi, 90)
            roi_filtered = roi[(roi > lower_bound) & (roi < upper_bound)]
            
            if roi_filtered.size == 0:
                roi_filtered = roi
            
            depth_value = np.median(roi_filtered)
            depths.append(depth_value)
            distances.append(real_distance)
    
    if len(depths) == 0:
        print("[Aviso] Nenhuma amostra válida para calibração. Usando padrão 0.40")
        return 0.40
    
    # Calcula scale_factor
    depths = np.array(depths)
    distances = np.array(distances)
    scale_factors = depths / distances
    
    # Usa mediana para robustez
    scale_factor = float(np.median(scale_factors))
    
    print(f"[Calibração] scale_factor calculado = {scale_factor:.4f} ({len(depths)} amostras)")
    
    return scale_factor


