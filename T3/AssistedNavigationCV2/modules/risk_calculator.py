"""
Módulo de Cálculo de Risco de Colisão
"""

import numpy as np


# def calculate_risk(obj, distance_threshold=0.7, velocity_weight=0.3):
#     """
#     Calcula o score de risco de colisão para um objeto rastreado

#     Args:
#         obj: TrackedObject com informações do objeto
#         distance_threshold: Limiar de distância para alto risco (0-1)
#         velocity_weight: Peso da velocidade no cálculo do risco

#     Returns:
#         Score de risco (0-1), onde 1 = risco máximo
#     """
#     if obj.smooth_dist is None:
#         return 0.0

#     # Componente de distância (quanto mais próximo, maior o risco)
#     distance_risk = max(0.0, 1.0 - (obj.smooth_dist / distance_threshold))

#     # Componente de velocidade (objetos se aproximando são mais perigosos)
#     velocity_risk = 0.0
#     if hasattr(obj, "velocity") and obj.velocity > 0:
#         velocity_risk = min(1.0, obj.velocity / 2.0)

#     # Risco combinado
#     risk_score = (1 - velocity_weight) * distance_risk + velocity_weight * velocity_risk

#     # Ajustar por classe de objeto (pedestres e veículos são mais críticos)
#     critical_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
#     if obj.class_name in critical_classes:
#         risk_score *= 1.2  # Aumentar risco para objetos críticos

#     # Limitar entre 0 e 1
#     risk_score = min(1.0, max(0.0, risk_score))

#     return risk_score


def categorize_risk(depth_distance):
    """
    Categoriza o nível de risco

    Args:
        risk_score: Score de risco (0-1)

    Returns:
        Categoria de risco ('low', 'medium', 'high', 'critical')
    """
    if depth_distance > 8 :
        return "low"
    elif 6 < depth_distance <= 8:
        return "medium"
    elif 3 < depth_distance <= 6:
        return "high"
    else:
        return "critical"


# def get_priority_objects(tracked_objects, top_n=3):
#     """
#     Retorna os objetos com maior risco de colisão

#     Args:
#         tracked_objects: Lista de TrackedObjects
#         top_n: Número de objetos prioritários a retornar

#     Returns:
#         Lista de objetos ordenados por risco (maior primeiro)
#     """
#     # Ordenar por risk_score decrescente
#     sorted_objects = sorted(
#         tracked_objects,
#         key=lambda x: x.risk_score,
#         reverse=True
#     )

#     return sorted_objects[:top_n]
