"""
Módulo de Visualização e Telemetria
"""

import cv2
import numpy as np
from modules.risk_calculator import categorize_risk


def create_roi_heatmap(roi_data, target_size=(100, 60)):
    """
    Cria um heatmap colorido da ROI de profundidade

    Args:
        roi_data: Dados da ROI (matriz 2D com valores de profundidade)
        target_size: Tamanho alvo (largura, altura) do heatmap

    Returns:
        Imagem colorida do heatmap (numpy array BGR)
    """
    if roi_data is None or roi_data.size == 0:
        # Retorna imagem preta se não há dados
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Normalizar valores para 0-255
    roi_normalized = roi_data.copy()
    roi_min, roi_max = roi_normalized.min(), roi_normalized.max()
    if roi_max > roi_min:
        roi_normalized = (roi_normalized - roi_min) / (roi_max - roi_min) * 255
    else:
        roi_normalized = np.ones_like(roi_normalized) * 127

    roi_normalized = roi_normalized.astype(np.uint8)

    # Redimensionar para tamanho alvo
    roi_resized = cv2.resize(
        roi_normalized, target_size, interpolation=cv2.INTER_LINEAR
    )

    # Aplicar colormap (viridis-like usando COLORMAP_VIRIDIS ou COLORMAP_TURBO)
    roi_colored = cv2.applyColorMap(roi_resized, cv2.COLORMAP_VIRIDIS)

    return roi_colored


def display_results(
    frame, tracked_objects, depth_map=None, show_depth=False, show_roi=True
):
    """
    Exibe resultados da detecção, rastreamento e análise de risco no frame

    Args:
        frame: Frame de vídeo
        tracked_objects: Lista de TrackedObjects
        depth_map: Mapa de profundidade (opcional)
        show_depth: Se True, mostra o mapa de profundidade
        show_roi: Se True, mostra heatmaps das ROIs

    Returns:
        Frame com visualizações
    """
    output_frame = frame.copy()

    # Marcar área central
    frame_height, frame_width = output_frame.shape[:2]
    central_margin = int(frame_width * 0.2)  # 20% de margem central
    x1_central = int(frame_width / 2 - central_margin)
    x2_central = int(frame_width / 2 + central_margin)
    y1_central = 0
    y2_central = frame_height
    # Desenha um retângulo azul claro na área central
    cv2.rectangle(output_frame, (x1_central, y1_central), (x2_central, y2_central), (255, 200, 100), 2)

    # Cores por categoria de risco
    risk_colors = {
        "low": (0, 255, 0),  # Verde
        "medium": (0, 255, 255),  # Amarelo
        "high": (0, 165, 255),  # Laranja
        "critical": (0, 0, 255),  # Vermelho
    }

    # Desenhar objetos rastreados
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj.bbox)

        # Determinar cor baseada no risco
        risk_category = categorize_risk(
            obj.smooth_dist if obj.smooth_dist is not None else 0
        )

        color = risk_colors.get(risk_category, (255, 255, 255))

        # Desenhar bounding box
        thickness = 3 if risk_category in ["high", "critical"] else 2
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)

        # Preparar texto
        label = f"ID:{obj.id} {obj.class_name}"
        label += f" C:{obj.confidence:.2f}"
        if obj.smooth_dist is not None:
            label += f" D:{obj.smooth_dist:.2f}"
        # label += f" R:{obj.risk_score:.2f}"

        # Desenhar fundo do texto
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            output_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1
        )

        # Desenhar texto
        cv2.putText(
            output_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Desenhar heatmap da ROI (se disponível e solicitado)
        if show_roi and hasattr(obj, "roi_data") and obj.roi_data is not None:
            # Criar heatmap da ROI
            roi_heatmap = create_roi_heatmap(obj.roi_data, target_size=(80, 50))

            # Posicionar heatmap no canto inferior direito do bbox
            heatmap_h, heatmap_w = roi_heatmap.shape[:2]
            heatmap_x = min(x2 - heatmap_w, output_frame.shape[1] - heatmap_w)
            heatmap_y = min(y2 + 5, output_frame.shape[0] - heatmap_h)

            # Garantir que está dentro dos limites
            if heatmap_x >= 0 and heatmap_y >= 0:
                # Adicionar borda branca ao heatmap
                roi_with_border = cv2.copyMakeBorder(
                    roi_heatmap, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )

                border_h, border_w = roi_with_border.shape[:2]
                end_y = min(heatmap_y + border_h, output_frame.shape[0])
                end_x = min(heatmap_x + border_w, output_frame.shape[1])

                output_frame[heatmap_y:end_y, heatmap_x:end_x] = roi_with_border[
                    : end_y - heatmap_y, : end_x - heatmap_x
                ]

                # Adicionar label do heatmap
                cv2.putText(
                    output_frame,
                    "ROI Depth",
                    (heatmap_x, heatmap_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return output_frame
