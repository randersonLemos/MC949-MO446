"""
Módulo de Lógica de Narração e Alertas
"""

from collections import defaultdict
import time

# from modules.risk_calculator import categorize_risk, get_priority_objects


class NarrationLogic:
    """Gerencia a lógica de narração e alertas"""

    def __init__(self, tts_engine):
        """
        Inicializa a lógica de narração

        Args:
            tts_engine: Instância do TTSEngine
        """
        self.tts = tts_engine
        self.last_alerts = defaultdict(float)  # obj_id -> timestamp
        self.alert_cooldown = 3.0  # segundos entre alertas do mesmo objeto

        # Mensagens por categoria de risco
        self.risk_messages = {
            "critical": "Atenção! Objeto muito próximo à frente!",
            "high": "Cuidado! Objeto se aproximando!",
            "medium": "Objeto detectado à frente",
            "low": "",  # Sem alerta para baixo risco
        }

        # Mensagens por classe de objeto
        self.class_messages = {
            "person": "Pedestre",
            "bicycle": "Bicicleta",
            "car": "Carro",
            "motorcycle": "Moto",
            "bus": "Ônibus",
            "truck": "Caminhão",
            "traffic light": "Semáforo",
            "stop sign": "Placa de pare",
        }

    def decide_and_speak(self, tracked_objects):
        """
        Decide quais alertas emitir baseado nos objetos rastreados

        Args:
            tracked_objects: Lista de TrackedObjects
        """
        if not tracked_objects:
            return

        # Obter objetos prioritários
        priority_objects = get_priority_objects(tracked_objects, top_n=2)

        current_time = time.time()

        for obj in priority_objects:
            # Categorizar risco
            risk_category = categorize_risk(obj.risk_score)

            # Verificar se deve alertar
            if risk_category == "low":
                continue

            # Verificar cooldown do objeto
            if current_time - self.last_alerts[obj.id] < self.alert_cooldown:
                continue

            # Construir mensagem
            message = self._build_message(obj, risk_category)

            if message:
                # Emitir alerta
                self.tts.speak(message)
                self.last_alerts[obj.id] = current_time

                # Apenas um alerta por vez
                break

    def _build_message(self, obj, risk_category):
        """
        Constrói a mensagem de alerta

        Args:
            obj: TrackedObject
            risk_category: Categoria de risco

        Returns:
            Mensagem formatada
        """
        # Mensagem base de risco
        base_message = self.risk_messages.get(risk_category, "")

        if not base_message:
            return ""

        # Adicionar informação do objeto
        obj_name = self.class_messages.get(obj.class_name, obj.class_name)

        # Adicionar direção aproximada
        bbox = obj.bbox
        x_center = (bbox[0] + bbox[2]) / 2

        # Assumindo frame de 640 pixels de largura (padrão YOLO)
        if x_center < 213:
            direction = "à esquerda"
        elif x_center > 427:
            direction = "à direita"
        else:
            direction = "à frente"

        # Montar mensagem completa
        if risk_category == "critical":
            message = f"Perigo! {obj_name} muito próximo {direction}!"
        elif risk_category == "high":
            message = f"Atenção! {obj_name} {direction}!"
        else:
            message = f"{obj_name} {direction}"

        return message

    def set_alert_cooldown(self, cooldown):
        """Define o tempo de cooldown entre alertas"""
        self.alert_cooldown = cooldown

    def clear_alerts(self):
        """Limpa histórico de alertas"""
        self.last_alerts.clear()
