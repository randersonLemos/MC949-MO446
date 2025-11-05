"""
Módulo de Filtro de Kalman para suavização de distâncias
"""
import numpy as np


class KalmanFilter:
    """Filtro de Kalman para suavizar medidas de distância"""
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        """
        Inicializa o Filtro de Kalman
        
        Args:
            process_variance: Variância do processo (Q)
            measurement_variance: Variância da medida (R)
        """
        # Estado inicial
        self.x = 0.0  # estimativa
        self.P = 1.0  # covariância do erro
        
        # Parâmetros do filtro
        self.Q = process_variance  # variância do processo
        self.R = measurement_variance  # variância da medida
        
        self.initialized = False
    
    def update(self, measurement):
        """
        Atualiza o filtro com uma nova medida
        
        Args:
            measurement: Nova medida de distância
            
        Returns:
            Estimativa filtrada
        """
        if not self.initialized:
            # Primeira medida - inicializar estado
            self.x = measurement
            self.initialized = True
            return self.x
        
        # Predição
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Atualização
        K = P_pred / (P_pred + self.R)  # Ganho de Kalman
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x
    
    def reset(self):
        """Reseta o filtro"""
        self.x = 0.0
        self.P = 1.0
        self.initialized = False


def initialize_kalman_filter(process_variance=1e-5, measurement_variance=1e-2):
    """
    Função auxiliar para inicializar um Filtro de Kalman
    
    Args:
        process_variance: Variância do processo
        measurement_variance: Variância da medida
        
    Returns:
        Instância do KalmanFilter
    """
    return KalmanFilter(process_variance, measurement_variance)
