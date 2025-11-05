# AssistedNavigationCV - Sistema de Detecção e Alerta de Colisão

Sistema de navegação assistida baseado em visão computacional para detecção de obstáculos e alerta de colisão em tempo real.

## Arquitetura do Sistema

### Módulos Implementados

1. **detector.py** - Detecção de objetos com YOLO11n
   - Usa modelo YOLO11n (nano por padrão)
   - Detecta múltiplas classes de objetos
   - Retorna bounding boxes e classes

2. **tracker.py** - Rastreamento de objetos
   - Implementação de rastreamento multi-objeto
   - Mantém IDs consistentes entre frames
   - Suporta ByteTrack

3. **depth_estimation.py** - Estimação de profundidade
   - Usa modelo MiDAS 
   - Gera mapas de profundidade densa
   - Calcula distância por bounding box

4. **kalman_filter.py** - Suavização de medidas
   - Filtro de Kalman 1D para distâncias
   - Reduz ruído nas estimativas
   - Um filtro por objeto rastreado

5. **risk_calculator.py** - Análise de risco
   - Calcula score de risco de colisão
   - Smooth distance com Kalman
   - Categoriza riscos (low/medium/high/critical)

6. **tts_engine.py** - Text-to-Speech
   - Piper (modelo ONNX) por padrão:
   - Execução assíncrona
   - Cooldown entre alertas

7. **narration_logic.py** - Lógica de alertas
   - Decide quando emitir alertas
   - Prioriza objetos mais perigosos
   - Mensagens contextuais

8. **visualization.py** - Visualização
   - Desenha bounding boxes com cores por risco
   - Mostra telemetria em tempo real
   - Opcional: visualização de mapa de profundidade

9. **main.py** - Loop principal
   - Integra todos os módulos
   - Processa vídeo frame a frame
   - Interface com usuário

## Instalação

### Requisitos
- Python 3.8+
- GPU NVIDIA (recomendado para melhor performance)

### Passos

1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. Download de modelos (automático na primeira execução):
   - YOLOv8n será baixado pela biblioteca ultralytics
   - MiDAS será baixado via torch.hub

## Uso

### Execução Básica
```bash
./run.sh
```

### Configurações no main.py

```python
# Fonte de vídeo
VIDEO_SOURCE = 0  # 0 = webcam, ou caminho para arquivo

# Modelo YOLOv8
model_name='yolov8n.pt'  # Opções: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Modelo MiDAS
model_type='DPT_Small'  # Opções: DPT_Large, DPT_Hybrid, DPT_Small, MiDaS_small

# Threshold de confiança
conf_threshold=0.3  # 0.0 a 1.0

# Visualização
SHOW_VISUALIZATION = True
SHOW_DEPTH_MAP = False
```

### Controles de Teclado
- `q` - Sair do programa
- `r` - Alternar visualização do mapa de profundidade

## Pipeline de Processamento

```
Frame de Vídeo
    ↓
[Detector YOLOv8] → Detecções
    ↓
[Tracker] → Objetos Rastreados com IDs
    ↓
[MiDAS] → Mapa de Profundidade
    ↓
[Cálculo de Distância] → Distância por objeto
    ↓
[Filtro de Kalman] → Distância Suavizada
    ↓
[Cálculo de Risco] → Score de Risco
    ↓
[Lógica de Narração] → Alertas Sonoros
    ↓
[Visualização] → Frame Anotado
```

## Performance

### Otimizações
- Use YOLO11n (nano) para velocidade máxima
- Use MiDAS_small ou DPT_Small para profundidade rápida
- Execute em GPU se disponível
- Reduza resolução do vídeo se necessário

## Referências

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **MiDAS**: https://github.com/isl-org/MiDAS
- **ByteTrack**: https://github.com/ifzhang/ByteTrack
- **Filtro de Kalman**: https://www.kalmanfilter.net/
