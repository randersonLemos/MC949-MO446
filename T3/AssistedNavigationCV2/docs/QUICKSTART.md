# 🚀 Guia Rápido de Início

## ✅ Instalação Completa!

Todas as dependências foram instaladas com sucesso. O sistema está pronto para uso!

## 🎯 Como Usar

### Opção 1: Script Rápido
```bash
./run.sh
```

### Opção 2: Comando Direto
```bash
source venv/bin/activate
python main.py
```

## 🎮 Controles

Durante a execução:
- **`q`** - Sair do sistema
- **`d`** - Alternar visualização do mapa de profundidade

## ⚙️ Configurações

Edite `main.py` para ajustar:

```python
# Fonte de vídeo
VIDEO_SOURCE = 0  # 0 = webcam, ou "video.mp4" para arquivo

# Modelo YOLOv8
model_name='yolov8n.pt'  # n=nano (rápido), s=small, m=medium, l=large, x=xlarge

# Modelo MiDAS
model_type='DPT_Small'  # Small (rápido), Hybrid, Large (preciso)

# Threshold de confiança
conf_threshold=0.3  # 0.0 a 1.0

# Visualização
SHOW_VISUALIZATION = True  # False para rodar sem interface
SHOW_DEPTH_MAP = False     # True para mostrar mapa de profundidade
```

## 🔊 Text-to-Speech (Opcional)

O TTS está configurado mas precisa do espeak no Linux:

```bash
# Ubuntu/Debian
sudo apt-get install espeak

# Arch Linux
sudo pacman -S espeak-ng
```

**Sem espeak**: Os alertas serão mostrados no console (funcionamento normal).

## 📊 Status Atual

✅ **Instalado e Funcionando:**
- OpenCV
- NumPy
- PyTorch (CPU)
- TorchVision
- Ultralytics (YOLOv8)
- pyttsx3
- SciPy
- timm

⚠️ **Observações:**
- **GPU**: Não detectada - usando CPU
  - Para GPU: Reinstale PyTorch com suporte CUDA
  - Comando: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- **TTS**: Funcionando via console (espeak não instalado)

## 🎥 Testando com Vídeo

Se não tiver webcam, baixe um vídeo de teste:

```bash
# Exemplo: vídeo de tráfego urbano
# Edite main.py e mude:
VIDEO_SOURCE = "caminho/para/video.mp4"
```

## 📈 Performance Esperada

### CPU (sem GPU):
- **YOLOv8n**: ~10-15 FPS
- **MiDAS Small**: ~3-5 FPS
- **Pipeline completo**: ~2-4 FPS

### GPU (se disponível):
- **YOLOv8n**: ~80-100 FPS
- **MiDAS Small**: ~25-30 FPS
- **Pipeline completo**: ~20-25 FPS

## 🔧 Solução de Problemas

### Erro: "No module named 'XXX'"
```bash
source venv/bin/activate
pip install XXX
```

### Erro: "Can't open camera"
```bash
# Verificar câmeras disponíveis
ls /dev/video*

# Testar com opencv
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERRO')"
```

### Sistema muito lento
1. Use modelo menor: `yolov8n.pt` (mais rápido)
2. Reduza resolução da câmera
3. Desative mapa de profundidade: `SHOW_DEPTH_MAP = False`
4. Use MiDAS menor: `model_type='MiDaS_small'`

### Muitos alertas
```python
# Em main.py, após inicialização:
narration_logic.set_alert_cooldown(5.0)  # 5 segundos entre alertas
```

## 📝 Próximos Passos

1. **Testar sistema básico**: `python main.py`
2. **Ajustar sensibilidade** em `risk_calculator.py`
3. **Personalizar mensagens** em `narration_logic.py`
4. **Otimizar performance** conforme necessário

## 📚 Documentação Completa

Veja `README_IMPLEMENTATION.md` para detalhes sobre:
- Arquitetura do sistema
- Descrição de cada módulo
- Personalização avançada
- Pipeline de processamento

## 🐛 Reportar Problemas

Se encontrar erros, verifique:
1. Ambiente virtual ativado: `echo $VIRTUAL_ENV`
2. Todas as dependências: `python test_installation.py`
3. Câmera funcionando: `ls /dev/video*`

---

**Pronto para começar!** Execute `./run.sh` ou `python main.py` 🚀
