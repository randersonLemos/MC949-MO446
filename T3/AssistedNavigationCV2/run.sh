#!/bin/bash

echo "🚀 SISTEMA DE NAVEGAÇÃO ASSISTIDA COM VISÃO COMPUTACIONAL"
printf '=%.0s' {1..60}
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "main.py" ]; then
    echo "❌ Erro: Execute este script a partir do diretório do projeto"
    echo "   Diretório atual: $(pwd)"
    echo "   Execute: cd /home/unjun/Desktop/VSCode/MC949/T3/AssistedNavigationCV"
    exit 1
fi

# Ativar/instalar ambiente virtual Python 3.11 a partir de requirements.txt
VENV_DIR=".venv"
REQ_FILE="requirements.txt"

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "📦 Preparando ambiente virtual em ./$VENV_DIR ..."
    if [ ! -d "$VENV_DIR" ]; then
        # escolher intérprete Python (preferir python3.11)
        if command -v python3.11 >/dev/null 2>&1; then
            PYTHON=python3.11
        elif command -v python3 >/dev/null 2>&1; then
            PYTHON=python3
        elif command -v python >/dev/null 2>&1; then
            PYTHON=python
        else
            echo "❌ Nenhum interpretador Python encontrado. Instale Python 3.11/3.x"
            exit 1
        fi

        echo "⚙️  Criando venv com $PYTHON ..."
        $PYTHON -m venv "$VENV_DIR" || { echo "❌ Falha ao criar venv"; exit 1; }
        echo "✅ Venv criado: $VENV_DIR"
    fi

    # Ativar venv
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        echo "✅ Ambiente virtual ativado: ${VIRTUAL_ENV:-$VENV_DIR}"

        # Instalar dependências se houver requirements.txt
        if [ -f "$REQ_FILE" ]; then
            echo "⬇️  Instalando dependências de $REQ_FILE ..."
            pip install --upgrade pip wheel setuptools >/dev/null 2>&1 || true
            if ! pip install -r "$REQ_FILE"; then
                echo "❌ Falha ao instalar dependências a partir de $REQ_FILE"
                exit 1
            fi
            echo "✅ Dependências instaladas"
        else
            echo "⚠️  Arquivo $REQ_FILE não encontrado, pulando instalação de dependências"
        fi
    else
        echo "❌ Arquivo de ativação não encontrado em $VENV_DIR/bin/activate"
        echo "   Verifique permissões ou crie o venv manualmente"
        exit 1
    fi
else
    echo "✅ Ambiente virtual já ativo: $VIRTUAL_ENV"
fi

echo ""

# Verificar modelos necessários
echo "🔍 Verificando modelos necessários..."

# Verificar modelo YOLO
if [ -f "models/yolo11n.pt" ]; then
    echo "✅ Modelo YOLO encontrado: models/yolo11n.pt"
else
    echo "⚠️  Modelo YOLO não encontrado em models/yolo11n.pt"
    echo "   O modelo será baixado automaticamente na primeira execução"
fi

# Verificar modelos TTS Piper
if [ -f "models/pt_BR-cadu-medium.onnx" ] && [ -f "models/pt_BR-cadu-medium.onnx.json" ]; then
    echo "✅ Modelos TTS Piper encontrados"
else
    echo "⚠️  Modelos TTS Piper não encontrados em models/"
    echo "   Verifique se os arquivos pt_BR-cadu-medium.onnx e .json estão na pasta models/"
fi

echo ""
echo "⌨️  CONTROLES DURANTE EXECUÇÃO:"
echo "   'q' ou ESC: Sair do sistema"
echo "   'r': Ativar/desativar visualização ROI"
echo ""

# Verificar dependências principais
echo "🔧 Verificando dependências..."
python3 -c "
import sys
required_modules = ['cv2', 'torch', 'ultralytics', 'numpy', 'piper']
missing = []

for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing.append(module)
        print(f'❌ {module} - NÃO ENCONTRADO')

if missing:
    print(f'\n⚠️  Módulos faltando: {missing}')
    print('   Execute: pip install opencv-python torch ultralytics numpy piper-tts')
    sys.exit(1)
else:
    print('\n✅ Todas as dependências encontradas')
" || exit 1

echo ""
echo "🚀 Iniciando Sistema de Navegação Assistida..."
echo "   Processamento: YOLO + MiDaS + Piper TTS"
echo "   Modo: Detecção de obstáculos com alerta por voz"
echo ""

echo "📹 SELEÇÃO DE FONTE DE VÍDEO"
printf '=%.0s' {1..60}
echo ""
echo "Opções disponíveis:"
echo "  0  - Webcam padrão"

if [ -d "videos/" ]; then
    echo ""
    echo "📁 Vídeos de teste disponíveis:"
    video_files=("videos/bicicleta.mp4" "videos/carro.mp4")
    counter=1
    any=false
    for video in "${video_files[@]}"; do
        if [ -f "$video" ]; then
            size=$(du -h "$video" | cut -f1)
            echo "  $counter  - $(basename "$video") ($size)"
            ((counter++))
            any=true
        fi
    done
    if [ "$any" = false ]; then
        echo "   (nenhum dos vídeos bicicleta.mp4 ou carro.mp4 encontrados em videos/)"
    fi
fi

echo ""
echo "  c  - Arquivo customizado (digite o caminho)"
echo "  Enter - Usar configuração atual do config.py"
echo ""

# Ler escolha do usuário
read -p "📝 Escolha uma opção: " choice

# Processar escolha
case "$choice" in
    "")
        echo "✅ Usando configuração atual do config.py"
        video_source=""
        ;;
    "0")
        echo "✅ Usando webcam padrão"
        video_source="0"
        ;;
    "1")
        if [ -f "videos/bicicleta.mp4" ]; then
            echo "✅ Usando vídeo: bicicleta.mp4"
            video_source="videos/bicicleta.mp4"
        else
            echo "❌ Vídeo bicicleta.mp4 não encontrado em videos/"
            echo "   Usando configuração padrão"
            video_source=""
        fi
        ;;
    "2")
        if [ -f "videos/carro.mp4" ]; then
            echo "✅ Usando vídeo: carro.mp4"
            video_source="videos/carro.mp4"
        else
            echo "❌ Vídeo carro.mp4 não encontrado em videos/"
            echo "   Usando configuração padrão"
            video_source=""
        fi
        ;;
    "c"|"C")
        read -p "📁 Digite o caminho do arquivo de vídeo: " custom_path
        if [ -f "$custom_path" ]; then
            echo "✅ Usando arquivo: $custom_path"
            video_source="$custom_path"
        else
            echo "❌ Arquivo não encontrado: $custom_path"
            echo "   Usando configuração padrão"
            video_source=""
        fi
        ;;
    *)
        echo "❌ Opção inválida, usando configuração padrão"
        video_source=""
        ;;
esac

echo ""

# Executar sistema principal com fonte de vídeo selecionada
if [ -n "$video_source" ]; then
    echo "🎬 Executando com fonte: $video_source"
    # Método 1: Passar via argumento de linha de comando
    python3 -u main.py --video-source "$video_source"
    
    # Método 2 (alternativo): Usar variável de ambiente
    # export VIDEO_SOURCE="$video_source"
    # python3 -u main.py
else
    echo "🎬 Executando com configuração padrão"
    python3 -u main.py
fi

# Verificar código de saída
exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Sistema encerrado com sucesso"
else
    echo "❌ Sistema encerrado com erro (código: $exit_code)"
fi